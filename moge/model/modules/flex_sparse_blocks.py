from typing import Literal, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .utils import zero_module

from flex_gemm.nn import (
    SubmanifoldConv3d,
    SparsePool3d,
    SparseUpsample3d,
    SparsePixelShuffle3d,
    SparsePixelShuffle,
    SparsePixelUnshuffle3d,
    SparsePixelUnshuffle,
)


def _pick_spconv_algorithm(in_channels: int, out_channels: int) -> str:
    """Pick the submanifold-conv index-GEMM variant by channel width.

    Wide GEMMs (max(in, out) >= 128) benefit from `masked_implicit_gemm`;
    narrow ones stay on the default `implicit_gemm`. Threshold determined
    empirically on v3_4 with model_channels=[32, 64, 128, 256, 1024].
    """
    if max(in_channels, out_channels) >= 128:
        return "masked_implicit_gemm"
    return "implicit_gemm"


def make_conv3d(in_channels: int, out_channels: int, kernel_size: int = 3) -> SubmanifoldConv3d:
    return SubmanifoldConv3d(
        in_channels, out_channels, kernel_size,
        algorithm=_pick_spconv_algorithm(in_channels, out_channels),
    )


def get_activation(activation: Literal["relu", "silu"]) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    elif activation == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation: {activation}")


def _with_channels(shape: torch.Size, channels: int) -> torch.Size:
    return torch.Size(list(shape[:-1]) + [channels])


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _ceil_downsample_shape(shape: torch.Size, sparse_dim: int, factor: int) -> torch.Size:
    sparse_shape = list(shape[:sparse_dim])
    dense_shape = list(shape[sparse_dim:])
    for dim in range(sparse_dim - 3, sparse_dim):
        sparse_shape[dim] = _ceil_div(int(sparse_shape[dim]), factor)
    return torch.Size([*sparse_shape, *dense_shape])


class PointwiseBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, activation: nn.Module):
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch)
        self.act = activation
        self.use_checkpoint = False

    def _forward(self, feats, coords, shape, neighbor_cache=None):
        return self.act(self.linear(feats)), neighbor_cache

    def forward(self, feats, coords, shape, neighbor_cache=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, feats, coords, shape, neighbor_cache,
                use_reentrant=False,
            )
        return self._forward(feats, coords, shape, neighbor_cache)


class SparseResBlock3d(nn.Module):
    def __init__(self, channels: int, out_channels: int = None,
                 norm: bool = True, activation: Literal["silu", "relu"] = "silu", norm2: bool = False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = False

        self.norm1 = nn.LayerNorm(channels, elementwise_affine=True, eps=1e-6) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(self.out_channels, elementwise_affine=False, eps=1e-6) if (norm and norm2) else nn.Identity()
        self.activation_fn = F.silu if activation == "silu" else F.relu

        self.conv1 = make_conv3d(channels, self.out_channels)
        self.conv2 = zero_module(make_conv3d(self.out_channels, self.out_channels))
        self.skip_connection = (
            nn.Linear(channels, self.out_channels)
            if channels != self.out_channels else nn.Identity()
        )

    def _forward(self, feats, coords, shape, neighbor_cache=None):
        h = self.activation_fn(self.norm1(feats).type_as(feats))
        h, neighbor_cache = self.conv1(h, coords, shape, neighbor_cache=neighbor_cache)
        h = self.activation_fn(self.norm2(h).type_as(h))
        h, neighbor_cache = self.conv2(h, coords, shape, neighbor_cache=neighbor_cache)
        return h + self.skip_connection(feats), neighbor_cache

    def forward(self, feats, coords, shape, neighbor_cache=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, feats, coords, shape, neighbor_cache,
                use_reentrant=False,
            )
        return self._forward(feats, coords, shape, neighbor_cache)


class SparseResBlockDownsample3d(nn.Module):
    def __init__(self, channels: int, out_channels: int = None,
                 downsample_factor: int = 2,
                 norm: bool = True, activation: Literal["silu", "relu"] = "silu"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.downsample_factor = downsample_factor
        self.use_checkpoint = False

        self.norm1 = nn.LayerNorm(channels, elementwise_affine=True, eps=1e-6) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(self.out_channels, elementwise_affine=False, eps=1e-6) if norm else nn.Identity()
        self.activation_fn = F.silu if activation == "silu" else F.relu

        self.conv1 = make_conv3d(channels, self.out_channels)
        self.conv2 = zero_module(make_conv3d(self.out_channels, self.out_channels))
        self.skip_connection = (
            nn.Linear(channels, self.out_channels)
            if channels != self.out_channels else nn.Identity()
        )
        self.pool = SparsePool3d(
            kernel_size=downsample_factor,
            stride=downsample_factor,
            reduce="mean",
        )

    def _forward(self, feats, coords, shape, conv_cache=None):
        h = self.activation_fn(self.norm1(feats).type_as(feats))
        ds_shape_hint = _ceil_downsample_shape(shape, coords.shape[1], self.downsample_factor)
        h, ds_coords, ds_shape, down_cache = self.pool(h, coords, shape, output_shape=ds_shape_hint)
        feats_ds, _, _, _ = self.pool(
            feats, coords, shape,
            output_coords=ds_coords, output_shape=ds_shape,
            neighbor_cache=down_cache,
        )
        # `conv_cache` (if given) is the submanifold-conv neighborhood for the
        # post-pool ds_coords at this block's output level; conv1 will build it
        # lazily if None.
        h, conv_cache = self.conv1(h, ds_coords, ds_shape, neighbor_cache=conv_cache)
        h = self.activation_fn(self.norm2(h).type_as(h))
        h, conv_cache = self.conv2(h, ds_coords, ds_shape, neighbor_cache=conv_cache)
        h = h + self.skip_connection(feats_ds)
        return h, ds_coords, _with_channels(ds_shape, self.out_channels), down_cache, conv_cache

    def forward(self, feats, coords, shape, conv_cache=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, feats, coords, shape, conv_cache,
                use_reentrant=False,
            )
        return self._forward(feats, coords, shape, conv_cache)


class SparseResBlockUpsample3d(nn.Module):
    def __init__(self, channels: int, out_channels: int = None,
                 upsample_factor: int = 2,
                 norm: bool = True, activation: Literal["silu", "relu"] = "silu"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = False

        self.norm1 = nn.LayerNorm(channels, elementwise_affine=True, eps=1e-6) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(self.out_channels, elementwise_affine=False, eps=1e-6) if norm else nn.Identity()
        self.activation_fn = F.silu if activation == "silu" else F.relu

        self.conv1 = make_conv3d(channels, self.out_channels)
        self.conv2 = zero_module(make_conv3d(self.out_channels, self.out_channels))
        self.skip_connection = (
            nn.Linear(channels, self.out_channels)
            if channels != self.out_channels else nn.Identity()
        )
        self.upsample = SparseUpsample3d(
            scale_factor=upsample_factor,
            mode="nearest",
        )

    def _forward(self, feats, coords, shape, target_coords, target_shape,
                 up_cache=None, conv_cache=None):
        h = self.activation_fn(self.norm1(feats).type_as(feats))
        skip_feats = self.skip_connection(feats)
        h_and_skip = torch.cat([h, skip_feats], dim=-1)
        h_and_skip, up_coords, up_shape, _ = self.upsample(
            h_and_skip,
            coords,
            _with_channels(shape, h_and_skip.shape[-1]),
            output_coords=target_coords,
            output_shape=_with_channels(target_shape, h_and_skip.shape[-1]),
            neighbor_cache=up_cache,
        )
        h, skip_feats_up = h_and_skip.split([self.channels, self.out_channels], dim=-1)
        # `conv_cache` (if given) is the submanifold-conv neighborhood for
        # up_coords at this block's output level; conv1 will build it lazily
        # if None.
        h, conv_cache = self.conv1(h, up_coords, up_shape, neighbor_cache=conv_cache)
        h = self.activation_fn(self.norm2(h).type_as(h))
        h, conv_cache = self.conv2(h, up_coords, up_shape, neighbor_cache=conv_cache)
        h = h + skip_feats_up
        return h, up_coords, _with_channels(up_shape, self.out_channels), conv_cache

    def forward(self, feats, coords, shape, target_coords, target_shape,
                up_cache=None, conv_cache=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, feats, coords, shape, target_coords, target_shape,
                up_cache, conv_cache,
                use_reentrant=False,
            )
        return self._forward(feats, coords, shape, target_coords, target_shape,
                             up_cache, conv_cache)


class PixelUnshuffleDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, factor_tuple: Tuple[int, ...]):
        super().__init__()
        ch_factor = math.prod(factor_tuple)
        if len(set(factor_tuple)) == 1:
            self.unshuffle = SparsePixelUnshuffle3d(factor_tuple[0])
        else:
            self.unshuffle = SparsePixelUnshuffle(factor_tuple)
        self.linear = nn.Linear(in_ch * ch_factor, out_ch)

    def forward(self, feats, coords, shape):
        feats, coords, shape, down_cache = self.unshuffle(feats, coords, shape)
        feats = self.linear(feats)
        return feats, coords, _with_channels(shape, feats.shape[-1]), down_cache


class PixelShuffleUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, factor_tuple: Tuple[int, ...]):
        super().__init__()
        ch_factor = math.prod(factor_tuple)
        self.linear = nn.Linear(in_ch, out_ch * ch_factor)
        if len(set(factor_tuple)) == 1:
            self.shuffle = SparsePixelShuffle3d(factor_tuple[0])
        else:
            self.shuffle = SparsePixelShuffle(factor_tuple)

    def forward(self, feats, coords, shape, target_coords, target_shape, up_cache=None):
        feats = self.linear(feats)
        shape = _with_channels(shape, feats.shape[-1])
        feats, coords, shape, _ = self.shuffle(
            feats, coords, shape,
            output_coords=target_coords, output_shape=target_shape,
            neighbor_cache=up_cache,
        )
        return feats, coords, shape


class PoolDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, factor: int):
        super().__init__()
        self.factor = factor
        self.pool = SparsePool3d(kernel_size=factor, stride=factor, reduce="mean")
        self.linear = nn.Linear(in_ch, out_ch)

    def forward(self, feats, coords, shape):
        output_shape = _ceil_downsample_shape(shape, coords.shape[1], self.factor)
        feats, coords, shape, down_cache = self.pool(feats, coords, shape, output_shape=output_shape)
        feats = self.linear(feats)
        return feats, coords, _with_channels(shape, feats.shape[-1]), down_cache


class NearestUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, factor: int):
        super().__init__()
        self.upsample = SparseUpsample3d(scale_factor=factor, mode="nearest")
        self.linear = nn.Linear(in_ch, out_ch)

    def forward(self, feats, coords, shape, target_coords, target_shape, up_cache=None):
        feats = self.linear(feats)
        shape = _with_channels(shape, feats.shape[-1])
        feats, coords, shape, _ = self.upsample(
            feats, coords, shape,
            output_coords=target_coords, output_shape=target_shape,
            neighbor_cache=up_cache,
        )
        return feats, coords, shape