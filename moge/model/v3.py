from typing import *
from numbers import Number

import torch
import torch.nn.functional as F
import utils3d

from ..utils.geometry_torch import normalized_view_plane_uv
from .v2 import MoGeModel as MoGeModelV2
from .modules import Sparse3DUNet


class MoGeModel(MoGeModelV2):
    def __init__(
        self,
        encoder: Dict[str, Any],
        neck: Dict[str, Any],
        points_head: Dict[str, Any] = None,
        mask_head: Dict[str, Any] = None,
        normal_head: Dict[str, Any] = None,
        scale_head: Dict[str, Any] = None,
        remap_output: Literal['linear', 'sinh', 'exp', 'sinh_exp'] = 'exp',
        num_tokens_range: List[int] = [1200, 3600],
        refiner: Optional[Dict[str, Any]] = None,
        **deprecated_kwargs,
    ):
        super().__init__(
            encoder=encoder,
            neck=neck,
            points_head=points_head,
            mask_head=mask_head,
            normal_head=normal_head,
            scale_head=scale_head,
            remap_output=remap_output,
            num_tokens_range=num_tokens_range,
            **deprecated_kwargs,
        )
        self.encoder_patch_size: int = self.encoder.backbone.patch_size

        if refiner is not None:
            refiner_cfg = dict(refiner)
            self.refiner_depth_resolution: float = refiner_cfg.pop('depth_resolution', 256)
            self.refiner = Sparse3DUNet(**refiner_cfg)
        else:
            print("Warning: refiner is not enabled.")

    def enable_refiner_gradient_checkpointing(self):
        self.refiner.enable_gradient_checkpointing()

    def enable_refiner_mixed_precision(self, dtype: torch.dtype = torch.bfloat16):
        from .utils import wrap_module_with_autocast, unwrap_module
        if getattr(self.refiner.__class__, 'is_autocast_wrapper', False):
            unwrap_module(self.refiner)
        wrap_module_with_autocast(self.refiner, device_type='cuda', dtype=dtype)

    def _refine_coord_to_points(self, old_coord: torch.Tensor, new_logz: torch.Tensor) -> torch.Tensor:
        uv = old_coord[..., :2]
        return torch.cat([uv, new_logz.unsqueeze(-1)], dim=-1)

    def _voxelize(
        self,
        point_coord: torch.Tensor,
        uv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Size, torch.Tensor]:
        """
        Convert dense point coordinates to a sparse representation.

        - point_coord: [B, H, W, 3] at (x/z, y/z, logz).
        - uv:   [H, W, 2] UV at the point-map resolution,

        Returns (feats, coords, shape, logz):
        - feats:  (M, in_channels) input features ([uv, logz] or [logz]).
        - coords: (M, 4) int32, columns (batch, i, j, z_bin).
        - shape:  Size([B, H, W, z_extent, in_channels]).
        - logz:   [B, H, W] dense log-depth (for the residual update).
        """
        if point_coord.ndim != 4 or point_coord.shape[-1] != 3:
            raise ValueError(f"point_coord must be [B, H, W, 3], got {point_coord.shape}")

        bsz, height, width, _ = point_coord.shape
        device = point_coord.device

        logz = point_coord[..., 2]
        zq = torch.round(logz * self.refiner_depth_resolution).long()
        z_offset = zq.amin(dim=(1, 2), keepdim=True)
        z_idx = zq - z_offset
        z_extent = z_idx.amax().item() + 1

        i = torch.arange(height, device=device, dtype=torch.long).view(1, height, 1).expand(bsz, height, width)
        j = torch.arange(width, device=device, dtype=torch.long).view(1, 1, width).expand(bsz, height, width)
        batch = torch.arange(bsz, device=device, dtype=torch.long).view(bsz, 1, 1).expand(bsz, height, width)

        coords = torch.stack([batch, i, j, z_idx], dim=-1).reshape(-1, 4).to(torch.int32)
        feats = torch.cat([uv.unsqueeze(0).expand(bsz, -1, -1, -1), logz.unsqueeze(-1)], dim=-1).reshape(-1, 3)
        shape = torch.Size([bsz, height, width, z_extent, feats.shape[-1]])
        return feats, coords, shape, logz

    def _refine_logz(
        self,
        point_coord: torch.Tensor,
        encoder_feature: torch.Tensor,
        uv: torch.Tensor,
        return_delta_z: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, height, width, _ = point_coord.shape
        feats, coords, shape, logz = self._voxelize(point_coord, uv)
        out = self.refiner(feats, coords, shape, encoder_feature)
        out_logz = out.squeeze(-1).reshape(bsz, height, width)
        refined_logz = logz + out_logz
        if return_delta_z:
            delta_z = (torch.exp(refined_logz) - torch.exp(logz)).detach()
        else:
            delta_z = None
        return refined_logz, delta_z

    def forward(
        self,
        image: torch.Tensor,
        num_tokens: Union[int, torch.LongTensor],
        refine_steps: int = 3,
        refiner_detach_backbone: bool = True,
        detach_refine_coords: bool = False,
        return_delta_z: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, img_h, img_w = image.shape
        device, dtype = image.device, image.dtype

        aspect_ratio = img_w / img_h
        base_h, base_w = (num_tokens / aspect_ratio) ** 0.5, (num_tokens * aspect_ratio) ** 0.5
        if isinstance(base_h, torch.Tensor):
            base_h, base_w = base_h.round().long(), base_w.round().long()
        else:
            base_h, base_w = round(base_h), round(base_w)

        # Backbones encoding
        feat_h, feat_w = base_h * self.encoder_patch_size, base_w * self.encoder_patch_size
        features, cls_token = self.encoder(image, base_h, base_w, return_class_token=True)
        features = [features, None, None, None, None]

        # Concat UVs for aspect ratio input
        uv_for_refiner: Optional[torch.Tensor] = None
        for level in range(5):
            uv = normalized_view_plane_uv(width=base_w * 2 ** level, height=base_h * 2 ** level, aspect_ratio=aspect_ratio, dtype=dtype, device=device)
            if level == 4:
                uv_for_refiner = uv
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)

            if features[level] is None:
                features[level] = uv
            else:
                features[level] = torch.concat([features[level], uv], dim=1)

        # Shared neck
        neck_features = self.neck(features)

        # Heads decoding
        raw_points = self.points_head(neck_features)[-1] if hasattr(self, 'points_head') else None
        normal, mask = (
            getattr(self, head)(neck_features)[-1] if hasattr(self, head) else None
            for head in ['normal_head', 'mask_head']
        )
        metric_scale = self.scale_head(cls_token) if hasattr(self, 'scale_head') else None

        # Resize
        resize_fn = lambda x: F.interpolate(x, (img_h, img_w), mode='bilinear', align_corners=False, antialias=False)
        normal, mask = (resize_fn(x) if x is not None else None for x in [normal, mask])

        def postprocess_points(points: torch.Tensor, hwc: bool, resize: bool) -> torch.Tensor:
            # input is BHW3 or B3HW at (x/z, y/z, logz). Output is BHW3 at (x, y, z).
            if hwc: # input is BHW3
                points = points.permute(0, 3, 1, 2) # to B3HW
            if resize:
                points = resize_fn(points)
            points = points.permute(0, 2, 3, 1) # to BHW3
            points = self._remap_points(points) # BHW3 at (x, y, z)
            return points

        # Process points and optionally refine them
        points_all: List[torch.Tensor] = []
        delta_z_all: List[torch.Tensor] = []
        if raw_points is not None: # raw_points is B3HW at (x/z, y/z, logz)
            points_all.append(postprocess_points(raw_points, hwc=False, resize=True))

            if refine_steps > 0:
                refiner_feature: torch.Tensor = features[0]

                current_points = raw_points.permute(0, 2, 3, 1) # BHW3 at (x/z, y/z, logz)
                for _ in range(refine_steps):
                    coord_for_refiner = current_points.detach()
                    feature_for_refiner = refiner_feature.detach() if refiner_detach_backbone else refiner_feature
                    refined_logz, delta_z = self._refine_logz(
                        coord_for_refiner, feature_for_refiner, uv_for_refiner, return_delta_z=return_delta_z,
                    )
                    current_points = self._refine_coord_to_points(coord_for_refiner, refined_logz)
                    points_all.append(postprocess_points(current_points, hwc=True, resize=True))
                    if return_delta_z:
                        delta_z_all.append(delta_z)

        # Remap
        if normal is not None:
            normal = normal.permute(0, 2, 3, 1)
            normal = F.normalize(normal, dim=-1)
        if mask is not None:
            mask = mask.squeeze(1).sigmoid()
        if metric_scale is not None:
            metric_scale = metric_scale.squeeze(1).exp()

        return_dict = {
            'points_all': points_all if len(points_all) > 0 else None,
            'delta_z_all': delta_z_all if len(delta_z_all) > 0 else None,
            'normal': normal,
            'mask': mask,
            'metric_scale': metric_scale,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        return return_dict

    @torch.inference_mode()
    def infer(
        self,
        image: torch.Tensor,
        num_tokens: int = None,
        resolution_level: int = 9,
        force_projection: bool = True,
        apply_mask: Literal[False, True, 'blend'] = True,
        fov_x: Optional[Union[Number, torch.Tensor]] = None,
        refine_steps: int = 3,
        use_fp16: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False
        image = image.to(dtype=self.dtype, device=self.device)

        original_height, original_width = image.shape[-2:]
        aspect_ratio = original_width / original_height

        # Determine the number of base tokens to use
        if num_tokens is None:
            min_tokens, max_tokens = self.num_tokens_range
            num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))

        # Forward pass
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_fp16 and self.dtype != torch.float16):
            output = self.forward(image, num_tokens=num_tokens, refine_steps=refine_steps)
        points_all, normal, mask, metric_scale = (output.get(k, None) for k in ['points_all', 'normal', 'mask', 'metric_scale'])

        # Always process the output in fp32 precision
        points_all = [p.float() for p in points_all] if points_all is not None else None
        normal, mask, metric_scale, fov_x = map(lambda x: x.float() if isinstance(x, torch.Tensor) else x, [normal, mask, metric_scale, fov_x])
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            if mask is not None:
                mask_binary = mask > 0.5
            else:
                mask_binary = None

            if points_all is not None:
                from ..utils.geometry_torch import recover_focal_shift

                # Per-step (focal, shift) recovery: refinement modifies logz which changes
                # the 3D shape, so focal recovered from each step's point map differs.
                # Jointly solving (focal, shift) on the same point map gives the optimal
                # affine->camera alignment for that step (matches v2_4's single-step logic).
                points_ref = points_all[-1]
                if fov_x is not None:
                    focal_fixed = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=points_ref.device, dtype=points_ref.dtype) / 2))
                    if focal_fixed.ndim == 0:
                        focal_fixed = focal_fixed[None].expand(points_ref.shape[0])
                else:
                    focal_fixed = None

                points_all_processed, depth_all, intrinsics_all = [], [], []
                for points in points_all:
                    if focal_fixed is None:
                        focal_i, shift_i = recover_focal_shift(points, mask_binary)
                    else:
                        focal_i = focal_fixed
                        _, shift_i = recover_focal_shift(points, mask_binary, focal=focal_i)
                    fx_i, fy_i = focal_i / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal_i / 2 * (1 + aspect_ratio ** 2) ** 0.5
                    intrinsics_i = utils3d.pt.intrinsics_from_focal_center(fx_i, fy_i, 0.5, 0.5)

                    points = points.clone()
                    points[..., 2] += shift_i[..., None, None]
                    depth = points[..., 2].clone()

                    if force_projection:
                        points = utils3d.pt.depth_map_to_point_map(depth, intrinsics=intrinsics_i)

                    if metric_scale is not None:
                        points *= metric_scale[:, None, None, None]
                        depth *= metric_scale[:, None, None]

                    points_all_processed.append(points)
                    depth_all.append(depth)
                    intrinsics_all.append(intrinsics_i)

                # Final intrinsics correspond to the final-step point map (the one used as
                # the canonical output `points` / `depth`).
                intrinsics = intrinsics_all[-1]

                # Build per-step masks so each step is self-consistently masked
                # against its own depth>0 (mirrors v2_4's `mask_binary &= points[..., 2] > 0`).
                if mask_binary is not None:
                    per_step_masks = [mask_binary & (d > 0) for d in depth_all]
                    mask_binary = per_step_masks[-1]
                else:
                    per_step_masks = None

                points_all = points_all_processed
                points = points_all[-1]
                depth = depth_all[-1]
            else:
                points_all = None
                depth_all = None
                intrinsics_all = None
                per_step_masks = None
                points, depth, intrinsics = None, None, None

            if apply_mask and per_step_masks is not None:
                points_all = [torch.where(m[..., None], p, torch.inf) for p, m in zip(points_all, per_step_masks)]
                depth_all = [torch.where(m, d, torch.inf) for d, m in zip(depth_all, per_step_masks)]
                points = points_all[-1]
                depth = depth_all[-1]
                normal = torch.where(mask_binary[..., None], normal, torch.zeros_like(normal)) if normal is not None else None

        return_dict = {
            'points': points,
            'points_all': points_all,
            'intrinsics': intrinsics,
            'intrinsics_all': intrinsics_all,
            'depth': depth,
            'depth_all': depth_all,
            'mask': mask_binary,
            'normal': normal,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        if omit_batch_dim:
            return_dict = {
                k: [item.squeeze(0) for item in v] if isinstance(v, list) else v.squeeze(0)
                for k, v in return_dict.items()
            }

        return return_dict
