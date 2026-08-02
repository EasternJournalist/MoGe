from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

def wrap_module_with_gradient_checkpointing(module: nn.Module):
    from torch.utils.checkpoint import checkpoint
    class _CheckpointingWrapper(module.__class__):
        _restore_cls = module.__class__
        def forward(self, *args, **kwargs):
            return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)
        
    module.__class__ = _CheckpointingWrapper
    return module


def unwrap_module_with_gradient_checkpointing(module: nn.Module):
    module.__class__ = module.__class__._restore_cls


def sync_ddp_hook(state, bucket: torch.distributed.GradBucket) -> torch.futures.Future[torch.Tensor]:
    group_to_use = torch.distributed.group.WORLD
    world_size = group_to_use.size()
    grad = bucket.buffer()
    grad.div_(world_size)
    torch.distributed.all_reduce(grad, group=group_to_use)
    fut = torch.futures.Future()
    fut.set_result(grad)
    return fut


def wrap_module_with_autocast(module: nn.Module, **autocast_kwargs):
    class _AutocastWrapper(module.__class__):
        _restore_cls = module.__class__
        is_autocast_wrapper = True
        def forward(self, *args, **kwargs):
            with torch.autocast(**autocast_kwargs):
                return super().forward(*args, **kwargs)

    module.__class__ = _AutocastWrapper
    return module


def unwrap_module(module: nn.Module):
    if hasattr(module.__class__, '_restore_cls'):
        module.__class__ = module.__class__._restore_cls
