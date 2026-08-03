import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import json
import math
import random
from typing import *
from concurrent.futures import ThreadPoolExecutor
import io
import gc
import traceback
from collections import deque
from datetime import timedelta
import numpy as np
import cv2
import torch
import torch.version
import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import set_seed
import utils3d
import click
from tqdm import tqdm
import git
from copy import deepcopy
import shutil
import warnings

from ..utils.tools import timeit
from moge.train.dataloader import TrainDataLoaderPipeline
from moge.train.losses import *
from .utils import (
    build_optimizer,
    build_lr_scheduler,
    to_device,
    get_raft_weight,
    materialize_log_records,
    to_log_scalar,
    write_optimizer_param_assignment_log,
)
from ..utils.tools import key_average, flatten_nested_dict
from .options import common_train_options
from .experiment import RunLogger, setup_accelerator
from .checkpoint import CheckpointSaver, load_checkpoint, restore_ma_buffer, restore_training_state
from .debug import DebugDumper
from .visualization import visualize_gt, visualize_predictions
from ..test.metrics import compute_metrics


warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
torch._dynamo.config.disable = True
torch.backends.cudnn.benchmark = False      # Varying input size, make sure cudnn benchmark is disabled


@click.command()
@common_train_options
@click.option('--refiner_gradient_checkpoint', type=bool, default=True, help='Use gradient checkpointing inside the sparse refiner')
@click.option('--refiner_bf16', type=bool, default=False, help='Wrap the sparse refiner with a bf16 autocast scope (only effective when --precision mixed_bf16; default is fp32 for the refiner)')
def main(
    config_path: str,
    experiment_name: str,
    workspace_path: str,
    base_checkpoint: Optional[str],
    checkpoint_path: str,
    batch_size_forward: int,
    gradient_accumulation_steps: int,
    backbone_gradient_checkpoint: bool,
    refiner_gradient_checkpoint: bool,
    refiner_bf16: bool,
    precision: str,
    enable_ema: bool,
    debug_mode: bool,
    num_iterations: int,
    checkpoint_every: int,
    rolling_checkpoint_every: int,
    log_every: int,
    vis_every: int,
    vis_gt: bool,
    num_vis_images: int,
    log_type: Tuple[str, ...],
    log_dir: Optional[str],
    gc_every: int,
    seed: Optional[int],
    wandb_project: str,
    max_invalid_batches: int,
    find_unused_parameters: bool,
    num_load_workers: int,
    num_process_workers: int,
):
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    refine_ratio = config['refine_ratio']
    if not 0.0 <= refine_ratio <= 1.0:
        raise ValueError(f"config['refine_ratio'] must be in [0, 1], got {refine_ratio}")

    accelerator, device, batch_size_total, workspace = setup_accelerator(
        gradient_accumulation_steps, find_unused_parameters, batch_size_forward, workspace_path,
    )
    logger = RunLogger(accelerator, log_type)
    logger.setup(
        workspace=workspace, config=config, experiment_name=experiment_name,
        log_dir=log_dir, wandb_project=wandb_project, batch_size_total=batch_size_total,
        extra_params={'refine_ratio': refine_ratio},
    )

    # Set seed
    if seed is not None:
        set_seed(seed, device_specific=True)

    # Initialize model
    print('Initialize model')
    with accelerator.local_main_process_first():
        from moge.model import import_model_class_by_version
        MoGeModel = import_model_class_by_version(config['model_version'])      
        model = MoGeModel(**config['model'])
    count_total_parameters = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {count_total_parameters}')

    # Set up EMA model
    if enable_ema and accelerator.is_main_process:
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: 0.999 * averaged_model_parameter + 0.001 * model_parameter
        ema_model = torch.optim.swa_utils.AveragedModel(model, device=accelerator.device, avg_fn=ema_avg_fn)

    # Set gradient checkpointing
    if backbone_gradient_checkpoint:
        model.enable_gradient_checkpointing()
    if refiner_gradient_checkpoint:
        model.refiner.enable_gradient_checkpointing()

    # Set precision
    if precision == 'fp32':
        pass
    elif precision == 'tf32':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = False
    elif precision == 'mixed_bf16':
        model.enable_mixed_precision()
    if refiner_bf16:
        model.refiner.enable_mixed_precision(torch.bfloat16)
        if precision != 'mixed_bf16':
            print(f"Warning: --refiner_bf16 is set but --precision is '{precision}'")

    # Initalize optimizer & lr scheduler
    optimizer = build_optimizer(model, config['optimizer'])
    lr_scheduler = build_lr_scheduler(optimizer, config['lr_scheduler'])

    count_grouped_parameters = [sum(p.numel() for p in param_group['params'] if p.requires_grad) for param_group in optimizer.param_groups]
    for i, count in enumerate(count_grouped_parameters):
        print(f'- Group {i}: {count} parameters')
    if accelerator.is_main_process:
        write_optimizer_param_assignment_log(
            model,
            optimizer,
            workspace,
        )

    # Attempt to load checkpoint; fall back to the base checkpoint when the workspace has none.
    checkpoint = load_checkpoint(checkpoint_path, workspace, accelerator, enable_ema)
    if checkpoint is None:
        checkpoint = load_checkpoint(base_checkpoint, workspace, accelerator, enable_ema)
    initial_step = restore_training_state(
        checkpoint, model, optimizer, lr_scheduler, ema_model if enable_ema and accelerator.is_main_process else None,
        accelerator, enable_ema,
    )
    del checkpoint

    model, optimizer = accelerator.prepare(model, optimizer)
    if torch.version.hip and isinstance(model, torch.nn.parallel.DistributedDataParallel):
        # Hacking potential gradient synchronization issue in ROCm backend
        from moge.model.utils import sync_ddp_hook
        model.register_comm_hook(None, sync_ddp_hook)

    def get_refine_accumulation_schedule(step: int) -> List[bool]:
        start_quota = math.floor(step * gradient_accumulation_steps * refine_ratio + 1e-9)
        end_quota = math.floor((step + 1) * gradient_accumulation_steps * refine_ratio + 1e-9)
        num_refine_accumulations = end_quota - start_quota
        if num_refine_accumulations <= 0:
            return [False] * gradient_accumulation_steps
        if num_refine_accumulations >= gradient_accumulation_steps:
            return [True] * gradient_accumulation_steps
        num_norefine_accumulations = gradient_accumulation_steps - num_refine_accumulations
        return [i >= num_norefine_accumulations for i in range(gradient_accumulation_steps)]

    # Initialize training data pipelines
    dataloader_seed = (seed + accelerator.process_index) if seed is not None else None
    with accelerator.local_main_process_first():
        refine_data_pipeline = TrainDataLoaderPipeline(deepcopy(config['refine_data']), batch_size_forward, workspace=workspace, num_load_workers=num_load_workers, num_process_workers=num_process_workers, seed=dataloader_seed)
        norefine_data_pipeline = TrainDataLoaderPipeline(deepcopy(config['norefine_data']), batch_size_forward, workspace=workspace, num_load_workers=num_load_workers, num_process_workers=num_process_workers, seed=dataloader_seed)

    # Restore data pipeline RNG state if resuming
    if initial_step > 0:
        for _pipeline_name, _pipeline in [('refine', refine_data_pipeline), ('norefine', norefine_data_pipeline)]:
            _pipeline_state_path = Path(workspace, 'checkpoint', 'data_pipeline', f'latest_{_pipeline_name}_data_pipeline_rank_{accelerator.process_index}.pt')
            if _pipeline_state_path.exists():
                _pipeline_state = torch.load(_pipeline_state_path, map_location='cpu', weights_only=False)
                _pipeline.load_state_dict(_pipeline_state)
                print(f"Restored {_pipeline_name} data pipeline state for rank {accelerator.process_index} from step {_pipeline_state.get('step', '?')}")
            else:
                print(f"Warning: No {_pipeline_name} data pipeline state found for rank {accelerator.process_index}, data ordering will restart from the beginning")

    records = []
    ma_buffer = restore_ma_buffer(workspace, initial_step, accelerator)

    model.train()

    with (
        refine_data_pipeline,
        norefine_data_pipeline,
        tqdm(initial=initial_step, total=num_iterations, desc='Training', disable=not accelerator.is_main_process) as pbar,
        ThreadPoolExecutor(max_workers=1) as save_checkpoint_executor,
    ):
        checkpoint_saver = CheckpointSaver(
            workspace=workspace, config=config, accelerator=accelerator, model=model,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            ema_model=ema_model if enable_ema and accelerator.is_main_process else None,
            enable_ema=enable_ema, ma_buffer=ma_buffer, executor=save_checkpoint_executor,
            pbar=pbar, num_iterations=num_iterations, checkpoint_every=checkpoint_every,
            rolling_checkpoint_every=rolling_checkpoint_every, initial_step=initial_step,
        )
        # Get some batches for visualization
        if accelerator.is_main_process:
            batches_for_vis: List[Dict[str, torch.Tensor]] = []
            num_vis_images = num_vis_images // 2 * 2 // batch_size_forward * batch_size_forward
            num_vis_batches = num_vis_images // batch_size_forward
            for _ in range(num_vis_batches // 2):
                batch = norefine_data_pipeline.get()
                batches_for_vis.append(batch)
            for _ in range(num_vis_batches - len(batches_for_vis)):
                batch = refine_data_pipeline.get()
                batches_for_vis.append(batch)

        # Visualize GT
        if vis_every > 0 and accelerator.is_main_process and vis_gt:
            visualize_gt(batches_for_vis, workspace, batch_size_forward, initial_step, logger)

        # Reset seed to avoid training on the same data when resuming training
        if seed is not None:
            set_seed(seed + initial_step, device_specific=True)   

        # Tags starting with 'nan_' are fatal and count toward the abort threshold; others
        # (e.g. large_grad_norm_*) are informational and capped to avoid filling disk.
        dumper = DebugDumper(
            workspace, accelerator,
            dump_grad_norm_above=config.get('dump_grad_norm_above', 10),
        )
        invalid_batch_encountered_times = 0
        refine_regression_tracker: Dict[Tuple, List[int]] = {}
        refine_regression_log: Dict[str, float] = {}
        delta_increase_tracker: Dict[Tuple, List[int]] = {}
        delta_increase_log: Dict[str, float] = {}
        error_decrease_tracker: Dict[Tuple, List[int]] = {}
        error_decrease_log: Dict[str, float] = {}

        # Training loop
        for i_step in range(initial_step, num_iterations):
            with timeit('Step', verbose=False) as timer_step:
                step_record_start = len(records)
                refine_accumulation_schedule = get_refine_accumulation_schedule(i_step)
                for i_accumulate in range(gradient_accumulation_steps):
                    use_refine_pipeline = refine_accumulation_schedule[i_accumulate]
                    active_data_pipeline = refine_data_pipeline if use_refine_pipeline else norefine_data_pipeline
                    # Load batch
                    with timeit('Load instance', verbose=False) as timer_load_instance:
                        batch = active_data_pipeline.get()
                        batch = to_device(batch, device)
                    records.append({'time/data': timer_load_instance.time})

                    is_invalid_batch = False

                    image, gt_depth, gt_normal, gt_mask_fin, gt_mask_inf, gt_intrinsics, label_type, is_metric, info = batch['image'], batch['depth'], batch['normal'], batch['depth_mask_fin'], batch['depth_mask_inf'], batch['intrinsics'], batch['label_type'], batch['is_metric'], batch['info']

                    current_batch_size = image.shape[0]

                    if all(label == 'invalid' for label in label_type):
                        is_invalid_batch = True
                        print(f"Rank {accelerator.process_index} all-invalid batch at step {i_step}, accumulation {i_accumulate}. Batch info: {info}")
                        invalid_batch_encountered_times += 1
                    
                    refine_steps = config['refine_steps'] if use_refine_pipeline else 0

                    gt_points = utils3d.pt.depth_map_to_point_map(gt_depth, intrinsics=gt_intrinsics)
                    gt_focal = 1 / (1 / gt_intrinsics[..., 0, 0] ** 2 + 1 / gt_intrinsics[..., 1, 1] ** 2) ** 0.5

                    with accelerator.accumulate(model):
                        # Forward
                        if i_step <= config.get('low_resolution_training_steps', 0):
                            num_tokens = config['model']['num_tokens_range'][0]
                        else:
                            num_tokens = random.Random(f'num_tokens-{seed}-{i_step}-{i_accumulate}').randint(*config['model']['num_tokens_range'])
                        
                        _detach_backbone = i_step < config['refiner_detach_backbone_until']
                        with timeit('Model forward', verbose=False) as timer_forward: # , sync=torch.cuda.synchronize
                            output = model(
                                image,
                                num_tokens=num_tokens,
                                refine_steps=refine_steps,
                                refiner_detach_backbone=_detach_backbone,
                                return_delta_z=True,
                            )
                        pred_points_all, delta_z_all = (output.get(k, None) for k in ['points_per_step', 'delta_z_per_update'])
                        pred_normal, pred_mask, pred_metric_scale = (output.get(k, None) for k in ['normal', 'mask', 'metric_scale'])

                        # Compute loss (per instance)
                        with timeit('Loss computation', verbose=False) as timer_loss_computation: # , sync=torch.cuda.synchronize
                            if is_invalid_batch:
                                loss = torch.tensor(0.0, device=device, requires_grad=True)
                            else:
                                loss_list, weight_list = [], []

                                for i in range(current_batch_size):
                                    mask_i = torch.isfinite(gt_points[i]).all(dim=-1)
                                    gt_points_i = torch.where(mask_i[..., None], gt_points[i], 1)
                                    gt_metric_scale = None
                                    step0_gt_metric_scale = None
                                    refine_step0_scale = None
                                    loss_dict, weight_dict, misc_dict, refine_stat_dict = {}, {}, {}, {}
                                    is_refine_instance = label_type[i] == "D"
                                    radial_loss_cache: Optional[RadialPartitionLocalLossCache] = None

                                    # points
                                    pred_points_i = [p[i] for p in pred_points_all]
                                    for pred_step, pred_points_iter in enumerate(pred_points_i):
                                        with torch.no_grad():
                                            misc_dict[f'monitoring_step_{pred_step}'] = monitoring(pred_points_iter.detach())
                                        if pred_step > 0:
                                            delta_stats = monitor_delta(delta_z_all[pred_step-1][i].detach())
                                            refine_stat_dict[f'delta_z_step_{pred_step}_'] = delta_stats
                                        for k, v in config['loss'][label_type[i]].get('points', {}).items():
                                            if pred_step not in v['apply_steps']:
                                                continue
                                            iter_key = f'{k}_step_{pred_step}' if pred_step > 0 else k

                                            if not is_refine_instance:
                                                weight_dict[iter_key] = v['weight']
                                            else:
                                                if config.get("loss_weight_balance", "mean") == "mean":
                                                    if _detach_backbone:
                                                        if pred_step == 0:
                                                            weight_dict[iter_key] = v['weight']
                                                        else:
                                                            weight_dict[iter_key] = v['weight'] / config['refine_steps']
                                                    else:
                                                        weight_dict[iter_key] = v['weight'] / len(v['apply_steps'])
                                                elif config["loss_weight_balance"] == "refine_steps_mean":
                                                    if pred_step == 0:
                                                        weight_dict[iter_key] = v['weight']
                                                    else:
                                                        weight_dict[iter_key] = v['weight'] / config['refine_steps']
                                                elif config["loss_weight_balance"] == "raft":
                                                    if _detach_backbone:
                                                        if pred_step == 0:
                                                            weight_dict[iter_key] = v['weight']
                                                        else:
                                                            weight_dict[iter_key] = v['weight'] * get_raft_weight(config['refine_steps'], pred_step - 1, config["raft_gamma"])
                                                    else:
                                                        weight_dict[iter_key] = v['weight'] * get_raft_weight(config['refine_steps'] + 1, pred_step, config["raft_gamma"])
                                                else:
                                                    raise ValueError(f"Unknown loss weight balance strategy: {config['loss_weight_balance']}")

                                            if v['function'] == 'affine_invariant_global_loss':
                                                # For refine steps (pred_step > 0), reuse the step-0 scale so the
                                                # alignment only solves the z-shift and the loss stays scale-sensitive.
                                                _fixed_scale = refine_step0_scale if pred_step > 0 else None
                                                loss_dict[iter_key], misc_dict[iter_key], gt_metric_scale, _ = affine_invariant_global_loss(
                                                    pred_points_iter,
                                                    gt_points_i,
                                                    mask_i,
                                                    fixed_scale=_fixed_scale,
                                                    **v['params'],
                                                )
                                                if pred_step == 0:
                                                    step0_gt_metric_scale = gt_metric_scale
                                                    # Detached so later refine steps pin their scale to step 0 without
                                                    # back-propagating step-k losses into the step-0 point map.
                                                    refine_step0_scale = gt_metric_scale.detach()
                                            elif v['function'] in {'radial_partition_local_loss', 'radial_partition_local_loss_rand_partition'}:
                                                scale_to_use = gt_metric_scale
                                                if scale_to_use is None:
                                                    raise RuntimeError(f"{v['function']} requires a preceding global scale loss in the same points config")
                                                if scale_to_use.dim() == 0:
                                                    scale_to_use = scale_to_use.unsqueeze(0)
                                                if radial_loss_cache is None:
                                                    radial_loss_cache = RadialPartitionLocalLossCache.from_inputs(gt_points_i, mask_i)
                                                radial_loss_fn = radial_partition_local_loss_rand_partition if v['function'] == 'radial_partition_local_loss_rand_partition' else radial_partition_local_loss
                                                loss_dict[iter_key], misc_dict[iter_key] = radial_loss_fn(
                                                    pred_points_iter,
                                                    gt_points_i,
                                                    mask_i,
                                                    scale_to_use,
                                                    packed=radial_loss_cache,
                                                    **v['params'],
                                                )
                                            elif v['function'] == 'edge_loss':
                                                loss_dict[iter_key], misc_dict[iter_key] = edge_loss(
                                                    pred_points_iter,
                                                    gt_points_i,
                                                    mask_i,
                                                )

                                    # normal
                                    for k, v in config['loss'][label_type[i]].get('normal', {}).items():
                                        weight_dict[k] = v['weight']
                                        if v['function'] == 'normal_map_loss':
                                            loss_dict[k], misc_dict[k] = normal_map_loss(pred_normal[i], gt_normal[i])
                                    
                                    # mask
                                    for k, v in config['loss'][label_type[i]].get('mask', {}).items():
                                        weight_dict[k] = v['weight']
                                        if v['function'] == 'mask_bce_loss':
                                            loss_dict[k], misc_dict[k] = mask_bce_loss(pred_mask[i], gt_mask_fin[i], gt_mask_inf[i])

                                    # metric_scale
                                    for k, v in config['loss'][label_type[i]].get('metric_scale', {}).items():
                                        weight_dict[k] = v['weight']
                                        if v['function'] == 'metric_scale_loss':
                                            if is_metric[i] and pred_metric_scale is not None and step0_gt_metric_scale is not None:
                                                loss_dict[k], misc_dict[k] = metric_scale_loss(pred_metric_scale[i], step0_gt_metric_scale.detach())

                                    weight_dict = {'.'.join(k): v for k, v in flatten_nested_dict(weight_dict).items()}
                                    loss_dict = {'.'.join(k): v for k, v in flatten_nested_dict(loss_dict).items()}
                                    loss_ = sum([weight_dict[k] * loss_dict[k] for k in loss_dict], start=torch.tensor(0.0, device=device))
                                    loss_list.append(loss_)
                                    
                                    # NaN loss check
                                    loss_finite_names, loss_finite_flags = [], []
                                    for loss_name, loss_value in loss_dict.items():
                                        loss_finite_names.append(loss_name)
                                        loss_finite_flags.append(torch.isfinite(loss_value.detach()).all())
                                    if loss_finite_flags:
                                        loss_finite_values = torch.stack(loss_finite_flags).cpu().tolist()
                                        for loss_name, is_finite in zip(loss_finite_names, loss_finite_values):
                                            if not is_finite:
                                                pbar.write(f'NaN loss in process {accelerator.process_index}, loss name: {loss_name}')
                                                dumper.add_reason(f'nan_loss_{loss_name}')

                                    misc_dict = {'.'.join(k): v for k, v in flatten_nested_dict(misc_dict).items()}
                                    refine_stat_dict = {'.'.join(k): v for k, v in flatten_nested_dict(refine_stat_dict).items()}
                                    loss_log_dict = {f"loss/{k}": v_ for k, v in loss_dict.items() if (v_ := to_log_scalar(v)) is not None}
                                    misc_log_dict = {f"misc/{k}": v_ for k, v in misc_dict.items() if (v_ := to_log_scalar(v)) is not None}
                                    refine_log_dict = {f"refine_stat/{k}": v_ for k, v in refine_stat_dict.items() if (v_ := to_log_scalar(v)) is not None}
                                    records.append({
                                        **loss_log_dict,
                                        **misc_log_dict,
                                        **refine_log_dict,
                                    })

                                    # Monitor refine step regression (only for refine instances)
                                    if is_refine_instance and accelerator.is_main_process:
                                        loss_dict_for_monitor, misc_dict_for_monitor = materialize_log_records([loss_dict, misc_dict])
                                        monitor_loss_by_step: Dict[str, Dict[int, float]] = {}
                                        for lk, lv in loss_dict_for_monitor.items():
                                            parts = lk.rsplit('_step_', 1)
                                            if len(parts) == 2:
                                                monitor_loss_by_step.setdefault(parts[0], {})[int(parts[1])] = lv
                                            else:
                                                # bare key = step 0
                                                monitor_loss_by_step.setdefault(lk, {})[0] = lv
                                        monitor_pairs = [(0, 1), (1, 2), (2, 3), (0, 3), (1, 3)]
                                        for name, step_vals in monitor_loss_by_step.items():
                                            for s_from, s_to in monitor_pairs:
                                                if s_from in step_vals and s_to in step_vals:
                                                    tracker_key = (name, s_from, s_to)
                                                    if tracker_key not in refine_regression_tracker:
                                                        refine_regression_tracker[tracker_key] = [0, 0]
                                                    refine_regression_tracker[tracker_key][1] += 1
                                                    if step_vals[s_to] > step_vals[s_from]:
                                                        refine_regression_tracker[tracker_key][0] += 1

                                        # Monitor misc delta increase and error decrease across refine steps
                                        misc_delta_by_step: Dict[str, Dict[int, float]] = {}
                                        misc_error_by_step: Dict[str, Dict[int, float]] = {}
                                        for mk, mv in misc_dict_for_monitor.items():
                                            dot_pos = mk.rfind('.')
                                            if dot_pos < 0:
                                                continue
                                            prefix = mk[:dot_pos]
                                            metric_name = mk[dot_pos + 1:]
                                            parts = prefix.rsplit('_step_', 1)
                                            if len(parts) == 2:
                                                base_name, step_str = parts
                                                step_num = int(step_str)
                                            else:
                                                base_name = parts[0]
                                                step_num = 0
                                            if metric_name == 'delta' or metric_name.startswith('delta_'):
                                                delta_key = base_name if metric_name == 'delta' else base_name + '.' + metric_name
                                                misc_delta_by_step.setdefault(delta_key, {})[step_num] = mv
                                            elif metric_name == 'truncated_error':
                                                misc_error_by_step.setdefault(base_name, {})[step_num] = mv

                                        for name, step_vals in misc_delta_by_step.items():
                                            for s_from, s_to in monitor_pairs:
                                                if s_from in step_vals and s_to in step_vals:
                                                    tracker_key = (name, s_from, s_to)
                                                    if tracker_key not in delta_increase_tracker:
                                                        delta_increase_tracker[tracker_key] = [0, 0]
                                                    delta_increase_tracker[tracker_key][1] += 1
                                                    if step_vals[s_to] > step_vals[s_from]:
                                                        delta_increase_tracker[tracker_key][0] += 1

                                        for name, step_vals in misc_error_by_step.items():
                                            for s_from, s_to in monitor_pairs:
                                                if s_from in step_vals and s_to in step_vals:
                                                    tracker_key = (name, s_from, s_to)
                                                    if tracker_key not in error_decrease_tracker:
                                                        error_decrease_tracker[tracker_key] = [0, 0]
                                                    error_decrease_tracker[tracker_key][1] += 1
                                                    if step_vals[s_to] < step_vals[s_from]:
                                                        error_decrease_tracker[tracker_key][0] += 1

                                loss = sum(loss_list) / len(loss_list)  # Average over the batch
                            records.append({'train/loss': to_log_scalar(loss)})

                        # Backward
                        with timeit('Backward', verbose=False) as timer_backward: # , sync=torch.cuda.synchronize
                            accelerator.backward(loss)

                        # Optimizer step
                        if accelerator.sync_gradients:
                            # Clip grad norm
                            grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                            records.append({'train/grad_norm': to_log_scalar(grad_norm)})
                            # Step only if grad is finite
                            grad_norm_is_finite = True if accelerator.scaler is not None else torch.isfinite(grad_norm.detach()).cpu().item()
                            if grad_norm_is_finite or accelerator.scaler is not None:
                                optimizer.step()
                            else:
                                pbar.write(f'Non-finite gradient norm {grad_norm} encountered in process {accelerator.process_index}, skip optimizer step.')
                                pbar.write(f'Batch info: {info}')
                                dumper.add_reason('nan_grad_norm')

                            # Extra dump trigger: large grad norm.
                            dumper.note_grad_norm(grad_norm, grad_norm_is_finite)

                        optimizer.zero_grad()

                        dumper.flush(i_step, i_accumulate, batch, output)

            records.append({'time/step': timer_step.time})

            lr_scheduler.step()

            # EMA update            
            if enable_ema and accelerator.is_main_process and accelerator.sync_gradients:
                ema_model.update_parameters(model)

            # Print raw loss values and refine loss regression stats every 100 steps
            if accelerator.is_main_process and i_step % 100 == 0:
                step_avg = key_average(materialize_log_records(records[step_record_start:]))
                loss_misc = {k: v for k, v in sorted(step_avg.items()) if k.startswith(('loss/',))}
                if loss_misc:
                    pbar.write(f'[Step {i_step}] ' + ' | '.join(f'{k}: {v:.6f}' for k, v in loss_misc.items()))
                if 'train/loss' in step_avg:
                    pbar.set_postfix({'loss': step_avg['train/loss']}, refresh=False)
            
                if i_step != initial_step:
                    if refine_regression_tracker:
                        pbar.write(f'[Step {i_step}] Refine loss decrease% monitor (bigger=better):')
                        names = sorted(set(n for n, _, _ in refine_regression_tracker))
                        name_width = max(len('loss'), *(len(name) for name in names))
                        pbar.write(f'  {"loss":<{name_width}s}  {"0->1":>7s}  {"1->2":>7s}  {"2->3":>7s}  {"0->3":>7s}  {"1->3":>7s}')
                        for name in names:
                            cells = []
                            for s_from, s_to in [(0, 1), (1, 2), (2, 3), (0, 3), (1, 3)]:
                                key = (name, s_from, s_to)
                                if key in refine_regression_tracker:
                                    n_inc, n_total = refine_regression_tracker[key]
                                    n_dec = n_total - n_inc
                                    pct = 100.0 * n_dec / n_total if n_total > 0 else 0.0
                                    cells.append(f'{pct:6.1f}%' if n_total > 0 else '   N/A')
                                    refine_regression_log[f'loss_decrease/{name}_{s_from}_to_{s_to}'] = pct
                                else:
                                    cells.append('      -')
                            pbar.write(f'  {name:<{name_width}s}  {"  ".join(cells)}')
                        refine_regression_tracker.clear()

                    if delta_increase_tracker:
                        pbar.write(f'[Step {i_step}] Misc delta increase% monitor (bigger=better):')
                        names = sorted(set(n for n, _, _ in delta_increase_tracker))
                        name_width = max(len('metric'), *(len(name) for name in names))
                        pbar.write(f'  {"metric":<{name_width}s}  {"0->1":>7s}  {"1->2":>7s}  {"2->3":>7s}  {"0->3":>7s}  {"1->3":>7s}')
                        for name in names:
                            cells = []
                            for s_from, s_to in [(0, 1), (1, 2), (2, 3), (0, 3), (1, 3)]:
                                key = (name, s_from, s_to)
                                if key in delta_increase_tracker:
                                    n_inc, n_total = delta_increase_tracker[key]
                                    pct = 100.0 * n_inc / n_total if n_total > 0 else 0.0
                                    cells.append(f'{pct:6.1f}%' if n_total > 0 else '   N/A')
                                    delta_increase_log[f'delta_increase/{name}_{s_from}_to_{s_to}'] = pct
                                else:
                                    cells.append('      -')
                            pbar.write(f'  {name:<{name_width}s}  {"  ".join(cells)}')
                        delta_increase_tracker.clear()

                    if error_decrease_tracker:
                        pbar.write(f'[Step {i_step}] Misc error decrease% monitor (bigger=better):')
                        names = sorted(set(n for n, _, _ in error_decrease_tracker))
                        name_width = max(len('metric'), *(len(name) for name in names))
                        pbar.write(f'  {"metric":<{name_width}s}  {"0->1":>7s}  {"1->2":>7s}  {"2->3":>7s}  {"0->3":>7s}  {"1->3":>7s}')
                        for name in names:
                            cells = []
                            for s_from, s_to in [(0, 1), (1, 2), (2, 3), (0, 3), (1, 3)]:
                                key = (name, s_from, s_to)
                                if key in error_decrease_tracker:
                                    n_dec, n_total = error_decrease_tracker[key]
                                    pct = 100.0 * n_dec / n_total if n_total > 0 else 0.0
                                    cells.append(f'{pct:6.1f}%' if n_total > 0 else '   N/A')
                                    error_decrease_log[f'error_decrease/{name}_{s_from}_to_{s_to}'] = pct
                                else:
                                    cells.append('      -')
                            pbar.write(f'  {name:<{name_width}s}  {"  ".join(cells)}')
                        error_decrease_tracker.clear()

            # Log metrics
            if i_step != initial_step and i_step % log_every == 0:
                _extra_scalars = {}
                for _log in (refine_regression_log, delta_increase_log, error_decrease_log):
                    _extra_scalars.update(_log)
                    _log.clear()
                records = logger.log_metrics(
                    records, ma_buffer, lr_scheduler, i_step, initial_step, extra_scalars=_extra_scalars,
                )

            checkpoint_saver.save_if_due(i_step)

            # Save data pipeline RNG state for all processes so data order can be resumed
            if checkpoint_saver.is_due(i_step):
                for _pipeline_name, _pipeline in [('refine', refine_data_pipeline), ('norefine', norefine_data_pipeline)]:
                    _pipeline_state_path = Path(workspace, 'checkpoint', 'data_pipeline', f'latest_{_pipeline_name}_data_pipeline_rank_{accelerator.process_index}.pt')
                    _pipeline_state_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({'step': i_step, **_pipeline.state_dict()}, _pipeline_state_path)

            if accelerator.is_main_process and i_step > 0 and i_step % 100 == 0:
                pbar.write(f'[Step {i_step}] refine data pipeline profile:\n{refine_data_pipeline.profile()}')
                pbar.write(f'[Step {i_step}] norefine data pipeline profile:\n{norefine_data_pipeline.profile()}')

            checkpoint_saver.poll_on_demand(i_step)

            if max_invalid_batches >= 0:
                invalid_batch_abort_flag = torch.tensor(
                    [int(invalid_batch_encountered_times >= max_invalid_batches)],
                    device=device,
                    dtype=torch.int32,
                )
                gathered_invalid_batch_abort_flags = accelerator.gather(invalid_batch_abort_flag)
                if gathered_invalid_batch_abort_flags.max().item() > 0:
                    if accelerator.is_main_process:
                        triggered_ranks = [
                            i for i, flag in enumerate(gathered_invalid_batch_abort_flags.tolist()) if flag > 0
                        ]
                        pbar.write(f'Invalid batch threshold {max_invalid_batches} reached on ranks {triggered_ranks}, saving checkpoint before abort.')
                        checkpoint_saver.save(i_step, async_save=False)
                    accelerator.wait_for_everyone()
                    raise RuntimeError('Encountered too many invalid batches on at least one rank, abort training.')

            # Visualize
            if vis_every > 0 and accelerator.is_main_process and (i_step == initial_step or i_step % vis_every == 0 or i_step == num_iterations - 1):
                visualize_predictions(
                    batches_for_vis, model, accelerator, workspace, device,
                    batch_size_forward, i_step, refine_steps=config['refine_steps'], logger=logger,
                )
            pbar.update(1)

            if accelerator.is_main_process and (i_step % 10 == 0):
                autotune_cache_path = Path.home() / '.flex_gemm' / 'autotune_cache.json'
                if autotune_cache_path.exists():
                    try:
                        shutil.copy(autotune_cache_path, Path(workspace, f'autotune_cache.json'))
                    except Exception as e:
                        pbar.write(f'Error copying autotune cache: {e}')
                        traceback.print_exc()

            # Garbage collection to reduce peak memory
            if (i_step % gc_every == 0) and (i_step != initial_step):
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()