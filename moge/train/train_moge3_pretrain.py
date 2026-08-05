import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import json
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
    append_group_log_dict,
    append_group_log_value,
    group_loss_values,
    materialize_log_records,
    to_log_scalar,
    write_optimizer_param_assignment_log,
)
from ..utils.tools import key_average
from .options import common_train_options
from .experiment import RunLogger, setup_accelerator
from .checkpoint import CheckpointSaver, load_checkpoint, restore_ma_buffer, restore_training_state
from .debug import DebugDumper
from .visualization import visualize_gt, visualize_predictions


warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")
torch._dynamo.config.disable = True
torch.backends.cudnn.benchmark = False      # Varying input size, make sure cudnn benchmark is disabled


@click.command()
@common_train_options
def main(
    config_path: str,
    experiment_name: str,
    workspace_path: str,
    base_checkpoint: Optional[str],
    checkpoint_path: str,
    batch_size_forward: int,
    gradient_accumulation_steps: int,
    backbone_gradient_checkpoint: bool,
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

    accelerator, device, batch_size_total, workspace = setup_accelerator(
        gradient_accumulation_steps, find_unused_parameters, batch_size_forward, workspace_path,
    )
    logger = RunLogger(accelerator, log_type)
    logger.setup(
        workspace=workspace, config=config, experiment_name=experiment_name,
        log_dir=log_dir, wandb_project=wandb_project, batch_size_total=batch_size_total,
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

    # Set precision
    if precision == 'fp32':
        pass
    elif precision == 'tf32':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = False
    elif precision == 'mixed_bf16':
        model.enable_mixed_precision()

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

    # Initialize training data pipeline
    dataloader_seed = (seed + accelerator.process_index) if seed is not None else None
    with accelerator.local_main_process_first():
        train_data_pipeline = TrainDataLoaderPipeline(config['data'], batch_size_forward, workspace=workspace, num_load_workers=num_load_workers, num_process_workers=num_process_workers, seed=dataloader_seed)

    # Restore data pipeline RNG state if resuming
    if initial_step > 0:
        _pipeline_state_path = Path(workspace, 'checkpoint', f'latest_data_pipeline_rank_{accelerator.process_index}.pt')
        if _pipeline_state_path.exists():
            _pipeline_state = torch.load(_pipeline_state_path, map_location='cpu', weights_only=False)
            train_data_pipeline.load_state_dict(_pipeline_state)
            print(f"Restored data pipeline state for rank {accelerator.process_index} from step {_pipeline_state.get('step', '?')}")
        else:
            print(f"Warning: No data pipeline state found for rank {accelerator.process_index}, data ordering will restart from the beginning")

    records = []
    ma_buffer = restore_ma_buffer(workspace, initial_step, accelerator)

    model.train()

    with (
        train_data_pipeline,
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
            for _ in range(num_vis_images // batch_size_forward):
                batch = train_data_pipeline.get()
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
            dump_grad_norm_above=config.get('dump_grad_norm_above'),
        )
        invalid_batch_encountered_times = 0

        # Training loop
        for i_step in range(initial_step, num_iterations):
            with timeit('Step', verbose=False) as timer_step:
                step_record_start = len(records)
                for i_accumulate in range(gradient_accumulation_steps):
                    # Load batch
                    with timeit('Load instance', verbose=False) as timer_load_instance:
                        batch = train_data_pipeline.get()
                        batch = to_device(batch, device)
                    records.append({'time/data': timer_load_instance.time})

                    is_invalid_batch = False

                    image, gt_depth, gt_normal, gt_mask_fin, gt_mask_inf, gt_intrinsics, label_type, is_metric, info = batch['image'], batch['depth'], batch['normal'], batch['depth_mask_fin'], batch['depth_mask_inf'], batch['intrinsics'], batch['label_type'], batch['is_metric'], batch['info']

                    current_batch_size = image.shape[0]

                    if all(label == 'invalid' for label in label_type):
                        is_invalid_batch = True
                        print(f"Rank {accelerator.process_index} all-invalid batch at step {i_step}, accumulation {i_accumulate}. Batch info: {info}")
                        invalid_batch_encountered_times += 1

                    gt_points = utils3d.pt.depth_map_to_point_map(gt_depth, intrinsics=gt_intrinsics)
                    gt_focal = 1 / (1 / gt_intrinsics[..., 0, 0] ** 2 + 1 / gt_intrinsics[..., 1, 1] ** 2) ** 0.5

                    with accelerator.accumulate(model):
                        # Forward
                        if i_step <= config.get('low_resolution_training_steps', 0):
                            num_tokens = config['model']['num_tokens_range'][0]
                        else:
                            num_tokens = random.Random(f'num_tokens-{seed}-{i_step}-{i_accumulate}').randint(*config['model']['num_tokens_range'])
                        
                        with timeit('Model forward', verbose=False) as timer_forward:
                            output = model(
                                image,
                                num_tokens=num_tokens,
                                refine_steps=0
                            )
                        pred_points_all, delta_z_all = (output.get(k, None) for k in ['points_per_step', 'delta_z_per_update'])
                        pred_normal, pred_mask, pred_metric_scale = (output.get(k, None) for k in ['normal', 'mask', 'metric_scale'])

                        # Compute loss (grouped by label type)
                        with timeit('Loss computation', verbose=False) as timer_loss_computation:
                            if is_invalid_batch:
                                loss = torch.tensor(0.0, device=device, requires_grad=True)
                            else:
                                loss_sum = torch.tensor(0.0, device=device)
                                label_to_indices: Dict[str, List[int]] = {}
                                for i, label in enumerate(label_type):
                                    label_to_indices.setdefault(label, []).append(i)

                                def accumulate_group_loss(loss_name: str, loss_value: torch.Tensor, weight: float, group_size: int):
                                    nonlocal loss_sum
                                    if not torch.isfinite(loss_value.detach()).all().cpu().item():
                                        pbar.write(f'NaN loss in process {accelerator.process_index}, loss name: {loss_name}')
                                        dumper.add_reason(f'nan_loss_{loss_name}')
                                    loss_sum = loss_sum + weight * group_loss_values(loss_value, group_size).sum()

                                for label, indices_list in label_to_indices.items():
                                    label_loss_config = config['loss'][label]
                                    if not label_loss_config:
                                        continue

                                    group_size = len(indices_list)
                                    group_indices = torch.as_tensor(indices_list, device=device, dtype=torch.long)
                                    group_records: List[Dict[str, Any]] = [{} for _ in range(group_size)]
                                    gt_points_group_raw = gt_points.index_select(0, group_indices)
                                    mask_group = torch.isfinite(gt_points_group_raw).all(dim=-1)
                                    gt_points_group = torch.where(mask_group[..., None], gt_points_group_raw, 1)
                                    gt_normal_group = gt_normal.index_select(0, group_indices)
                                    gt_mask_fin_group = gt_mask_fin.index_select(0, group_indices)
                                    gt_mask_inf_group = gt_mask_inf.index_select(0, group_indices)
                                    gt_focal_group = gt_focal.index_select(0, group_indices)

                                    gt_metric_scale = None
                                    step0_gt_metric_scale = None
                                    radial_loss_cache: Optional[RadialPartitionLocalLossCache] = None

                                    # points
                                    # pred_points_all is a list of length refine_steps+1, each element is a tensor of shape (B, H, W, 3)
                                    for pred_step, pred_points_iter_all in enumerate(pred_points_all):
                                        pred_points_group = pred_points_iter_all.index_select(0, group_indices)
                                        with torch.no_grad():
                                            append_group_log_dict(
                                                group_records,
                                                f'misc/monitoring_step_{pred_step}',
                                                {'std': pred_points_group.detach().reshape(group_size, -1).std(dim=1)},
                                            )

                                        for k, v in label_loss_config.get('points', {}).items():
                                            if pred_step not in v['apply_steps']:
                                                continue
                                            iter_key = f'{k}_step_{pred_step}' if pred_step > 0 else k
                                            weight = v['weight']

                                            if v['function'] == 'affine_invariant_global_loss':
                                                loss_value, misc_value, gt_metric_scale, _ = affine_invariant_global_loss(
                                                    pred_points_group,
                                                    gt_points_group,
                                                    mask_group,
                                                    **v['params'],
                                                )
                                                if pred_step == 0:
                                                    step0_gt_metric_scale = gt_metric_scale
                                            elif v['function'] in {'radial_partition_local_loss', 'radial_partition_local_loss_rand_partition'}:
                                                scale_to_use = gt_metric_scale
                                                if scale_to_use is None:
                                                    raise RuntimeError(f"{v['function']} requires a preceding global scale loss in the same points config")
                                                if scale_to_use.dim() == 0:
                                                    scale_to_use = scale_to_use.unsqueeze(0)
                                                if radial_loss_cache is None:
                                                    radial_loss_cache = RadialPartitionLocalLossCache.from_inputs(gt_points_group, mask_group)
                                                radial_loss_fn = radial_partition_local_loss_rand_partition if v['function'] == 'radial_partition_local_loss_rand_partition' else radial_partition_local_loss
                                                loss_value, misc_value = radial_loss_fn(
                                                    pred_points_group,
                                                    gt_points_group,
                                                    mask_group,
                                                    scale_to_use,
                                                    packed=radial_loss_cache,
                                                    **v['params'],
                                                )
                                            elif v['function'] == 'affine_invariant_local_loss':
                                                loss_value, misc_value = affine_invariant_local_loss(
                                                    pred_points_group, 
                                                    gt_points_group, 
                                                    mask_group,
                                                    gt_focal_group, 
                                                    gt_metric_scale, 
                                                    **v['params']
                                                )
                                            elif v['function'] == 'edge_loss':
                                                loss_value, misc_value = edge_loss(
                                                    pred_points_group,
                                                    gt_points_group,
                                                    mask_group,
                                                )
                                            else:
                                                raise ValueError(f"Unknown points loss function: {v['function']}")

                                            accumulate_group_loss(iter_key, loss_value, weight, group_size)
                                            append_group_log_value(group_records, f'loss/{iter_key}', loss_value)
                                            append_group_log_dict(group_records, f'misc/{iter_key}', misc_value)

                                    # normal
                                    for k, v in label_loss_config.get('normal', {}).items():
                                        weight = v['weight']
                                        if v['function'] == 'normal_map_loss':
                                            loss_value, misc_value = normal_map_loss(pred_normal.index_select(0, group_indices), gt_normal_group)
                                        else:
                                            raise ValueError(f"Unknown normal loss function: {v['function']}")
                                        accumulate_group_loss(k, loss_value, weight, group_size)
                                        append_group_log_value(group_records, f'loss/{k}', loss_value)
                                        append_group_log_dict(group_records, f'misc/{k}', misc_value)
                                    
                                    # mask
                                    for k, v in label_loss_config.get('mask', {}).items():
                                        weight = v['weight']
                                        if v['function'] == 'mask_bce_loss':
                                            loss_value, misc_value = mask_bce_loss(pred_mask.index_select(0, group_indices), gt_mask_fin_group, gt_mask_inf_group)
                                        else:
                                            raise ValueError(f"Unknown mask loss function: {v['function']}")
                                        accumulate_group_loss(k, loss_value, weight, group_size)
                                        append_group_log_value(group_records, f'loss/{k}', loss_value)
                                        append_group_log_dict(group_records, f'misc/{k}', misc_value)

                                    # metric_scale
                                    metric_positions = [pos for pos, batch_idx in enumerate(indices_list) if is_metric[batch_idx]]
                                    if pred_metric_scale is not None and step0_gt_metric_scale is not None and metric_positions:
                                        metric_positions_tensor = torch.as_tensor(metric_positions, device=device, dtype=torch.long)
                                        metric_indices = group_indices.index_select(0, metric_positions_tensor)
                                        metric_scale_gt = step0_gt_metric_scale
                                        if metric_scale_gt.dim() == 0:
                                            metric_scale_gt = metric_scale_gt.unsqueeze(0)
                                        metric_scale_gt = metric_scale_gt.index_select(0, metric_positions_tensor)
                                        for k, v in label_loss_config.get('metric_scale', {}).items():
                                            weight = v['weight']
                                            if v['function'] == 'metric_scale_loss':
                                                loss_value, misc_value = metric_scale_loss(pred_metric_scale.index_select(0, metric_indices), metric_scale_gt.detach())
                                            else:
                                                raise ValueError(f"Unknown metric_scale loss function: {v['function']}")
                                            accumulate_group_loss(k, loss_value, weight, len(metric_positions))
                                            append_group_log_value(group_records, f'loss/{k}', loss_value, positions=metric_positions)
                                            append_group_log_dict(group_records, f'misc/{k}', misc_value, positions=metric_positions)

                                    records.extend(group_records)

                                loss = loss_sum / current_batch_size  # Average over the batch
                            records.append({'train/loss': to_log_scalar(loss)})

                        # Backward
                        with timeit('Backward', verbose=False) as timer_backward:
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

                            # Extra dump trigger: large (but finite) grad norm.
                            # Controlled by config['dump_grad_norm_above']; capped by max_extra_dumps.
                            dumper.note_grad_norm(grad_norm, grad_norm_is_finite)

                        optimizer.zero_grad()

                        dumper.flush(i_step, i_accumulate, batch, output)

            records.append({'time/step': timer_step.time})

            lr_scheduler.step()

            # EMA update            
            if enable_ema and accelerator.is_main_process and accelerator.sync_gradients:
                ema_model.update_parameters(model)

            # Print raw loss values every 100 steps
            if accelerator.is_main_process and i_step % 100 == 0:
                step_avg = key_average(materialize_log_records(records[step_record_start:]))
                loss_misc = {k: v for k, v in sorted(step_avg.items()) if k.startswith(('loss/',))}
                if loss_misc:
                    pbar.write(f'[Step {i_step}] ' + ' | '.join(f'{k}: {v:.6f}' for k, v in loss_misc.items()))
                if 'train/loss' in step_avg:
                    pbar.set_postfix({'loss': step_avg['train/loss']}, refresh=False)

            # Log metrics
            if i_step != initial_step and i_step % log_every == 0:
                records = logger.log_metrics(records, ma_buffer, lr_scheduler, i_step, initial_step)

            checkpoint_saver.save_if_due(i_step)

            # Save data pipeline RNG state for all processes so data order can be resumed
            if checkpoint_saver.is_due(i_step):
                _pipeline_state_path = Path(workspace, 'checkpoint', f'latest_data_pipeline_rank_{accelerator.process_index}.pt')
                _pipeline_state_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'step': i_step, **train_data_pipeline.state_dict()}, _pipeline_state_path)

            if accelerator.is_main_process and i_step > 0 and i_step % 100 == 0:
                pbar.write(f'[Step {i_step}] data pipeline profile:\n{train_data_pipeline.profile()}')

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
                    batch_size_forward, i_step, refine_steps=0, logger=logger,
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