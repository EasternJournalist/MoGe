import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import gc
import json
import random
import warnings
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import *

import click
import torch
import torch.version
try:
    import utils3d_moge as utils3d
except ImportError:
    import utils3d
from accelerate.utils import set_seed
from tqdm import tqdm

from moge.train.dataloader import TrainDataLoaderPipeline
from moge.train.losses import (
    affine_invariant_global_loss,
    affine_invariant_local_loss,
    edge_loss,
    mask_bce_loss,
    mask_l2_loss,
    metric_scale_loss,
    monitoring,
    normal_loss,
    normal_map_loss,
)
from .checkpoint import (
    CheckpointSaver,
    load_checkpoint,
    restore_data_pipeline_states,
    restore_ma_buffer,
    restore_training_state,
    save_data_pipeline_states,
)
from .debug import DebugDumper
from .experiment import RunLogger, setup_accelerator
from .utils import (
    build_lr_scheduler,
    build_optimizer,
    materialize_log_records,
    to_device,
    to_log_scalar,
    write_optimizer_param_assignment_log,
)
from .visualization import visualize_gt, visualize_predictions
from ..utils.tools import flatten_nested_dict, key_average, timeit


warnings.filterwarnings('ignore', category=FutureWarning, module='torch.utils.checkpoint')
torch.backends.cudnn.benchmark = False


@click.command()
@click.option('--config', 'config_path', type=str, default='configs/debug.json')
@click.option('--name', 'experiment_name', type=str, default='debug', help='Experiment name')
@click.option('--workspace', '--workspace_path', 'workspace_path', type=str, default='workspace/debug', help='Workspace for logs, visualizations and checkpoints')
@click.option('--base_checkpoint', type=str, default='', help='Checkpoint used when the workspace has no requested checkpoint')
@click.option('--checkpoint', 'checkpoint_path', type=str, default=None, help='Checkpoint path, step number, "latest", or "none"')
@click.option('--batch_size_forward', type=int, default=8, help='Batch size for each forward pass on each device')
@click.option('--gradient_accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')
@click.option('--backbone_gradient_checkpoint', '--enable_gradient_checkpointing', type=bool, default=True, help='Use gradient checkpointing in the backbone')
@click.option('--enable_mixed_precision', type=bool, default=False, help='Legacy alias for --precision mixed_fp16')
@click.option('--precision', type=click.Choice(['fp32', 'tf32', 'mixed_fp16', 'mixed_bf16']), default=None, help='Numerical precision; overrides --enable_mixed_precision')
@click.option('--enable_ema', type=bool, default=True, help='Maintain an exponential moving average of model weights')
@click.option('--debug', 'debug_mode', type=bool, default=False, help='Enable additional debug dumps')
@click.option('--num_iterations', type=int, default=1000000, help='Number of iterations to train the model')
@click.option('--checkpoint_every', '--save_every', type=int, default=10000, help='Save a permanent checkpoint every n iterations')
@click.option('--rolling_checkpoint_every', type=int, default=500, help='Save a rolling checkpoint every n iterations')
@click.option('--log_every', type=int, default=1000, help='Log metrics every n iterations')
@click.option('--vis_every', type=int, default=0, help='Visualize every n iterations')
@click.option('--vis_gt', type=bool, default=True, help='Visualize ground truth')
@click.option('--num_vis_images', type=int, default=32, help='Number of images to visualize')
@click.option('--enable_mlflow', type=bool, default=True, help='Legacy switch: use MLflow when --log_type is omitted')
@click.option('--log_type', type=click.Choice(['mlflow', 'tensorboard', 'wandb']), multiple=True, default=(), help='Logging backend; may be specified more than once')
@click.option('--log_dir', type=str, default=None, help='Root directory for TensorBoard logs')
@click.option('--gc_every', type=int, default=1000, help='Run garbage collection every n iterations')
@click.option('--seed', type=int, default=0, help='Random seed')
@click.option('--wandb_project', type=str, default='MoGe', help='Weights & Biases project name')
@click.option('--max_invalid_batches', type=int, default=-1, help='Maximum all-invalid batches before abort; -1 disables the check')
@click.option('--find_unused_parameters', type=bool, default=True, help='Whether DDP should look for unused parameters')
@click.option('--num_load_workers', type=int, default=4, help='Number of workers for loading data')
@click.option('--num_process_workers', type=int, default=8, help='Number of workers for processing data')
def main(
    config_path: str,
    experiment_name: str,
    workspace_path: str,
    base_checkpoint: Optional[str],
    checkpoint_path: Optional[str],
    batch_size_forward: int,
    gradient_accumulation_steps: int,
    backbone_gradient_checkpoint: bool,
    enable_mixed_precision: bool,
    precision: Optional[str],
    enable_ema: bool,
    debug_mode: bool,
    num_iterations: int,
    checkpoint_every: int,
    rolling_checkpoint_every: int,
    log_every: int,
    vis_every: int,
    vis_gt: bool,
    num_vis_images: int,
    enable_mlflow: bool,
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
    with open(config_path, 'r') as f:
        config = json.load(f)

    precision = precision or ('mixed_fp16' if enable_mixed_precision else 'fp32')
    accelerator_precision = {
        'mixed_fp16': 'fp16',
        'mixed_bf16': 'bf16',
    }.get(precision)
    accelerator, device, batch_size_total, workspace = setup_accelerator(
        gradient_accumulation_steps,
        find_unused_parameters,
        batch_size_forward,
        workspace_path,
        mixed_precision=accelerator_precision,
    )

    # An explicit --log_type takes precedence; otherwise preserve the old
    # --enable_mlflow behavior used by docs and existing launch scripts.
    effective_log_type = log_type or (('mlflow',) if enable_mlflow else ())
    logger = RunLogger(accelerator, effective_log_type)
    logger.setup(
        workspace=workspace,
        config=config,
        experiment_name=experiment_name,
        log_dir=log_dir,
        wandb_project=wandb_project,
        batch_size_total=batch_size_total,
        extra_params={'effective_precision': precision},
    )

    if seed is not None:
        set_seed(seed, device_specific=True)

    print('Initialize model')
    with accelerator.local_main_process_first():
        from moge.model import import_model_class_by_version
        model_class = import_model_class_by_version(config['model_version'])
        model = model_class(**config['model'])
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}')

    if enable_ema and accelerator.is_main_process:
        ema_avg_fn = lambda averaged, current, _: 0.999 * averaged + 0.001 * current
        ema_model = torch.optim.swa_utils.AveragedModel(model, device=device, avg_fn=ema_avg_fn)

    if backbone_gradient_checkpoint:
        model.enable_gradient_checkpointing()
    if precision == 'tf32':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = False

    optimizer = build_optimizer(model, config['optimizer'])
    lr_scheduler = build_lr_scheduler(optimizer, config['lr_scheduler'])
    for i, group in enumerate(optimizer.param_groups):
        print(f"- Group {i}: {sum(p.numel() for p in group['params'] if p.requires_grad)} parameters")
    if accelerator.is_main_process:
        write_optimizer_param_assignment_log(model, optimizer, workspace)

    checkpoint = load_checkpoint(checkpoint_path, workspace, accelerator, enable_ema)
    if checkpoint is None:
        checkpoint = load_checkpoint(base_checkpoint, workspace, accelerator, enable_ema)
    initial_step = restore_training_state(
        checkpoint,
        model,
        optimizer,
        lr_scheduler,
        ema_model if enable_ema and accelerator.is_main_process else None,
        accelerator,
        enable_ema,
    )
    del checkpoint

    model, optimizer = accelerator.prepare(model, optimizer)
    if torch.version.hip and isinstance(model, torch.nn.parallel.DistributedDataParallel):
        from moge.model.utils import sync_ddp_hook
        model.register_comm_hook(None, sync_ddp_hook)

    dataloader_seed = seed + accelerator.process_index if seed is not None else None
    with accelerator.local_main_process_first():
        train_data_pipe = TrainDataLoaderPipeline(
            deepcopy(config['data']),
            batch_size_forward,
            workspace=workspace,
            num_load_workers=num_load_workers,
            num_process_workers=num_process_workers,
            seed=dataloader_seed,
        )
    data_pipelines = {'train': train_data_pipe}
    restore_data_pipeline_states(workspace, initial_step, accelerator, data_pipelines)

    records: List[Dict[str, Any]] = []
    ma_buffer = restore_ma_buffer(workspace, initial_step, accelerator)
    model.train()

    with (
        train_data_pipe,
        tqdm(initial=initial_step, total=num_iterations, desc='Training', disable=not accelerator.is_main_process) as pbar,
        ThreadPoolExecutor(max_workers=1) as checkpoint_executor,
    ):
        checkpoint_saver = CheckpointSaver(
            workspace=workspace,
            config=config,
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema_model=ema_model if enable_ema and accelerator.is_main_process else None,
            enable_ema=enable_ema,
            ma_buffer=ma_buffer,
            executor=checkpoint_executor,
            pbar=pbar,
            num_iterations=num_iterations,
            checkpoint_every=checkpoint_every,
            rolling_checkpoint_every=rolling_checkpoint_every,
            initial_step=initial_step,
        )

        batches_for_vis: List[Dict[str, torch.Tensor]] = []
        if accelerator.is_main_process and vis_every > 0:
            num_vis_images = num_vis_images // batch_size_forward * batch_size_forward
            batches_for_vis = [train_data_pipe.get() for _ in range(num_vis_images // batch_size_forward)]
            if vis_gt:
                visualize_gt(batches_for_vis, workspace, batch_size_forward, initial_step, logger)

        if seed is not None:
            set_seed(seed + initial_step, device_specific=True)

        dumper = DebugDumper(
            workspace,
            accelerator,
            dump_grad_norm_above=config.get('dump_grad_norm_above', 10 if debug_mode else None),
        )
        invalid_batch_count = 0
        autocast_dtype = torch.bfloat16 if precision == 'mixed_bf16' else torch.float16
        autocast_enabled = precision in {'mixed_fp16', 'mixed_bf16'}

        for i_step in range(initial_step, num_iterations):
            step_record_start = len(records)
            with timeit('Step', verbose=False) as timer_step:
                for i_accumulate in range(gradient_accumulation_steps):
                    with timeit('Load instance', verbose=False) as timer_load:
                        batch = to_device(train_data_pipe.get(), device)
                    records.append({'time/data': timer_load.time})

                    image = batch['image']
                    label_type = batch['label_type']
                    info = batch.get('info')
                    is_invalid_batch = all(label == 'invalid' for label in label_type)
                    if is_invalid_batch:
                        invalid_batch_count += 1
                        pbar.write(
                            f'Rank {accelerator.process_index} all-invalid batch at step {i_step}, '
                            f'accumulation {i_accumulate}. Batch info: {info}'
                        )

                    gt_points_raw = utils3d.pt.depth_map_to_point_map(
                        batch['depth'], intrinsics=batch['intrinsics'],
                    )
                    gt_points_mask = torch.isfinite(gt_points_raw).all(dim=-1)
                    gt_points = torch.where(gt_points_mask[..., None], gt_points_raw, 1)
                    gt_focal = 1 / (
                        1 / batch['intrinsics'][..., 0, 0] ** 2
                        + 1 / batch['intrinsics'][..., 1, 1] ** 2
                    ) ** 0.5

                    with accelerator.accumulate(model):
                        if i_step <= config.get('low_resolution_training_steps', 0):
                            num_tokens = config['model']['num_tokens_range'][0]
                        else:
                            num_tokens = random.Random(
                                f'num_tokens-{seed}-{i_step}-{i_accumulate}'
                            ).randint(*config['model']['num_tokens_range'])

                        with timeit('Model forward', verbose=False) as timer_forward:
                            with torch.autocast(
                                device_type=device.type,
                                dtype=autocast_dtype,
                                enabled=autocast_enabled,
                            ):
                                output = model(image, num_tokens=num_tokens)
                        records.append({'time/forward': timer_forward.time})
                        pred_points, pred_mask, pred_normal, pred_metric_scale = (
                            output.get(k) for k in ('points', 'mask', 'normal', 'metric_scale')
                        )

                        if is_invalid_batch:
                            loss = torch.tensor(0.0, device=device, requires_grad=True)
                        else:
                            instance_losses = []
                            for i in range(image.shape[0]):
                                gt_metric_scale = None
                                loss_dict: Dict[str, Any] = {}
                                weight_dict: Dict[str, Any] = {}
                                misc_dict: Dict[str, Any] = {
                                    'monitoring': monitoring(pred_points[i].detach())
                                }

                                for name, spec in config['loss'][label_type[i]].items():
                                    weight_dict[name] = spec['weight']
                                    function = spec['function']
                                    params = spec.get('params', {})
                                    if function == 'affine_invariant_global_loss':
                                        loss_dict[name], misc_dict[name], gt_metric_scale, _ = affine_invariant_global_loss(
                                            pred_points[i], gt_points[i], gt_points_mask[i], **params,
                                        )
                                        gt_metric_scale = gt_metric_scale.detach()
                                    elif function == 'affine_invariant_local_loss':
                                        if gt_metric_scale is None:
                                            raise RuntimeError(
                                                'affine_invariant_local_loss requires a preceding global loss'
                                            )
                                        loss_dict[name], misc_dict[name] = affine_invariant_local_loss(
                                            pred_points[i], gt_points[i], gt_points_mask[i],
                                            gt_focal[i], gt_metric_scale, **params,
                                        )
                                    elif function == 'normal_loss':
                                        loss_dict[name], misc_dict[name] = normal_loss(
                                            pred_points[i], gt_points_raw[i],
                                        )
                                    elif function == 'edge_loss':
                                        loss_dict[name], misc_dict[name] = edge_loss(
                                            pred_points[i], gt_points[i], gt_points_mask[i],
                                        )
                                    elif function == 'normal_map_loss':
                                        loss_dict[name], misc_dict[name] = normal_map_loss(
                                            pred_normal[i], batch['normal'][i],
                                        )
                                    elif function == 'mask_bce_loss':
                                        loss_dict[name], misc_dict[name] = mask_bce_loss(
                                            pred_mask[i], batch['depth_mask_fin'][i],
                                            batch['depth_mask_inf'][i],
                                        )
                                    elif function == 'mask_l2_loss':
                                        loss_dict[name], misc_dict[name] = mask_l2_loss(
                                            pred_mask[i], batch['depth_mask_fin'][i],
                                            batch['depth_mask_inf'][i],
                                        )
                                    elif function == 'metric_scale_loss':
                                        if (
                                            bool(batch['is_metric'][i])
                                            and pred_metric_scale is not None
                                            and gt_metric_scale is not None
                                        ):
                                            loss_dict[name], misc_dict[name] = metric_scale_loss(
                                                pred_metric_scale[i], gt_metric_scale,
                                            )
                                    else:
                                        raise ValueError(f'Undefined loss function: {function}')

                                weight_dict = {
                                    '.'.join(k): v for k, v in flatten_nested_dict(weight_dict).items()
                                }
                                loss_dict = {
                                    '.'.join(k): v for k, v in flatten_nested_dict(loss_dict).items()
                                }
                                misc_dict = {
                                    '.'.join(k): v for k, v in flatten_nested_dict(misc_dict).items()
                                }
                                instance_loss = sum(
                                    (weight_dict[name] * value for name, value in loss_dict.items()),
                                    start=torch.tensor(0.0, device=device),
                                )
                                instance_losses.append(instance_loss)
                                for name, value in loss_dict.items():
                                    if not torch.isfinite(value.detach()).all():
                                        pbar.write(
                                            f'NaN loss in process {accelerator.process_index}: {name}'
                                        )
                                        dumper.add_reason(f'nan_loss_{name}')
                                records.append({
                                    **{f'loss/{k}': to_log_scalar(v) for k, v in loss_dict.items()},
                                    **{f'misc/{k}': to_log_scalar(v) for k, v in misc_dict.items()},
                                })
                            loss = sum(instance_losses) / len(instance_losses)
                        records.append({'train/loss': to_log_scalar(loss)})

                        with timeit('Backward', verbose=False) as timer_backward:
                            accelerator.backward(loss)
                        records.append({'time/backward': timer_backward.time})

                        if accelerator.sync_gradients:
                            grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                            records.append({'train/grad_norm': to_log_scalar(grad_norm)})
                            grad_is_finite = (
                                True if accelerator.scaler is not None
                                else bool(torch.isfinite(grad_norm.detach()).cpu().item())
                            )
                            if grad_is_finite or accelerator.scaler is not None:
                                optimizer.step()
                            else:
                                pbar.write(f'Non-finite gradient norm {grad_norm}; skip optimizer step')
                                pbar.write(f'Batch info: {info}')
                                dumper.add_reason('nan_grad_norm')
                            dumper.note_grad_norm(grad_norm, grad_is_finite)
                        optimizer.zero_grad()
                        dumper.flush(i_step, i_accumulate, batch, output)

            records.append({'time/step': timer_step.time})
            lr_scheduler.step()

            if enable_ema and accelerator.is_main_process and accelerator.sync_gradients:
                ema_model.update_parameters(model)

            if accelerator.is_main_process and i_step % 100 == 0:
                step_average = key_average(materialize_log_records(records[step_record_start:]))
                if 'train/loss' in step_average:
                    pbar.set_postfix({'loss': step_average['train/loss']}, refresh=False)

            if log_every > 0 and (i_step == initial_step or i_step % log_every == 0):
                records = logger.log_metrics(records, ma_buffer, lr_scheduler, i_step, initial_step)

            checkpoint_saver.save_if_due(i_step)
            if checkpoint_saver.is_due(i_step):
                save_data_pipeline_states(workspace, i_step, accelerator, data_pipelines)
            checkpoint_saver.poll_on_demand(i_step)

            if accelerator.is_main_process and i_step > 0 and i_step % 100 == 0:
                pbar.write(f'[Step {i_step}] train data pipeline profile:\n{train_data_pipe.profile()}')

            if max_invalid_batches >= 0:
                abort_flag = torch.tensor(
                    [int(invalid_batch_count > max_invalid_batches)], device=device, dtype=torch.int32,
                )
                gathered_flags = accelerator.gather(abort_flag)
                if gathered_flags.max().item() > 0:
                    if accelerator.is_main_process:
                        ranks = [i for i, flag in enumerate(gathered_flags.tolist()) if flag > 0]
                        pbar.write(
                            f'Invalid batch threshold {max_invalid_batches} reached on ranks {ranks}; '
                            'saving before abort'
                        )
                        checkpoint_saver.save(i_step, async_save=False)
                    accelerator.wait_for_everyone()
                    raise RuntimeError('Encountered too many invalid batches on at least one rank')

            if (
                vis_every > 0
                and accelerator.is_main_process
                and (i_step == initial_step or i_step % vis_every == 0 or i_step == num_iterations - 1)
            ):
                visualize_predictions(
                    batches_for_vis,
                    model,
                    accelerator,
                    workspace,
                    device,
                    batch_size_forward,
                    i_step,
                    refine_steps=None,
                    logger=logger,
                )

            pbar.update(1)
            if gc_every > 0 and i_step != initial_step and i_step % gc_every == 0:
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
