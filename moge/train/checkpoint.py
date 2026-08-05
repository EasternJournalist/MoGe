"""Checkpoint loading, saving and scheduling for the MoGe training entry points."""
import io
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import *

import torch

from .utils import cleanup_old_rolling_ckpts, record_rolling_ckpt, write_bytes_retry_loop

MA_BUFFER_MAXLEN = 1000


def load_checkpoint(
    ckpt_path: Optional[str],
    workspace: Path,
    accelerator,
    enable_ema: bool,
) -> Optional[Dict[str, Any]]:
    """Load a checkpoint by explicit path, by "latest", or by step number.

    "latest" and step-number forms read `latest.pt` (which holds only a step
    pointer) and then hydrate the model / optimizer / EMA shards written
    alongside it. Returns None when no checkpoint is requested or found.
    """
    with accelerator.local_main_process_first():
        checkpoint = None
        if not ckpt_path or ckpt_path == 'none':
            # - No checkpoint requested
            pass
        elif ckpt_path.endswith('.pt'):
            # - Load specific checkpoint file
            print(f'Load checkpoint: {ckpt_path}')
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        elif ckpt_path == 'latest':
            # - Load latest
            latest_path = Path(workspace, 'checkpoint', 'latest.pt')
            if latest_path.exists():
                print(f'Load checkpoint: {latest_path}')
                checkpoint = torch.load(latest_path, map_location='cpu', weights_only=True)
                checkpoint = _hydrate_shards(checkpoint, checkpoint['step'], workspace, accelerator, enable_ema)
        elif ckpt_path.isdigit():
            # - Load by step number
            i_step = int(ckpt_path)
            checkpoint = _hydrate_shards({'step': i_step}, i_step, workspace, accelerator, enable_ema)
    return checkpoint


def _hydrate_shards(checkpoint, i_step, workspace, accelerator, enable_ema):
    ckpt_name = f'{i_step:08d}' if isinstance(i_step, int) else i_step
    if 'model' not in checkpoint and (path := Path(workspace, 'checkpoint', f'{ckpt_name}.pt')).exists():
        print(f'Load model checkpoint: {path}')
        checkpoint['model'] = torch.load(path, map_location='cpu', weights_only=True)['model']
    if 'optimizer' not in checkpoint and (path := Path(workspace, 'checkpoint', f'{ckpt_name}_optimizer.pt')).exists():
        print(f'Load optimizer checkpoint: {path}')
        checkpoint.update(torch.load(path, map_location='cpu', weights_only=True))
    if enable_ema and accelerator.is_main_process:
        if 'ema_model' not in checkpoint and (path := Path(workspace, 'checkpoint', f'{ckpt_name}_ema.pt')).exists():
            print(f'Load EMA model checkpoint: {path}')
            checkpoint['ema_model'] = torch.load(path, map_location='cpu', weights_only=True)['model']
    return checkpoint


def restore_training_state(
    checkpoint: Optional[Dict[str, Any]],
    model,
    optimizer,
    lr_scheduler,
    ema_model,
    accelerator,
    enable_ema: bool,
) -> int:
    """Apply a checkpoint to the training state, or initialise from scratch.

    `strict=False` is load-bearing: a base checkpoint from a non-refiner run has
    no `refiner.*` keys, and the refiner must keep its fresh initialisation.
    Returns the step to resume from.
    """
    if checkpoint is None:
        print('Initialize model weights')
        with accelerator.local_main_process_first():
            model.init_weights()
        if enable_ema and accelerator.is_main_process:
            ema_model.module.load_state_dict(model.state_dict())
        return 0

    if 'model' not in checkpoint:
        raise FileNotFoundError(
            f"Checkpoint step {checkpoint.get('step', '?')} has no model state; "
            "the checkpoint may be incomplete or the requested step may not exist"
        )

    model.load_state_dict(checkpoint['model'], strict=False)
    if 'step' in checkpoint:
        initial_step = checkpoint['step'] + 1
        print(f'Resume from step {initial_step}')
    else:
        initial_step = 0
        print('No step info found in checkpoint, start from step 0')
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print('Warning: No optimizer state found in checkpoint, optimizer is re-initialized')
    if enable_ema and accelerator.is_main_process:
        if 'ema_model' in checkpoint:
            ema_model.module.load_state_dict(checkpoint['ema_model'], strict=False)
        else:
            ema_model.module.load_state_dict(model.state_dict())
            print('Warning: EMA enabled but no EMA state found; initialized EMA from the loaded model')
    if 'lr_scheduler' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    else:
        print('Warning: No lr_scheduler state found in checkpoint, lr_scheduler is re-initialized')
    return initial_step


def restore_ma_buffer(workspace: Path, initial_step: int, accelerator) -> deque:
    """Restore the moving-average buffer so ma1000/avg1000 curves stay continuous on resume."""
    ma_buffer = deque(maxlen=MA_BUFFER_MAXLEN)
    if initial_step > 0 and accelerator.is_main_process:
        path = Path(workspace, 'checkpoint', 'latest_ma_buffer.pt')
        if path.exists():
            state = torch.load(path, map_location='cpu', weights_only=False)
            ma_buffer = deque(state.get('ma_buffer', []), maxlen=MA_BUFFER_MAXLEN)
            print(f"Restored ma_buffer ({len(ma_buffer)} entries) from step {state.get('step', '?')}")
        else:
            print('Warning: No ma_buffer state found, ma1000/avg1000 metrics will restart from the beginning')
    return ma_buffer


def restore_data_pipeline_states(
    workspace: Path,
    initial_step: int,
    accelerator,
    pipelines: Dict[str, Any],
) -> None:
    """Restore per-rank dataloader RNG states that match the resumed step."""
    if initial_step <= 0:
        return
    expected_step = initial_step - 1
    for name, pipeline in pipelines.items():
        path = Path(
            workspace, 'checkpoint', 'data_pipeline',
            f'latest_{name}_data_pipeline_rank_{accelerator.process_index}.pt',
        )
        if not path.exists():
            print(f'Warning: No {name} data pipeline state found; data ordering will restart')
            continue
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('step') != expected_step:
            print(
                f"Warning: {name} data pipeline state is from step {state.get('step', '?')}, "
                f'not requested step {expected_step}; data ordering will restart'
            )
            continue
        pipeline.load_state_dict(state)
        print(f'Restored {name} data pipeline state for rank {accelerator.process_index} from step {expected_step}')


def save_data_pipeline_states(
    workspace: Path,
    i_step: int,
    accelerator,
    pipelines: Dict[str, Any],
) -> None:
    """Save each dataloader RNG state separately for every distributed rank."""
    state_dir = Path(workspace, 'checkpoint', 'data_pipeline')
    state_dir.mkdir(parents=True, exist_ok=True)
    for name, pipeline in pipelines.items():
        path = state_dir / f'latest_{name}_data_pipeline_rank_{accelerator.process_index}.pt'
        torch.save({'step': i_step, **pipeline.state_dict()}, path)


class CheckpointSaver:
    """Writes the checkpoint shards for a run and decides when to write them.

    Writing goes through a single-worker executor so a slow (often network-mounted)
    filesystem does not stall training. Built once before the loop rather than as a
    per-step closure.
    """

    def __init__(
        self,
        workspace: Path,
        config: Dict[str, Any],
        accelerator,
        model,
        optimizer,
        lr_scheduler,
        ema_model,
        enable_ema: bool,
        ma_buffer: deque,
        executor: ThreadPoolExecutor,
        pbar,
        num_iterations: int,
        checkpoint_every: int,
        rolling_checkpoint_every: int,
        initial_step: int,
    ):
        self.workspace = workspace
        self.config = config
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.ema_model = ema_model
        self.enable_ema = enable_ema
        self.ma_buffer = ma_buffer
        self.executor = executor
        self.pbar = pbar
        self.num_iterations = num_iterations
        self.checkpoint_every = checkpoint_every
        self.rolling_checkpoint_every = rolling_checkpoint_every
        self.initial_step = initial_step

    def _write(self, name: str, payload: Dict[str, Any], async_save: bool):
        with io.BytesIO() as f:
            torch.save(payload, f)
            data = f.getvalue()
        path = Path(self.workspace, 'checkpoint', name)
        if async_save:
            self.executor.submit(write_bytes_retry_loop, path, data)
        else:
            write_bytes_retry_loop(path, data)

    def save(self, i_step: int, async_save: bool = True):
        ckpt_name = 'final' if i_step == self.num_iterations - 1 else f'{i_step:08d}'
        self.pbar.write(f'Save checkpoint: {i_step:08d}')
        Path(self.workspace, 'checkpoint').mkdir(parents=True, exist_ok=True)

        model_config = self.config['model']
        self._write(f'{ckpt_name}.pt', {
            'model_config': model_config,
            'model': self.accelerator.unwrap_model(self.model).state_dict(),
        }, async_save)
        self._write(f'{ckpt_name}_optimizer.pt', {
            'model_config': model_config,
            'step': i_step,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }, async_save)
        if self.enable_ema:
            # NOTE: always async, matching the original behaviour.
            self._write(f'{ckpt_name}_ema.pt', {
                'model_config': model_config,
                'model': self.ema_model.module.state_dict(),
            }, True)
        latest_step = 'final' if ckpt_name == 'final' else i_step
        self._write('latest.pt', {'model_config': model_config, 'step': latest_step}, async_save)
        self._write('latest_ma_buffer.pt', {'step': i_step, 'ma_buffer': list(self.ma_buffer)}, async_save)

    def is_due(self, i_step: int) -> bool:
        """Whether this step warrants a checkpoint (permanent, rolling or final)."""
        return self._classify(i_step)[0]

    def _classify(self, i_step: int) -> Tuple[bool, bool]:
        is_permanent = (
            self.checkpoint_every > 0
            and i_step % self.checkpoint_every == 0
            and i_step != self.initial_step
        )
        is_rolling = (
            self.rolling_checkpoint_every > 0
            and i_step % self.rolling_checkpoint_every == 0
            and i_step != self.initial_step
            and not is_permanent
        )
        is_final = (i_step == self.num_iterations - 1)
        return is_permanent or is_rolling or is_final, is_rolling

    def save_if_due(self, i_step: int) -> bool:
        """Write a permanent, rolling or final checkpoint if this step calls for one."""
        due, is_rolling = self._classify(i_step)
        if self.accelerator.is_main_process and due:
            self.save(i_step)
            # For rolling checkpoints, drop the previous rolling one. Record this step
            # first so a crash before cleanup leaves it tracked rather than orphaned.
            if is_rolling:
                record_rolling_ckpt(self.workspace, i_step)
                self.executor.submit(cleanup_old_rolling_ckpts, self.workspace, i_step)
        return due

    def poll_on_demand(self, i_step: int):
        """Honour an out-of-band request to checkpoint at specific steps.

        Reads `<workspace>/save_ckpt_at.txt`, a whitespace-separated list of step
        numbers, and rewrites it with the steps still in the future.
        """
        if not (self.accelerator.is_main_process and i_step % 100 == 0):
            return
        path = Path(self.workspace, 'save_ckpt_at.txt')
        if not path.exists():
            return
        try:
            steps = {int(s) for s in path.read_text().split() if s.strip().isdigit()}
            if i_step in steps:
                self.pbar.write(f'On-demand checkpoint triggered at step {i_step}')
                self.save(i_step)
            remaining = {s for s in steps if s > i_step}
            if remaining:
                path.write_text(' '.join(str(s) for s in sorted(remaining)) + '\n')
            else:
                path.unlink()
        except Exception as e:
            self.pbar.write(f'Error reading on-demand checkpoint file: {e}')
