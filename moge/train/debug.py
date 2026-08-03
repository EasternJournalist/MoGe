"""Debug dumping for training runs that hit NaNs or unusually large gradients."""
from pathlib import Path
from typing import *

import torch

from .utils import detach_to_cpu


class DebugDumper:
    """Collects dump triggers during a step and writes the offending state to disk.

    Call `add_reason(tag)` from anywhere in the step to flag it. Tags starting with
    `nan_` are fatal and count toward the abort threshold; anything else (e.g.
    `large_grad_norm_12.34`) is informational and capped so a pathological run
    cannot fill the disk. Call `flush(...)` once per accumulation step.
    """

    def __init__(
        self,
        workspace: Path,
        accelerator,
        dump_grad_norm_above: Optional[float] = None,
        max_nan_dumps_before_abort: int = 10,
        max_extra_dumps: int = 25,
    ):
        self.workspace = workspace
        self.accelerator = accelerator
        self.dump_grad_norm_above = dump_grad_norm_above
        self.max_nan_dumps_before_abort = max_nan_dumps_before_abort
        self.max_extra_dumps = max_extra_dumps
        self.reasons: List[str] = []
        self.nan_encountered_times = 0
        self.extra_dump_count = 0

    def add_reason(self, tag: str):
        self.reasons.append(tag)

    def note_grad_norm(self, grad_norm, grad_norm_is_finite: bool):
        """Flag an unusually large but finite gradient norm, if a threshold was configured.

        The threshold and quota are checked before reading the value, so the
        `.cpu()` sync only happens when the feature is actually armed.
        """
        if (
            self.dump_grad_norm_above is not None
            and grad_norm_is_finite
            and self.extra_dump_count < self.max_extra_dumps
        ):
            grad_norm_value = float(grad_norm.detach().cpu().item())
            if grad_norm_value > self.dump_grad_norm_above:
                self.add_reason(f'large_grad_norm_{grad_norm_value:.2f}')

    def _dump(self, step: int, accumulate_step: int, batch: Any, output: Any) -> Path:
        """Dump (batch, output, reasons) so the failing forward pass can be replayed offline."""
        dump_path = Path(
            self.workspace,
            'debug',
            f'step_{step:08d}_accum_{accumulate_step}_proc_{self.accelerator.process_index}'
            f'_reasons_{self.reasons[0]}.pkl',
        )
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open('wb') as f:
            torch.save({
                'batch': detach_to_cpu(batch),
                'output': detach_to_cpu(output),
                'reasons': self.reasons,
            }, f)
        return dump_path

    def flush(self, i_step: int, i_accumulate: int, batch: Any, output: Any):
        """Write a dump if anything flagged this step, and abort after too many NaNs."""
        if not self.reasons:
            return
        if any(r.startswith('nan_') for r in self.reasons):
            self.nan_encountered_times += 1
        else:
            self.extra_dump_count += 1
        self._dump(i_step, i_accumulate, batch, output)
        self.reasons = []
        if self.nan_encountered_times >= self.max_nan_dumps_before_abort:
            raise RuntimeError('NaN encountered too many times, abort training.')
