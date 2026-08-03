"""Click options shared by the MoGe-3 training entry points.

`train_moge3_pretrain.py` and `train_moge3_refiner.py` took the same 26 options
verbatim. Keeping them here means a flag's type, default and help text are
defined once; each script adds only the options unique to it.
"""
from typing import *

import click


def common_train_options(fn: Callable) -> Callable:
    """Attach the options shared by every MoGe-3 training entry point.

    Click applies decorators bottom-up, so these are listed in reverse of how
    they appear in `--help`.
    """
    options = [
        click.option('--num_process_workers', type=int, default=8, help='Number of workers for processing data'),
        click.option('--num_load_workers', type=int, default=4, help='Number of workers for loading data'),
        click.option('--find_unused_parameters', type=bool, default=True, help='Whether to set find_unused_parameters=True for DistributedDataParallel, which may be necessary if not all model parameters receive gradients in each iteration'),
        click.option('--max_invalid_batches', type=int, default=-1, help='Maximum number of all-invalid batches before abort; set to -1 to disable this check'),
        click.option('--wandb_project', type=str, default='MoGe', help='Weights & Biases project name'),
        click.option('--seed', type=int, default=0, help='Random seed'),
        click.option('--gc_every', type=int, default=1000, help='Run garbage collection every n iterations to reduce memory usage'),
        click.option('--log_dir', type=str, default=None, help='Root directory for tensorboard logs'),
        click.option('--log_type', type=click.Choice(['mlflow', 'tensorboard', 'wandb']), multiple=True, default=('tensorboard',), help='log type to use (can specify multiple)'),
        click.option('--num_vis_images', type=int, default=32, help='Number of images to visualize, must be a multiple of divided batch size'),
        click.option('--vis_gt', type=bool, default=True, help='Visualize ground truth'),
        click.option('--vis_every', type=int, default=0, help='Visualize every n iterations'),
        click.option('--log_every', type=int, default=1000, help='Log metrics every n iterations'),
        click.option('--rolling_checkpoint_every', type=int, default=500, help='Save rolling checkpoint every n iterations (only keeps the latest)'),
        click.option('--checkpoint_every', type=int, default=5000, help='Save permanent checkpoint every n iterations'),
        click.option('--num_iterations', type=int, default=1000000, help='Number of iterations to train the model'),
        click.option('--debug', 'debug_mode', type=bool, default=False, help='Enable debug mode'),
        click.option('--enable_ema', type=bool, default=True, help='Maintain an exponential moving average of the model weights'),
        click.option('--precision', type=click.Choice(['fp32', 'tf32', 'mixed_bf16']), default='fp32', help='Numerical precision to use'),
        click.option('--backbone_gradient_checkpoint', type=bool, default=False, help='Use gradient checkpointing in backbone'),
        click.option('--gradient_accumulation_steps', type=int, default=2, help='Number of steps to accumulate gradients'),
        click.option('--batch_size_forward', type=int, default=1, help='Batch size for each forward pass on each device'),
        click.option('--checkpoint', 'checkpoint_path', type=str, default='latest', help='Path to the checkpoint to load, step number, "latest", or "none"'),
        # NOTE: "" rather than None. Both mean "no base checkpoint" to
        # load_checkpoint's `not ckpt_path` guard, but "" also survives being
        # passed to code that expects a str.
        click.option('--base_checkpoint', type=str, default='', help='Checkpoint to fall back to when the workspace has none of its own'),
        click.option('--workspace_path', type=str, default='./workspace/', help='Path of workspace for saving visualizations and checkpoints'),
        click.option('--name', 'experiment_name', type=str, default='debug', help='Name of the experiment'),
        click.option('--config', 'config_path', type=str, default='configs/debug.json'),
    ]
    for option in options:
        fn = option(fn)
    return fn
