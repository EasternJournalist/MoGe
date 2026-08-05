"""Training-time visualisation dumps shared by the MoGe-3 entry points."""
import json
from pathlib import Path
from typing import *

import cv2
import numpy as np
import torch
try:
    import utils3d_moge as utils3d
except ImportError:
    import utils3d
from tqdm import tqdm

from ..utils.vis import colorize_depth, colorize_normal

EXR_FLOAT = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]


def _write_rgb(path: Path, image: np.ndarray, params: Optional[List[int]] = None):
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), params or [])


def visualize_gt(
    batches_for_vis: List[Dict[str, Any]],
    workspace: Path,
    batch_size_forward: int,
    initial_step: int,
    logger,
):
    """Dump the ground truth of the held-out visualisation batches once."""
    save_dir = Path(workspace).joinpath('vis/gt')
    for i_batch, batch in enumerate(tqdm(batches_for_vis, desc='Visualize GT', leave=False)):
        image, gt_depth, gt_normal, gt_intrinsics, info = (
            batch['image'], batch['depth'], batch['normal'], batch['intrinsics'], batch['info']
        )
        gt_points = utils3d.pt.depth_map_to_point_map(gt_depth, intrinsics=gt_intrinsics)
        for i_instance in range(batch['image'].shape[0]):
            idx = i_batch * batch_size_forward + i_instance
            image_i = (image[i_instance].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            gt_depth_i = gt_depth[i_instance].numpy()
            instance_dir = save_dir.joinpath(f'{idx:04d}')
            instance_dir.mkdir(parents=True, exist_ok=True)
            _write_rgb(instance_dir / 'image.jpg', image_i)
            _write_rgb(instance_dir / 'points.exr', gt_points[i_instance].numpy(), EXR_FLOAT)
            _write_rgb(instance_dir / 'depth_vis.png', colorize_depth(gt_depth_i))
            _write_rgb(instance_dir / 'normal.png', colorize_normal(gt_normal[i_instance].numpy()))
            logger.log_images({
                f'{idx:04d}-image-gt': image_i,
                f'{idx:04d}-depth_vis-gt': colorize_depth(gt_depth_i),
            }, step=initial_step)
            with instance_dir.joinpath('info.json').open('w') as f:
                json.dump(info[i_instance], f)


def visualize_predictions(
    batches_for_vis: List[Dict[str, Any]],
    model,
    accelerator,
    workspace: Path,
    device,
    batch_size_forward: int,
    i_step: int,
    refine_steps: int,
    logger,
):
    """Run inference on the visualisation batches and dump one set of maps per refine step."""
    unwrapped_model = accelerator.unwrap_model(model)
    save_dir = Path(workspace).joinpath(f'vis/step_{i_step:08d}')
    save_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for i_batch, batch in enumerate(tqdm(batches_for_vis, desc=f'Visualize: {i_step:08d}', leave=False)):
            image = batch['image'].to(device)
            output = unwrapped_model.infer(image, refine_steps=refine_steps)
            pred_points_all = [step.cpu().numpy() for step in output['points_per_step']]
            if 'depth_per_step' in output:
                pred_depth_all = [step.cpu().numpy() for step in output['depth_per_step']]
            else:
                pred_depth_all = [output['depth'].cpu().numpy()]
            pred_mask = output['mask'].cpu().numpy()
            image = image.cpu().numpy()

            for i_instance in range(image.shape[0]):
                idx = i_batch * batch_size_forward + i_instance
                pred_mask_i = pred_mask[i_instance]
                instance_dir = save_dir.joinpath(f'{idx:04d}')
                instance_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(instance_dir / f'mask_train_step_{i_step:08d}.png'), pred_mask_i * 255)
                images_to_log = {}
                for i_refine_step, (points_step, depth_step) in enumerate(zip(pred_points_all, pred_depth_all)):
                    suffix = f'train_step_{i_step:08d}_refine_step_{i_refine_step:02d}'
                    depth_vis = colorize_depth(depth_step[i_instance], pred_mask_i)
                    _write_rgb(instance_dir / f'points_{suffix}.exr', points_step[i_instance], EXR_FLOAT)
                    _write_rgb(instance_dir / f'depth_vis_{suffix}.png', depth_vis)
                    images_to_log[
                        f'{idx:04d}-depth_vis-pred-train-step-{i_step:06d}-refine-step-{i_refine_step:02d}'
                    ] = depth_vis
                logger.log_images(images_to_log, step=i_step)
