"""Export validation point clouds in their original camera coordinates."""
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _numpy(value):
    return value.detach().float().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def _write_point_cloud(path, points, colors):
    """Binary PLY with float32 XYZ and uint8 RGB, readable by standard 3D viewers."""
    vertices = np.empty(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for index, name in enumerate(('x', 'y', 'z')):
        vertices[name] = points[:, index]
    for index, name in enumerate(('red', 'green', 'blue')):
        vertices[name] = colors[:, index]
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {len(points)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with Path(path).open('wb') as output:
        output.write(header.encode('ascii'))
        output.write(vertices.tobytes())


def save_validation_point_clouds(directory, sample, output, max_points):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rgb = np.round(_numpy(sample['image']).transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
    gt = _numpy(sample['points']).reshape(-1, 3)
    pred = _numpy(output['points']).reshape(-1, 3)
    colors = rgb.reshape(-1, 3)
    valid_gt = _numpy(sample['depth_mask']).astype(bool).reshape(-1)
    valid_gt &= np.isfinite(gt).all(axis=1) & (gt[:, 2] > 0)
    indices = np.flatnonzero(valid_gt)
    if len(indices) > max_points:
        indices = indices[np.linspace(0, len(indices) - 1, max_points, dtype=int)]
    pred_valid = np.isfinite(pred[indices]).all(axis=1) & (pred[indices, 2] > 0)
    pred_indices = indices[pred_valid]
    _write_point_cloud(directory / 'gt.ply', gt[indices], colors[indices])
    _write_point_cloud(directory / 'pred.ply', pred[pred_indices], colors[pred_indices])
    overlay_points = np.concatenate([gt[indices], pred[pred_indices]])
    overlay_colors = np.concatenate([
        np.tile(np.array([0, 200, 255], dtype=np.uint8), (len(indices), 1)),
        np.tile(np.array([255, 120, 0], dtype=np.uint8), (len(pred_indices), 1)),
    ])
    _write_point_cloud(directory / 'overlay.ply', overlay_points, overlay_colors)
    Image.fromarray(rgb).save(directory / 'image.png')
    metadata = {
        'filename': sample['filename'],
        'coordinate_frame': 'OpenCV camera: +X right, +Y down, +Z forward',
        'gt_units': 'meters' if sample['is_metric'] else 'relative',
        'prediction_units': 'meters',
        'alignment': 'none; original GT and predicted scale, shift and camera intrinsics',
        'overlay_colors': {'gt': [0, 200, 255], 'prediction': [255, 120, 0]},
        'gt_points': len(indices),
        'prediction_points': len(pred_indices),
        'prediction_points_removed': int((~pred_valid).sum()),
        'sampling': 'evenly spaced valid GT pixels; prediction uses the same selected pixels',
        'mask': 'GT valid depth; prediction mask is not applied, matching validation metrics',
        'gt_intrinsics': _numpy(sample['intrinsics']).tolist(),
        'pred_intrinsics': _numpy(output['intrinsics']).tolist(),
    }
    (directory / 'info.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
