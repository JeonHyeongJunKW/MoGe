"""Report real-image agreement with a cached Teacher, separately from GT validation."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ..utils.io import read_depth, read_image


def validate_reference(root):
    root = Path(root)
    manifest_path = root / 'dataset.json'
    manifest = json.loads(manifest_path.read_text())
    rows = manifest.get('samples', [])
    if (manifest.get('format') != 'moge_v4l2_teacher_reference_v1' or
            manifest.get('status') != 'complete' or not rows or
            any(row['split'] != 'test' or row['camera'] != 'front_left' for row in rows) or
            len({row['session'] for row in rows}) != 1 or
            len({row['sample'] for row in rows}) != len(rows)):
        raise ValueError(f'Expected one complete independent front_left Teacher reference: {root}')
    for row in rows:
        relative = Path(row['sample'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe sample path: {relative}')
        for name in ('image.jpg', 'rectification_valid.png', 'teacher_labels.npz'):
            if not (root / relative / name).is_file():
                raise FileNotFoundError(root / relative / name)
    return manifest, manifest_path.read_bytes()


def validate_pseudo_train(root):
    root = Path(root)
    manifest_path = root / 'dataset.json'
    manifest = json.loads(manifest_path.read_text())
    rows = [row for row in manifest.get('samples', []) if row.get('split') == 'train']
    if (manifest.get('format') != 'moge_v4l2_multitask_v1' or
            manifest.get('status') != 'complete' or not rows or
            len({row['sample'] for row in rows}) != len(rows)):
        raise ValueError(f'Expected a complete V4L2 multitask training dataset: {root}')
    declared = (root / '.train.txt').read_text().splitlines()
    if set(declared) != {row['sample'] for row in rows} or len(declared) != len(rows):
        raise ValueError('V4L2 training index differs from manifest')
    for row in rows:
        relative = Path(row['sample'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe sample path: {relative}')
        for name in ('image.jpg', 'depth.png', 'teacher_labels.npz'):
            if not (root / relative / name).is_file():
                raise FileNotFoundError(root / relative / name)
    return rows, manifest_path.read_bytes(), (root / '.train.txt').read_bytes()


@torch.no_grad()
def evaluate_pseudo_train(model, device, root):
    """Mean per-image metric Abs Rel to frozen Teacher depth on all V4L2 train frames."""
    root = Path(root)
    rows, _, _ = validate_pseudo_train(root)
    rows.sort(key=lambda row: (row['session'], row['camera'], int(Path(row['sample']).name)))
    records = []
    for row in rows:
        folder = root / row['sample']
        rgb = read_image(folder / 'image.jpg')
        depth = read_depth(folder / 'depth.png').astype(np.float32)
        with np.load(folder / 'teacher_labels.npz', allow_pickle=False) as labels:
            valid = labels['depth_valid'].astype(bool)
        if depth.shape != rgb.shape[:2] or valid.shape != depth.shape:
            raise ValueError(f'Invalid V4L2 train depth shape: {folder}')
        x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().to(device) / 255
        prediction = model.infer(x, num_tokens=400, use_fp16=True, apply_mask=False)
        predicted = prediction['depth'].cpu().numpy()
        target = valid & np.isfinite(depth) & (depth >= .1) & (depth <= 100)
        both = target & np.isfinite(predicted) & (predicted > 0)
        if not both.any():
            raise ValueError(f'No valid V4L2 train depth pixels: {folder}')
        ratio = np.maximum(predicted[both] / depth[both], depth[both] / predicted[both])
        absolute_error = np.abs(predicted[both] - depth[both])
        records.append((float(np.mean(absolute_error / depth[both])),
                        float(np.mean(ratio < 1.25)), float(both.sum() / max(target.sum(), 1)),
                        float(np.mean(absolute_error))))
    values = np.asarray(records)
    return {'depth_teacher/rel': float(values[:, 0].mean()),
            'depth_teacher/delta1': float(values[:, 1].mean()),
            'depth_teacher/coverage': float(values[:, 2].mean()),
            'depth_teacher/mae_m': float(values[:, 3].mean()),
            'num_samples': len(rows)}


@torch.no_grad()
def evaluate_teacher_reference(model, device, root):
    root = Path(root)
    manifest, _ = validate_reference(root)
    rows = sorted(manifest['samples'], key=lambda r: int(Path(r['sample']).name))
    metrics = []
    for row in rows:
        folder = root / row['sample']
        rgb = read_image(folder / 'image.jpg')
        rect_image = cv2.imread(str(folder / 'rectification_valid.png'), cv2.IMREAD_GRAYSCALE)
        if rect_image is None or rect_image.shape != rgb.shape[:2]:
            raise ValueError(f'Invalid rectification mask: {folder}')
        rect = rect_image > 0
        with np.load(folder / 'teacher_labels.npz', allow_pickle=False) as labels:
            td = labels['depth'].astype(np.float32)
            tm = labels['mask'].astype(bool)
            tp = labels['mask_probability'].astype(np.float32)
            tn = labels['normal'].astype(np.float32)
        if td.shape != rect.shape or tm.shape != rect.shape or tp.shape != rect.shape or tn.shape != (*rect.shape, 3):
            raise ValueError(f'Invalid Teacher label shapes: {folder}')
        x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().to(device) / 255
        pred = model.infer(x, num_tokens=400, use_fp16=True,
                           apply_mask=False, return_mask_probability=True)
        d, mask, normal, prob = (pred[k].cpu().numpy() for k in
                                 ('depth', 'mask', 'normal', 'mask_probability'))
        tv = rect & tm & np.isfinite(td) & (td >= .1) & (td <= 100)
        both = tv & np.isfinite(d) & (d > 0)
        if not both.any() or not rect.any():
            raise ValueError(f'No valid comparison pixels: {folder}')
        ratio = np.maximum(d[both] / td[both], td[both] / d[both])
        record = {
            'depth_teacher/rel': float(np.mean(np.abs(d[both] - td[both]) / td[both])),
            'depth_teacher/delta1': float(np.mean(ratio < 1.25)),
            'depth_teacher/coverage': float(both.sum() / max(tv.sum(), 1)),
            'mask_teacher/agreement': float(np.mean((prob[rect] > .5) == (tp[rect] > .5))),
        }
        negatives = rect & (tp <= .5)
        if negatives.any():
            record['mask_teacher/negative_recall'] = float(np.mean(prob[negatives] <= .5))
        nv = tv & rect & mask & np.isfinite(normal).all(-1) & np.isfinite(tn).all(-1)
        if nv.any():
            dot = np.sum(normal[nv] * tn[nv], axis=-1)
            norms = np.linalg.norm(normal[nv], axis=-1) * np.linalg.norm(tn[nv], axis=-1)
            record['normal_teacher/angle_deg'] = float(np.degrees(
                np.arccos(np.clip(dot / np.maximum(norms, 1e-6), -1, 1))).mean())
        metrics.append(record)
    keys = sorted({key for record in metrics for key in record})
    result = {key: float(np.mean([record[key] for record in metrics if key in record])) for key in keys}
    result['num_samples'] = len(rows)
    return result
