"""Pixel-weighted depth errors grouped by ground-truth camera-Z distance."""
import csv
import json
import math
from pathlib import Path

import numpy as np


def validate_depth_bins(config):
    if config is None:
        return None
    if not isinstance(config, dict) or set(config) - {'width_m', 'max_depth_m'}:
        raise ValueError('validation.depth_bins accepts width_m and max_depth_m only')
    width, maximum = config.get('width_m', 1.0), config.get('max_depth_m', 1000.0)
    for value in (width, maximum):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('depth_bins width_m and max_depth_m must be positive finite numbers')
    count = maximum / width
    if count > 10000 or not math.isclose(count, round(count), rel_tol=0, abs_tol=1e-8) or count < 1:
        raise ValueError('depth_bins max_depth_m must be an integer multiple of width_m, with at most 10000 bins')
    return {'width_m': float(width), 'max_depth_m': float(maximum)}


class DepthBinAccumulator:
    def __init__(self, config):
        self.config = validate_depth_bins(config)
        self.width = self.config['width_m']
        self.maximum = self.config['max_depth_m']
        self.regular_bins = round(self.maximum / self.width)
        size = self.regular_bins + 1  # Overflow is reported, never silently discarded.
        self.pixels = np.zeros(size, dtype=np.int64)
        self.images = np.zeros(size, dtype=np.int64)
        self.sums = np.zeros((4, size), dtype=np.float64)
        self.skipped_nonmetric_images = 0

    def update(self, prediction, target, mask, is_metric=True):
        if not is_metric:
            self.skipped_nonmetric_images += 1
            return
        def array(value):
            return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
        pred, gt, valid = array(prediction), array(target), array(mask).astype(bool)
        if pred.shape != gt.shape or valid.shape != gt.shape:
            raise ValueError('Depth-bin prediction, target and mask shapes must match')
        valid = valid & np.isfinite(gt) & (gt > 0)
        pred, gt = pred[valid].astype(np.float64), gt[valid].astype(np.float64)
        if not np.isfinite(pred).all():
            raise ValueError('Non-finite prediction in valid depth-bin pixels')
        # Clip before converting to integers, including extreme but finite depths.
        indices = np.floor(np.minimum(gt, self.maximum) / self.width).astype(np.int64)
        indices = np.minimum(indices, self.regular_bins)
        counts = np.bincount(indices, minlength=len(self.pixels))
        self.pixels += counts
        self.images += counts > 0
        absolute = np.abs(pred - gt)
        # Match the overall depth metric epsilon convention; nonpositive predictions fail delta1.
        with np.errstate(divide='ignore', invalid='ignore'):
            delta1 = (pred > 0) & (np.maximum(gt / (pred + 1e-6), pred / (gt + 1e-6)) < 1.25)
        values = (absolute / (gt + 1e-6), delta1.astype(float), absolute, np.square(absolute))
        for row, values_for_metric in zip(self.sums, values):
            row += np.bincount(indices, weights=values_for_metric, minlength=len(self.pixels))

    def merge(self, other):
        if self.config != other.config:
            raise ValueError('Cannot merge different depth-bin configurations')
        self.pixels += other.pixels
        self.images += other.images
        self.sums += other.sums
        self.skipped_nonmetric_images += other.skipped_nonmetric_images

    def rows(self):
        rows = []
        for i, count in enumerate(self.pixels):
            means = self.sums[:, i] / count if count else None
            rows.append({
                'lower_m': i * self.width,
                'upper_m': (i + 1) * self.width if i < self.regular_bins else None,
                'num_pixels': int(count), 'num_images': int(self.images[i]),
                'rel': float(means[0]) if count else None,
                'delta1': float(means[1]) if count else None,
                'mae_m': float(means[2]) if count else None,
                'rmse_m': float(np.sqrt(means[3])) if count else None,
            })
        return rows


def depth_bin_report(accumulators, config):
    combined = DepthBinAccumulator(config)
    for accumulator in accumulators.values():
        combined.merge(accumulator)
    return {
        **combined.config,
        'basis': 'GT camera-Z depth in meters after validation preprocessing; no scale/shift alignment',
        'interval': 'lower_m <= GT < upper_m; null upper_m means overflow',
        'aggregation': 'pixel-weighted across eligible images; empty-bin metrics are null',
        'mask': 'same valid GT depth mask as overall metrics; predicted mask does not exclude pixels',
        'skipped_nonmetric_images': {k: a.skipped_nonmetric_images for k, a in accumulators.items()},
        'datasets': {k: a.rows() for k, a in accumulators.items()},
        'all_datasets': combined.rows(),
    }


def format_depth_bin_report(report):
    """Console tables for populated bins; all bins remain in JSON/CSV."""
    lines = [f'[Depth bins step {report["step"]}] GT camera-Z (m), pixel-weighted; empty bins omitted']
    tables = [('dataset: ' + name, rows) for name, rows in report['datasets'].items()]
    if len(tables) > 1:
        tables.append(('all datasets', report['all_datasets']))
    for label, rows in tables:
        lines += ['', label,
                  f'{"GT depth (m)":>18} | {"pixels":>12} | {"images":>7} | '
                  f'{"REL":>10} | {"delta1 (%)":>10} | {"MAE (m)":>10} | {"RMSE (m)":>10}']
        populated = [row for row in rows if row['num_pixels'] > 0]
        for row in populated:
            lower, upper = row['lower_m'], row['upper_m']
            interval = f'[{lower:g}, {upper:g})' if upper is not None else f'>= {lower:g}'
            lines.append(
                f'{interval:>18} | {row["num_pixels"]:12,d} | {row["num_images"]:7,d} | '
                f'{row["rel"]:10.6f} | {100 * row["delta1"]:10.2f} | '
                f'{row["mae_m"]:10.6f} | {row["rmse_m"]:10.6f}'
            )
        if not populated:
            lines.append('No valid metric-depth pixels.')
    skipped = sum(report['skipped_nonmetric_images'].values())
    if skipped:
        lines.append(f'Skipped non-metric images: {skipped}')
    return '\n'.join(lines)


def write_depth_bin_report(prefix, report):
    prefix = Path(prefix)
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    fields = ['scope', 'dataset', 'lower_m', 'upper_m', 'num_pixels', 'num_images', 'rel', 'delta1', 'mae_m', 'rmse_m']
    with prefix.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, rows in report['datasets'].items():
            writer.writerows({'scope': 'dataset', 'dataset': name, **row} for row in rows)
        writer.writerows({'scope': 'all', 'dataset': '', **row} for row in report['all_datasets'])
