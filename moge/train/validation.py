"""Periodic, deterministic MoGe-2 validation and model-only best checkpoints."""
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


class ValidationRunner:
    def __init__(self, config, workspace):
        self.config = dict(config)
        self.workspace = Path(workspace)
        self.every = config.get('every', 1000)
        self.monitor = config.get('monitor', 'depth_metric/rel')
        self.mode = config.get('mode', 'min')
        self.max_samples = config.get('max_samples')
        self.resolution_level = config.get('resolution_level', 9)
        self.seed = config.get('seed', 0)
        self.datasets = config.get('datasets', {})
        visualization = config.get('visualization', {})
        if not isinstance(visualization, dict) or set(visualization) - {'num_samples', 'max_points'}:
            raise ValueError('validation.visualization accepts num_samples and max_points only')
        self.num_vis_samples = visualization.get('num_samples', 0)
        self.vis_max_points = visualization.get('max_points', 50000)
        if type(self.num_vis_samples) is not int or self.num_vis_samples < 0:
            raise ValueError('validation.visualization.num_samples must be a non-negative integer')
        if type(self.vis_max_points) is not int or self.vis_max_points < 1:
            raise ValueError('validation.visualization.max_points must be a positive integer')
        unknown = set(config) - {'every', 'monitor', 'mode', 'max_samples', 'resolution_level', 'seed', 'datasets', 'metric_groups', 'visualization'}
        if unknown:
            raise ValueError(f'Unknown validation options: {sorted(unknown)}')
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 32:
            raise ValueError('validation.seed must be an integer in [0, 2**32)')
        if not isinstance(self.monitor, str) or '/' not in self.monitor:
            raise ValueError('validation.monitor must be a metric path, e.g. depth_metric/rel')
        if type(self.every) is not int or self.every < 1:
            raise ValueError('validation.every must be a positive integer')
        if self.mode not in ('min', 'max'):
            raise ValueError('validation.mode must be min or max')
        if self.max_samples is not None and (type(self.max_samples) is not int or self.max_samples < 1):
            raise ValueError('validation.max_samples must be null or a positive integer')
        if type(self.resolution_level) is not int or not 0 <= self.resolution_level <= 9:
            raise ValueError('validation.resolution_level must be an integer from 0 to 9')
        if not isinstance(self.datasets, dict) or not self.datasets:
            raise ValueError('validation.datasets must be a non-empty mapping of names to evaluation configs')
        # Include split contents so resumed runs cannot compare different validation sets.
        # Visualization does not affect metrics; allow enabling it on resumed runs.
        identity = {'config': {k: v for k, v in self.config.items() if k != 'visualization'}, 'splits': {}}
        for name, dataset in self.datasets.items():
            if '/' in name or '\\' in name or name in ('', '.', '..', 'mean'):
                raise ValueError('Validation dataset names must be safe directory names and cannot equal mean')
            if 'index' in dataset:
                raise ValueError(f'Validation dataset {name}: use split, not index')
            for dimension in ('width', 'height'):
                if type(dataset.get(dimension)) is not int or dataset[dimension] < 1:
                    raise ValueError(f'Validation dataset {name}: {dimension} must be a positive integer')
            subset = dataset.get('subset')
            if subset is not None and (type(subset) is not int or subset < 1):
                raise ValueError(f'Validation dataset {name}: subset must be a positive integer')
            split = Path(dataset['path'], dataset.get('split', '.index.txt'))
            contents = split.read_text(encoding='utf-8')
            if not contents.splitlines() or any(not line.strip() for line in contents.splitlines()):
                raise ValueError(f'Validation split must be non-empty and contain no blank lines: {split}')
            identity['splits'][name] = contents
        self.signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.best_path = self.workspace / 'checkpoint' / 'best.pt'
        self.best_info_path = self.workspace / 'validation' / 'best.json'
        self.best_score = None
        if self.best_info_path.exists():
            best = json.loads(self.best_info_path.read_text())
            if best['signature'] != self.signature:
                raise ValueError('Validation settings/splits changed; use a new workspace for a new best metric')
            if not self.best_path.exists():
                raise FileNotFoundError(f'Best validation metadata exists but checkpoint is missing: {self.best_path}')
            self.best_score = best['score']

    def is_due(self, step, num_iterations):
        # Step numbers in saved metrics match the trainer's zero-based steps.
        return (step + 1) % self.every == 0 or step == num_iterations - 1

    @torch.no_grad()
    def evaluate(self, model, device, step=None):
        from ..test.dataloader import EvalDataLoaderPipeline
        from ..test.metrics import compute_metrics
        from ..utils.io import read_json

        # Preserve training's per-sample scale semantics without changing the
        # standalone benchmark loader's existing behavior.
        class ValidationDataLoader(EvalDataLoaderPipeline):
            def __init__(self, metric_from_meta=False, **kwargs):
                self.metric_from_meta = metric_from_meta
                super().__init__(**kwargs)

            def _load_instance(self, idx):
                sample = super()._load_instance(idx)
                if sample is not None:
                    meta = read_json(self.path / sample['filename'] / 'meta.json')
                    scale = meta.get('metric_scale', meta.get('depth_scale'))
                    sample['has_metric_annotation'] = scale is not None
                    if scale is not None:
                        sample['depth'] *= scale
                    sample['depth_mask'] &= np.isfinite(sample['depth']) & (sample['depth'] > 0)
                return sample

            def _process_instance(self, sample):
                sample = super()._process_instance(sample)
                if sample is not None and self.metric_from_meta:
                    sample['is_metric'] = sample['has_metric_annotation']
                return sample

        training_modes = [(module, module.training) for module in model.modules()]
        python_state, numpy_state = random.getstate(), np.random.get_state()
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        results, totals, counts = {}, defaultdict(float), defaultdict(int)
        try:
            model.eval()
            with torch.random.fork_rng(devices=cuda_devices):
                random.seed(self.seed)
                np.random.seed(self.seed)
                torch.manual_seed(self.seed)
                for name, dataset in self.datasets.items():
                    sums, metric_counts = defaultdict(float), defaultdict(int)
                    evaluated = skipped = 0
                    # Recreate the finite pipeline on every validation pass.
                    with ValidationDataLoader(**dataset) as loader:
                        limit = min(len(loader), self.max_samples) if self.max_samples else len(loader)
                        vis_indices = set(np.linspace(0, limit - 1, min(limit, self.num_vis_samples), dtype=int))
                        for sample_index in tqdm(
                            range(limit),
                            desc=f'Validation {name}',
                            unit='sample',
                            dynamic_ncols=True,
                        ):
                            sample = loader.get()
                            if sample is None or sample.get('label_type') == 'invalid':
                                skipped += 1
                                continue
                            sample = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
                            sample['depth_mask'] &= torch.isfinite(sample['depth']) & (sample['depth'] > 0)
                            if not sample['depth_mask'].any():
                                skipped += 1
                                continue
                            output = model.infer(sample['image'], resolution_level=self.resolution_level,
                                                 apply_mask=False, use_fp16=False)
                            for key in ('depth', 'points'):
                                if not torch.isfinite(output[key][sample['depth_mask']]).all():
                                    raise ValueError(f'Non-finite validation prediction: {name}/{sample["filename"]}/{key}')
                            if step is not None and sample_index in vis_indices:
                                from .validation_visualization import save_validation_point_clouds
                                save_validation_point_clouds(
                                    self.workspace / 'validation' / f'step_{step:08d}' / name / f'{sample_index:06d}',
                                    sample, output, self.vis_max_points,
                                )
                            pred = {'depth_metric': output['depth'], 'points_metric': output['points'],
                                    'intrinsics': output['intrinsics']}
                            metrics, _ = compute_metrics(pred, sample, mg=self.config.get('metric_groups', 'global,metric'))
                            for group, values in metrics.items():
                                for key, value in values.items():
                                    value = float(value)
                                    if not math.isfinite(value):
                                        raise ValueError(f'Non-finite validation metric: {name}/{group}/{key}')
                                    metric = f'{group}/{key}'
                                    sums[metric] += value
                                    metric_counts[metric] += 1
                            evaluated += 1
                    if not evaluated or not metric_counts:
                        raise ValueError(f'Validation dataset {name} has no evaluable samples/metrics')
                    results[name] = {key: sums[key] / metric_counts[key] for key in sums}
                    results[name].update(num_samples=evaluated, num_skipped=skipped)
                    for key in sums:
                        totals[key] += sums[key]
                        counts[key] += metric_counts[key]
        finally:
            for module, training in training_modes:
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        # Each eligible image has equal weight, rather than each dataset.
        results['mean'] = {key: totals[key] / counts[key] for key in totals}
        return results

    def record(self, results, model, model_config, step, logger):
        if self.monitor not in results['mean']:
            raise ValueError(f'Validation monitor {self.monitor!r} is missing; check metric_groups and depth_unit')
        score = float(results['mean'][self.monitor])
        if not math.isfinite(score):
            raise ValueError(f'Validation monitor is not finite: {score}')
        improved = self.best_score is None or (score < self.best_score if self.mode == 'min' else score > self.best_score)
        info = {'step': step, 'monitor': self.monitor, 'mode': self.mode, 'score': score,
                'signature': self.signature, 'metrics': results}
        output_dir = self.workspace / 'validation'
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f'step_{step:08d}.json').write_text(json.dumps(info, indent=2, allow_nan=False))
        if improved:
            self.best_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.best_path.with_suffix('.pt.tmp')
            # A model-only checkpoint is compatible with v2.from_pretrained and
            # --initial_checkpoint. Do not change the resume pointer latest.pt.
            torch.save({'model_config': model_config,
                        'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'validation': info}, temporary)
            temporary.replace(self.best_path)
            temporary_info = self.best_info_path.with_suffix('.json.tmp')
            temporary_info.write_text(json.dumps(info, indent=2, allow_nan=False))
            temporary_info.replace(self.best_info_path)
            self.best_score = score
        scalars = {f'val/{name}/{key}': value for name, metrics in results.items() for key, value in metrics.items()}
        scalars['val/best_score'] = self.best_score
        logger.log_validation(scalars, step)
        print(f'[Validation step {step}] {self.monitor}={score:.6g}, best={self.best_score:.6g}'
              + (' (saved best.pt)' if improved else ''))

    def run(self, model, accelerator, model_config, step, logger):
        """Called on every rank; only rank zero evaluates the unwrapped model."""
        accelerator.wait_for_everyone()
        status = [None]
        if accelerator.is_main_process:
            try:
                unwrapped = accelerator.unwrap_model(model)
                results = self.evaluate(unwrapped, accelerator.device, step=step)
                self.record(results, unwrapped, model_config, step, logger)
            except Exception as exc:
                status[0] = f'{type(exc).__name__}: {exc}'
        if accelerator.num_processes > 1:
            from accelerate.utils import broadcast_object_list
            broadcast_object_list(status, from_process=0)
        if status[0] is not None:
            raise RuntimeError(f'Validation failed: {status[0]}')
        accelerator.wait_for_everyone()
