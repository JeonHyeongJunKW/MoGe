"""CPU regression tests; no pretrained downloads or evaluation datasets required."""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from moge.train.validation import ValidationRunner


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.child = torch.nn.Dropout()
        self.calls = 0

    def infer(self, image, **kwargs):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert kwargs['apply_mask'] is False
        self.calls += 1
        random.random()
        np.random.rand()
        torch.rand(1)
        depth = torch.ones(2, 2) * self.weight
        return {'depth': depth, 'points': depth[..., None].expand(2, 2, 3), 'intrinsics': torch.eye(3)}


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / '.val.txt').write_text('sample_a\nsample_b\n')
        self.config = {
            'every': 2, 'metric_groups': 'depth_metric',
            'datasets': {'val': {'path': str(self.root), 'split': '.val.txt',
                                 'width': 2, 'height': 2, 'depth_unit': 1.0}},
        }
        self.model = TinyModel()
        self.logger = SimpleNamespace(log_validation=Mock())

    def runner(self, config=None):
        return ValidationRunner(config or self.config, self.root / 'workspace')

    def fake_dependencies(self, invalid=False, fail=False):
        class Loader:
            opened = closed = 0

            def __init__(self, path, depth_unit=None, **kwargs):
                self.path = Path(path)
                self.depth_unit = depth_unit
                self.index = 0

            def __len__(self):
                return 2

            def __enter__(self):
                Loader.opened += 1
                return self

            def __exit__(self, *args):
                Loader.closed += 1

            def get(self):
                sample = self._process_instance(self._load_instance(self.index))
                self.index += 1
                return sample

            def _load_instance(self, index):
                sample = {'filename': f'sample_{index}', 'image': torch.ones(3, 2, 2),
                          'depth': np.full((2, 2), float(index + 1), dtype=np.float32),
                          'depth_mask': np.ones((2, 2), dtype=bool)}
                if invalid:
                    sample['label_type'] = 'invalid'
                return sample

            def _process_instance(self, sample):
                sample['depth'] = torch.from_numpy(sample['depth']) * (self.depth_unit or 1)
                sample['depth_mask'] = torch.from_numpy(sample['depth_mask'])
                sample['is_metric'] = self.depth_unit is not None
                return sample

        def metrics(pred, sample, **kwargs):
            if fail:
                raise ValueError('metric failure')
            if not sample['is_metric']:
                return {}, {}
            rel = ((pred['depth_metric'] - sample['depth']).abs() / sample['depth'])[sample['depth_mask']].mean()
            return {'depth_metric': {'rel': rel.item()}}, {}

        loader_module = ModuleType('moge.test.dataloader')
        loader_module.EvalDataLoaderPipeline = Loader
        metric_module = ModuleType('moge.test.metrics')
        metric_module.compute_metrics = metrics
        io_module = ModuleType('moge.utils.io')
        io_module.read_json = lambda path: {'metric_scale': 2.0}
        return patch.dict('sys.modules', {'moge.test.dataloader': loader_module,
                                         'moge.test.metrics': metric_module,
                                         'moge.utils.io': io_module}), Loader

    def test_schedule_and_config_errors(self):
        runner = self.runner()
        self.assertFalse(runner.is_due(0, 5))
        self.assertTrue(runner.is_due(1, 5))
        self.assertTrue(runner.is_due(4, 5))
        for overrides in ({'every': 0}, {'max_samples': 0}, {'mode': 'other'},
                          {'resolution_level': 10}, {'datasets': {}}, {'everry': 1}):
            with self.assertRaises(ValueError):
                self.runner({**self.config, **overrides})
        (self.root / '.val.txt').write_text('')
        with self.assertRaisesRegex(ValueError, 'non-empty'):
            self.runner()

    def test_repeated_evaluation_restores_modes_rng_and_applies_meta_scale(self):
        runner = self.runner()
        self.model.train()
        self.model.child.eval()
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        deps, loader = self.fake_dependencies()
        with deps:
            first = runner.evaluate(self.model, torch.device('cpu'))
            second = runner.evaluate(self.model, torch.device('cpu'))
        # Raw GT 1,2 becomes 2,4 via meta scale; relative errors .5,.75.
        self.assertAlmostEqual(first['mean']['depth_metric/rel'], .625)
        self.assertEqual(first, second)
        self.assertEqual(first['val']['num_samples'], 2)
        self.assertEqual(loader.opened, loader.closed)
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.child.training)
        self.assertEqual(py_state, random.getstate())
        self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))
        self.assertEqual(np_state[2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertIsNone(self.model.weight.grad)

    def test_sample_weighted_mean_and_limit(self):
        cfg = copy.deepcopy(self.config)
        cfg['datasets']['other'] = {**cfg['datasets']['val'], 'depth_unit': 2.0}
        deps, _ = self.fake_dependencies()
        with deps:
            result = self.runner(cfg).evaluate(self.model, torch.device('cpu'))
        self.assertAlmostEqual(result['mean']['depth_metric/rel'], (.5 + .75 + .75 + .875) / 4)
        cfg['max_samples'] = 1
        deps, _ = self.fake_dependencies()
        with deps:
            result = self.runner(cfg).evaluate(self.model, torch.device('cpu'))
        self.assertAlmostEqual(result['mean']['depth_metric/rel'], (.5 + .75) / 2)
        self.assertEqual(result['val']['num_samples'], 1)

    def test_invalid_samples_and_failure_restore_state(self):
        for invalid, fail in ((True, False), (False, True)):
            deps, loader = self.fake_dependencies(invalid=invalid, fail=fail)
            state = torch.get_rng_state()
            with deps, self.assertRaises(ValueError):
                self.runner().evaluate(self.model, torch.device('cpu'))
            self.assertTrue(self.model.training)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertEqual(loader.opened, loader.closed)

    def test_best_checkpoint_improvement_resume_and_signature(self):
        runner = self.runner()
        def record(score, step):
            runner.record({'mean': {'depth_metric/rel': score}}, self.model, {'encoder': {}}, step, self.logger)
        record(.3, 1)
        with torch.no_grad():
            self.model.weight.fill_(2)
        record(.4, 3)
        checkpoint = torch.load(runner.best_path, map_location='cpu')
        self.assertEqual(checkpoint['model']['weight'].item(), 1)
        self.assertNotIn('optimizer', checkpoint)
        self.assertNotIn('step', checkpoint)
        self.assertIn('model_config', checkpoint)
        self.assertFalse((runner.best_path.parent / 'latest.pt').exists())
        record(.2, 5)
        self.assertEqual(torch.load(runner.best_path)['model']['weight'].item(), 2)
        self.assertAlmostEqual(self.runner().best_score, .2)
        self.assertEqual(self.logger.log_validation.call_count, 3)
        with self.assertRaisesRegex(ValueError, 'not finite'):
            record(float('nan'), 7)
        with self.assertRaisesRegex(ValueError, 'missing'):
            runner.record({'mean': {}}, self.model, {}, 7, self.logger)
        (self.root / '.val.txt').write_text('changed_sample\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.runner()

    def test_max_mode_and_metric_from_meta(self):
        cfg = copy.deepcopy(self.config)
        cfg['mode'] = 'max'
        cfg['datasets']['val']['depth_unit'] = None
        cfg['datasets']['val']['metric_from_meta'] = True
        runner = self.runner(cfg)
        deps, _ = self.fake_dependencies()
        with deps:
            result = runner.evaluate(self.model, torch.device('cpu'))
        self.assertAlmostEqual(result['mean']['depth_metric/rel'], .625)
        for step, score in enumerate((.1, .2, .15)):
            runner.record({'mean': {'depth_metric/rel': score}}, self.model, {}, step, self.logger)
        self.assertEqual(runner.best_score, .2)

    def test_run_and_error_propagation(self):
        runner = self.runner()
        accelerator = SimpleNamespace(is_main_process=True, num_processes=1, device=torch.device('cpu'),
                                      unwrap_model=lambda model: model, wait_for_everyone=Mock())
        deps, _ = self.fake_dependencies()
        with deps:
            runner.run(self.model, accelerator, {}, 1, self.logger)
        self.assertTrue(runner.best_path.exists())
        self.assertEqual(accelerator.wait_for_everyone.call_count, 2)
        deps, _ = self.fake_dependencies(fail=True)
        with deps, self.assertRaisesRegex(RuntimeError, 'metric failure'):
            runner.run(self.model, accelerator, {}, 3, self.logger)
        # A waiting non-main rank must also fail if rank zero reported an error.
        accelerator.is_main_process = False
        accelerator.num_processes = 2
        module = ModuleType('accelerate.utils')
        def broadcast(status, from_process):
            status[0] = 'ValueError: rank zero failed'
        module.broadcast_object_list = broadcast
        with patch.dict('sys.modules', {'accelerate.utils': module}), self.assertRaisesRegex(RuntimeError, 'rank zero failed'):
            runner.run(self.model, accelerator, {}, 3, self.logger)

    def test_visualization_selection_and_resume_compatibility(self):
        original = self.runner()
        original.record({'mean': {'depth_metric/rel': .5}}, self.model, {}, 1, self.logger)
        cfg = {**self.config, 'visualization': {'num_samples': 1, 'max_points': 10}}
        runner = self.runner(cfg)
        self.assertEqual(runner.best_score, .5)
        deps, _ = self.fake_dependencies()
        module = ModuleType('moge.train.validation_visualization')
        module.save_validation_point_clouds = Mock()
        with deps, patch.dict('sys.modules', {'moge.train.validation_visualization': module}):
            runner.evaluate(self.model, torch.device('cpu'), step=3)
            runner.evaluate(self.model, torch.device('cpu'), step=5)
        calls = module.save_validation_point_clouds.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0][1]['filename'], calls[1][0][1]['filename'])
        self.assertEqual(calls[0][0][0].name, '000000')
        self.assertIn('step_00000003', str(calls[0][0][0]))
        self.assertIn('step_00000005', str(calls[1][0][0]))


if __name__ == '__main__':
    unittest.main()
