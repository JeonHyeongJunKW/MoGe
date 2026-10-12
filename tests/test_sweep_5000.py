import copy
import fnmatch
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from run_script.sweep_5000 import REPO, resolve_experiment


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((REPO / 'configs/experiments/gaemi_finetune_sweep_5000.json').read_text())
        self.base = json.loads((REPO / self.manifest['base_config']).read_text())

    def test_groups_losses_schedule_and_effective_batch(self):
        for experiment in self.manifest['experiments']:
            config, batch, accumulation = resolve_experiment(self.base, experiment, 200, 32, 1)
            self.assertEqual(batch * accumulation, 32)
            groups = config['optimizer']['params']
            self.assertEqual(len(config['lr_scheduler']['params']['lr_lambda']), len(groups))
            for component, lr in experiment['learning_rates'].items():
                matches = [g for g in groups if any(fnmatch.fnmatch(f'{component}.weight', p)
                                                   for p in g['params']['include'])]
                self.assertEqual(len(matches), 1)
                self.assertEqual(matches[0]['lr'], lr)
            self.assertEqual(config['loss'], self.base['loss'])
            self.assertEqual(config['data']['image_augmentation'], [])
            self.assertTrue(all(not fnmatch.fnmatch('encoder.backbone.weight', pattern)
                                for g in groups for pattern in g['params']['include']))
        self.assertEqual(self.base['data']['image_augmentation'], ['jittering'])

    def test_reject_unsafe_or_unsupervised_configuration(self):
        e = self.manifest['experiments'][1]
        with self.assertRaisesRegex(ValueError, 'divisible'):
            resolve_experiment(self.base, e, 200, 32, 1, 3)
        for head in ('normal', 'mask'):
            base = copy.deepcopy(self.base)
            base['loss']['A'].pop(head)
            with self.assertRaisesRegex(ValueError, 'requires a positive'):
                resolve_experiment(base, e, 200, 32, 1)

    def test_dry_run_emits_executable_flags_and_smoke_has_no_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            for smoke in (False, True):
                root = Path(temporary) / str(smoke)
                args = [str(REPO / 'run_script/run_gaemi_finetune_sweep_5000.sh'),
                        '--dry-run', '--root', str(root)]
                if smoke:
                    args.append('--smoke')
                subprocess.run(args, check=True, capture_output=True, text=True)
                for e in self.manifest['experiments']:
                    cmd = json.loads((root / e['name'] / 'command.json').read_text())
                    self.assertEqual(cmd[cmd.index('--num_iterations') + 1], '10' if smoke else '5000')
                    self.assertEqual('--val_config' in cmd, not smoke)
                    for name in ('neck', 'normal_head', 'mask_head', 'scale_head', 'points_head'):
                        self.assertEqual(cmd[cmd.index('--train_' + name) + 1], str(name in e['learning_rates']))
                    self.assertEqual(cmd[cmd.index('--checkpoint') + 1], 'none')
                    self.assertFalse((root / e['name'] / 'exit_code').exists())
                self.assertEqual((root / 'pretrained_baseline').exists(), not smoke)
                self.assertIn('dry_run', (root / 'summary.csv').read_text())
                rerun = subprocess.run(args, capture_output=True, text=True)
                self.assertNotEqual(rerun.returncode, 0)
                self.assertIn('not empty', rerun.stderr)


if __name__ == '__main__':
    unittest.main()
