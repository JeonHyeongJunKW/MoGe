import copy
import fnmatch
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from run_script.sweep_5000 import REPO, resolve_experiment


class SweepTests(unittest.TestCase):
    def test_teacher_dry_run_precedes_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([
                sys.executable, str(REPO / 'run_script/sweep_5000.py'),
                '--manifest', str(REPO / 'configs/experiments/gaemi_flip_continue_tokens_total30000.json'),
                '--root', temporary, '--dry-run',
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            root = Path(temporary)
            teacher = json.loads((root / 'teacher_moge2_vitl_normal_t1200/command.json').read_text())
            self.assertIn(str(REPO / 'run_script/evaluate_teacher.py'), teacher)
            self.assertNotIn('--num_iterations', teacher)
            self.assertEqual(teacher[teacher.index('--num-tokens') + 1], '1200')
            self.assertEqual(teacher[teacher.index('--batch-size') + 1], '1')
            self.assertLess(result.stdout.index('evaluate_teacher.py'), result.stdout.index('accelerate.commands.launch'))
            large = root / 'teacher_vitl_t1200_train2000'
            large_command = json.loads((large / 'command.json').read_text())
            self.assertIn(str(REPO / 'run_script/run_checkpoint_experiment.py'), large_command)
            self.assertEqual(large_command[large_command.index('--num_iterations') + 1], '2000')
            self.assertEqual(large_command[large_command.index('--gradient_accumulation_steps') + 1], '32')
            self.assertEqual(json.loads((large / 'validation_config.json').read_text())['batch_size'], 1)
            self.assertFalse((root / 'teacher_vitl_t1200_train2000_initial').exists())
            for tokens in (400, 160):
                command = json.loads((root / f'flip_mild_continue_t{tokens}_add20000/command.json').read_text())
                self.assertEqual(command[command.index('--num_iterations') + 1], '20000')
                self.assertIn(f'best_flip_mild_t{tokens}_total10000.pt', command[command.index('--initial_checkpoint') + 1])

    def test_large_training_architecture_and_batch(self):
        from run_script.run_checkpoint_experiment import configure_model
        manifest = json.loads((REPO / 'configs/experiments/gaemi_flip_continue_tokens_total30000.json').read_text())
        base = json.loads((REPO / manifest['base_config']).read_text())
        experiment = manifest['experiments'][0]
        self.assertEqual(experiment['name'], 'teacher_vitl_t1200_train2000')
        config, batch, accumulation = resolve_experiment(base, experiment, 200, 32, 1)
        self.assertEqual((batch, accumulation), (1, 32))
        checkpoint_config = copy.deepcopy(base['model'])
        checkpoint_config['encoder'].update(backbone='dinov2_vitl14', intermediate_layers=[11, 23], dim_out=1024)
        checkpoint_config['scale_head']['dims'] = [1024, 1024, 1024, 1]
        result = configure_model(config, checkpoint_config)
        self.assertEqual(result['model']['encoder'], checkpoint_config['encoder'])
        self.assertEqual(result['model']['scale_head'], checkpoint_config['scale_head'])
        self.assertEqual(result['model']['num_tokens_range'], [1200, 1200])
        self.assertEqual(result['optimizer'], config['optimizer'])
        self.assertEqual(result['data'], config['data'])
        self.assertNotEqual(config['model']['encoder']['backbone'], 'dinov2_vitl14')
        del checkpoint_config['normal_head']
        with self.assertRaises(ValueError):
            configure_model(config, checkpoint_config)

    def test_rgb_detail_manifest_optimizer_group(self):
        manifest = json.loads((REPO / 'configs/experiments/gaemi_robot160_rgb_detail.json').read_text())
        base = json.loads((REPO / manifest['base_config']).read_text())
        experiment = manifest['experiments'][0]
        config, batch, accumulation = resolve_experiment(base, experiment, 200, 32, 1)
        self.assertTrue(config['model']['rgb_detail']['enabled'])
        self.assertNotIn('rgb_detail', base['model'])
        groups = config['optimizer']['params']
        matches = [g for g in groups if any(fnmatch.fnmatch('rgb_detail.stages.0.weight', p)
                                          for p in g['params']['include'])]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['lr'], experiment['learning_rates']['rgb_detail'])
        self.assertEqual(batch * accumulation, 32)
        broken = copy.deepcopy(experiment)
        broken['rgb_detail']['enabled'] = False
        with self.assertRaises(ValueError):
            resolve_experiment(base, broken, 200, 32, 1)

    def test_progress_carriage_returns_and_exit_code_are_preserved(self):
        payload = '\rValidation 0/2\rValidation 1/2\rValidation 2/2\n완료\n'.encode()
        with tempfile.TemporaryDirectory() as temporary:
            child = [sys.executable, '-c', f'import os; os.write(1, {payload!r}); raise SystemExit(3)']
            relay = (
                'from pathlib import Path\n'
                'from run_script.sweep_5000 import execute\n'
                f'raise SystemExit(execute({child!r}, Path({temporary!r}), False))\n'
            )
            result = subprocess.run([sys.executable, '-c', relay], cwd=REPO, capture_output=True)
            self.assertEqual(result.returncode, 3)
            self.assertTrue(result.stdout.endswith(payload), result.stdout)
            self.assertEqual((Path(temporary) / 'run.log').read_bytes(), payload)
            self.assertEqual((Path(temporary) / 'exit_code').read_text().strip(), '3')

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

    def test_backbone_sweep_groups_and_commands(self):
        manifest = json.loads((REPO / 'configs/experiments/gaemi_backbone_sweep_1500.json').read_text())
        self.assertEqual(len(manifest['experiments']), 7)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'sweep'
            subprocess.run([str(REPO / 'run_script/run_gaemi_backbone_sweep_1500.sh'),
                            '--dry-run', '--root', str(root)], check=True, capture_output=True)
            for e in manifest['experiments']:
                config, batch, accumulation = resolve_experiment(self.base, e, 200, 32, 1)
                self.assertEqual((batch, accumulation), (8, 4))
                self.assertEqual(config['loss'], self.base['loss'])
                self.assertEqual([d['path'] for d in config['data']['datasets']],
                                 [d['path'] for d in self.base['data']['datasets']])
                groups = config['optimizer']['params']
                matches = [g for g in groups if any(fnmatch.fnmatch('encoder.backbone.blocks.11.attn.qkv.weight', p)
                                                   for p in g['params']['include'])]
                self.assertEqual(len(matches), int(e['train_backbone_last_n_blocks'] > 0))
                if matches:
                    self.assertEqual(matches[0]['lr'], e['learning_rates']['backbone'])
                self.assertEqual(len(config['lr_scheduler']['params']['lr_lambda']), len(groups))
                cmd = json.loads((root / e['name'] / 'command.json').read_text())
                for flag, value in {'--train_backbone_last_n_blocks': str(e['train_backbone_last_n_blocks']),
                                    '--num_iterations': '1500', '--batch_size_forward': '8',
                                    '--gradient_accumulation_steps': '4', '--checkpoint': 'none'}.items():
                    self.assertEqual(cmd[cmd.index(flag) + 1], value)
                self.assertIn('--val_config', cmd)
            self.assertTrue((root / 'pretrained_baseline' / 'command.json').exists())

    def test_backbone_requires_count_and_rate_together(self):
        original = self.manifest['experiments'][1]
        for count, rate in ((2, None), (0, 1e-6), (-1, 1e-6), (True, 1e-6)):
            e = copy.deepcopy(original)
            e['train_backbone_last_n_blocks'] = count
            if rate is not None:
                e['learning_rates']['backbone'] = rate
            with self.assertRaises(ValueError):
                resolve_experiment(self.base, e, 200, 32, 1)

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
