import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from moge.model.modules.dinov2.models.vision_transformer import DinoVisionTransformer
from moge.model.modules.lora import inject_lora, merged_checkpoint
from moge.model.utils import wrap_module_with_gradient_checkpointing
from moge.train.finetuning import configure_moge2_heads, set_moge2_head_training_modes
from moge.train.utils import build_optimizer
from run_script.run_checkpoint_experiment import configure_model
from run_script.run_v4l2_distillation import build_config

ROOT = Path(__file__).resolve().parents[1]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.backbone = DinoVisionTransformer(img_size=28, patch_size=14,
                                                      embed_dim=24, depth=4, num_heads=3, block_chunks=0)
        self.scale_head = nn.Linear(24, 1)

    def forward(self, x):
        result = self.encoder.backbone.forward_features(x)['x_norm_clstoken']
        return self.scale_head(result)


class LoRATests(unittest.TestCase):
    def test_zero_initialization_freeze_updates_ema_reload_and_merge(self):
        torch.set_num_threads(1)
        torch.manual_seed(15)
        plain = TinyModel().eval()
        model = copy.deepcopy(plain)
        lora = dict(rank=4, alpha=8, last_n_blocks=2, targets=['qkv', 'proj'])
        model.lora_config = lora
        rng = torch.get_rng_state().clone()
        inject_lora(model.encoder.backbone, lora)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        image = torch.randn(2, 3, 28, 28)
        torch.testing.assert_close(plain(image), model(image), rtol=0, atol=0)
        configure_moge2_heads(model, {'scale_head': True}, 2)
        for name, p in model.named_parameters():
            self.assertEqual(p.requires_grad, '.lora_' in name or name.startswith('scale_head.'), name)
        set_moge2_head_training_modes(model, {'scale_head': True}, 2)
        for block in model.encoder.backbone.blocks:
            wrap_module_with_gradient_checkpointing(block)
        config = {'params': [dict(name='backbone', type='AdamW', lr=.01,
                                  params={'include': ['encoder.backbone.*']}),
                             dict(name='head', type='AdamW', lr=.01,
                                  params={'include': ['scale_head.*']})]}
        opt = build_optimizer(model, config)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        ema = torch.optim.swa_utils.AveragedModel(model)
        for _ in range(2):
            opt.zero_grad()
            model(image).square().mean().backward()
            for name, p in model.named_parameters():
                if '.lora_' in name:
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all())
            opt.step()
            ema.update_parameters(model)
        for name, value in model.state_dict().items():
            if '.lora_' not in name and not name.startswith('scale_head.'):
                self.assertTrue(torch.equal(value, before[name]), name)
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in model.state_dict().items() if '.lora_B' in k))
        model.eval()
        expected = model(image)
        checkpoint = dict(model_config={'lora': lora}, model=model.state_dict())
        reloaded = copy.deepcopy(plain)
        inject_lora(reloaded.encoder.backbone, lora)
        reloaded.load_state_dict(checkpoint['model'], strict=True)
        torch.testing.assert_close(expected, reloaded(image))
        merged = merged_checkpoint(checkpoint)
        self.assertNotIn('lora', merged['model_config'])
        self.assertFalse(any('.lora_' in k for k in merged['model']))
        plain.load_state_dict(merged['model'], strict=True)
        torch.testing.assert_close(expected, plain(image), rtol=2e-5, atol=2e-6)
        ema_plain = TinyModel().eval()
        ema_plain.load_state_dict(merged_checkpoint(dict(model_config={'lora': lora}, model=ema.module.state_dict()))['model'])
        ema.eval()
        torch.testing.assert_close(ema(image), ema_plain(image), rtol=2e-5, atol=2e-6)

    def test_corrupt_adapter_checkpoint_and_block_selection_rejected(self):
        model = TinyModel()
        model.lora_config = dict(rank=4, alpha=8, last_n_blocks=2, targets=['qkv', 'proj'])
        inject_lora(model.encoder.backbone, model.lora_config)
        with self.assertRaisesRegex(ValueError, 'must match'):
            configure_moge2_heads(model, {'scale_head': True}, 4)
        state = dict(model.state_dict())
        state.pop(next(k for k in state if k.endswith('.lora_A')))
        with self.assertRaises(ValueError):
            merged_checkpoint(dict(model_config={'lora': model.lora_config}, model=state))

    def test_recipe_only_changes_backbone_learning_and_preserves_checkpoint_architecture(self):
        baseline, val, _, _ = build_config('/data', '/pseudo')
        for rank in (4, 8):
            config, lora_val, _, _ = build_config('/data', '/pseudo', lora_rank=rank)
            self.assertEqual(config['data'], baseline['data'])
            self.assertEqual(config['loss'], baseline['loss'])
            self.assertEqual(lora_val, val)
            self.assertEqual(config['lr_scheduler'], baseline['lr_scheduler'])
            for a, b in zip(config['optimizer']['params'], baseline['optimizer']['params']):
                if a['name'] != 'backbone':
                    self.assertEqual(a, b)
                else:
                    self.assertEqual(a['lr'], 1e-5)
            resolved = configure_model(config, baseline['model'])
            self.assertEqual(resolved['model']['lora']['rank'], rank)
            self.assertEqual(resolved['model']['num_tokens_range'], [400, 400])

    def test_sweep_dry_run_shared_initial_model_and_independent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(ROOT / 'run_script/run_v4l2_lora_sweep.py'),
                                     '--root', tmp, '--dry-run'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            root = Path(tmp)
            summary = json.loads((root / 'summary.json').read_text())
            self.assertEqual([r['rank'] for r in summary], [0, 4, 8])
            for row in summary:
                self.assertEqual(row['status'], 'dry_run')
                command = json.loads((root / row['experiment'] / 'command.json').read_text())
                self.assertTrue(command[command.index('--initial_checkpoint') + 1].endswith('best_flip_mild_t400_total18500.pt'))
                self.assertEqual(command[command.index('--num_iterations') + 1], '3000')
                self.assertEqual(command[command.index('--checkpoint') + 1], 'none')


if __name__ == '__main__':
    unittest.main()
