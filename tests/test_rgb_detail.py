from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from moge.model.v2 import MoGeModel
from moge.train.finetuning import configure_moge2_heads, set_moge2_head_training_modes
from moge.train.utils import build_optimizer


class TinyEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.proj = nn.Conv2d(3, 8, 1)

    def forward(self, image, rows, cols, return_class_token=True):
        x = self.proj(torch.nn.functional.interpolate(image, (rows, cols)))
        return x, x.mean((2, 3))


class RGBDetailTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.encoder_patch = patch('moge.model.v2.DINOv2Encoder', TinyEncoder)
        self.encoder_patch.start()
        self.addCleanup(self.encoder_patch.stop)
        neck = dict(dim_in=[10, 2, 2, 2, 2], dim_res_blocks=[8]*5,
                    dim_out=None, resamplers=['bilinear']*4, num_res_blocks=0)
        head = dict(neck, dim_in=[8]*5, dim_out=[None]*4+[3])
        self.config = dict(encoder={}, neck=neck, points_head=head,
                           scale_head={'dims': [8, 8, 1]}, remap_output='linear')

    def test_disabled_and_zero_initialization_match_pretrained(self):
        base = MoGeModel(**self.config).eval()
        disabled = MoGeModel(**self.config, rgb_detail={'enabled': False}).eval()
        disabled.load_state_dict(base.state_dict(), strict=True)
        enabled = MoGeModel(**self.config, rgb_detail={'enabled': True, 'channels': 4}).eval()
        missing, unexpected = enabled.load_state_dict(base.state_dict(), strict=False)
        self.assertTrue(missing)
        self.assertTrue(all(k.startswith('rgb_detail.') for k in missing))
        self.assertFalse(unexpected)
        for h, w in [(36, 64), (65, 37)]:
            image = torch.rand(1, 3, h, w)
            with torch.no_grad():
                expected = base(image, 16)
                for model in (disabled, enabled):
                    actual = model(image, 16)
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_branch_training_optimizer_and_checkpoint(self):
        config = dict(self.config, rgb_detail={'enabled': True, 'channels': 4})
        model = MoGeModel(**config)
        policy = {'rgb_detail': True, 'points_head': False, 'scale_head': False}
        configure_moge2_heads(model, policy)
        set_moge2_head_training_modes(model, policy)
        self.assertTrue(model.rgb_detail.training)
        self.assertFalse(model.encoder.training)
        optimizer = build_optimizer(model, {'params': [dict(name='rgb_detail', type='AdamW',
            params={'include': ['rgb_detail.*']}, lr=1e-3)]})
        selected = {id(p) for g in optimizer.param_groups for p in g['params']}
        self.assertEqual(selected, {id(p) for p in model.rgb_detail.parameters()})
        image = torch.rand(1, 3, 36, 64)
        # Zero projections first learn to use features; then gradients reach CNN.
        for _ in range(2):
            optimizer.zero_grad()
            result = model(image, 16)
            result['points'].square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.rgb_detail.parameters()))
            optimizer.step()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.rgb_detail.stages.parameters()), 0)
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        model.eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pt'
            torch.save({'model_config': config, 'model': model.state_dict()}, path)
            restored = MoGeModel.from_pretrained(path).eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(image, 16)['points'], model(image, 16)['points'])

    def test_bad_configuration(self):
        for config in [{'enabled': 'yes'}, {'enabled': True, 'channels': 0}, {'enabeld': True}]:
            with self.assertRaises(ValueError):
                MoGeModel(**self.config, rgb_detail=config)


if __name__ == '__main__':
    unittest.main()
