import json
from pathlib import Path
import unittest

import torch
from torch import nn

from moge.model.modules.dinov2.models.vision_transformer import DinoVisionTransformer
from moge.model.utils import wrap_module_with_gradient_checkpointing
from moge.train.finetuning import configure_moge2_heads, set_moge2_head_training_modes
from moge.train.utils import build_optimizer


class BackboneFinetuningTests(unittest.TestCase):
    def model(self, chunks=0):
        model = nn.Module()
        model.encoder = nn.Module()
        model.encoder.backbone = DinoVisionTransformer(
            img_size=28, patch_size=14, embed_dim=24, depth=4,
            num_heads=3, block_chunks=chunks)
        model.scale_head = nn.Linear(24, 1)
        return model

    def test_gradients_updates_modes_and_optimizer_with_checkpointing(self):
        torch.set_num_threads(1)
        for chunks in (0, 2):
            with self.subTest(chunks=chunks):
                torch.manual_seed(42)
                model = self.model(chunks)
                # Frozen decoder still propagates gradients to the selected blocks.
                configure_moge2_heads(model, {'scale_head': False}, 2)
                backbone = model.encoder.backbone
                for block in backbone.blocks:
                    wrap_module_with_gradient_checkpointing(block)
                set_moge2_head_training_modes(model, {'scale_head': False}, 2)
                blocks = (list(backbone.blocks) if not chunks else
                          [b for chunk in backbone.blocks for b in chunk
                           if not isinstance(b, nn.Identity)])
                for index, block in enumerate(blocks):
                    self.assertEqual(block.training, index >= 2)
                self.assertFalse(backbone.patch_embed.training)
                self.assertFalse(backbone.norm.training)
                self.assertFalse(model.scale_head.training)
                selected = {id(p) for b in blocks[-2:] for p in b.parameters()}
                for name, p in model.named_parameters():
                    self.assertEqual(p.requires_grad, id(p) in selected, name)
                config_path = Path(__file__).resolve().parents[1] / 'configs/train/gaemi_moge2_vits_small_augmentation.json'
                optimizer = build_optimizer(model, json.loads(config_path.read_text())['optimizer'])
                grouped = [id(p) for g in optimizer.param_groups for p in g['params']]
                self.assertEqual(set(grouped), selected)
                self.assertEqual(len(grouped), len(selected))
                before = {name: p.detach().clone() for name, p in model.named_parameters()}
                features = backbone.get_intermediate_layers(
                    torch.randn(2, 3, 28, 28), n=1, return_class_token=True)
                model.scale_head(features[-1][1]).square().mean().backward()
                for name, p in model.named_parameters():
                    self.assertEqual(p.grad is not None, p.requires_grad, name)
                    if p.grad is not None:
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                optimizer.step()
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        self.assertTrue(torch.equal(before[name], p), name)
                for block in blocks[-2:]:
                    self.assertGreater(sum(p.grad.abs().sum().item() for p in block.parameters()), 0)
                self.assertTrue(any(not torch.equal(before[n], p) for n, p in model.named_parameters()
                                    if p.requires_grad))

    def test_invalid_counts_do_not_change_freeze_policy(self):
        for count in (-1, 5, 1.5, True):
            model = self.model()
            with self.assertRaises(ValueError):
                configure_moge2_heads(model, {'scale_head': True}, count)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_zero_preserves_frozen_backbone(self):
        model = self.model()
        configure_moge2_heads(model, {'scale_head': True})
        set_moge2_head_training_modes(model, {'scale_head': True})
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
        self.assertFalse(any(m.training for m in model.encoder.modules()))


if __name__ == '__main__':
    unittest.main()
