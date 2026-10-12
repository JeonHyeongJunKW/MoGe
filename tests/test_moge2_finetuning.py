import unittest

import torch
from torch import nn

from moge.train.finetuning import configure_moge2_heads, set_moge2_head_training_modes


class TinyMoGe(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Dropout(.5))
        self.neck = nn.Linear(4, 4)
        self.scale_head = nn.Linear(4, 1)
        self.points_head = nn.Linear(4, 3)
        self.normal_head = nn.Linear(4, 3)
        self.mask_head = nn.Linear(4, 1)

    def forward(self, images):
        encoded = self.encoder(images)
        features = self.neck(encoded)
        return self.scale_head(encoded).square().sum() + sum(
            head(features).square().sum() for head in
            (self.points_head, self.normal_head, self.mask_head))


class FinetuningTests(unittest.TestCase):
    def test_neck_and_all_heads_train_while_encoder_and_buffers_stay_frozen(self):
        torch.manual_seed(1)
        model = TinyMoGe()
        components = {name: True for name in
                      ('neck', 'scale_head', 'points_head', 'normal_head', 'mask_head')}
        configure_moge2_heads(model, components)
        set_moge2_head_training_modes(model, components)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        model(torch.ones(2, 3)).backward()
        optimizer.step()
        for name, parameter in model.named_parameters():
            enabled = not name.startswith('encoder.')
            self.assertEqual(parameter.grad is not None, enabled, name)
            self.assertEqual(not torch.equal(before[name], parameter), enabled, name)
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(before[name], value), name)
        self.assertFalse(model.encoder.training)
        self.assertTrue(model.neck.training)
        self.assertTrue(model.normal_head.training)
        self.assertTrue(model.mask_head.training)

    def test_frozen_normal_head_still_backpropagates_into_trainable_neck(self):
        model = TinyMoGe()
        configure_moge2_heads(model, {'neck': True})
        set_moge2_head_training_modes(model, {'neck': True})
        model.normal_head(model.neck(model.encoder(torch.ones(2, 3)))).square().sum().backward()
        self.assertIsNotNone(model.neck.weight.grad)
        self.assertIsNone(model.normal_head.weight.grad)
        self.assertIsNone(model.encoder[0].weight.grad)

    def test_encoder_cannot_be_enabled_by_decoder_policy(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            configure_moge2_heads(TinyMoGe(), {'encoder': True})

    def test_only_selected_heads_change(self):
        for scale, points in ((True, True), (True, False), (False, True)):
            with self.subTest(scale=scale, points=points):
                torch.manual_seed(1)
                model = TinyMoGe()
                heads = {'scale_head': scale, 'points_head': points}
                configure_moge2_heads(model, heads)
                # Loading pretrained weights must not change the freeze policy.
                model.load_state_dict(model.state_dict())
                model.train()
                set_moge2_head_training_modes(model, heads)
                for name, module in model.named_children():
                    self.assertEqual(module.training, heads.get(name, False))
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
                before = {name: value.clone() for name, value in model.state_dict().items()}
                model(torch.ones(2, 3)).backward()
                optimizer.step()
                for name, parameter in model.named_parameters():
                    enabled = heads.get(name.split('.')[0], False)
                    self.assertEqual(parameter.requires_grad, enabled)
                    self.assertEqual(parameter.grad is not None, enabled)
                    self.assertEqual(not torch.equal(before[name], parameter), enabled)
                for name, buffer in model.named_buffers():
                    self.assertTrue(torch.equal(before[name], buffer), name)

    def test_reject_no_trainable_heads_and_missing_enabled_head(self):
        with self.assertRaisesRegex(ValueError, 'at least one'):
            configure_moge2_heads(TinyMoGe(), {'scale_head': False, 'points_head': False})
        model = TinyMoGe()
        del model.scale_head
        with self.assertRaisesRegex(ValueError, 'absent'):
            configure_moge2_heads(model, {'scale_head': True, 'points_head': False})


if __name__ == '__main__':
    unittest.main()
