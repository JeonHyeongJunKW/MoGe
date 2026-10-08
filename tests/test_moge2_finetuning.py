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
        features = self.neck(self.encoder(images))
        return sum(head(features).square().sum() for head in
                   (self.scale_head, self.points_head, self.normal_head, self.mask_head))


class FinetuningTests(unittest.TestCase):
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
