import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from moge.train.dataloader import TrainDataLoaderPipeline
from moge.train.losses import teacher_mask_bce_loss, teacher_normal_cosine_loss
from moge.utils.io import write_image, write_depth
from run_script.run_v4l2_distillation import build_config
from tools.v4l2_distillation import write_json


class MultitaskTests(unittest.TestCase):
    def test_soft_mask_loss_negative_gradient_and_ignore(self):
        pred = torch.tensor([[.8, .2, .4]], requires_grad=True)
        loss, _ = teacher_mask_bce_loss(pred, torch.tensor([[.1, .9, float('nan')]]),
                                       torch.tensor([[True, True, False]]))
        loss.backward()
        self.assertGreater(pred.grad[0, 0], 0)
        self.assertLess(pred.grad[0, 1], 0)
        self.assertEqual(pred.grad[0, 2], 0)
        empty, _ = teacher_mask_bce_loss(pred, torch.zeros_like(pred), torch.zeros_like(pred, dtype=torch.bool))
        self.assertEqual(empty.item(), 0)

    def test_normal_sign_and_empty_supervision(self):
        pred = torch.tensor([[[1., 0, 0], [0, 1., 0]]], requires_grad=True)
        target = torch.tensor([[[-1., 0, 0], [float('nan')]*3]])
        loss, _ = teacher_normal_cosine_loss(pred, target)
        self.assertEqual(loss.item(), 2)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        pred.grad.zero_()
        orthogonal = torch.tensor([[[0., 1, 0], [float('nan')]*3]])
        loss, _ = teacher_normal_cosine_loss(pred, orthogonal)
        loss.backward()
        self.assertLess(pred.grad[0, 0, 1], 0)
        self.assertEqual(pred.grad[0, 1].abs().sum(), 0)
        empty, _ = teacher_normal_cosine_loss(pred, torch.full_like(pred, float('nan')))
        self.assertEqual(empty.item(), 0)

    def test_recipe_weights_and_controls(self):
        for mask_weight, normal_weight in [(0, 0), (.1, 0), (.1, .1)]:
            cfg, _, _, _ = build_config('/data', '/pseudo', lora_rank=8,
                multitask_labels=True, mask_weight=mask_weight, normal_weight=normal_weight)
            loss = cfg['loss']['PSEUDO']
            self.assertEqual('mask' in loss, bool(mask_weight))
            self.assertEqual('normal' in loss, bool(normal_weight))
            if mask_weight:
                self.assertAlmostEqual(loss['mask']['teacher_mask']['weight'], .025)
            if normal_weight:
                self.assertAlmostEqual(loss['normal']['teacher_normal']['weight'], .025)
            self.assertIn('mask', cfg['loss']['A'])
            self.assertEqual(cfg['data']['datasets'][-1]['teacher_labels'], 'teacher_labels.npz')
        with self.assertRaises(ValueError):
            build_config('/data', '/pseudo', mask_weight=.1)

    def test_loader_preserves_teacher_normals_negatives_and_flip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / 'sample'
            sample.mkdir()
            (root / '.train.txt').write_text('sample\n')
            h, w = 18, 32
            write_image(sample / 'image.jpg', np.full((h, w, 3), 128, np.uint8))
            depth = np.full((h, w), 2., np.float32)
            depth[:, :8] = np.nan
            write_depth(sample / 'depth.png', depth)
            write_json(sample / 'meta.json', dict(intrinsics=[[1, 0, .5], [0, 1, .5], [0, 0, 1]], metric_scale=1.))
            normal = np.broadcast_to(np.array([.6, 0, .8], np.float32), (h, w, 3)).copy()
            valid = np.ones((h, w), bool)
            normal_valid = valid.copy()
            normal_valid[:, :8] = False
            probability = np.full((h, w), .9, np.float32)
            probability[:, :8] = .1
            np.savez(sample / 'teacher_labels.npz', normal=normal, normal_valid=normal_valid,
                     mask_probability=probability, mask_valid=valid)
            cfg, _, _, _ = build_config(root, root, multitask_labels=True, mask_weight=.1, normal_weight=.1)
            cfg['data']['datasets'] = [cfg['data']['datasets'][-1]]
            cfg['data'].update(flip_augmentation=True, image_augmentation=[], resize=False, perspective_warp=False)
            loader = TrainDataLoaderPipeline(cfg['data'], batch_size=1)
            base = dict(dataset=cfg['data']['datasets'][0]['name'], filename='sample', path=sample,
                        label_type='PSEUDO', width=w, height=h)
            seen = set()
            for seed in range(8):
                item = loader._process_instance(loader._load_instance(dict(base, seed=seed)))
                self.assertEqual(item['label_type'], 'PSEUDO')
                flipped = bool(np.random.default_rng(seed).choice([True, False]))
                seen.add(flipped)
                expected = normal.copy()
                expected[~normal_valid] = np.nan
                if flipped:
                    expected = expected[:, ::-1] * [-1, 1, 1]
                np.testing.assert_allclose(item['normal'], expected, atol=1e-6)
                self.assertTrue(item['teacher_mask_valid'].all())
                self.assertEqual(int((item['teacher_mask_probability'] < .5).sum()), h * 8)
                batch = loader._collate_batch([item])
                self.assertIn('teacher_mask_valid', batch)
            self.assertEqual(seen, {True, False})
            # Resize exercises the same spatial transform as RGB and restores unit normals.
            loader.datasets[base['dataset']].update(resize=True, flip_augmentation=False)
            item = loader._process_instance(loader._load_instance(dict(base, seed=0, width=16, height=9)))
            self.assertEqual(item['teacher_mask_probability'].shape, (9, 16))
            lengths = item['normal'].norm(dim=-1)
            torch.testing.assert_close(lengths[torch.isfinite(lengths)], torch.ones_like(lengths[torch.isfinite(lengths)]))


if __name__ == '__main__':
    unittest.main()
