import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

from tools.v4l2_distillation import (filter_depth, prepare, pseudo, read_json,
                                     rectification_maps, select_frames, validate_sessions, write_json)
from run_script.run_v4l2_distillation import build_config, preflight, validate_student
from moge.utils.io import read_depth


class PseudoDepthTests(unittest.TestCase):
    def config(self):
        return dict(cameras=['front_left'], min_interval_seconds=.5,
                    rectification=dict(width=64, height=36, K=[[40, 0, 31.5], [0, 40, 17.5], [0, 0, 1]]),
                    sessions=[dict(id='s1', path='s1', scene_group='location1', split='train')])

    def test_splits_reject_same_session_alias_and_scene_leakage(self):
        for extra in [dict(id='s2', path='s1', scene_group='location2', split='val'),
                      dict(id='s2', path='s2', scene_group='location1', split='val')]:
            cfg = self.config()
            cfg['sessions'].append(extra)
            with self.assertRaises(ValueError):
                validate_sessions(cfg, '/tmp/raw')
        cfg = self.config()
        cfg['sessions'].append(dict(id='s2', path='s2', scene_group='location2', split='val'))
        validate_sessions(cfg, '/tmp/raw')

    def test_timestamp_sampling_sorts_numerically_and_rejects_duplicates(self):
        files = [Path(f'{t}.bmp') for t in [1000000000, 9, 500000009, 20]]
        self.assertEqual([p.stem for p in select_frames(files, .5)], ['9', '500000009'])
        with self.assertRaises(ValueError):
            select_frames([Path('1.bmp'), Path('1.png')], 0)

    def test_rectification_normalization_uses_pixel_centers(self):
        profile = self.config()['rectification']
        raw = dict(width=64, height=36, K=profile['K'], D=[0, 0, 0, 0, 0])
        mx, my, valid, K = rectification_maps(raw, profile)
        self.assertEqual(K[0, 2], .5)
        self.assertEqual(K[1, 2], .5)
        self.assertAlmostEqual(K[0, 0], 40 / 64)
        np.testing.assert_allclose(mx[10, 10], 10, atol=1e-5)
        self.assertFalse(valid[-1, -1])
        with self.assertRaises(ValueError):
            rectification_maps({**raw, 'width': 128}, profile)

    def test_invalid_and_inconsistent_predictions_are_unknown_not_sky(self):
        d = np.array([[1., 2., np.inf, -1., 200., 3.]], dtype=np.float32)
        mask = np.array([[True, False, True, True, True, True]])
        flip = np.array([[1.05, 2., 1., 1., 1., 9.]])
        out = filter_depth(d, mask, np.ones_like(mask), .1, 100, flip, np.ones_like(mask), .2)
        self.assertEqual(out[0, 0], 1)
        self.assertTrue(np.isnan(out[0, 1:]).all())
        self.assertFalse(np.isinf(out).any())

    def test_training_mixture_preserves_gt_and_has_separate_pseudo_loss(self):
        config, val, batch, accumulation = build_config('/data', '/pseudo')
        datasets = config['data']['datasets']
        self.assertAlmostEqual(sum(d['weight'] for d in datasets[:-1]), .75)
        self.assertEqual(datasets[-1]['weight'], .25)
        self.assertEqual(datasets[-1]['finite_depth_mask'], 'only_known')
        self.assertEqual(config['model']['num_tokens_range'], [400, 400])
        self.assertEqual(batch * accumulation, 32)
        self.assertNotIn('normal', config['loss']['PSEUDO'])
        self.assertNotIn('mask', config['loss']['PSEUDO'])
        self.assertEqual(config['loss']['A']['metric_scale']['metric_scale']['weight'], .1)
        self.assertEqual(config['loss']['PSEUDO']['metric_scale']['metric_scale']['weight'], .025)
        self.assertTrue(all('pseudo' not in d['path'] for d in val['datasets'].values()))
        control, _, _, _ = build_config('/data', '/missing', real_fraction=0)
        self.assertEqual(len(control['data']['datasets']), 2)
        self.assertNotIn('PSEUDO', control['loss'])
        with self.assertRaises(ValueError):
            build_config('/data', '/pseudo', real_fraction=1)
        with self.assertRaises(ValueError):
            build_config('/data', '/pseudo', batch_size=7)

    def test_reject_resume_checkpoint_that_would_skip_new_training(self):
        checkpoint = dict(model={}, model_config=dict(encoder=dict(backbone='dinov2_vits14')))
        validate_student(checkpoint)
        with self.assertRaisesRegex(ValueError, 'model-only'):
            validate_student({**checkpoint, 'step': 8499})
        with self.assertRaisesRegex(ValueError, 'ViT-Small'):
            validate_student(dict(model={}, model_config=dict(encoder=dict(backbone='dinov2_vitl14'))))

    def test_prepare_pseudo_roundtrip_and_loader_masks(self):
        from moge.model.v2 import MoGeModel
        from moge.train.dataloader import TrainDataLoaderPipeline

        class FixedTeacher:
            scale_head = True

            def to(self, device):
                return self

            def eval(self):
                return self

            def requires_grad_(self, enabled):
                assert not enabled
                return self

            def infer(self, image, **kwargs):
                h, w = image.shape[-2:]
                depth = torch.full((h, w), 2.)
                mask = torch.ones(h, w, dtype=torch.bool)
                mask[:, :5] = False
                return dict(depth=depth, mask=mask, intrinsics=torch.eye(3))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.config()
            manifest = root / 'sessions.json'
            write_json(manifest, cfg)
            raw_dir = root / 's1/camera/v4l2_driver/front_view/front_left/color'
            raw_dir.mkdir(parents=True)
            cv2.imwrite(str(raw_dir / '1000000000.bmp'), np.full((36, 64, 3), 128, np.uint8))
            raw = dict(width=64, height=36, K=cfg['rectification']['K'], D=[0.] * 5)
            rgb = root / 'rgb'
            with patch('tools.v4l2_distillation.read_camera_info', return_value=(raw, 1)):
                prepare(SimpleNamespace(manifest=manifest, source_root=root, output=rgb))
            teacher = root / 'teacher.pt'
            teacher.write_bytes(b'fake teacher for IO integration test')
            target = root / 'pseudo'
            args = SimpleNamespace(dataset=rgb, output=target, teacher=teacher, num_tokens=1200,
                                   device='cpu', fp16=False, min_depth=.1, max_depth=100.,
                                   min_valid_fraction=.1, flip_consistency_threshold=0, resume=False)
            with patch.object(MoGeModel, 'from_pretrained', return_value=FixedTeacher()):
                pseudo(args)
            info = read_json(target / 'dataset.json')
            self.assertEqual(info['counts'], dict(train=1, val=0, test=0))
            name = info['samples'][0]['sample']
            depth = read_depth(target / name / 'depth.png')
            self.assertTrue(np.isnan(depth[:, :5]).all())
            np.testing.assert_allclose(depth[10, 10], 2., rtol=1e-4)
            self.assertFalse(np.isinf(depth).any())
            with self.assertRaises(FileExistsError):
                pseudo(args)
            # Resume reuses verified outputs, but not when the teacher changed.
            target.rename(target.with_name(target.name + '.partial'))
            args.resume = True
            teacher.write_bytes(b'different checkpoint')
            with self.assertRaisesRegex(ValueError, 'identical inputs/settings'):
                pseudo(args)
            teacher.write_bytes(b'fake teacher for IO integration test')
            with patch.object(MoGeModel, 'from_pretrained', return_value=FixedTeacher()), \
                    patch.object(FixedTeacher, 'infer', side_effect=AssertionError('Should reuse cached output')):
                pseudo(args)
            # Exercise the actual MoGe loader, including normal construction.
            cfg, validation, _, _ = build_config(root, target)
            ds = cfg['data']['datasets'][-1]
            cfg['data']['datasets'] = [ds]
            cfg['data']['image_sizes'] = [[64, 36]]
            cfg['data']['flip_augmentation'] = False
            cfg['data']['image_augmentation'] = []
            loader = TrainDataLoaderPipeline(cfg['data'], batch_size=1)
            instance = dict(dataset=ds['name'], filename=name, path=target / name,
                            label_type='PSEUDO', width=64, height=36, seed=0)
            item = loader._process_instance(loader._load_instance(instance))
            self.assertEqual(item['label_type'], 'PSEUDO')
            self.assertTrue(item['is_metric'])
            self.assertFalse(item['depth_mask_inf'].any())
            self.assertFalse(item['depth_mask_fin'][:, :5].any())
            preflight(cfg, {'datasets': {}})
            # The launcher must not accept a held-out sample inserted in train.
            info['samples'][0]['split'] = 'val'
            write_json(target / 'dataset.json', info)
            with self.assertRaisesRegex(ValueError, 'train split'):
                preflight(cfg, {'datasets': {}})


if __name__ == '__main__':
    unittest.main()
