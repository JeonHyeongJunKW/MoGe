import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from moge.model.v2 import MoGeModel
from tools.v4l2_distillation import auxiliary_labels


class AuxiliaryLabelTests(unittest.TestCase):
    def test_normal_png_roundtrip_for_non_square_image(self):
        from moge.utils.io import read_normal, write_normal
        normal = np.broadcast_to(np.array([.6, 0, -.8], np.float32), (4, 7, 3)).copy()
        normal[0, 0] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'normal.png'
            write_normal(path, normal)
            restored = read_normal(path)
        self.assertTrue(np.isnan(restored[0, 0]).all())
        np.testing.assert_allclose(restored[1:], normal[1:], atol=1e-4)

    def test_negative_mask_targets_and_geometry_filters_are_separate(self):
        probability = np.array([[.1, .9, .8, .7, .9]], np.float32)
        normal = np.array([[[0, 0, 2], [0, 3, 0], [1, 0, 0],
                            [0, 0, 0], [np.nan, 0, 0]]], np.float32)
        labels = auxiliary_labels(probability, normal,
                                  [[False, False, True, True, True]],
                                  [[True, True, False, True, True]])
        np.testing.assert_array_equal(labels['mask_valid'], [[True, True, False, True, True]])
        np.testing.assert_array_equal(labels['teacher_mask'], [[False, True, True, True, True]])
        self.assertFalse(labels['normal_valid'].any())
        np.testing.assert_allclose(labels['normal'][0, 1], [0, 1, 0])
        self.assertTrue(np.isfinite(labels['normal']).all())
        labels = auxiliary_labels(probability, normal, np.ones((1, 5), bool), np.ones((1, 5), bool))
        np.testing.assert_array_equal(labels['normal_valid'], [[True, True, True, False, False]])

    def test_rejects_invalid_probability_and_shapes(self):
        for probability in [np.array([[np.nan]]), np.array([[1.1]])]:
            with self.assertRaises(ValueError):
                auxiliary_labels(probability, np.ones((1, 1, 3)), [[True]], [[True]])
        with self.assertRaises(ValueError):
            auxiliary_labels(np.ones((1, 1)), np.ones((1, 2, 3)), [[True]], [[True]])

    def test_infer_probability_is_opt_in_and_unmasked(self):
        class FixedModel(MoGeModel):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def forward(self, image, num_tokens):
                return {'mask': torch.tensor([[[.1, .9]]]),
                        'normal': torch.ones(1, 1, 2, 3)}

        model = FixedModel()
        image = torch.zeros(3, 1, 2)
        default = model.infer(image, num_tokens=1, use_fp16=False)
        output = model.infer(image, num_tokens=1, use_fp16=False, return_mask_probability=True)
        self.assertNotIn('mask_probability', default)
        torch.testing.assert_close(output['mask_probability'], torch.tensor([[.1, .9]]))
        self.assertEqual(tuple(output['mask'].shape), (1, 2))
        self.assertEqual(output['normal'][0, 0].sum(), 0)


if __name__ == '__main__':
    unittest.main()
