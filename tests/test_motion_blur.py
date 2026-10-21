import unittest
import numpy as np
from moge.utils.motion_blur import motion_blur, validate_motion_blur
from moge.utils.data_augmentation import image_color_augmentation


class MotionBlurTests(unittest.TestCase):
    def test_seed_dtype_shape_and_depth_preserved(self):
        image = np.random.default_rng(4).integers(0, 256, (36, 64, 3), dtype=np.uint8)
        depth = np.full((36, 64), 2., dtype=np.float32)
        before = image.copy()
        cfg = dict(type='motion_blur', probability=1, kernel_sizes=[7], angle_range=[0, 0])
        a = image_color_augmentation(image, [cfg], np.random.default_rng(0), depth)
        b = image_color_augmentation(image, [cfg], np.random.default_rng(0), depth)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(image, before)
        np.testing.assert_array_equal(depth, 2.)
        self.assertEqual(a.shape, image.shape)
        self.assertEqual(a.dtype, image.dtype)
        self.assertLess(float(a.var()), float(image.var()))

    def test_skip_and_constant_image(self):
        im = np.full((30, 40, 3), 91, np.uint8)
        for probability in (0, 1):
            result = motion_blur(im, dict(probability=probability), np.random.default_rng(0))
            np.testing.assert_array_equal(result, im)

    def test_bad_parameters(self):
        for cfg in ({'probability':2}, {'kernel_sizes':[4]}, {'angle_range':[10,0]}):
            with self.assertRaises(ValueError):
                validate_motion_blur(cfg)


if __name__ == '__main__':
    unittest.main()
