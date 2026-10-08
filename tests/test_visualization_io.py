import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from moge.train.visualization_io import write_rgb


class VisualizationIOTests(unittest.TestCase):
    def test_missing_exr_writer_preserves_point_map(self):
        points = np.array([[[1., 2., 3.], [np.nan, np.inf, -np.inf]]], dtype=np.float32)
        for capability in (False, cv2.error('EXR disabled')):
            with self.subTest(capability=capability), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'points.exr'
                kwargs = {'side_effect': capability} if isinstance(capability, Exception) else {'return_value': capability}
                with patch('cv2.haveImageWriter', **kwargs), patch('cv2.imwrite') as imwrite:
                    with self.assertWarnsRegex(RuntimeWarning, 'original XYZ'):
                        write_rgb(path, points)
                imwrite.assert_not_called()
                actual = np.load(path.with_suffix('.npy'), allow_pickle=False)
                np.testing.assert_array_equal(actual, points)
                self.assertEqual(actual.dtype, points.dtype)

    def test_supported_exr_writer_keeps_existing_export(self):
        image = np.array([[[1., 2., 3.]]], dtype=np.float32)
        with patch('cv2.haveImageWriter', return_value=True), patch('cv2.imwrite', return_value=True) as write:
            write_rgb('points.exr', image, [1, 2])
        np.testing.assert_array_equal(write.call_args.args[1], image[..., ::-1])
        self.assertEqual(write.call_args.args[2], [1, 2])

    def test_png_color_order_and_write_failure(self):
        image = np.array([[[255, 128, 0]]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            write_rgb(path, image)
            np.testing.assert_array_equal(cv2.imread(str(path)), image[..., ::-1])
            with patch('cv2.imwrite', return_value=False), self.assertRaises(OSError):
                write_rgb(path, image)


if __name__ == '__main__':
    unittest.main()
