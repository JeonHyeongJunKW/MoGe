import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from moge.train.validation_visualization import save_validation_point_clouds


def read_ply(path):
    header, payload = path.read_bytes().split(b'end_header\n', 1)
    count = int(next(line.split()[-1] for line in header.decode().splitlines() if line.startswith('element vertex')))
    dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertices = np.frombuffer(payload, dtype=dtype)
    assert len(vertices) == count
    return np.column_stack([vertices[k] for k in ('x', 'y', 'z')]), np.column_stack([vertices[k] for k in ('red', 'green', 'blue')])


class PointCloudTests(unittest.TestCase):
    def test_coordinates_colors_filtering_and_overlay(self):
        points = np.arange(18, dtype=np.float32).reshape(2, 3, 3) + 1
        sample = {'filename': 'scene/sample/camera', 'image': np.ones((3, 2, 3)) * .5,
                  'points': points, 'depth_mask': np.ones((2, 3), dtype=bool),
                  'intrinsics': np.eye(3), 'is_metric': True}
        prediction = points * 2
        prediction[0, 2] = np.inf
        output = {'points': prediction, 'intrinsics': np.eye(3) * 2}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_validation_point_clouds(path, sample, output, 3)
            gt, colors = read_ply(path / 'gt.ply')
            pred, _ = read_ply(path / 'pred.ply')
            overlay, overlay_colors = read_ply(path / 'overlay.ply')
            np.testing.assert_array_equal(gt, points.reshape(-1, 3)[[0, 2, 5]])
            np.testing.assert_array_equal(pred, prediction.reshape(-1, 3)[[0, 5]])
            np.testing.assert_array_equal(colors, np.full((3, 3), 128, dtype=np.uint8))
            np.testing.assert_array_equal(overlay, np.concatenate([gt, pred]))
            np.testing.assert_array_equal(overlay_colors[:3], np.tile([0, 200, 255], (3, 1)))
            np.testing.assert_array_equal(overlay_colors[3:], np.tile([255, 120, 0], (2, 1)))
            self.assertTrue((path / 'image.png').exists())
            info = json.loads((path / 'info.json').read_text())
            self.assertEqual(info['prediction_points_removed'], 1)
            self.assertEqual(info['gt_units'], 'meters')
            self.assertEqual(info['filename'], sample['filename'])

    def test_empty_clouds_are_valid_files(self):
        sample = {'filename': 'empty', 'image': np.zeros((3, 2, 2)), 'points': np.zeros((2, 2, 3)),
                  'depth_mask': np.zeros((2, 2), dtype=bool), 'intrinsics': np.eye(3), 'is_metric': False}
        with tempfile.TemporaryDirectory() as directory:
            save_validation_point_clouds(directory, sample, {'points': sample['points'], 'intrinsics': np.eye(3)}, 10)
            for name in ('gt.ply', 'pred.ply', 'overlay.ply'):
                points, colors = read_ply(Path(directory) / name)
                self.assertEqual(points.shape, (0, 3))


if __name__ == '__main__':
    unittest.main()
