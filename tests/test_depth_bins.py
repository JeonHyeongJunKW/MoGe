import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from moge.train.depth_bins import DepthBinAccumulator, depth_bin_report, validate_depth_bins, write_depth_bin_report


class DepthBinTests(unittest.TestCase):
    config = {'width_m': 1, 'max_depth_m': 3}

    def test_boundaries_overflow_masks_and_invalid_gt(self):
        a = DepthBinAccumulator(self.config)
        gt = np.array([.5, 1, 1.99, 2, 3, 1000, np.inf, np.nan, 0, -1, 2.5])
        pred = np.nan_to_num(gt)
        mask = np.ones(gt.shape, bool)
        mask[-1] = False
        a.update(pred, gt, mask)
        self.assertEqual([r['num_pixels'] for r in a.rows()], [1, 2, 1, 2])
        self.assertEqual(a.rows()[-1]['upper_m'], None)
        self.assertEqual(a.rows()[1]['lower_m'], 1)
        self.assertTrue(all(r['mae_m'] == 0 and r['delta1'] == 1 for r in a.rows()))

    def test_pixel_weighted_accumulation_and_rmse(self):
        a = DepthBinAccumulator(self.config)
        a.update(np.array([.5, .75]), np.array([.5, .5]), np.ones(2, bool))
        b = DepthBinAccumulator(self.config)
        b.update(np.array([1.]), np.array([.5]), np.ones(1, bool))
        a.merge(b)
        row = a.rows()[0]
        self.assertEqual(row['num_images'], 2)
        self.assertEqual(row['num_pixels'], 3)
        self.assertAlmostEqual(row['rel'], .25 / (.5 + 1e-6))
        self.assertAlmostEqual(row['mae_m'], .25)
        self.assertAlmostEqual(row['rmse_m'], math.sqrt((.25**2 + .5**2) / 3))
        self.assertAlmostEqual(row['delta1'], 1/3)
        self.assertIsNone(a.rows()[1]['rel'])

    def test_nonmetric_nonpositive_prediction_and_bad_inputs(self):
        a = DepthBinAccumulator(self.config)
        a.update(np.array([-1.]), np.array([.5]), np.array([True]))
        self.assertEqual(a.rows()[0]['delta1'], 0)
        a.update(None, None, None, is_metric=False)
        self.assertEqual(a.skipped_nonmetric_images, 1)
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            a.update(np.array([np.nan]), np.array([.5]), np.array([True]))
        with self.assertRaisesRegex(ValueError, 'shapes'):
            a.update(np.ones(2), np.ones(1), np.ones(1))
        for bad in [False, {'width_m': 0}, {'max_depth_m': float('inf')},
                    {'width_m': True}, {'width_m': 2, 'max_depth_m': 3}, {'max_depth_m': 10001}, {'typo': 1}]:
            with self.subTest(config=bad), self.assertRaises(ValueError):
                validate_depth_bins(bad)

    def test_reports_keep_empty_bins_and_combined_pixel_counts(self):
        a = DepthBinAccumulator(self.config)
        a.update(np.array([1.]), np.array([1.]), np.array([True]))
        report = depth_bin_report({'val': a}, self.config)
        with tempfile.TemporaryDirectory() as d:
            prefix = Path(d) / 'depth_bins_step_00000249'
            write_depth_bin_report(prefix, report)
            loaded = json.loads(prefix.with_suffix('.json').read_text())
            self.assertIsNone(loaded['all_datasets'][0]['mae_m'])
            with prefix.with_suffix('.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 8)
            self.assertEqual(rows[0]['mae_m'], '')
            self.assertEqual(rows[-3]['num_pixels'], '1')


if __name__ == '__main__':
    unittest.main()
