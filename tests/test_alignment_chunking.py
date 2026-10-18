"""Alignment chunking must preserve candidate selection and final transforms."""
import unittest
from unittest.mock import patch

import torch
from moge.utils import alignment


class AlignmentChunkingTests(unittest.TestCase):
    def test_xyz_alignment_matches_unsplit_search(self):
        torch.manual_seed(42)
        source = torch.randn(1, 640, 3)
        target = source * 1.7 + torch.tensor([.2, -.1, 2.0]) + .01 * torch.randn_like(source)
        weights = torch.rand(1, 640) + .1
        original = alignment.split_batch_fwd
        observed = []

        def track(fn, chunk_size, *args, **kwargs):
            observed.append((chunk_size, args[0].shape))
            return original(fn, chunk_size, *args, **kwargs)

        with patch.object(alignment, 'split_batch_fwd', side_effect=track):
            actual = alignment.align_points_scale_xyz_shift(source, target, weights)
        with patch.object(alignment, 'split_batch_fwd', side_effect=lambda fn, size, *args, **kw: fn(*args, **kw)):
            expected = alignment.align_points_scale_xyz_shift(source, target, weights)
        self.assertGreater(observed[0][1][0], observed[0][0])
        self.assertLessEqual(observed[0][0] * observed[0][1][1], 2 ** 20)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)


if __name__ == '__main__':
    unittest.main()
