import contextlib
import tempfile
import unittest
from pathlib import Path

import torch

from moge.train.checkpoint import load_checkpoint


class _AcceleratorStub:
    is_main_process = True

    def local_main_process_first(self):
        return contextlib.nullcontext()


class CheckpointLoadingTests(unittest.TestCase):
    def test_explicit_checkpoint_file_without_extension(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'content-addressed-blob'
            expected = {'model': {'weight': torch.tensor([1.0])}}
            torch.save(expected, path)

            actual = load_checkpoint(str(path), Path(temporary), _AcceleratorStub(), False)

            self.assertTrue(torch.equal(actual['model']['weight'], expected['model']['weight']))


if __name__ == '__main__':
    unittest.main()
