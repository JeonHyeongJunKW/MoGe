"""Image exports with a lossless fallback when OpenCV has no EXR writer."""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import warnings
from pathlib import Path

import cv2
import numpy as np


def write_rgb(path, image, params=None):
    path = Path(path)
    if path.suffix.lower() == '.exr':
        try:
            supports_exr = cv2.haveImageWriter(str(path))
        except cv2.error:
            supports_exr = False
        if not supports_exr:
            # Preserve XYZ channel order, floating-point precision and NaN/Inf.
            # These are geometry arrays, not color previews.
            np.save(path.with_suffix('.npy'), image, allow_pickle=False)
            warnings.warn(
                'OpenCV has no usable EXR writer; saving point maps as .npy '
                'in their original XYZ channel order instead.',
                RuntimeWarning,
            )
            return
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), params or []):
        raise OSError(f'Failed to write visualization: {path}')
