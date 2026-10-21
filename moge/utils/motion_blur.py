"""Seeded linear motion blur in input-image pixel units."""
import math
import cv2
import numpy as np


def validate_motion_blur(config):
    if set(config) - {'type', 'probability', 'kernel_sizes', 'angle_range'}:
        raise ValueError('Unknown motion_blur option')
    probability = config.get('probability', 0.3)
    sizes = config.get('kernel_sizes', [3, 5, 7])
    angles = config.get('angle_range', [-180, 180])
    if not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('motion_blur probability must be in [0, 1]')
    if not sizes or any(type(k) is not int or k < 3 or k > 63 or k % 2 == 0 for k in sizes):
        raise ValueError('motion_blur kernel_sizes must be odd integers from 3 to 63')
    if len(angles) != 2 or not all(math.isfinite(a) for a in angles) or angles[0] > angles[1]:
        raise ValueError('motion_blur angle_range must be a finite ordered pair')
    return probability, sizes, angles


def motion_blur(image, config, rng):
    probability, sizes, angles = validate_motion_blur(config)
    if rng.random() >= probability:
        return image
    size = int(rng.choice(sizes))
    angle = np.deg2rad(rng.uniform(*angles))
    # Supersampling avoids jagged binary kernels at shallow angles.
    scale = 8
    kernel = np.zeros((size * scale, size * scale), np.uint8)
    center = (size * scale - 1) / 2
    radius = (size - 1) * scale / 2
    offset = np.array([np.cos(angle), np.sin(angle)]) * radius
    start = tuple(np.round(center - offset).astype(int))
    end = tuple(np.round(center + offset).astype(int))
    cv2.line(kernel, start, end, 255, thickness=scale, lineType=cv2.LINE_AA)
    kernel = cv2.resize(kernel.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
