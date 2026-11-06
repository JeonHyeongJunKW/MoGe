"""Verify every saved multitask label and produce a compact visual preview."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from moge.utils.io import read_depth, read_image, read_normal
from tools.v4l2_distillation import read_json, sha256, write_json


def verify(root, source):
    info = read_json(root / 'dataset.json')
    assert info['status'] == 'complete'
    assert info['format'] == 'moge_v4l2_multitask_v1'
    counts = Counter()
    max_error = 0.
    for row in info['samples']:
        folder = root / row['sample']
        result = read_json(folder / 'result.json')
        for name, digest in result['output_sha256'].items():
            assert sha256(folder / name) == digest, folder / name
        with np.load(folder / 'teacher_labels.npz', allow_pickle=False) as labels:
            p, n = labels['mask_probability'], labels['normal']
            mv, nv, dv = (labels[k] for k in ('mask_valid', 'normal_valid', 'depth_valid'))
            assert p.dtype == n.dtype == np.float32
            assert n.shape == (*p.shape, 3)
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            assert np.isfinite(n).all()
            assert all(labels[k].dtype == bool for k in
                       ('teacher_mask', 'geometry_mask', 'mask_valid', 'normal_valid', 'depth_valid'))
            np.testing.assert_array_equal(labels['teacher_mask'], p > .5)
            assert not (labels['geometry_mask'] & ~labels['teacher_mask']).any()
            assert not (nv & ~(mv & labels['geometry_mask'])).any()
            assert not (dv & ~(mv & labels['geometry_mask'])).any()
            rect = cv2.imread(str(source / row['sample'] / 'rectification_valid.png'), 0)
            np.testing.assert_array_equal(mv, rect > 0)
            np.testing.assert_array_equal(dv, np.isfinite(read_depth(folder / 'depth.png')))
            for name in ('teacher_mask', 'mask_valid', 'normal_valid'):
                np.testing.assert_array_equal(labels[name], cv2.imread(str(folder / f'{name}.png'), 0) > 0)
            error = float(np.max(np.abs(np.linalg.norm(n[nv], axis=-1) - 1))) if nv.any() else 0.
            max_error = max(max_error, error)
            assert error < 1e-5
            normal_png = read_normal(folder / 'normal.png')
            np.testing.assert_array_equal(np.isfinite(normal_png).all(-1), nv)
            np.testing.assert_allclose(normal_png[nv], n[nv], atol=1e-4)
            counts.update(mask_supervised=int(mv.sum()), mask_positive=int((mv & (p > .5)).sum()),
                          mask_negative=int((mv & (p <= .5)).sum()), normal_valid=int(nv.sum()),
                          depth_valid=int(dv.sum()))
    cameras = Counter(r['camera'] for r in info['samples'])
    report = dict(samples=len(info['samples']), cameras=dict(cameras), splits=info['counts'],
                  teacher_sha256=info['settings']['teacher_sha256'], pixel_counts=dict(counts),
                  max_normal_unit_length_error=max_error, all_sample_hashes_verified=True,
                  note='Teacher pseudo labels; all existing training frames are not independent validation.')
    write_json(root / 'verification.json', report)
    selected = []
    for camera in cameras:
        rows = [r for r in info['samples'] if r['camera'] == camera]
        selected.extend(rows[i] for i in (0, len(rows)//2, len(rows)-1))
    fig, axes = plt.subplots(len(selected), 5, figsize=(18, 2.3 * len(selected)), squeeze=False)
    for row_axes, row in zip(axes, selected):
        folder = root / row['sample']
        with np.load(folder / 'teacher_labels.npz') as labels:
            normal = np.where(labels['normal_valid'][..., None], labels['normal'] * .5 + .5, 0)
            panels = [read_image(folder / 'image.jpg'), labels['mask_probability'],
                      labels['teacher_mask'] & labels['mask_valid'], read_depth(folder / 'depth.png'), normal]
            for ax, panel, title in zip(row_axes, panels, ['RGB', 'Teacher mask probability', 'Teacher mask (rectified)', 'Depth (m, 0-100)', 'Teacher normal (camera XYZ)']):
                if title == 'Depth (m, 0-100)':
                    ax.imshow(panel, cmap='turbo', vmin=0, vmax=100)
                else:
                    ax.imshow(panel, cmap='gray', vmin=0, vmax=1)
                ax.set_title(title, fontsize=9)
                ax.axis('off')
            row_axes[0].set_title(f"{row['camera']} / {folder.name}", fontsize=9)
    fig.tight_layout()
    fig.savefig(root / 'preview.png', dpi=140)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    verify(args.dataset, args.source)
