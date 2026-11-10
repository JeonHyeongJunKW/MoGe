"""Cache a frozen Teacher's outputs for an independent V4L2 RGB test session."""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch

from moge.model.v2 import MoGeModel
from moge.utils.io import read_image
from tools.v4l2_distillation import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rgb-root', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source = json.loads((a.rgb_root / 'dataset.json').read_text())
    if source.get('format') != 'moge_v4l2_rgb_v1':
        p.error('Expected a rectified V4L2 RGB dataset')
    rows = [r for r in source['samples'] if r['camera'] == 'front_left' and r['split'] == 'test']
    if not rows or len({r['session'] for r in rows}) != 1:
        p.error('Expected one independent front_left test session')
    rows.sort(key=lambda r: int(Path(r['sample']).name))
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'dataset.json').write_text(json.dumps({'format': 'moge_v4l2_teacher_reference_v1',
        'status': 'building', 'source_dataset_sha256': sha256(a.rgb_root / 'dataset.json'),
        'teacher_sha256': sha256(a.teacher), 'teacher_tokens': 1200,
        'samples': rows}, indent=2) + '\n')
    torch.set_num_threads(2)
    c = torch.load(a.teacher, map_location='cpu', weights_only=True)
    model = MoGeModel(**c['model_config'])
    model.load_state_dict(c['model'], strict=True)
    model = model.cuda().eval()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            src = a.rgb_root / row['sample']
            dst = a.output / row['sample']
            dst.mkdir(parents=True)
            for name in ('image.jpg', 'rectification_valid.png'):
                shutil.copyfile(src / name, dst / name)
            rgb = read_image(dst / 'image.jpg')
            rect = cv2.imread(str(dst / 'rectification_valid.png'), cv2.IMREAD_GRAYSCALE)
            if rect is None or rect.shape != rgb.shape[:2]:
                raise ValueError(f'Invalid rectification mask: {src}')
            x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().cuda() / 255
            pred = model.infer(x, num_tokens=1200, use_fp16=True,
                               apply_mask=False, return_mask_probability=True)
            labels = {k: pred[k].cpu().numpy() for k in ('depth', 'normal', 'mask', 'mask_probability')}
            np.savez_compressed(dst / 'teacher_labels.npz', **labels)
            if (i + 1) % 10 == 0:
                print(f'{i + 1}/{len(rows)}', flush=True)
    manifest = json.loads((a.output / 'dataset.json').read_text())
    manifest['status'] = 'complete'
    (a.output / 'dataset.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Prepared {len(rows)} independent frames: {a.output}', flush=True)


if __name__ == '__main__':
    main()
