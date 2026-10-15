"""Portable per-image validation reports and deduplicated RGB previews."""
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def make_image_record(sample, metrics, dataset, sample_index, directory, dataset_root=None):
    filename = str(sample['filename'])
    record = {'dataset': dataset, 'filename': filename, 'sample_index': sample_index,
              'metrics': {f'{group}/{key}': float(value) for group, values in metrics.items()
                          for key, value in values.items()}}
    if dataset_root is not None:
        root = Path(dataset_root).resolve()
        record.update(dataset_root=str(root), data_path=str(root / filename),
                      image_path=(Path(filename) / 'image.jpg').as_posix())
    town = re.search(r'Town[0-9A-Za-z]+', filename)
    record['town'] = town.group() if town else 'unknown'
    record['camera'] = filename.rstrip('/').split('/')[-1]
    record['environment'] = next((v for v in ('indoor-proxy', 'night', 'day') if f'__{v}' in filename), 'unknown')
    if directory is not None:
        image = sample['image'].detach().float().cpu().numpy().transpose(1, 2, 0)
        image = np.round(image.clip(0, 1) * 255).astype(np.uint8)
        digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        relative = Path('sample_images') / f'{digest}.jpg'
        path = Path(directory) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix('.tmp')
            Image.fromarray(image).save(temporary, format='JPEG', quality=85)
            temporary.replace(path)
        record['image'] = relative.as_posix()
    return record


def write_image_report(directory, step, signature, records, metadata=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = {'metadata': metadata or {}, 'schema_version': 1, 'step': step, 'validation_signature': signature,
              'description': 'Per-image evaluation metrics, not the training objective loss. No GT scale alignment for depth_metric.',
              'samples': records}
    path = directory / f'per_image_step_{step:08d}.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)
    metrics = sorted({key for row in records for key in row['metrics']})
    fields = ['dataset', 'filename', 'dataset_root', 'data_path', 'image_path', 'sample_index', 'town', 'environment', 'camera', 'image', *metrics]
    with path.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({**{k: row.get(k, '') for k in fields if k not in metrics}, **row['metrics']})
    return path
