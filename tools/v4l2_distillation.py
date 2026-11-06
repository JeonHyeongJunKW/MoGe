"""Prepare rectified v4l2 RGB and cache fixed-teacher metric pseudo labels.

Both outputs are standalone, relocatable datasets. Nothing is published as a
complete dataset until every input has been processed. The source is read-only.
"""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPLITS = ('train', 'val', 'test')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError(f'Unsafe dataset identifier: {value!r}')
    return value


def validate_sessions(config, source_root):
    """Prevent a recording or a declared physical scene group crossing splits."""
    sessions = config['sessions']
    if not sessions:
        raise ValueError('No sessions specified')
    ids, paths, groups = set(), set(), {}
    for session in sessions:
        sid = safe_id(session['id'])
        path = (Path(source_root) / session['path']).resolve()
        split = session['split']
        group = safe_id(session['scene_group'])
        if split not in SPLITS or sid in ids or path in paths:
            raise ValueError(f'Duplicate session/path or invalid split: {sid}')
        if group in groups and groups[group] != split:
            raise ValueError(f'Scene group crosses splits: {group}')
        ids.add(sid)
        paths.add(path)
        groups[group] = split
    cameras = config['cameras']
    if not cameras or len(set(cameras)) != len(cameras):
        raise ValueError('Specify distinct cameras')
    for camera in cameras:
        safe_id(camera)
    interval = config['min_interval_seconds']
    if not math.isfinite(interval) or interval < 0:
        raise ValueError('min_interval_seconds must be finite and non-negative')


def select_frames(files, interval_seconds):
    """The supplied export uses integer nanosecond timestamps as filenames."""
    ordered = sorted(files, key=lambda p: int(p.stem))
    selected, last = [], None
    interval_ns = round(interval_seconds * 1e9)
    seen = set()
    for path in ordered:
        stamp = int(path.stem)
        if stamp < 0 or stamp in seen:
            raise ValueError(f'Invalid or duplicate timestamp: {path}')
        seen.add(stamp)
        if last is None or stamp - last >= interval_ns:
            selected.append(path)
            last = stamp
    return selected


def read_camera_info(directory):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    files = sorted(Path(directory).glob('*.mcap'))
    if not files:
        raise FileNotFoundError(f'No camera-info MCAP in {directory}')
    calibration, count = None, 0
    for path in files:
        with path.open('rb') as stream:
            for _, _, _, message in make_reader(stream, decoder_factories=[DecoderFactory()]).iter_decoded_messages():
                info = getattr(message, 'camera_info', message)
                if not hasattr(info, 'k') or not hasattr(info, 'distortion_model'):
                    continue
                current = dict(width=int(info.width), height=int(info.height),
                               distortion_model=info.distortion_model, K=list(info.k), D=list(info.d))
                if calibration is not None and current != calibration:
                    raise ValueError(f'Camera calibration changes within recording: {directory}')
                calibration = current
                count += 1
    if calibration is None:
        raise ValueError(f'No CameraInfo messages: {directory}')
    if calibration['distortion_model'] != 'plumb_bob':
        raise ValueError(f'Unsupported distortion model: {calibration["distortion_model"]}')
    return calibration, count


def rectification_maps(raw, profile):
    import cv2
    import numpy as np
    width, height = profile['width'], profile['height']
    if (raw['width'], raw['height']) != (width, height):
        raise ValueError('Source resolution differs from rectification profile; resizing is not implicit')
    K = np.asarray(profile['K'], dtype=np.float64).reshape(3, 3)
    raw_K = np.asarray(raw['K'], dtype=np.float64).reshape(3, 3)
    D = np.asarray(raw['D'], dtype=np.float64)
    for matrix in (K, raw_K):
        if not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 0:
            raise ValueError('Invalid intrinsics')
        if not np.allclose(matrix[2], [0, 0, 1]):
            raise ValueError('Invalid intrinsic last row')
    if not np.isfinite(D).all():
        raise ValueError('Non-finite distortion')
    mx, my = cv2.initUndistortRectifyMap(raw_K, D, None, K, (width, height), cv2.CV_32FC1)
    valid = (mx >= 0) & (mx < width - 1) & (my >= 0) & (my < height - 1)
    if valid.mean() < .5:
        raise ValueError('Rectification has less than 50% source coverage; check calibration')
    # OpenCV integer pixel centers -> MoGe pixel-edge normalized coordinates.
    normalized = K.copy()
    normalized[0, 2] += .5
    normalized[1, 2] += .5
    normalized[0] /= width
    normalized[1] /= height
    return mx, my, valid, normalized


def write_indices(root, records):
    for split in SPLITS:
        paths = [r['sample'] for r in records if r['split'] == split]
        (root / f'.{split}.txt').write_text(''.join(p + '\n' for p in paths))
    (root / '.index.txt').write_text(''.join(r['sample'] + '\n' for r in records))


def prepare(args):
    import cv2
    import numpy as np
    from moge.utils.io import write_image
    config = read_json(args.manifest)
    validate_sessions(config, args.source_root)
    output = args.output.resolve()
    staging = output.with_name(output.name + '.partial')
    if output.exists() or staging.exists():
        raise FileExistsError(f'Use a new output directory (including .partial): {output}')
    profile = config['rectification']
    records, calibrations = [], {}
    # Validate all camera sources/calibrations before writing any images.
    jobs = []
    for session in config['sessions']:
        source = (args.source_root / session['path']).resolve()
        for camera in config['cameras']:
            folder = source / 'camera/v4l2_driver/front_view' / camera / 'color'
            files = [p for p in folder.iterdir() if p.suffix.lower() in ('.bmp', '.png', '.jpg', '.jpeg')]
            files = select_frames(files, config['min_interval_seconds'])
            if not files:
                raise ValueError(f'No images: {folder}')
            raw, count = read_camera_info(source / 'camera/info/v4l2_driver' / camera)
            mx, my, valid, K = rectification_maps(raw, profile)
            key = f'{session["id"]}/{camera}'
            calibrations[key] = dict(raw=raw, camera_info_messages=count, rectified_K=profile['K'],
                                     intrinsics=K.tolist(), valid_fraction=float(valid.mean()))
            jobs.append((session, camera, files, mx, my, valid, K))
    staging.mkdir(parents=True)
    for session, camera, files, mx, my, valid, K in jobs:
        for source in files:
            sample = f'samples/{session["id"]}/{camera}/{source.stem}'
            destination = staging / sample
            destination.mkdir(parents=True)
            bgr = cv2.imread(str(source))
            if bgr is None or bgr.shape != (profile['height'], profile['width'], 3):
                raise ValueError(f'Unreadable or incorrect-sized RGB: {source}')
            bgr = cv2.remap(bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            write_image(destination / 'image.jpg', cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), quality=100)
            if not cv2.imwrite(str(destination / 'rectification_valid.png'), valid.astype(np.uint8) * 255):
                raise OSError('Could not write rectification mask')
            meta = dict(intrinsics=K.tolist(), camera=camera, session=session['id'],
                        scene_group=session['scene_group'], split=session['split'],
                        timestamp_ns=int(source.stem), source_image=str(source),
                        source_sha256=sha256(source), rectified_K=profile['K'],
                        image_geometry='rectified RGB; no resize/crop; pixel-edge normalized intrinsics')
            write_json(destination / 'meta.json', meta)
            records.append(dict(sample=sample, split=session['split'], session=session['id'],
                                scene_group=session['scene_group'], camera=camera,
                                image_sha256=sha256(destination / 'image.jpg'),
                                mask_sha256=sha256(destination / 'rectification_valid.png'),
                                meta_sha256=sha256(destination / 'meta.json')))
        print(f'{session["id"]}/{camera}: {len(files)} rectified frames', flush=True)
    write_indices(staging, records)
    write_json(staging / 'dataset.json', dict(format='moge_v4l2_rgb_v1', status='complete',
               config=config, calibrations=calibrations, samples=records,
               counts={s: sum(r['split'] == s for r in records) for s in SPLITS}))
    staging.rename(output)
    print(f'RGB dataset: {output} ({len(records)} images). No depth labels yet.')
    if not any(r['split'] != 'train' for r in records):
        print('No independent real holdout session; CARLA validation remains the quantitative monitor.')


def filter_depth(depth, model_mask, rectification_valid, min_depth, max_depth,
                 flipped_depth=None, flipped_mask=None, consistency_threshold=0):
    """Rejected/unknown pixels are NaN, never infinity (known sky)."""
    import numpy as np
    valid = (model_mask.astype(bool) & rectification_valid.astype(bool) &
             np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth))
    if consistency_threshold > 0:
        if flipped_depth is None or flipped_mask is None:
            raise ValueError('Flip consistency requires both predictions')
        valid &= (flipped_mask.astype(bool) & np.isfinite(flipped_depth) & (flipped_depth > 0))
        with np.errstate(invalid='ignore', divide='ignore'):
            relative_difference = np.abs(depth - flipped_depth) / np.maximum(np.maximum(depth, flipped_depth), 1e-6)
        valid &= relative_difference <= consistency_threshold
    return np.where(valid, depth, np.nan).astype(np.float32)


def auxiliary_labels(probability, normal, geometry_mask, rectification_valid):
    """Keep mask negatives separate from depth filters; normals use camera XYZ."""
    import numpy as np
    probability = np.asarray(probability, dtype=np.float32)
    normal = np.asarray(normal, dtype=np.float32)
    shape = probability.shape
    if (len(shape) != 2 or normal.shape != (*shape, 3) or
            np.shape(geometry_mask) != shape or np.shape(rectification_valid) != shape):
        raise ValueError('Auxiliary label shape mismatch')
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError('Invalid teacher mask probabilities')
    norm = np.linalg.norm(normal, axis=-1)
    normal_finite = np.isfinite(normal).all(axis=-1) & np.isfinite(norm) & (norm > 1e-6)
    unit = np.zeros_like(normal)
    unit[normal_finite] = normal[normal_finite] / norm[normal_finite, None]
    rect = np.asarray(rectification_valid, dtype=bool)
    geometry_mask = np.asarray(geometry_mask, dtype=bool)
    return dict(mask_probability=probability, teacher_mask=probability > .5,
                geometry_mask=np.asarray(geometry_mask, dtype=bool), mask_valid=rect,
                normal=unit, normal_valid=rect & geometry_mask & normal_finite)


def pseudo(args):
    import shutil
    import cv2
    import numpy as np
    import torch
    from moge.model.v2 import MoGeModel
    from moge.utils.io import read_image, write_depth, write_normal
    include_aux = getattr(args, 'include_auxiliary_labels', False)
    if (not math.isfinite(args.min_depth) or not math.isfinite(args.max_depth) or
            not 0 < args.min_depth < args.max_depth or not 0 < args.min_valid_fraction <= 1 or
            not 0 <= args.flip_consistency_threshold <= 1 or args.num_tokens < 1):
        raise ValueError('Invalid depth bounds, valid fraction, consistency threshold or token count')
    source = args.dataset.resolve()
    info = read_json(source / 'dataset.json')
    if info.get('format') != 'moge_v4l2_rgb_v1' or info.get('status') != 'complete':
        raise ValueError('Expected complete prepared RGB dataset')
    teacher = args.teacher.resolve()
    if not teacher.is_file():
        raise FileNotFoundError(teacher)
    output = args.output.resolve()
    staging = output.with_name(output.name + '.partial')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    settings = dict(teacher_sha256=sha256(teacher), source_manifest_sha256=sha256(source / 'dataset.json'),
                    num_tokens=args.num_tokens, precision='fp16' if args.fp16 else 'fp32',
                    fov_override=None, min_depth=args.min_depth, max_depth=args.max_depth,
                    min_valid_fraction=args.min_valid_fraction,
                    flip_consistency_threshold=args.flip_consistency_threshold)
    if include_aux:
        settings['auxiliary_labels'] = 'mask_probability_and_camera_normal_v1'
    if staging.exists():
        if not args.resume or read_json(staging / 'generation.json')['settings'] != settings:
            raise ValueError('Partial output exists. Resume only with identical inputs/settings, or use a new output.')
    else:
        staging.mkdir(parents=True)
        write_json(staging / 'generation.json', dict(settings=settings, teacher_path=str(teacher),
                                                    source_dataset=str(source), status='in_progress'))
    device = torch.device(args.device)
    if args.fp16 and device.type != 'cuda':
        raise ValueError('--fp16 requires CUDA')
    model = MoGeModel.from_pretrained(str(teacher)).to(device).eval().requires_grad_(False)
    if not hasattr(model, 'scale_head') or model.scale_head is None:
        raise ValueError('Teacher must have a metric scale head')
    if include_aux and any(getattr(model, name, None) is None for name in ('mask_head', 'normal_head')):
        raise ValueError('Auxiliary labels require teacher mask and normal heads')
    records, rejected = [], []
    with torch.inference_mode():
        for i, row in enumerate(info['samples']):
            src = source / row['sample']
            for name, key in [('image.jpg', 'image_sha256'), ('meta.json', 'meta_sha256'),
                              ('rectification_valid.png', 'mask_sha256')]:
                if sha256(src / name) != row[key]:
                    raise ValueError(f'Prepared source changed: {src / name}')
            dst = staging / row['sample']
            done = dst / 'result.json'
            if done.exists():
                result = read_json(done)
                if result['accepted']:
                    for name, digest in result['output_sha256'].items():
                        if sha256(dst / name) != digest:
                            raise ValueError(f'Cached sample corrupted: {dst / name}')
            else:
                dst.mkdir(parents=True, exist_ok=True)
                rgb = read_image(src / 'image.jpg')
                valid = cv2.imread(str(src / 'rectification_valid.png'), cv2.IMREAD_GRAYSCALE) > 0
                x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().to(device) / 255
                extra = {'return_mask_probability': True} if include_aux else {}
                pred = model.infer(x, num_tokens=args.num_tokens, use_fp16=args.fp16, apply_mask=False, **extra)
                depth, mask = pred['depth'].cpu().numpy(), pred['mask'].cpu().numpy()
                fd = fm = None
                if args.flip_consistency_threshold:
                    flip = model.infer(x.flip(-1), num_tokens=args.num_tokens,
                                       use_fp16=args.fp16, apply_mask=False)
                    fd = flip['depth'].flip(-1).cpu().numpy()
                    fm = flip['mask'].flip(-1).cpu().numpy()
                depth = filter_depth(depth, mask, valid, args.min_depth, args.max_depth,
                                     fd, fm, args.flip_consistency_threshold)
                fraction = float(np.isfinite(depth).mean())
                result = dict(accepted=fraction >= args.min_valid_fraction, valid_fraction=fraction)
                if result['accepted']:
                    shutil.copyfile(src / 'image.jpg', dst / 'image.jpg')
                    write_depth(dst / 'depth.png', depth)
                    extra_files = []
                    if include_aux:
                        labels = auxiliary_labels(pred['mask_probability'].cpu().numpy(),
                                                  pred['normal'].cpu().numpy(), mask, valid)
                        labels['depth_valid'] = np.isfinite(depth)
                        np.savez_compressed(dst / 'teacher_labels.npz', **labels)
                        for filename, array in [('teacher_mask.png', labels['teacher_mask']),
                                                ('mask_valid.png', labels['mask_valid']),
                                                ('normal_valid.png', labels['normal_valid'])]:
                            if not cv2.imwrite(str(dst / filename), array.astype(np.uint8) * 255):
                                raise IOError(f'Failed to write {filename}')
                        write_normal(dst / 'normal.png', np.where(labels['normal_valid'][..., None],
                                                                 labels['normal'], np.nan))
                        extra_files = ['teacher_labels.npz', 'teacher_mask.png', 'mask_valid.png',
                                       'normal_valid.png', 'normal.png']
                    meta = read_json(src / 'meta.json')
                    meta.update(metric_scale=1.0, depth_unit='meter', depth_type='camera_z',
                                label_source='teacher_prediction_not_ground_truth',
                                teacher_sha256=settings['teacher_sha256'],
                                teacher_num_tokens=args.num_tokens,
                                teacher_predicted_intrinsics=pred['intrinsics'].cpu().tolist(),
                                valid_fraction=fraction)
                    if include_aux:
                        meta['auxiliary_labels'] = dict(
                            file='teacher_labels.npz', normal_coordinates='OpenCV camera: x right, y down, z forward',
                            normal_source='teacher normal head, renormalized to unit vectors; no sign flip',
                            mask_target='raw sigmoid probability, before positive-depth/range filters',
                            mask_valid='rectification-valid pixels, INCLUDING teacher negatives',
                            normal_valid='rectification AND teacher infer geometry mask AND finite nonzero normal; no depth-range filter',
                            depth_valid='finite filtered depth', normal_horizontal_flip='reverse width and negate x component')
                    write_json(dst / 'meta.json', meta)
                    result['output_sha256'] = {name: sha256(dst / name)
                                               for name in ['image.jpg', 'depth.png', 'meta.json', *extra_files]}
                write_json(done, result)
            entry = {**row, 'valid_fraction': result['valid_fraction']}
            if result['accepted']:
                records.append(entry)
            else:
                rejected.append(entry)
            if (i + 1) % 10 == 0 or i + 1 == len(info['samples']):
                print(f'{i+1}/{len(info["samples"])}: accepted={len(records)}, rejected={len(rejected)}', flush=True)
    if not records:
        raise ValueError('No accepted pseudo labels; adjust data/model before training')
    write_indices(staging, records)
    write_json(staging / 'dataset.json', dict(format='moge_v4l2_multitask_v1' if include_aux else 'moge_v4l2_pseudo_v1', status='complete',
               settings=settings, samples=records, rejected=rejected,
               counts={s: sum(r['split'] == s for r in records) for s in SPLITS},
               evaluation_note='Real pseudo labels are teacher agreement targets, not independent ground truth.'))
    generation = read_json(staging / 'generation.json')
    generation['status'] = 'complete'
    write_json(staging / 'generation.json', generation)
    staging.rename(output)
    print(f'Pseudo dataset: {output}; ready to copy as a whole directory.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare', help='Read exported RGB and camera-info MCAP; rectify and sample by timestamp')
    p.add_argument('--manifest', type=Path, default=ROOT / 'configs/data/v4l2_real_sessions.json')
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser('pseudo', help='Frozen offline teacher; output MoGe metric depth dataset')
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--teacher', type=Path, default=ROOT / 'pretrained/best_teacher_vitl_t1200_last12_total5000.pt')
    p.add_argument('--num-tokens', type=int, default=1200)
    p.add_argument('--device', default='cuda')
    p.add_argument('--fp16', action='store_true', help='Match reviewed FP16 inference; default FP32')
    p.add_argument('--include-auxiliary-labels', action='store_true',
                   help='Also save raw mask probabilities, camera normals and separate supervision masks (new multitask format)')
    p.add_argument('--min-depth', type=float, default=0.1)
    p.add_argument('--max-depth', type=float, default=100.0)
    p.add_argument('--min-valid-fraction', type=float, default=0.1)
    p.add_argument('--flip-consistency-threshold', type=float, default=0,
                   help='Optional relative disagreement filter, 0 disables; not calibrated confidence')
    p.add_argument('--resume', action='store_true', help='Resume matching .partial output; never overwrite completed datasets')
    p.set_defaults(func=pseudo)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
