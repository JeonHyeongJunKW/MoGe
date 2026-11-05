"""Fine-tune the 400-token Student with CARLA GT and offline v4l2 pseudo depth.

Reuses the existing trainer, optimizer grouping, checkpoint architecture resolver
and CARLA validation. Pseudo targets never enter the validation monitor.
"""
import argparse
import copy
import datetime
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_script.sweep_5000 import COMPONENTS, execute, resolve_experiment, write_json


def build_config(data_root, pseudo_root, real_fraction=.25, pseudo_loss_weight=.25,
                 iterations=3000, batch_size=8, lora_rank=0, lora_lr=1e-5):
    if type(lora_rank) is not int or lora_rank < 0 or not math.isfinite(lora_lr) or lora_lr <= 0:
        raise ValueError('Invalid LoRA rank or learning rate')
    if not math.isfinite(real_fraction) or not 0 <= real_fraction < 1:
        raise ValueError('real_fraction must be in [0, 1); retain CARLA supervision')
    if not math.isfinite(pseudo_loss_weight) or not 0 < pseudo_loss_weight <= 1:
        raise ValueError('pseudo_loss_weight must be in (0, 1]')
    if iterations < 1:
        raise ValueError('iterations must be positive')
    base = json.loads((ROOT / 'configs/train/gaemi_moge2_robot160.json').read_text())
    validation = json.loads((ROOT / 'configs/validation/gaemi_moge2_robot160.json').read_text())
    for ds in [*base['data']['datasets'], *validation['datasets'].values()]:
        ds['path'] = str(Path(data_root).resolve() / Path(ds['path']).relative_to('/home/coder/dataset'))
    experiment = dict(
        name='v4l2_pseudo_student400' if real_fraction else 'carla_only_student400',
        train_backbone_last_n_blocks=4, num_tokens=400, batch_size_forward=batch_size,
        learning_rates=dict(backbone=5e-7, neck=5e-7, scale_head=5e-6,
                            points_head=1.5e-6, normal_head=1.5e-6, mask_head=1.5e-6),
        image_augmentation=[dict(type='motion_blur', probability=.3,
                                kernel_sizes=[3, 5, 7], angle_range=[-180, 180])],
        flip_augmentation=True,
    )
    # A short smoke also keeps the normal 3,000-update schedule.
    schedule_end = max(300, iterations)
    experiment['cosine_decay'] = dict(start_step=200, end_step=schedule_end, min_factor=.2)
    config, batch, accumulation = resolve_experiment(base, experiment, 200, 32, 1)
    if lora_rank:
        lora = dict(rank=lora_rank, alpha=2 * lora_rank, last_n_blocks=4, targets=['qkv', 'proj'])
        config['model']['lora'] = lora
        config['sweep_experiment']['lora'] = copy.deepcopy(lora)
        config['sweep_experiment']['name'] += f'_lora_r{lora_rank}'
        config['sweep_experiment']['learning_rates']['backbone'] = lora_lr
        for group in config['optimizer']['params']:
            if group['name'] == 'backbone':
                group['lr'] = lora_lr
    total = sum(ds['weight'] for ds in config['data']['datasets'])
    for ds in config['data']['datasets']:
        ds['weight'] = (1 - real_fraction) * ds['weight'] / total
    if real_fraction:
        pseudo = dict(name='v4l2_teacher_pseudo_train', path=str(Path(pseudo_root).resolve()),
                      index='.train.txt', label_type='PSEUDO', weight=real_fraction,
                      depth='depth.png', depth_unit=1.0, finite_depth_mask='only_known')
        config['data']['datasets'].append(pseudo)
        # Supervise geometry + metric scale, not teacher mask/normal heads.
        # Existing normal/mask heads still receive CARLA supervision through A.
        loss = copy.deepcopy(config['loss']['A'])
        loss.pop('normal', None)
        loss.pop('mask', None)
        for group in loss.values():
            for spec in group.values():
                spec['weight'] *= pseudo_loss_weight
        config['loss']['PSEUDO'] = loss
    config['distillation'] = dict(method='offline_metric_depth_pseudo_labels',
                                  real_sampling_fraction=real_fraction,
                                  pseudo_loss_multiplier=pseudo_loss_weight,
                                  real_labels_are_ground_truth=False)
    config['data'].update(center_augmentation=0.0, fov_range_relative=[1.0, 1.0])
    validation.update(batch_size=8, every=250, per_image_metrics=False, depth_bins=None)
    return config, validation, batch, accumulation


def check_index(root, index):
    root = Path(root)
    names = (root / index).read_text().splitlines()
    if not names or len(set(names)) != len(names) or any(not n.strip() for n in names):
        raise ValueError(f'Empty/duplicate entries in {root / index}')
    for name in names:
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f'Index escapes dataset root: {name}')
        for filename in ('image.jpg', 'depth.png', 'meta.json'):
            if not (path / filename).is_file():
                raise FileNotFoundError(path / filename)
    return names


def preflight(config, validation):
    from tools.v4l2_distillation import sha256
    for ds in config['data']['datasets']:
        names = check_index(ds['path'], ds['index'])
        if ds['label_type'] != 'PSEUDO':
            continue
        root = Path(ds['path'])
        manifest = json.loads((root / 'dataset.json').read_text())
        if manifest.get('status') != 'complete' or manifest.get('format') != 'moge_v4l2_pseudo_v1':
            raise ValueError('Expected complete v4l2 pseudo dataset')
        records = manifest['samples']
        expected = {r['sample'] for r in records if r['split'] == 'train'}
        if set(names) != expected:
            raise ValueError('Pseudo training index differs from declared train split')
        groups, sessions, samples = {}, {}, set()
        for row in records:
            if row['split'] not in ('train', 'val', 'test') or row['sample'] in samples:
                raise ValueError('Invalid/duplicate pseudo sample')
            samples.add(row['sample'])
            for mapping, key in [(groups, row['scene_group']), (sessions, row['session'])]:
                if key in mapping and mapping[key] != row['split']:
                    raise ValueError('Pseudo scene group/session crosses splits')
                mapping[key] = row['split']
        for name in names:
            sample = root / name
            result = json.loads((sample / 'result.json').read_text())
            if not result['accepted']:
                raise ValueError(f'Rejected pseudo sample in training: {name}')
            for filename in ('image.jpg', 'depth.png', 'meta.json'):
                if sha256(sample / filename) != result['output_sha256'][filename]:
                    raise ValueError(f'Pseudo sample changed: {sample / filename}')
            meta = json.loads((sample / 'meta.json').read_text())
            if meta.get('teacher_sha256') != manifest['settings']['teacher_sha256']:
                raise ValueError('Mixed teacher versions in pseudo dataset')
        config['distillation']['pseudo_dataset'] = dict(
            manifest_sha256=sha256(root / 'dataset.json'), settings=manifest['settings'], counts=manifest['counts'])
    for ds in validation['datasets'].values():
        check_index(ds['path'], ds['split'])


def validate_student(checkpoint):
    """This recipe is an additional-training run from a model-only Small best."""
    if 'model' not in checkpoint or 'model_config' not in checkpoint:
        raise ValueError('Student checkpoint requires model and model_config')
    if checkpoint['model_config']['encoder']['backbone'] != 'dinov2_vits14':
        raise ValueError('This Student recipe requires MoGe-2 ViT-Small')
    if checkpoint['model_config'].get('lora') or any('.lora_' in k for k in checkpoint['model']):
        raise ValueError('Start comparison experiments from the same plain Student checkpoint; merge adapters first if needed')
    if any(key in checkpoint for key in ('step', 'optimizer', 'lr_scheduler', 'ema_model')):
        raise ValueError('Use a model-only best.pt, not a resumable training-state checkpoint')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/home/coder/dataset'))
    parser.add_argument('--pseudo-root', type=Path, help='Default: DATA_ROOT/v4l2_pseudo_v1')
    parser.add_argument('--student', type=Path, default=ROOT / 'pretrained/best_flip_mild_t400_total18500.pt')
    parser.add_argument('--root', type=Path, help='New empty output directory')
    parser.add_argument('--real-fraction', type=float, default=.25, help='0 for CARLA-only control; default .25')
    parser.add_argument('--pseudo-loss-weight', type=float, default=.25)
    parser.add_argument('--num-iterations', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=8, help='Effective batch remains 32 on one GPU')
    parser.add_argument('--lora-rank', type=int, default=0, help='0: ordinary last-4 tuning; positive: attention LoRA only in backbone')
    parser.add_argument('--lora-lr', type=float, default=1e-5)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Write configs/commands only; no torch or data required')
    mode.add_argument('--smoke', action='store_true',
                      help='10 training updates; no validation/periodic checkpoints; trainer still writes final')
    args = parser.parse_args()
    pseudo_root = args.pseudo_root or args.data_root / 'v4l2_pseudo_v1'
    config, validation, batch, accumulation = build_config(
        args.data_root, pseudo_root, args.real_fraction, args.pseudo_loss_weight,
        args.num_iterations, args.batch_size, args.lora_rank, args.lora_lr)
    if not args.dry_run:
        if not args.student.is_file():
            parser.error(f'Student checkpoint not found: {args.student}')
        import torch
        checkpoint = torch.load(args.student, map_location='cpu', weights_only=True)
        validate_student(checkpoint)
        del checkpoint
        preflight(config, validation)
    mode_name = 'dry_run' if args.dry_run else 'smoke' if args.smoke else 'train'
    root = (args.root or ROOT / 'workspace/v4l2_distillation' /
            (datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + mode_name)).resolve()
    if root.exists() and any(root.iterdir()):
        parser.error(f'Use a new empty output directory: {root}')
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'train_config.json', config)
    write_json(root / 'validation_config.json', validation)
    write_json(root / 'invocation.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    if not args.dry_run:
        from tools.v4l2_distillation import sha256
        write_json(root / 'initial_student.json', dict(path=str(args.student.resolve()), sha256=sha256(args.student)))
    command = [sys.executable, '-u', '-m', 'accelerate.commands.launch', '--num_processes', '1',
               '--module', 'moge.train.train_moge12', '--config', str(root / 'train_config.json'),
               '--initial_checkpoint', str(args.student.resolve()), '--checkpoint', 'none',
               '--workspace', str(root), '--name', config['sweep_experiment']['name'],
               '--num_iterations', str(min(args.num_iterations, 10) if args.smoke else args.num_iterations),
               '--batch_size_forward', str(batch), '--gradient_accumulation_steps', str(accumulation),
               '--enable_gradient_checkpointing', 'True', '--precision', 'mixed_bf16', '--enable_ema', 'True',
               '--seed', '0', '--checkpoint_every', '0' if args.smoke else '1000',
               '--rolling_checkpoint_every', '0' if args.smoke else '500',
               '--log_every', '1' if args.smoke else '50', '--log_type', 'tensorboard',
               '--tb_log_root', str(root / 'tensorboard'), '--vis_every', '0', '--num_vis_images', '0',
               '--train_backbone_last_n_blocks', '4']
    rates = config['sweep_experiment']['learning_rates']
    for name in COMPONENTS:
        command += ['--train_' + name, str(name in rates)]
    if not args.smoke:
        command += ['--val_config', str(root / 'validation_config.json'), '--validate_before_training', 'True']
    command = [sys.executable, '-u', str(ROOT / 'run_script/run_checkpoint_experiment.py'),
               '--checkpoint-source', str(args.student.resolve()), '--', *command]
    print(f'CARLA / pseudo sampling: {1-args.real_fraction:.0%} / {args.real_fraction:.0%}; '
          f'pseudo loss multiplier: {args.pseudo_loss_weight}; Student: 400 tokens')
    print('Validation monitor: CARLA GT only. Real teacher agreement is not a ground-truth accuracy metric.')
    return execute(command, root, args.dry_run)


if __name__ == '__main__':
    raise SystemExit(main())
