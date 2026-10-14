"""Prepare/run the E vs. convolutional-decoder sweep without importing torch."""
import argparse
import copy
import csv
import datetime
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMPONENTS = ('scale_head', 'points_head', 'neck', 'normal_head', 'mask_head')


def write_json(path, data):
    path.write_text(json.dumps(data, indent=4) + '\n', encoding='utf-8')


def resolve_experiment(base, experiment, warmup_steps, effective_batch, processes, batch_override=None):
    """Build disjoint optimizer groups with one scheduler multiplier per group."""
    rates = experiment['learning_rates']
    if not rates or set(rates) - set(COMPONENTS):
        raise ValueError(f'Invalid trainable components: {rates}')
    if any(not math.isfinite(lr) or lr <= 0 for lr in rates.values()):
        raise ValueError('Learning rates must be positive and finite')
    batch = batch_override if batch_override is not None else experiment['batch_size_forward']
    if batch < 1 or processes < 1 or effective_batch < 1 or effective_batch % (batch * processes):
        raise ValueError('effective_batch_size must be divisible by batch_size_forward * num_processes')
    if warmup_steps < 1:
        raise ValueError('warmup_steps must be positive')
    config = copy.deepcopy(base)
    config.pop('validation', None)
    config['data']['image_augmentation'] = []
    for dataset in config['data']['datasets']:
        dataset['image_augmentation'] = []
    for scope in [config['data'], *config['data']['datasets']]:
        scope.update(flip_augmentation=False, perspective_warp=False, resize=False)
    groups = []
    template = next(g for g in config['optimizer']['params'] if g['name'] == 'head')
    for name in COMPONENTS:
        if name not in rates:
            continue
        if config['model'].get(name) is None:
            raise ValueError(f'Missing model component: {name}')
        group = copy.deepcopy(template)
        group.update(name=name, lr=rates[name], params={'include': [f'*{name}.*']})
        groups.append(group)
    for head, loss in [('normal_head', 'normal'), ('mask_head', 'mask')]:
        if head in rates:
            for dataset in config['data']['datasets']:
                specs = config['loss'][dataset['label_type']].get(loss, {})
                if not any(spec.get('weight', 0) > 0 for spec in specs.values()):
                    raise ValueError(f'{head} requires a positive {loss} loss for {dataset["name"]}')
    config['optimizer']['params'] = groups
    config['lr_scheduler'] = {'type': 'LambdaLR', 'params': {
        'lr_lambda': [f'min(1.0, (epoch + 1) / {warmup_steps})'] * len(groups)}}
    config['sweep_experiment'] = copy.deepcopy(experiment)
    return config, batch, effective_batch // (batch * processes)


def execute(command, directory, dry_run):
    write_json(directory / 'command.json', command)
    print(shlex.join(command), flush=True)
    if dry_run:
        return 0
    # tqdm redraws with carriage returns. Text-mode universal-newline decoding
    # turns those into newlines, so relay raw bytes without line buffering.
    with (directory / 'run.log').open('wb') as log:
        with subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=False) as process:
            for chunk in iter(lambda: process.stdout.read1(65536), b''):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                log.write(chunk)
                log.flush()
            code = process.wait()
    (directory / 'exit_code').write_text(str(code) + '\n')
    return code


def summarize(root, runs, dry_run, smoke):
    rows = []
    for name, rates, batch, accumulate, code in runs:
        folder = root / name
        best_path = folder / 'validation' / 'best.json'
        latest_paths = sorted((folder / 'validation').glob('step_*.json'))
        best = json.loads(best_path.read_text()) if best_path.exists() else {}
        latest = json.loads(latest_paths[-1].read_text()) if latest_paths else {}
        metrics = best.get('metrics', {}).get('mean', {})
        memory_path = folder / 'training_memory.json'
        memory = json.loads(memory_path.read_text()) if memory_path.exists() else {}
        rows.append({'experiment': name,
                     'status': 'dry_run' if dry_run else ('ok' if code == 0 else f'failed({code})'),
                     'batch': f'{batch} x {accumulate}',
                     'learning_rates': ', '.join(f'{k}={v:g}' for k, v in rates.items()),
                     'best_step': best.get('step', ''), 'best_rel': best.get('score', ''),
                     'best_delta1': metrics.get('depth_metric/delta1', ''),
                     'best_points_rel': metrics.get('points_metric/rel', ''),
                     'best_points_scale_rel': metrics.get('points_scale_invariant/rel', ''),
                     'best_fov_mae': metrics.get('fov_x/mae', ''),
                     'latest_rel': latest.get('metrics', {}).get('mean', {}).get('depth_metric/rel', ''),
                     'peak_allocated_gib': memory.get('peak_allocated_gib', ''),
                     'peak_reserved_gib': memory.get('peak_reserved_gib', '')})
    with (root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    headers = list(rows[0])
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    lines += ['| ' + ' | '.join(str(row[h]) for h in headers) + ' |' for row in rows]
    if smoke:
        lines += ['', 'Smoke run: training memory check only; no validation or quality comparison.']
    text = '\n'.join(lines) + '\n'
    (root / 'summary.md').write_text(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=REPO / 'configs/experiments/gaemi_finetune_sweep_5000.json')
    parser.add_argument('--pretrained', type=Path, default=Path('/home/coder/MoGe/pretrained/moge-2-vits.pt'))
    parser.add_argument('--root', type=Path, help='New, empty sweep output directory')
    parser.add_argument('--experiment', action='append', help='Run only this experiment (repeatable)')
    parser.add_argument('--num-iterations', type=int)
    parser.add_argument('--num-processes', type=int, default=1)
    parser.add_argument('--batch-size', type=int, help='Override forward batch; accumulation preserves effective batch 32')
    parser.add_argument('--validation-batch-size', type=int)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Write resolved configs/commands without starting training')
    parser.add_argument('--smoke', action='store_true', help='Ten training steps, no validation/checkpoints/PLY; same loss, precision and EMA')
    parser.add_argument('--skip-baseline', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    base = json.loads((REPO / manifest['base_config']).read_text())
    validation = json.loads((REPO / manifest['validation_config']).read_text())
    if args.validation_batch_size is not None:
        if args.validation_batch_size < 1:
            parser.error('--validation-batch-size must be positive')
        validation['batch_size'] = args.validation_batch_size
    if args.data_root:
        for dataset in [*base['data']['datasets'], *validation['datasets'].values()]:
            dataset['path'] = str(args.data_root.resolve())
    iterations = args.num_iterations if args.num_iterations is not None else manifest['num_iterations']
    if iterations < 1:
        parser.error('--num-iterations must be positive')
    if args.smoke:
        iterations = min(iterations, 10)
    experiments = manifest['experiments']
    if args.experiment:
        unknown = set(args.experiment) - {e['name'] for e in experiments}
        if unknown:
            parser.error(f'Unknown experiments: {sorted(unknown)}')
        experiments = [e for e in experiments if e['name'] in args.experiment]
    resolved = [(e, *resolve_experiment(base, e, manifest['warmup_steps'], manifest['effective_batch_size'],
                                       args.num_processes, args.batch_size)) for e in experiments]
    if not args.dry_run and not args.pretrained.is_file():
        parser.error(f'Pretrained checkpoint not found: {args.pretrained}')
    mode = 'smoke' if args.smoke else 'dry_run' if args.dry_run else 'train'
    root = (args.root or REPO / 'workspace/gaemi_finetune_sweep_5000' /
            (datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + mode)).resolve()
    if root.exists() and any(root.iterdir()):
        parser.error(f'Output directory is not empty: {root}. Use a new directory.')
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'manifest.json', manifest)
    write_json(root / 'validation_config.json', validation)
    write_json(root / 'invocation.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})

    def command(directory, rates, batch, accumulation, baseline=False):
        cmd = [sys.executable, '-u', '-m', 'accelerate.commands.launch', '--num_processes', str(args.num_processes),
               '--module', 'moge.train.train_moge12', '--config', str(directory / 'train_config.json'),
               '--initial_checkpoint', str(args.pretrained.resolve()), '--checkpoint', 'none',
               '--name', directory.name, '--workspace', str(directory),
               '--num_iterations', str(0 if baseline else iterations),
               '--batch_size_forward', str(batch), '--gradient_accumulation_steps', str(accumulation),
               '--enable_gradient_checkpointing', 'True', '--precision', 'mixed_bf16', '--enable_ema', 'True',
               '--seed', str(manifest['seed']), '--checkpoint_every', '0' if args.smoke or baseline else '1000',
               '--rolling_checkpoint_every', '0' if args.smoke or baseline else '500',
               '--log_every', '1' if args.smoke else '50', '--log_type', 'tensorboard',
               '--tb_log_root', str(root / 'tensorboard'), '--vis_every', '0', '--num_vis_images', '0']
        for name in COMPONENTS:
            cmd += ['--train_' + name, str(name in rates)]
        if not args.smoke:
            cmd += ['--val_config', str(root / 'validation_config.json')]
        if baseline:
            cmd += ['--validate_before_training', 'True']
        return cmd

    runs = []
    if not args.smoke and not args.skip_baseline:
        _, config, batch, accumulation = resolved[0]
        baseline_name = manifest.get('baseline_name', 'pretrained_baseline')
        directory = root / baseline_name
        directory.mkdir()
        write_json(directory / 'train_config.json', config)
        rates = resolved[0][0]['learning_rates']
        code = execute(command(directory, rates, batch, accumulation, baseline=True), directory, args.dry_run)
        runs.append((directory.name, {}, batch, accumulation, code))
    for experiment, config, batch, accumulation in resolved:
        directory = root / experiment['name']
        directory.mkdir()
        write_json(directory / 'train_config.json', config)
        code = execute(command(directory, experiment['learning_rates'], batch, accumulation), directory, args.dry_run)
        runs.append((directory.name, experiment['learning_rates'], batch, accumulation, code))
        summarize(root, runs, args.dry_run, args.smoke)
    print(f'Results: {root}')
    return int(any(code for *_, code in runs))


if __name__ == '__main__':
    sys.exit(main())
