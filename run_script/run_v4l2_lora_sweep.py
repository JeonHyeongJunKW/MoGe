"""Compare ordinary last-four-block tuning with rank-4/rank-8 LoRA sequentially."""
import argparse
import csv
import datetime
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_optional(path):
    return json.loads(path.read_text()) if path.exists() else {}


def summarize(root, runs):
    rows = []
    for name, rank, status in runs:
        directory = root / name
        initial = read_optional(directory / 'validation/step_00000000.json')
        best = read_optional(directory / 'validation/best.json')
        memory = read_optional(directory / 'training_memory.json')
        assignments = read_optional(directory / 'optimizer_param_assignments.json')
        config = read_optional(directory / 'train_config.json')
        pseudo_loss = config.get('loss', {}).get('PSEUDO', {})
        parameters = assignments.get('parameters', [])
        metrics = best.get('metrics', {}).get('mean', {})
        trainable = [p for p in parameters if p['requires_grad']]
        rows.append(dict(experiment=name, rank=rank, status=status,
                         teacher_mask_weight=sum(v['weight'] for v in pseudo_loss.get('mask', {}).values()),
                         teacher_normal_weight=sum(v['weight'] for v in pseudo_loss.get('normal', {}).values()),
                         initial_rel=initial.get('score', ''), best_rel=best.get('score', ''),
                         best_delta1=metrics.get('depth_metric/delta1', ''), best_step=best.get('step', ''),
                         trainable_parameters=sum(p['numel'] for p in trainable) if parameters else '',
                         lora_parameters=sum(p['numel'] for p in trainable if '.lora_' in p['name']) if parameters else '',
                         peak_allocated_gib=memory.get('peak_allocated_gib', ''),
                         peak_reserved_gib=memory.get('peak_reserved_gib', '')))
    with (root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows, indent=2), flush=True)


def main(default_manifest=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=default_manifest or ROOT / 'configs/experiments/v4l2_lora_sweep.json')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--data-root', type=Path, default=Path('/home/coder/dataset'))
    parser.add_argument('--pseudo-root', type=Path)
    parser.add_argument('--student', type=Path, default=ROOT / 'pretrained/best_flip_mild_t400_total18500.pt')
    parser.add_argument('--experiment', action='append', help='Experiment name from manifest; repeatable')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-iterations', type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--smoke', action='store_true', help='Up to 10 updates per experiment; no validation; final saved')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    experiments = manifest['experiments']
    names = [e['name'] for e in experiments]
    if len(names) != len(set(names)) or any(not n.replace('_', '').isalnum() for n in names):
        parser.error('Experiment names must be unique simple identifiers')
    if args.experiment:
        if set(args.experiment) - set(names):
            parser.error('Unknown experiment selection')
        experiments = [e for e in experiments if e['name'] in args.experiment]
    if not experiments:
        parser.error('No experiments selected')
    batch = args.batch_size if args.batch_size is not None else manifest['batch_size']
    iterations = args.num_iterations if args.num_iterations is not None else manifest['num_iterations']
    # Validate all recipes before creating outputs or launching any training.
    sys.path.insert(0, str(ROOT))
    from run_script.run_v4l2_distillation import build_config
    pseudo = args.pseudo_root or Path(manifest.get('pseudo_root', args.data_root / 'v4l2_pseudo_v1'))
    for e in experiments:
        build_config(args.data_root, pseudo, manifest['real_fraction'], manifest['pseudo_loss_weight'],
                     iterations, batch, e['lora_rank'], e['lora_lr'],
                     manifest.get('multitask_labels', False), e.get('mask_weight', 0.), e.get('normal_weight', 0.))
    label = 'dry_run' if args.dry_run else 'smoke' if args.smoke else 'train'
    root = (args.root or ROOT / 'workspace' / manifest.get('workspace_group', 'v4l2_lora_sweep') /
            (datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + label)).resolve()
    if root.exists() and any(root.iterdir()):
        parser.error(f'Output directory is not empty: {root}')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (root / 'invocation.json').write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
                                                    for k, v in vars(args).items()}, indent=2) + '\n')
    runs = []
    for experiment in experiments:
        name = experiment['name']
        command = [sys.executable, str(ROOT / 'run_script/run_v4l2_distillation.py'),
                   '--root', str(root / name), '--data-root', str(args.data_root.resolve()),
                   '--pseudo-root', str(pseudo.resolve()), '--student', str(args.student.resolve()),
                   '--real-fraction', str(manifest['real_fraction']),
                   '--pseudo-loss-weight', str(manifest['pseudo_loss_weight']),
                   '--batch-size', str(batch), '--num-iterations', str(iterations),
                   '--lora-rank', str(experiment['lora_rank']), '--lora-lr', str(experiment['lora_lr'])]
        if manifest.get('multitask_labels'):
            command += ['--multitask-labels', '--mask-weight', str(experiment.get('mask_weight', 0.)),
                        '--normal-weight', str(experiment.get('normal_weight', 0.))]
        if args.dry_run:
            command.append('--dry-run')
        if args.smoke:
            command.append('--smoke')
        print(f'Experiment: {name}', flush=True)
        code = subprocess.call(command, cwd=ROOT)
        status = f'failed({code})' if code else 'dry_run' if args.dry_run else 'smoke_ok' if args.smoke else 'ok'
        runs.append((name, experiment['lora_rank'], status))
        summarize(root, runs)
        if code:
            print('Stopping sweep after failed experiment; inspect its log before restarting.', flush=True)
            break
    print(f'Results: {root}', flush=True)
    return int(any(status.startswith('failed') for _, _, status in runs))


if __name__ == '__main__':
    raise SystemExit(main())
