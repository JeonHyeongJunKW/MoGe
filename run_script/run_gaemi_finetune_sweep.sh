#!/bin/bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SWEEP_CONFIG="${SWEEP_CONFIG:-configs/experiments/gaemi_finetune_sweep.json}"
SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${SWEEP_ROOT:-workspace/gaemi_finetune_sweep/$SWEEP_ID}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/home/coder/MoGe/pretrained/moge-2-vits.pt}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1000}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
BATCH_SIZE_FORWARD="${BATCH_SIZE_FORWARD:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    else
        echo "Python interpreter not found (tried: $PYTHON_BIN, python3)" >&2
        exit 1
    fi
fi

mkdir -p "$SWEEP_ROOT"

if [[ ! -f "$SWEEP_CONFIG" ]]; then
    echo "Sweep config not found: $SWEEP_CONFIG" >&2
    exit 1
fi

if [[ "$DRY_RUN" != "1" && ! -f "$PRETRAINED_CHECKPOINT" ]]; then
    echo "Initial checkpoint not found: $PRETRAINED_CHECKPOINT" >&2
    exit 1
fi

mapfile -t EXPERIMENT_ROWS < <(
    "$PYTHON_BIN" - "$SWEEP_CONFIG" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding='utf-8'))
for experiment in manifest['experiments']:
    print('\t'.join([
        experiment['name'],
        str(experiment['train_scale_head']),
        str(experiment['train_points_head']),
    ]))
PY
)

if [[ "${#EXPERIMENT_ROWS[@]}" -eq 0 ]]; then
    echo "No experiments found in: $SWEEP_CONFIG" >&2
    exit 1
fi

for row in "${EXPERIMENT_ROWS[@]}"; do
    IFS=$'\t' read -r experiment_name train_scale_head train_points_head <<< "$row"
    experiment_dir="$SWEEP_ROOT/$experiment_name"
    resolved_config="$experiment_dir/train_config.json"
    log_file="$experiment_dir/train.log"
    mkdir -p "$experiment_dir"

    "$PYTHON_BIN" - "$SWEEP_CONFIG" "$experiment_name" "$resolved_config" <<'PY'
import copy
import json
import sys
from pathlib import Path

manifest_path, experiment_name, output_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
experiment = next(item for item in manifest['experiments'] if item['name'] == str(experiment_name))
config = copy.deepcopy(json.loads(Path(manifest['base_config']).read_text(encoding='utf-8')))

config['data']['image_augmentation'] = experiment['image_augmentation']
for group in config['optimizer']['params']:
    if group.get('name') == 'head':
        group['lr'] = experiment['head_lr']
        break
else:
    raise KeyError('Optimizer group named "head" was not found')

config['sweep_experiment'] = {
    'name': experiment['name'],
    'description': experiment['description'],
    'train_scale_head': experiment['train_scale_head'],
    'train_points_head': experiment['train_points_head'],
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(config, indent=4) + '\n', encoding='utf-8')
PY

    validation_config="$("$PYTHON_BIN" - "$SWEEP_CONFIG" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['validation_config'])
PY
)"

    command=(
        "$PYTHON_BIN" -m accelerate.commands.launch
        --num_processes "$NUM_PROCESSES"
        --module moge.train.train_moge12
        --config "$resolved_config"
        --val_config "$validation_config"
        --initial_checkpoint "$PRETRAINED_CHECKPOINT"
        --checkpoint none
        --name "$experiment_name"
        --workspace "$experiment_dir"
        --num_iterations "$NUM_ITERATIONS"
        --batch_size_forward "$BATCH_SIZE_FORWARD"
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
        --enable_gradient_checkpointing True
        --precision mixed_bf16
        --train_scale_head "$train_scale_head"
        --train_points_head "$train_points_head"
        --enable_ema True
        --checkpoint_every 250
        --rolling_checkpoint_every 500
        --log_every 50
        --log_type tensorboard
        --tb_log_root "$SWEEP_ROOT/tensorboard"
        --vis_every 0
        --num_vis_images 0
    )

    echo
    echo "================================================================"
    echo "Experiment: $experiment_name"
    echo "Workspace:  $experiment_dir"
    echo "================================================================"

    if [[ "$DRY_RUN" == "1" ]]; then
        printf ' %q' "${command[@]}"
        printf '\n'
        echo "0" > "$experiment_dir/exit_code"
        continue
    fi

    "${command[@]}" 2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}
    echo "$exit_code" > "$experiment_dir/exit_code"
    if [[ "$exit_code" -ne 0 ]]; then
        echo "Experiment failed with exit code $exit_code; continuing with the remaining experiments." >&2
    fi
done

"$PYTHON_BIN" - "$SWEEP_CONFIG" "$SWEEP_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

manifest_path, sweep_root = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
rows = []

for experiment in manifest['experiments']:
    directory = sweep_root / experiment['name']
    exit_code_path = directory / 'exit_code'
    exit_code = exit_code_path.read_text().strip() if exit_code_path.exists() else 'not_run'
    best_path = directory / 'validation' / 'best.json'
    validation_files = sorted((directory / 'validation').glob('step_*.json'))
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    latest = json.loads(validation_files[-1].read_text()) if validation_files else None
    best_metrics = best.get('metrics', {}).get('mean', {}) if best else {}
    latest_metrics = latest.get('metrics', {}).get('mean', {}) if latest else {}
    rows.append({
        'experiment': experiment['name'],
        'exit_code': exit_code,
        'augmentations': ','.join(experiment['image_augmentation']) or 'none',
        'scale_head': experiment['train_scale_head'],
        'points_head': experiment['train_points_head'],
        'head_lr': experiment['head_lr'],
        'best_step': best.get('step', '') if best else '',
        'best_depth_rel': best.get('score', '') if best else '',
        'best_depth_delta1': best_metrics.get('depth_metric/delta1', ''),
        'latest_step': latest.get('step', '') if latest else '',
        'latest_depth_rel': latest.get('score', '') if latest else '',
        'latest_depth_delta1': latest_metrics.get('depth_metric/delta1', ''),
    })

fieldnames = list(rows[0]) if rows else []
with (sweep_root / 'summary.csv').open('w', newline='', encoding='utf-8') as output:
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

headers = ['experiment', 'status', 'aug', 'heads', 'lr', 'best step', 'best rel', 'best delta1', 'latest rel']
markdown = [
    '| ' + ' | '.join(headers) + ' |',
    '| ' + ' | '.join(['---'] * len(headers)) + ' |',
]
for row in rows:
    heads = 'scale+points' if row['points_head'] else 'scale only'
    def number(value):
        return f'{float(value):.6g}' if value != '' else '-'
    markdown.append('| ' + ' | '.join([
        row['experiment'],
        'ok' if row['exit_code'] == '0' else f"failed({row['exit_code']})",
        row['augmentations'],
        heads,
        f"{row['head_lr']:.1e}",
        str(row['best_step']) if row['best_step'] != '' else '-',
        number(row['best_depth_rel']),
        number(row['best_depth_delta1']),
        number(row['latest_depth_rel']),
    ]) + ' |')

successful = [row for row in rows if row['best_depth_rel'] != '']
if successful:
    winner = min(successful, key=lambda row: float(row['best_depth_rel']))
    markdown.extend([
        '',
        f"Best experiment: **{winner['experiment']}** ",
        f"(`depth_metric/rel={float(winner['best_depth_rel']):.6g}` at step {winner['best_step']})",
    ])
else:
    markdown.extend(['', 'No successful validation result was found.'])

summary = '\n'.join(markdown) + '\n'
(sweep_root / 'summary.md').write_text(summary, encoding='utf-8')
print('\nSweep summary')
print(summary)
print(f"CSV:      {sweep_root / 'summary.csv'}")
print(f"Markdown: {sweep_root / 'summary.md'}")
PY

echo "All experiments finished. Results: $SWEEP_ROOT"
