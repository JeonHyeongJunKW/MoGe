#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_BEST_CHECKPOINT="/home/coder/MoGe/workspace/gaemi_finetune_sweep_5000/20261012_050306_517345_train/e_neck_all_heads_5000/checkpoint/best.pt"
BEST_CHECKPOINT="${BEST_CHECKPOINT:-$DEFAULT_BEST_CHECKPOINT}"
SWEEP_ROOT="${SWEEP_ROOT:-$REPO_ROOT/workspace/gaemi_finetune_best_head_polish/$(date +%Y%m%d_%H%M%S)_train}"
MANIFEST="${MANIFEST:-$REPO_ROOT/configs/experiments/gaemi_finetune_best_head_polish.json}"

if [[ ! -f "$BEST_CHECKPOINT" && " $* " != *" --dry-run "* ]]; then
    echo "Best checkpoint not found: $BEST_CHECKPOINT" >&2
    echo "Set BEST_CHECKPOINT to e_neck_all_heads_5000/checkpoint/best.pt." >&2
    exit 1
fi

exec "${PYTHON_BIN:-python3}" "$REPO_ROOT/run_script/sweep_5000.py" \
    --manifest "$MANIFEST" \
    --pretrained "$BEST_CHECKPOINT" \
    --root "$SWEEP_ROOT" \
    "$@"
