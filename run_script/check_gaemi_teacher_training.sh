#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SWEEP_ROOT="${SWEEP_ROOT:-$REPO_ROOT/workspace/gaemi_teacher_training_check/$(date +%Y%m%d_%H%M%S_%N)}"
SMOKE_STEPS="${SMOKE_STEPS:-2}"

"${PYTHON_BIN:-python3}" "$REPO_ROOT/run_script/sweep_5000.py" \
    --manifest "$REPO_ROOT/configs/experiments/gaemi_flip_continue_tokens_total30000.json" \
    --root "$SWEEP_ROOT" \
    --experiment teacher_vitl_t1200_train2000 \
    --num-iterations "$SMOKE_STEPS" \
    --smoke \
    "$@"

echo "Teacher training smoke test completed successfully."
