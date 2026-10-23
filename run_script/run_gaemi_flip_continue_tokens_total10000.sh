#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SWEEP_ROOT="${SWEEP_ROOT:-$REPO_ROOT/workspace/gaemi_flip_continue_tokens_total10000/$(date +%Y%m%d_%H%M%S_%N)}"
exec "${PYTHON_BIN:-python3}" "$REPO_ROOT/run_script/sweep_5000.py" \
    --manifest "$REPO_ROOT/configs/experiments/gaemi_flip_continue_tokens_total10000.json" \
    --root "$SWEEP_ROOT" \
    "$@"
