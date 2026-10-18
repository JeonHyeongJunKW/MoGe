#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SWEEP_ROOT="${SWEEP_ROOT:-$REPO_ROOT/workspace/gaemi_backbone_sweep_1500/$(date +%Y%m%d_%H%M%S_%N)}"
exec "${PYTHON_BIN:-python3}" "$REPO_ROOT/run_script/sweep_5000.py" \
    --manifest "$REPO_ROOT/configs/experiments/gaemi_backbone_sweep_1500.json" \
    --root "$SWEEP_ROOT" \
    "$@"
