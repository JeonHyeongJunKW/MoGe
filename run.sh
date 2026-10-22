#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Local defaults; explicit CLI arguments below take precedence for scalar options.
VIEWER_CHECKPOINT="${MOGE_VIEWER_CHECKPOINT:-/home/jeon/mono_depth/model/best.pt}"
VIEWER_DATA_ROOT="${MOGE_VIEWER_DATA_ROOT:-/home/jeon/gaemi_simulations/datasets/mono-depth/20261007_20261008}"
VIEWER_NEAR_ROOT="${MOGE_VIEWER_NEAR_ROOT:-/home/jeon/gaemi_simulations/datasets/mono-depth/near_obstacles_train_20261020}"
VIEWER_INFERENCE_PYTHON="${MOGE_VIEWER_INFERENCE_PYTHON:-$REPO_ROOT/.venv-eval/bin/python}"

defaults=(--root "$REPO_ROOT/workspace")
[[ ! -f "$VIEWER_CHECKPOINT" ]] || defaults+=(--checkpoint "$VIEWER_CHECKPOINT")
[[ ! -x "$VIEWER_INFERENCE_PYTHON" ]] || defaults+=(--inference-python "$VIEWER_INFERENCE_PYTHON")
[[ ! -d "$VIEWER_DATA_ROOT" ]] || defaults+=(--data-root "$VIEWER_DATA_ROOT")
[[ ! -d "$VIEWER_NEAR_ROOT" ]] || defaults+=(--data-root "$VIEWER_NEAR_ROOT")

exec "${PYTHON_BIN:-python3}" "$REPO_ROOT/tools/ply_viewer/server.py" "${defaults[@]}" "$@"
