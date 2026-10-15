#!/usr/bin/env bash
# Evaluate only; save per-data-path validation metrics in JSON and CSV.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
exec "${PYTHON_BIN:-python3}" -m moge.scripts.analyze_validation "$@"
