#!/usr/bin/env bash
# Compatibility entry point: now adds 8,000 iterations (10,000 total).
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$REPO_ROOT/run_script/run_gaemi_flip_continue_tokens_total10000.sh" "$@"
