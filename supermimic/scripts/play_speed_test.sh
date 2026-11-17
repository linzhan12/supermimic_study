#!/usr/bin/env bash
# First argument: local checkpoint directory containing local_config.json.
set -euo pipefail
if [[ $# -eq 0 ]]; then
  echo "Usage: bash $0 /absolute/path/to/checkpoint-directory [extra play arguments]" >&2
  exit 2
fi
run_path="$(realpath -- "$1")"
shift
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec uv run scripts/play_speed_test.py \
  --checkpoint-dir "$run_path" \
  "$@"
