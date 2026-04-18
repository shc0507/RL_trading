#!/bin/bash
# Non-Slurm version — run directly or via nohup/tmux/screen.
#
# Usage:
#   bash scripts/run_walkforward.sh              # auto-detect device
#   bash scripts/run_walkforward.sh cuda          # force GPU
#   nohup bash scripts/run_walkforward.sh &       # background

set -euo pipefail

DEVICE="${1:-}"
DEVICE_FLAG=""
[ -n "$DEVICE" ] && DEVICE_FLAG="--device $DEVICE"

cd "$(dirname "$0")/.."
mkdir -p logs

LOG="logs/rl_trading_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

uv sync --quiet

uv run python -m rl_trading.run \
    --walk-forward \
    --epochs 200 \
    --patience 20 \
    $DEVICE_FLAG \
    --output-dir "artifacts/walkforward" \
    2>&1 | tee "$LOG"
