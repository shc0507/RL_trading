#!/bin/bash
# Non-Slurm version — run directly or via nohup/tmux/screen.
#
# Usage:
#   bash scripts/run_walkforward.sh               # auto-detect device
#   bash scripts/run_walkforward.sh cuda          # force GPU
#   bash scripts/run_walkforward.sh cuda --epochs 20 --symbols SPY QQQ
#   nohup bash scripts/run_walkforward.sh &       # background
#
# Default experiment args are defined below so the script runs the full
# wandb-backed walk-forward experiment unless you override them explicitly.

set -euo pipefail

DEVICE="${1:-}"
if [ -n "$DEVICE" ] && [[ "$DEVICE" != -* ]]; then
    shift
else
    DEVICE=""
fi

DEVICE_ARGS=()
[ -n "$DEVICE" ] && DEVICE_ARGS=(--device "$DEVICE")

DEFAULT_ARGS=(
    --walk-forward
    --epochs 200
    --patience 20
    --wandb
    --wandb-project rl-trading
    --wandb-name wf-full
    --wandb-tags full gpu walkforward
)

cd "$(dirname "$0")/.."
mkdir -p logs artifacts

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export PYTHONUNBUFFERED=1

LOG="logs/rl_trading_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

SYNC_ARGS=(--locked --quiet --extra wandb)
for arg in "$@"; do
    if [ "$arg" = "--wandb" ]; then
        break
    fi
done

uv sync "${SYNC_ARGS[@]}"

uv run python -u -m rl_trading.run \
    "${DEFAULT_ARGS[@]}" \
    "${DEVICE_ARGS[@]}" \
    --output-dir "artifacts/walkforward" \
    "$@" \
    2>&1 | tee "$LOG"
