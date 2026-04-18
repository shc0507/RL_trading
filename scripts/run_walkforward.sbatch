#!/bin/bash
#SBATCH --job-name=rl-trading-wf
#SBATCH --array=0-2
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# This repo requires Python >=3.11. On this cluster the default login-node
# Python is 3.9, so rely on `uv` to provision the project interpreter into
# `.venv` instead of loading a site Python module.
#
# Examples:
#   sbatch scripts/run_walkforward.sbatch
#   sbatch --partition=general-preempt-gpu scripts/run_walkforward.sbatch
#   sbatch scripts/run_walkforward.sbatch --symbols SPY QQQ --epochs 20
#
# This script submits one array task per agent:
#   0 -> dqn
#   1 -> pg
#   2 -> a2c
# User-supplied `--agents` arguments are intentionally ignored here.
#
# Default experiment args are defined below so a plain
# `sbatch scripts/run_walkforward.sbatch` runs the full wandb-backed job.
# Pass explicit CLI flags at submit time to override named values like
# `--epochs`, `--patience`, `--wandb-project`, `--wandb-name`, and `--wandb-tags`.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs artifacts

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Node: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi 2>/dev/null || echo "(no nvidia-smi)"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not available in PATH"
    exit 1
fi

AGENTS=(dqn pg a2c)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#AGENTS[@]}" ]; then
    echo "Invalid SLURM_ARRAY_TASK_ID: $TASK_ID"
    exit 1
fi
AGENT_NAME="${AGENTS[$TASK_ID]}"

DEFAULT_ARGS=(
    --walk-forward
    --epochs 200
    --patience 20
    --wandb
    --wandb-project rl-trading
    --wandb-name wf-full-etfs
    --wandb-tags full gpu walkforward etf
)

USER_ARGS=()
SYNC_ARGS=(--locked --quiet --extra wandb)
while [ "$#" -gt 0 ]; do
    case "$1" in
        --wandb)
            USER_ARGS+=("$1")
            shift
            ;;
        --agents)
            shift
            while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
                shift
            done
            ;;
        --agents=*)
            shift
            ;;
        *)
            USER_ARGS+=("$1")
            shift
            ;;
    esac
done

# Install dependencies (first run only; cached afterwards).
uv sync "${SYNC_ARGS[@]}"

RUN_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
OUTPUT_DIR="artifacts/walkforward_${RUN_ID}_${AGENT_NAME}"

# Run full walk-forward experiment
#   defaults come from DEFAULT_ARGS above
#   device is auto-detected (CUDA on GPU nodes, CPU otherwise)
echo "Array task: ${TASK_ID}"
echo "Agent: ${AGENT_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
uv run python -u -m rl_trading.run \
    --agents "$AGENT_NAME" \
    "${DEFAULT_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    "${USER_ARGS[@]}"

echo "Job $SLURM_JOB_ID finished at $(date)"
