#!/bin/bash
#SBATCH --job-name=rl-wf-train
#SBATCH --array=0-71%8
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Stage 2: one training task per (agent, fold, asset_class, seed). Array size
# 3 agents × 2 folds × 4 classes × 3 seeds = 72; train_one.py decodes the
# array index. The `%8` cap matches the per-user general-gpu concurrency QoS.
# Pass --run-id on sbatch like `--export=RUN_ID=foo` to label the output.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID) started at $(date)"
echo "Node: $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi 2>/dev/null || echo "(no nvidia-smi)"

uv sync --locked --quiet --extra wandb

RUN_ID="${RUN_ID:-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}}"
RUN_DIR="artifacts/wf_${RUN_ID}"
CACHE_PATH="${FEATURE_CACHE:-artifacts/cache/features.parquet}"

EPOCHS="${EPOCHS:-200}"
# Patience: pass through only if explicitly set; otherwise let
# train_one.py pick its per-agent default (DQN/PG: 50 OR-multi,
# A2C: 30 sharpe-only).
PATIENCE_FLAG=()
if [ -n "${PATIENCE:-}" ]; then
    PATIENCE_FLAG=(--patience "$PATIENCE")
fi

WANDB_ARGS=()
if [ "${WANDB:-1}" = "1" ]; then
    WANDB_ARGS=(
        --wandb
        --wandb-project "${WANDB_PROJECT:-rl-trading}"
        --wandb-tags full gpu walkforward parallel
    )
fi

# train_one.py decodes SLURM_ARRAY_TASK_ID -> (agent, fold, class, seed)
# and writes to ${RUN_DIR}/<agent>/<class>_fold<n>_seed<s>/.
uv run python -u -m rl_trading.train_one \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    --feature-cache "$CACHE_PATH" \
    --run-dir "$RUN_DIR" \
    --epochs "$EPOCHS" \
    "${PATIENCE_FLAG[@]}" \
    "${WANDB_ARGS[@]}"

echo "Task finished at $(date)"
