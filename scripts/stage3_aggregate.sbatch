#!/bin/bash
#SBATCH --job-name=rl-wf-aggregate
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=general-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Stage 3: glob stage-2 outputs, recompute baselines against the cached
# features, and produce the unified Zhang-style results tables and plots.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export PYTHONUNBUFFERED=1

uv sync --locked --quiet --extra wandb

RUN_ID="${RUN_ID:-$SLURM_JOB_ID}"
RUN_DIR="artifacts/wf_${RUN_ID}"
CACHE_PATH="${FEATURE_CACHE:-artifacts/cache/features.parquet}"
OUTPUT_DIR="${RUN_DIR}/aggregated"

uv run python -u -m rl_trading.aggregate \
    --run-dir "$RUN_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --feature-cache "$CACHE_PATH"

echo "Aggregation written to $OUTPUT_DIR"
