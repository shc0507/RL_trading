#!/bin/bash
#SBATCH --job-name=rl-wf-prefetch
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=general-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Stage 1 of the parallel walk-forward pipeline: fetch bars and build the
# shared feature frame used by every stage-2 worker. Writes
# `artifacts/cache/features.parquet` and is a no-op if the cache exists.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs artifacts/cache

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export PYTHONUNBUFFERED=1

uv sync --locked --quiet --extra wandb

CACHE_PATH="artifacts/cache/features.parquet"

uv run python -u -c "
from rl_trading.config import UNIVERSE
from rl_trading.evaluate import load_or_build_feature_cache
syms = [u['symbol'] for u in UNIVERSE]
feat = load_or_build_feature_cache(syms, '${CACHE_PATH}')
print(f'Feature cache ready: rows={len(feat)}, symbols={feat[\"symbol\"].nunique()}')
"
