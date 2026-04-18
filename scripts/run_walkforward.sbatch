#!/bin/bash
#SBATCH --job-name=rl-trading-wf
#SBATCH --output=logs/rl_trading_%j.out
#SBATCH --error=logs/rl_trading_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu            # adjust to your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ── Adapt these to your cluster ─────────────────────────────────────
# module load python/3.12 cuda/12.x    # uncomment & edit as needed
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi 2>/dev/null || echo "(no nvidia-smi)"

# Install dependencies (first run only; cached afterwards)
uv sync --quiet

# Run full walk-forward experiment
#   --walk-forward : 5 expanding-window folds (2013–2025 OOS)
#   --epochs 200   : per Zhang et al.
#   --patience 20  : early stopping on val Sharpe
#   --device cuda   : use GPU (auto-detects if omitted)
uv run python -m rl_trading.run \
    --walk-forward \
    --epochs 200 \
    --patience 20 \
    --device cuda \
    --output-dir "artifacts/walkforward_${SLURM_JOB_ID}"

echo "Job $SLURM_JOB_ID finished at $(date)"
