#!/bin/bash
#SBATCH --job-name=zhang-pg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --output=slurm-%x-%j.out

source "$(dirname "$0")/common.sh"

PG_TOTAL_STEPS="${PG_TOTAL_STEPS:-250000}"
PG_ACTOR_LR="${PG_ACTOR_LR:-1e-4}"

"${PYTHON_BIN}" scripts/slurm/run_zhang_production.py \
  "${COMMON_ARGS[@]}" \
  --algorithms pg \
  --pg-total-steps "${PG_TOTAL_STEPS}" \
  --pg-actor-lr "${PG_ACTOR_LR}"
