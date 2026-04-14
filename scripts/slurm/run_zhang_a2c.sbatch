#!/bin/bash
#SBATCH --job-name=zhang-a2c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --output=slurm-%x-%j.out

source "$(dirname "$0")/common.sh"

A2C_TOTAL_STEPS="${A2C_TOTAL_STEPS:-250000}"
A2C_BATCH_SIZE="${A2C_BATCH_SIZE:-128}"
A2C_ACTOR_LR="${A2C_ACTOR_LR:-1e-4}"
A2C_CRITIC_LR="${A2C_CRITIC_LR:-1e-3}"

"${PYTHON_BIN}" scripts/slurm/run_zhang_production.py \
  "${COMMON_ARGS[@]}" \
  --algorithms a2c \
  --a2c-total-steps "${A2C_TOTAL_STEPS}" \
  --a2c-batch-size "${A2C_BATCH_SIZE}" \
  --a2c-actor-lr "${A2C_ACTOR_LR}" \
  --a2c-critic-lr "${A2C_CRITIC_LR}"
