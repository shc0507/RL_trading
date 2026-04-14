#!/bin/bash
#SBATCH --job-name=zhang-dqn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --output=slurm-%x-%j.out

source "$(dirname "$0")/common.sh"

DQN_OPTIMIZER_UPDATES="${DQN_OPTIMIZER_UPDATES:-250000}"
DQN_BATCH_SIZE="${DQN_BATCH_SIZE:-64}"
DQN_WARMUP_STEPS="${DQN_WARMUP_STEPS:-1000}"

"${PYTHON_BIN}" scripts/slurm/run_zhang_production.py \
  "${COMMON_ARGS[@]}" \
  --algorithms dqn \
  --dqn-optimizer-updates "${DQN_OPTIMIZER_UPDATES}" \
  --dqn-batch-size "${DQN_BATCH_SIZE}" \
  --dqn-warmup-steps "${DQN_WARMUP_STEPS}"
