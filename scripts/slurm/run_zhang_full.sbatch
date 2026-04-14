#!/bin/bash
#SBATCH --job-name=zhang-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --output=slurm-%x-%j.out

source "$(dirname "$0")/common.sh"

DQN_OPTIMIZER_UPDATES="${DQN_OPTIMIZER_UPDATES:-250000}"
DQN_BATCH_SIZE="${DQN_BATCH_SIZE:-64}"
DQN_WARMUP_STEPS="${DQN_WARMUP_STEPS:-1000}"
PG_TOTAL_STEPS="${PG_TOTAL_STEPS:-250000}"
PG_ACTOR_LR="${PG_ACTOR_LR:-1e-4}"
A2C_TOTAL_STEPS="${A2C_TOTAL_STEPS:-250000}"
A2C_BATCH_SIZE="${A2C_BATCH_SIZE:-128}"
A2C_ACTOR_LR="${A2C_ACTOR_LR:-1e-4}"
A2C_CRITIC_LR="${A2C_CRITIC_LR:-1e-3}"

"${PYTHON_BIN}" scripts/slurm/run_zhang_production.py \
  "${COMMON_ARGS[@]}" \
  --algorithms dqn,pg,a2c \
  --dqn-optimizer-updates "${DQN_OPTIMIZER_UPDATES}" \
  --dqn-batch-size "${DQN_BATCH_SIZE}" \
  --dqn-warmup-steps "${DQN_WARMUP_STEPS}" \
  --pg-total-steps "${PG_TOTAL_STEPS}" \
  --pg-actor-lr "${PG_ACTOR_LR}" \
  --a2c-total-steps "${A2C_TOTAL_STEPS}" \
  --a2c-batch-size "${A2C_BATCH_SIZE}" \
  --a2c-actor-lr "${A2C_ACTOR_LR}" \
  --a2c-critic-lr "${A2C_CRITIC_LR}"
