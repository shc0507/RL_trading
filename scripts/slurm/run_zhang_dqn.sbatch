#!/bin/bash
#SBATCH --job-name=zhang-dqn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=general-gpu
#SBATCH --gpus=1
#SBATCH --output=slurm-%x-%j.out

COMMON_SH="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}/scripts/slurm/common.sh"
if [[ ! -f "${COMMON_SH}" ]]; then
  echo "Could not find common.sh at ${COMMON_SH}. Submit from the repo root or set PROJECT_ROOT=/abs/path/to/repo." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${COMMON_SH}"

DQN_OPTIMIZER_UPDATES="${DQN_OPTIMIZER_UPDATES:-250000}"
DQN_BATCH_SIZE="${DQN_BATCH_SIZE:-64}"
DQN_WARMUP_STEPS="${DQN_WARMUP_STEPS:-1000}"

"${PYTHON_BIN}" "${RUN_SCRIPT}" \
  "${COMMON_ARGS[@]}" \
  --algorithms dqn \
  --dqn-optimizer-updates "${DQN_OPTIMIZER_UPDATES}" \
  --dqn-batch-size "${DQN_BATCH_SIZE}" \
  --dqn-warmup-steps "${DQN_WARMUP_STEPS}"
