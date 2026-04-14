#!/bin/bash
#SBATCH --job-name=zhang-pg
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

PG_TOTAL_STEPS="${PG_TOTAL_STEPS:-250000}"
PG_ACTOR_LR="${PG_ACTOR_LR:-1e-4}"

"${PYTHON_BIN}" "${RUN_SCRIPT}" \
  "${COMMON_ARGS[@]}" \
  --algorithms pg \
  --pg-total-steps "${PG_TOTAL_STEPS}" \
  --pg-actor-lr "${PG_ACTOR_LR}"
