#!/bin/bash

set -euo pipefail

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

if [[ -n "${MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  for module_name in ${MODULES}; do
    module load "${module_name}"
  done
fi

if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
    conda activate "${CONDA_ENV_NAME}"
  fi
fi

if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
elif [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/slurm/${SLURM_JOB_NAME:-zhang}_${SLURM_JOB_ID:-manual}}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-101}"
CV_FRACTION="${CV_FRACTION:-0.10}"
EARLY_STOPPING_PATIENCE_EPOCHS="${EARLY_STOPPING_PATIENCE_EPOCHS:-20}"
SELECTION_METRIC="${SELECTION_METRIC:-sharpe}"
SELECTION_SPLIT="${SELECTION_SPLIT:-cv}"

mkdir -p "${RUN_OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

if [[ -z "${DATA_ROOT:-}" && -z "${SOURCE_DATASET_DIR:-}" ]]; then
  echo "Set DATA_ROOT or SOURCE_DATASET_DIR before submitting the job." >&2
  exit 1
fi

COMMON_ARGS=(
  --project-root "${PROJECT_ROOT}"
  --output-dir "${RUN_OUTPUT_DIR}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --cv-fraction "${CV_FRACTION}"
  --early-stopping-patience-epochs "${EARLY_STOPPING_PATIENCE_EPOCHS}"
  --selection-metric "${SELECTION_METRIC}"
  --selection-split "${SELECTION_SPLIT}"
)

if [[ -n "${DATA_ROOT:-}" ]]; then
  COMMON_ARGS+=(--data-root "${DATA_ROOT}")
fi
if [[ -n "${SOURCE_DATASET_DIR:-}" ]]; then
  COMMON_ARGS+=(--source-dataset-dir "${SOURCE_DATASET_DIR}")
fi
if [[ -n "${UNIVERSE_MANIFEST:-}" ]]; then
  COMMON_ARGS+=(--universe-manifest "${UNIVERSE_MANIFEST}")
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Output dir: ${RUN_OUTPUT_DIR}"
echo "Python: $(${PYTHON_BIN} --version 2>&1)"
echo "Device request: ${DEVICE}"
echo "Data root: ${DATA_ROOT:-<unset>}"
echo "Source dataset dir: ${SOURCE_DATASET_DIR:-<unset>}"
echo "Universe manifest: ${UNIVERSE_MANIFEST:-<default>}"
