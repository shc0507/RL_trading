#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

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

PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/slurm/${SLURM_JOB_NAME:-zhang}_${SLURM_JOB_ID:-manual}}"
if [[ "${RUN_OUTPUT_DIR}" != /* ]]; then
  RUN_OUTPUT_DIR="${PROJECT_ROOT}/${RUN_OUTPUT_DIR}"
fi
RUN_SCRIPT="${RUN_SCRIPT:-${PROJECT_ROOT}/scripts/slurm/run_zhang_production.py}"

if [[ -n "${VENV_PATH:-}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
elif [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if [[ -z "${PYTHON_BIN:-}" && -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-101}"
CV_FRACTION="${CV_FRACTION:-0.10}"
EARLY_STOPPING_PATIENCE_EPOCHS="${EARLY_STOPPING_PATIENCE_EPOCHS:-20}"
SELECTION_METRIC="${SELECTION_METRIC:-sharpe}"
SELECTION_SPLIT="${SELECTION_SPLIT:-cv}"

mkdir -p "${RUN_OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "Could not find Slurm runner at ${RUN_SCRIPT}." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "Python 3.11+ is required. Set PYTHON_BIN, VENV_PATH, or CONDA_SH/CONDA_ENV_NAME before submitting." >&2
  exit 1
fi

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
echo "Runner: ${RUN_SCRIPT}"
echo "Device request: ${DEVICE}"
echo "Data root: ${DATA_ROOT:-<unset>}"
echo "Source dataset dir: ${SOURCE_DATASET_DIR:-<unset>}"
echo "Universe manifest: ${UNIVERSE_MANIFEST:-<default>}"
