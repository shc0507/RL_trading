#!/usr/bin/env bash
# Driver for the parallel walk-forward pipeline:
#   stage1_prefetch  -> build shared feature cache
#   stage2_train     -> slurm array, one GPU job per (agent, fold, class)
#   stage3_aggregate -> glob stage-2 outputs, produce unified results
#
# Usage:
#   bash scripts/run_walkforward_parallel.sh
#   RUN_ID=mytag bash scripts/run_walkforward_parallel.sh
#   EPOCHS=50 PATIENCE=10 ARRAY_LIMIT=3 bash scripts/run_walkforward_parallel.sh
#
# Env vars:
#   RUN_ID       label for the run (defaults to the stage-2 array job id)
#   EPOCHS       stage-2 epochs (default 200)
#   PATIENCE     stage-2 patience (default 20)
#   ARRAY_LIMIT  restrict stage-2 array to 0..N-1 (default 24 = full sweep: 3 agents x 2 folds x 4 classes)
#   WANDB        0 to disable wandb, 1 to enable (default 1)

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs artifacts

EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-20}"
ARRAY_LIMIT="${ARRAY_LIMIT:-24}"
WANDB="${WANDB:-1}"
FEATURE_CACHE="artifacts/cache/features.parquet"

# --- Stage 1: prefetch (skipped if the cache already exists) ---
if [ -f "$FEATURE_CACHE" ]; then
    echo "Using existing feature cache: $FEATURE_CACHE"
    DEP_FLAG=""
else
    echo "Submitting stage 1 (prefetch) ..."
    STAGE1_ID=$(sbatch --parsable scripts/stage1_prefetch.sbatch)
    echo "  stage1 job id: $STAGE1_ID"
    DEP_FLAG="--dependency=afterok:${STAGE1_ID}"
fi

# --- Stage 2: training array ---
ARRAY_SPEC="0-$((ARRAY_LIMIT - 1))%8"
echo "Submitting stage 2 (train, array=${ARRAY_SPEC}) ..."
RUN_ID_PARAM="${RUN_ID:-}"
STAGE2_ID=$(sbatch --parsable \
    ${DEP_FLAG:+$DEP_FLAG} \
    --array="${ARRAY_SPEC}" \
    --export=ALL,RUN_ID="${RUN_ID_PARAM}",EPOCHS="${EPOCHS}",PATIENCE="${PATIENCE}",WANDB="${WANDB}",FEATURE_CACHE="${FEATURE_CACHE}" \
    scripts/stage2_train.sbatch)
echo "  stage2 job id: $STAGE2_ID"

# --- Stage 3: aggregate (afterany so partial results still aggregate) ---
# Inherit the array job id as RUN_ID when user didn't set one.
EFFECTIVE_RUN_ID="${RUN_ID_PARAM:-$STAGE2_ID}"
echo "Submitting stage 3 (aggregate) with RUN_ID=${EFFECTIVE_RUN_ID} ..."
STAGE3_ID=$(sbatch --parsable \
    --dependency=afterany:"${STAGE2_ID}" \
    --export=ALL,RUN_ID="${EFFECTIVE_RUN_ID}",FEATURE_CACHE="${FEATURE_CACHE}" \
    scripts/stage3_aggregate.sbatch)
echo "  stage3 job id: $STAGE3_ID"

echo
echo "Pipeline submitted:"
echo "  stage1 = ${STAGE1_ID:-(skipped, cache exists)}"
echo "  stage2 = ${STAGE2_ID}"
echo "  stage3 = ${STAGE3_ID}"
echo "Expect final results in artifacts/wf_${EFFECTIVE_RUN_ID}/aggregated/"
