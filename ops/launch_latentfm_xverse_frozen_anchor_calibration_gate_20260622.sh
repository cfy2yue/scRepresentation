#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/data/cyx/1030/scLatent/runs/latentfm_xverse_frozen_anchor_calibration_20260622/xverse_anchor_calibration_light_ode10_cell128"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"

source /data/cyx/1030/scLatent/init-scdfm.sh >/dev/null
cd /data/cyx/1030/scLatent/CoupledFM

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-4}"

set +e
python /data/cyx/1030/scLatent/ops/audit_latentfm_xverse_frozen_anchor_calibration_gate_20260622.py \
  --device cuda:0 \
  --gpu 0 \
  --ode-steps 10 \
  --max-cells 128 \
  --max-chunk 128 \
  --max-train-conditions-per-dataset 32 \
  --max-val-conditions-per-dataset 8 \
  --bootstrap 2000 \
  --run-root "${RUN_ROOT}" \
  > "${LOG_DIR}/run.log" 2>&1
status="$?"
set -e
echo "${status}" > "${RUN_ROOT}/EXIT_CODE"
date > "${RUN_ROOT}/FINISHED"
exit "${status}"
