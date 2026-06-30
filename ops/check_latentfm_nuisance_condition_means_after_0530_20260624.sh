#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

NOT_BEFORE_EPOCH=$(date -d '2026-06-24 05:30:00' +%s)
NOW_EPOCH=$(date +%s)
if (( NOW_EPOCH < NOT_BEFORE_EPOCH )); then
  echo "Refusing to check nuisance condition-means job before 2026-06-24 05:30:00 CST; now=$(date '+%F %T %Z')" >&2
  exit 3
fi

RUN_DIR=${ROOT}/runs/latentfm_xverse_nuisance_condition_means_20260624
exit_code=$(cat "${RUN_DIR}/EXIT_CODE" 2>/dev/null || true)
echo "[$(date '+%F %T %Z')] nuisance_condition_means_exit=${exit_code:-missing}"

if [[ "${exit_code}" == "0" ]]; then
  echo "Running nuisance residual CPU gate"
  "${PYTHON}" "${ROOT}/ops/audit_latentfm_xverse_nuisance_residual_gate_20260624.py"
else
  echo "Nuisance condition-means job not complete with exit 0; not running gate."
fi
