#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

echo "[$(date '+%F %T %Z')] true cell-count post-materialization gates"
echo "[gate] sample provenance"
"${PYTHON}" "${ROOT}/ops/backfill_latentfm_true_cell_count_sample_provenance_20260624.py" --write

echo "[gate] capped-H5 schema/provenance"
"${PYTHON}" "${ROOT}/ops/audit_latentfm_true_cell_count_capped_h5_schema_gate_20260624.py"

echo "[gate] dry-load"
"${PYTHON}" "${ROOT}/ops/audit_latentfm_true_cell_count_dryload_gate_20260624.py"

echo "[gate] design controls"
"${PYTHON}" "${ROOT}/ops/audit_latentfm_true_cell_count_design_controls_20260624.py"

echo "[$(date '+%F %T %Z')] true cell-count post-materialization gates complete"
