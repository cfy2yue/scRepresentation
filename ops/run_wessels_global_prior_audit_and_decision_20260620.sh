#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

ORIG_JSON=${ROOT}/CoupledFM/output/latentfm_runs/wessels_global_prior_20260620/scf_globalprior010_add005_wessels_4k/posthoc_eval_global_prior/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json
SWEEP020_JSON=${ROOT}/CoupledFM/output/latentfm_runs/wessels_global_prior_sweep_20260620/scf_globalprior020_add010_wessels_4k/posthoc_eval_global_prior_sweep/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json
SWEEP000_JSON=${ROOT}/CoupledFM/output/latentfm_runs/wessels_global_prior_sweep_20260620/scf_globalprior000_add010_wessels_4k/posthoc_eval_global_prior_sweep/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json

for path in "${ORIG_JSON}" "${SWEEP020_JSON}" "${SWEEP000_JSON}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Missing required posthoc JSON: ${path}" >&2
    exit 3
  fi
done

"${PYTHON}" "${ROOT}/ops/audit_wessels_global_prior_gate_20260620.py" \
  --run scf_globalprior010_add005_wessels_4k "${ORIG_JSON}" \
  --run scf_globalprior020_add010_wessels_4k "${SWEEP020_JSON}" \
  --run scf_globalprior000_add010_wessels_4k "${SWEEP000_JSON}" \
  --out-json "${ROOT}/reports/latentfm_wessels_global_prior_gate_audit_20260620.json" \
  --out-md "${ROOT}/reports/LATENTFM_WESSELS_GLOBAL_PRIOR_GATE_AUDIT_20260620.md"

"${PYTHON}" "${ROOT}/ops/decide_wessels_global_prior_next_action_20260620.py" \
  --out-json "${ROOT}/reports/latentfm_wessels_global_prior_next_action_20260620.json" \
  --out-md "${ROOT}/reports/LATENTFM_WESSELS_GLOBAL_PRIOR_NEXT_ACTION_20260620.md"
