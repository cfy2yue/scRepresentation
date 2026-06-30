#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

BASELINE_JSON=${COUPLED}/output/latentfm_runs/condition_prior_teacher_injection_20260619/scf_prior010_inject_e2_4k/posthoc_eval_focus_nwg/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json
RUN_A_JSON=${COUPLED}/output/latentfm_runs/focus_learnability_20260619/scf_prior010_inject_nwg_focus_4k/posthoc_eval_focus_nwg/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json
RUN_B_JSON=${COUPLED}/output/latentfm_runs/focus_learnability_20260619/scf_prior010_inject_nwg_focus_dsloss05_4k/posthoc_eval_focus_nwg/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json

AUDIT_JSON=${ROOT}/reports/latentfm_focus_learnability_gate_audit_20260619.json
AUDIT_MD=${ROOT}/reports/LATENTFM_FOCUS_LEARNABILITY_GATE_AUDIT_20260619.md
DECISION_JSON=${ROOT}/reports/latentfm_focus_next_action_decision_20260619.json
DECISION_MD=${ROOT}/reports/LATENTFM_FOCUS_NEXT_ACTION_DECISION_20260619.md

for path in "${BASELINE_JSON}" "${RUN_A_JSON}" "${RUN_B_JSON}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing required focus posthoc JSON: ${path}" >&2
    exit 2
  fi
done

"${PYTHON}" "${ROOT}/ops/audit_latentfm_focus_learnability_gate_20260619.py" \
  --baseline-name scf_prior010_inject_e2_4k \
  --baseline-json "${BASELINE_JSON}" \
  --run scf_prior010_inject_nwg_focus_4k "${RUN_A_JSON}" \
  --run scf_prior010_inject_nwg_focus_dsloss05_4k "${RUN_B_JSON}" \
  --out-json "${AUDIT_JSON}" \
  --out-md "${AUDIT_MD}"

"${PYTHON}" "${ROOT}/ops/decide_latentfm_focus_next_action_20260619.py" \
  --audit-json "${AUDIT_JSON}" \
  --out-json "${DECISION_JSON}" \
  --out-md "${DECISION_MD}"

echo "audit_json=${AUDIT_JSON}"
echo "audit_md=${AUDIT_MD}"
echo "decision_json=${DECISION_JSON}"
echo "decision_md=${DECISION_MD}"
