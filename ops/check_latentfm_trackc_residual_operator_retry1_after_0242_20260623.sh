#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PY=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PY}" ]]; then
  PY=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi
RUN_NAME=xverse_trackc_residual_operator_memall_resid_ep050_replay2_2k_seed42_retry1
RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_residual_operator_20260623/${RUN_NAME}
DECISION_MD=${ROOT}/reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_${RUN_NAME}.md
DECISION_JSON=${ROOT}/reports/latentfm_trackc_routed_distill_smoke_decision_${RUN_NAME}.json
ROUTE_GAP_JSON=${ROOT}/reports/latentfm_trackc_residual_operator_route_gap_gate_${RUN_NAME}.json
ROUTE_GAP_MD=${ROOT}/reports/LATENTFM_TRACKC_RESIDUAL_OPERATOR_ROUTE_GAP_GATE_${RUN_NAME}.md
WINDOW="2026-06-23 02:42:00"

now_epoch=$(date +%s)
window_epoch=$(date -d "${WINDOW}" +%s)
if (( now_epoch < window_epoch )); then
  echo "Refusing to check before ${WINDOW} CST; residual-operator retry1 is a long GPU task." >&2
  exit 3
fi

echo "[$(date '+%F %T %Z')] Track C residual-operator retry1 guarded check"
echo "run_root=${RUN_ROOT}"
cat "${RUN_ROOT}/${RUN_NAME}.EXIT_CODE" 2>/dev/null || echo "train still running or marker absent"
cat "${RUN_ROOT}/${RUN_NAME}.POSTHOC_EXIT_CODE" 2>/dev/null || echo "posthoc still running or marker absent"
if [[ -f "${RUN_ROOT}/posthoc_eval/support_anchor_split_ode20.json" && -f "${RUN_ROOT}/posthoc_eval/support_candidate_split_ode20.json" ]]; then
  "${PY}" "${ROOT}/ops/evaluate_latentfm_trackc_residual_operator_route_gap_gate_20260623.py" \
    --run-name "${RUN_NAME}" \
    --anchor-json "${RUN_ROOT}/posthoc_eval/support_anchor_split_ode20.json" \
    --candidate-json "${RUN_ROOT}/posthoc_eval/support_candidate_split_ode20.json" \
    --out-json "${ROUTE_GAP_JSON}" \
    --out-md "${ROUTE_GAP_MD}" \
    || true
fi
if [[ -f "${DECISION_MD}" ]]; then
  sed -n '1,120p' "${DECISION_MD}"
elif [[ -f "${ROUTE_GAP_MD}" ]]; then
  sed -n '1,120p' "${ROUTE_GAP_MD}"
elif [[ -f "${DECISION_JSON}" ]]; then
  "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${DECISION_JSON}")
payload = json.loads(p.read_text(encoding="utf-8"))
print(json.dumps(payload.get("decision") or payload.get("status") or payload, indent=2)[:4000])
PY
else
  echo "decision report not available yet"
fi
