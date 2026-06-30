#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PY=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PY}" ]]; then
  PY=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi
RUN_NAME=xverse_trackc_support_film_absroute_2k_seed42_retry1
RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_support_film_20260623/${RUN_NAME}
POSTHOC_SESSION=trackc_route_posthoc_${RUN_NAME}
DECISION_MD=${ROOT}/reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_${RUN_NAME}.md
DECISION_JSON=${ROOT}/reports/latentfm_trackc_routed_distill_smoke_decision_${RUN_NAME}.json
ROUTE_GAP_JSON=${ROOT}/reports/latentfm_trackc_support_film_route_gap_gate_${RUN_NAME}.json
ROUTE_GAP_MD=${ROOT}/reports/LATENTFM_TRACKC_SUPPORT_FILM_ROUTE_GAP_GATE_${RUN_NAME}.md
CPU_GATE_JSON=${ROOT}/reports/latentfm_trackc_alternative_support_conditioning_cpu_gate_20260623.json
WINDOW="2026-06-23 03:54:00"

now_epoch=$(date +%s)
window_epoch=$(date -d "${WINDOW}" +%s)
if (( now_epoch < window_epoch )); then
  echo "Refusing to check before ${WINDOW} CST; support-FiLM retry1 is a long GPU task." >&2
  exit 3
fi

echo "[$(date '+%F %T %Z')] Track C support-FiLM retry1 guarded check"
echo "run_root=${RUN_ROOT}"
cat "${RUN_ROOT}/${RUN_NAME}.EXIT_CODE" 2>/dev/null || echo "train still running or marker absent"
cat "${RUN_ROOT}/${RUN_NAME}.POSTHOC_EXIT_CODE" 2>/dev/null || echo "posthoc still running or marker absent"

if [[ -f "${RUN_ROOT}/posthoc_eval/support_anchor_split_ode20.json" && -f "${RUN_ROOT}/posthoc_eval/support_candidate_split_ode20.json" ]]; then
  "${PY}" "${ROOT}/ops/evaluate_latentfm_trackc_support_film_route_gap_gate_20260623.py" \
    --run-name "${RUN_NAME}" \
    --anchor-json "${RUN_ROOT}/posthoc_eval/support_anchor_split_ode20.json" \
    --candidate-json "${RUN_ROOT}/posthoc_eval/support_candidate_split_ode20.json" \
    --cpu-gate-json "${CPU_GATE_JSON}" \
    --out-json "${ROUTE_GAP_JSON}" \
    --out-md "${ROUTE_GAP_MD}" \
    || true
fi

if [[ -f "${ROUTE_GAP_JSON}" ]]; then
  status="$("${PY}" - <<PY
import json
from pathlib import Path
p = Path("${ROUTE_GAP_JSON}")
print((json.loads(p.read_text(encoding="utf-8")).get("decision") or {}).get("status", "unknown"))
PY
)"
  if [[ "${status}" != "support_film_route_gap_gate_pass" ]]; then
    if tmux has-session -t "${POSTHOC_SESSION}" 2>/dev/null; then
      tmux kill-session -t "${POSTHOC_SESSION}" || true
      date '+%F %T %Z' > "${RUN_ROOT}/POSTHOC_STOPPED_AFTER_SUPPORT_FILM_ROUTE_GAP_FAIL"
      echo "Stopped ${POSTHOC_SESSION} after route-gap fail status=${status}"
    fi
  fi
fi

if [[ -f "${DECISION_MD}" ]]; then
  sed -n '1,140p' "${DECISION_MD}"
elif [[ -f "${ROUTE_GAP_MD}" ]]; then
  sed -n '1,140p' "${ROUTE_GAP_MD}"
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
