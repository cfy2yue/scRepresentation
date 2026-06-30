#!/usr/bin/env bash
set -u -o pipefail

ROOT="/data/cyx/1030/scLatent"
TARGET_TIME="2026-06-27 03:10:00"
RUN_ROOT="${ROOT}/runs/latentfm_sciplex_dose_specific_anchor_posthoc_20260627/xverse_anchor_sciplex_logdose_doseeval_seed42"
STATUS_FILE="${RUN_ROOT}/RUN_STATUS.md"
EXIT_CODE_FILE="${RUN_ROOT}/EXIT_CODE"
EVAL_JSON="${RUN_ROOT}/condition_family_eval_anchor_sciplex_doseeval_ode20.json"
GATE_SCRIPT="${ROOT}/ops/build_latentfm_sciplex_dose_specific_outcome_gate_20260627.py"
GATE_JSON="${ROOT}/reports/latentfm_sciplex_dose_specific_outcome_gate_20260627.json"
GATE_MD="${ROOT}/reports/LATENTFM_SCIPLEX_DOSE_SPECIFIC_OUTCOME_GATE_20260627.md"

now_epoch="$(date +%s)"
target_epoch="$(date -d "${TARGET_TIME}" +%s)"
if (( now_epoch < target_epoch )); then
  echo "[check] refusing before ${TARGET_TIME}; now=$(date '+%F %T %Z')"
  exit 3
fi

append_status() {
  {
    echo
    echo "## Automated check update: $(date '+%F %T %Z')"
    echo
    echo "$1"
  } >> "${STATUS_FILE}"
}

echo "[check] start=$(date '+%F %T %Z')"
echo "[check] run=${RUN_ROOT}"

if [[ ! -f "${EXIT_CODE_FILE}" ]]; then
  append_status "Posthoc has not written EXIT_CODE at the scheduled 30-minute check. Status remains running; do not poll again immediately."
  echo "[check] EXIT_CODE missing; leaving job as running"
  exit 4
fi

posthoc_rc="$(tr -d '[:space:]' < "${EXIT_CODE_FILE}")"
echo "[check] posthoc_exit=${posthoc_rc}"
if [[ "${posthoc_rc}" != "0" ]]; then
  append_status "Posthoc wrote nonzero EXIT_CODE=${posthoc_rc}. Treat as implementation/provenance failure until log tail is inspected once."
  exit 1
fi

if [[ ! -s "${EVAL_JSON}" ]]; then
  append_status "Posthoc EXIT_CODE=0 but expected eval JSON is missing or empty: \`${EVAL_JSON}\`."
  echo "[check] missing eval json ${EVAL_JSON}"
  exit 2
fi

python "${GATE_SCRIPT}" --eval-json "${EVAL_JSON}"
gate_rc="$?"
if [[ "${gate_rc}" != "0" ]]; then
  append_status "CPU dose-specific gate script failed with exit code ${gate_rc}; expected report path: \`${GATE_MD}\`."
  exit "${gate_rc}"
fi

gate_status="$(
  python - <<'PY'
import json
from pathlib import Path
path = Path("/data/cyx/1030/scLatent/reports/latentfm_sciplex_dose_specific_outcome_gate_20260627.json")
payload = json.loads(path.read_text(encoding="utf-8"))
print(payload.get("status", "unknown"))
gate = payload.get("gate") or {}
print(gate.get("within_drug_pairs", "NA"))
print((gate.get("pp_high_minus_low_bootstrap") or {}).get("mean", "NA"))
print(gate.get("reasons", []))
PY
)"
status_line="$(printf '%s\n' "${gate_status}" | sed -n '1p')"
pairs_line="$(printf '%s\n' "${gate_status}" | sed -n '2p')"
pp_line="$(printf '%s\n' "${gate_status}" | sed -n '3p')"
reasons_line="$(printf '%s\n' "${gate_status}" | sed -n '4p')"

append_status "Posthoc EXIT_CODE=0 and dose-specific CPU gate completed. Gate status: \`${status_line}\`; within-drug pairs: \`${pairs_line}\`; pp high-low mean: \`${pp_line}\`; reasons: \`${reasons_line}\`. Report: \`${GATE_MD}\`."

echo "[check] gate_status=${status_line}"
echo "[check] gate_report=${GATE_MD}"
exit 0
