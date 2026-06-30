#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_NAME=xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42
RUN_ROOT="${ROOT}/runs/latentfm_xverse_trackc_support_context_v2_20260623/${RUN_NAME}"
REPORT_JSON="${ROOT}/reports/latentfm_trackc_routed_distill_smoke_decision_${RUN_NAME}.json"
REPORT_MD="${ROOT}/reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_${RUN_NAME}.md"
PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
WINDOW="2026-06-23 09:33:00 CST"

now_epoch=$(date +%s)
window_epoch=$(date -d "${WINDOW}" +%s)
if (( now_epoch < window_epoch )); then
  echo "Refusing to check before ${WINDOW}; v2 smoke is a GPU long task." >&2
  exit 3
fi

train_exit="${RUN_ROOT}/${RUN_NAME}.EXIT_CODE"
posthoc_exit="${RUN_ROOT}/${RUN_NAME}.POSTHOC_EXIT_CODE"
if [[ ! -f "${train_exit}" ]]; then
  echo "still running: training exit code missing for ${RUN_NAME}" >&2
  exit 3
fi

rc=$(cat "${train_exit}")
if [[ "${rc}" != "0" ]]; then
  echo "training failed with exit code ${rc}" >&2
  echo "Inspect: ${RUN_ROOT}/logs/${RUN_NAME}.train.log" >&2
  exit 2
fi

if [[ ! -f "${posthoc_exit}" ]]; then
  echo "training exit 0 but posthoc watcher has not finished yet" >&2
  exit 3
fi

posthoc_rc=$(cat "${posthoc_exit}")
if [[ "${posthoc_rc}" != "0" ]]; then
  echo "posthoc failed with exit code ${posthoc_rc}" >&2
  echo "Inspect: ${RUN_ROOT}/logs/${RUN_NAME}.posthoc.log" >&2
  exit 2
fi

if [[ ! -f "${REPORT_JSON}" || ! -f "${REPORT_MD}" ]]; then
  echo "posthoc exit 0 but decision report is missing" >&2
  exit 2
fi

"${PYTHON}" - "${RUN_ROOT}" "${REPORT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
report_json = Path(sys.argv[2])
posthoc = run_root / "posthoc_eval"
required = {
    "support_anchor_split": (posthoc / "support_anchor_split_ode20.json", False),
    "support_candidate_split": (posthoc / "support_candidate_split_ode20.json", False),
    "support_anchor_family": (posthoc / "support_anchor_family_ode20.json", False),
    "support_candidate_family": (posthoc / "support_candidate_family_ode20.json", False),
    "canonical_anchor_split": (posthoc / "canonical_anchor_split_ode20_stablecaps.json", True),
    "canonical_candidate_split": (posthoc / "canonical_candidate_split_ode20_stablecaps.json", True),
    "canonical_anchor_family": (posthoc / "canonical_anchor_family_ode20_stablecaps.json", True),
    "canonical_candidate_family": (posthoc / "canonical_candidate_family_ode20_stablecaps.json", True),
}
failures = []
for name, (path, expect_forced_absent) in required.items():
    if not path.exists():
        failures.append(f"missing_{name}")
        continue
    obj = json.loads(path.read_text(encoding="utf-8"))
    forced = bool(obj.get("support_context_forced_absent", False))
    if forced != expect_forced_absent:
        failures.append(f"{name}_support_context_forced_absent_{forced}_expected_{expect_forced_absent}")
    cfg = obj.get("config") or {}
    if expect_forced_absent and cfg.get("trackc_support_context_source") not in {"off", "", None}:
        failures.append(f"{name}_config_support_context_source_not_off")
report = json.loads(report_json.read_text(encoding="utf-8"))
status = report.get("status") or (report.get("decision") or {}).get("status") or "missing_status"
if failures:
    print(json.dumps({"status": status, "provenance_failures": failures}, indent=2))
    raise SystemExit(2)
print(json.dumps({"status": status, "provenance_failures": []}, indent=2))
PY

status=$("${PYTHON}" - "${REPORT_JSON}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("status") or (payload.get("decision") or {}).get("status") or "missing_status")
PY
)

echo "training_exit=${rc}"
echo "posthoc_exit=${posthoc_rc}"
echo "decision_status=${status}"
echo "report_md=${REPORT_MD}"
