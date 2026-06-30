#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
NOT_BEFORE_EPOCH=$(date -d '2026-06-23 07:10:00 CST' +%s)
NOW_EPOCH=$(date +%s)

if (( NOW_EPOCH < NOT_BEFORE_EPOCH )); then
  echo "Refusing to check before 2026-06-23 07:10:00 CST" >&2
  exit 3
fi

RUN_ROOT="${ROOT}/runs/latentfm_trackc_anchor_gated_support_teacher_blend_posthoc_20260623/xverse_support_film_retry1_anchor_gated_blend_posthoc_ode20_retry1"
REPORT_MD="${ROOT}/reports/LATENTFM_TRACKC_ANCHOR_GATED_SUPPORT_TEACHER_BLEND_POSTHOC_GATE_20260623.md"
REPORT_JSON="${ROOT}/reports/latentfm_trackc_anchor_gated_support_teacher_blend_posthoc_gate_20260623.json"

if [[ ! -f "${RUN_ROOT}/EXIT_CODE" ]]; then
  echo "still running: ${RUN_ROOT}" >&2
  exit 3
fi

rc=$(cat "${RUN_ROOT}/EXIT_CODE")
if [[ "${rc}" != "0" ]]; then
  echo "blend posthoc failed with exit code ${rc}" >&2
  echo "Inspect: ${RUN_ROOT}/logs/run.log" >&2
  exit 2
fi

if [[ ! -f "${REPORT_MD}" || ! -f "${REPORT_JSON}" ]]; then
  echo "posthoc exit 0 but decision report is missing" >&2
  exit 2
fi

status=$(
  PYTHONPATH="${ROOT}/CoupledFM${PYTHONPATH:+:${PYTHONPATH}}" \
  /data/cyx/software/miniconda3/envs/scdfm/bin/python - <<PY
import json
from pathlib import Path
p = Path("${REPORT_JSON}")
obj = json.loads(p.read_text(encoding="utf-8"))
print(obj.get("status", "missing_status"))
PY
)

echo "exit_code=${rc}"
echo "decision_status=${status}"
echo "report_md=${REPORT_MD}"
