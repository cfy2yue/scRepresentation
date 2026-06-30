#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
NOT_BEFORE_EPOCH=$(date -d '2026-06-23 07:41:00 CST' +%s)
NOW_EPOCH=$(date +%s)

if (( NOW_EPOCH < NOT_BEFORE_EPOCH )); then
  echo "Refusing to check before 2026-06-23 07:41:00 CST" >&2
  exit 3
fi

RUN_ROOT="${ROOT}/runs/latentfm_trackc_anchor_gated_blend_query_once_20260623_retry1"
REPORT_MD="${ROOT}/reports/LATENTFM_TRACKC_ANCHOR_GATED_BLEND_QUERY_ONCE_DECISION_20260623.md"
REPORT_JSON="${ROOT}/reports/latentfm_trackc_anchor_gated_blend_query_once_decision_20260623.json"

if [[ ! -f "${RUN_ROOT}/EXIT_CODE" ]]; then
  echo "still running: ${RUN_ROOT}" >&2
  exit 3
fi

rc=$(cat "${RUN_ROOT}/EXIT_CODE")
if [[ "${rc}" != "0" ]]; then
  echo "query eval failed with exit code ${rc}" >&2
  echo "Inspect: ${RUN_ROOT}/logs/query_eval.log" >&2
  exit 2
fi

if [[ ! -f "${REPORT_MD}" || ! -f "${REPORT_JSON}" ]]; then
  echo "query exit 0 but decision report is missing" >&2
  exit 2
fi

status=$(
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
