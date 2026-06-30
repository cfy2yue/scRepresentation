#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
WINDOW_EPOCH=$(TZ=Asia/Shanghai date -d '2026-06-23 11:03:00' +%s)
NOW_EPOCH=$(TZ=Asia/Shanghai date +%s)
if (( NOW_EPOCH < WINDOW_EPOCH )); then
  echo "Refusing to check before 2026-06-23 11:03:00 CST; v2 query eval is a long GPU task." >&2
  exit 3
fi

RUN_NAME=xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42
LABEL=latentfm_trackc_support_context_v2_query_once_${RUN_NAME}_20260623
RUN_ROOT=${ROOT}/runs/${LABEL}
DECISION_JSON=${ROOT}/reports/latentfm_trackc_support_context_v2_query_once_decision_${RUN_NAME}_20260623.json
DECISION_MD=${ROOT}/reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_QUERY_ONCE_DECISION_${RUN_NAME}_20260623.md
EXIT_CODE=${RUN_ROOT}/EXIT_CODE

if [[ ! -f "${EXIT_CODE}" ]]; then
  echo "query eval still running or exit code missing: ${RUN_ROOT}" >&2
  exit 3
fi

rc=$(cat "${EXIT_CODE}")
if [[ "${rc}" != "0" ]]; then
  echo "query eval failed with exit code ${rc}" >&2
  echo "Inspect: ${RUN_ROOT}/logs/query_eval.log" >&2
  exit 2
fi

if [[ ! -f "${DECISION_JSON}" || ! -f "${DECISION_MD}" ]]; then
  echo "query exit 0 but decision report is missing" >&2
  exit 2
fi

PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

status=$("${PYTHON}" - "${DECISION_JSON}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("status") or (payload.get("decision") or {}).get("status") or "missing_status")
PY
)

echo "query_exit=${rc}"
echo "decision_status=${status}"
echo "decision_md=${DECISION_MD}"
