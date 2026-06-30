#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_NAME=xverse_trackc_endpoint_route_w05_replay1_2k_seed42
RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_endpoint_routed_20260622/${RUN_NAME}
RUN_STATUS=${RUN_ROOT}/RUN_STATUS.md
DECISION_MD=${ROOT}/reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_${RUN_NAME}.md
DECISION_JSON=${ROOT}/reports/latentfm_trackc_routed_distill_smoke_decision_${RUN_NAME}.json
EARLIEST_EPOCH=$(date -d '2026-06-22 17:10:00 CST' +%s)
NOW_EPOCH=$(date +%s)

if (( NOW_EPOCH < EARLIEST_EPOCH )); then
  echo "Refusing endpoint-routed long-job check before 2026-06-22 17:10:00 CST" >&2
  exit 3
fi

if [[ ! -f "${RUN_STATUS}" ]]; then
  echo "Missing RUN_STATUS: ${RUN_STATUS}" >&2
  exit 2
fi

echo "[$(date '+%F %T %Z')] endpoint-routed lightweight status check"
echo "RUN_STATUS=${RUN_STATUS}"

echo "tmux sessions:"
tmux ls 2>/dev/null | grep -E "trackc_route_(train|posthoc)_${RUN_NAME}" || true

echo "exit codes:"
cat "${RUN_ROOT}/${RUN_NAME}.EXIT_CODE" 2>/dev/null || echo "training_still_running_or_exit_missing"
cat "${RUN_ROOT}/${RUN_NAME}.POSTHOC_EXIT_CODE" 2>/dev/null || echo "posthoc_still_running_or_exit_missing"

if [[ -f "${DECISION_JSON}" || -f "${DECISION_MD}" ]]; then
  echo "decision_artifacts:"
  [[ -f "${DECISION_JSON}" ]] && echo "${DECISION_JSON}"
  [[ -f "${DECISION_MD}" ]] && echo "${DECISION_MD}"
  if [[ -f "${DECISION_JSON}" ]]; then
    "${ROOT}/software/miniconda3/envs/scdfm/bin/python" - "${DECISION_JSON}" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
print("decision_status=" + str((payload.get("decision") or {}).get("status", "")))
print("decision_action=" + str((payload.get("decision") or {}).get("action", "")))
PY
  fi
else
  echo "decision_artifacts_missing"
fi
