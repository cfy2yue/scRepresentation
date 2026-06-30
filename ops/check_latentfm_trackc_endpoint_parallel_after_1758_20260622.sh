#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_endpoint_parallel_20260622
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi
EARLIEST_EPOCH=$(date -d '2026-06-22 17:58:00 CST' +%s)
NOW_EPOCH=$(date +%s)

if (( NOW_EPOCH < EARLIEST_EPOCH )); then
  echo "Refusing endpoint parallel block check before 2026-06-22 17:58:00 CST" >&2
  exit 3
fi

runs=(
  xverse_trackc_endpoint_w025_replay1_2k_seed42
  xverse_trackc_endpoint_w100_replay1_2k_seed42
  xverse_trackc_endpoint_w050_head010_replay1_2k_seed42
  xverse_trackc_endpoint_w050_delayed500_1500_replay1_2k_seed42
)

echo "[$(date '+%F %T %Z')] endpoint parallel block lightweight status check"
for run in "${runs[@]}"; do
  run_dir=${RUN_ROOT}/${run}
  decision_json=${ROOT}/reports/latentfm_trackc_routed_distill_smoke_decision_${run}.json
  decision_md=${ROOT}/reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_${run}.md
  echo "RUN=${run}"
  cat "${run_dir}/${run}.EXIT_CODE" 2>/dev/null || echo "training_still_running_or_exit_missing"
  cat "${run_dir}/${run}.POSTHOC_EXIT_CODE" 2>/dev/null || echo "posthoc_still_running_or_exit_missing"
  if [[ -f "${decision_json}" || -f "${decision_md}" ]]; then
    echo "decision_artifacts_present"
    [[ -f "${decision_json}" ]] && echo "${decision_json}"
    [[ -f "${decision_md}" ]] && echo "${decision_md}"
    if [[ -f "${decision_json}" ]]; then
      "${PYTHON}" - "${decision_json}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
decision = payload.get("decision") or {}
print("decision_status=" + str(decision.get("status", "")))
print("decision_action=" + str(decision.get("action", "")))
PY
    fi
  else
    echo "decision_artifacts_missing"
  fi
done
