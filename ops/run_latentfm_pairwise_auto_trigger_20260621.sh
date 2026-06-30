#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_pairwise_auto_trigger_20260621
LOG_DIR=${RUN_ROOT}/logs
DECISION_JSON=${ROOT}/reports/latentfm_active_posthoc_decision_20260621.json
PAIRWISE_LAUNCHER=${ROOT}/ops/launch_latentfm_pairwise_condition_smoke_20260621.sh
PAIRWISE_OUT=${ROOT}/CoupledFM/output/latentfm_runs/pairwise_condition_20260621/scf_pair_hadamard_prior010_inject_e2_4k
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

mkdir -p "${LOG_DIR}"
rm -f "${RUN_ROOT}/EXIT_CODE" "${RUN_ROOT}/FINISHED"
date '+%F %T %Z' > "${RUN_ROOT}/STARTED"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_pairwise_auto_trigger_20260621

## Command

\`\`\`bash
bash ${ROOT}/ops/run_latentfm_pairwise_auto_trigger_20260621.sh
\`\`\`

## Runtime classification

Long CPU watcher. It checks the active decision JSON every 30 minutes and only
launches pairwise if the decision status is \`launch_pairwise_next\`.

## Start time

$(cat "${RUN_ROOT}/STARTED")

## PID / tmux / scheduler ID

tmux session: \`latentfm_pairwise_auto_trigger_20260621\`

## Log path

\`${LOG_DIR}/pairwise_auto_trigger.log\`

## Expected outputs

If triggered:

* \`${ROOT}/runs/latentfm_pairwise_condition_20260621/RUN_STATUS.md\`
* \`${PAIRWISE_OUT}/best.pt\`

## How to check manually

\`\`\`bash
tmux ls | grep latentfm_pairwise_auto_trigger_20260621 || true
tail -n 50 ${LOG_DIR}/pairwise_auto_trigger.log
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo "still running"
\`\`\`

## Current status

Started.

## Notes

Does not inspect training logs or GPUs. The pairwise launcher performs its own
GPU/RAM audit if this watcher triggers it.
EOF

trap 'rc=$?; echo "$rc" > "${RUN_ROOT}/EXIT_CODE"; date "+%F %T %Z" > "${RUN_ROOT}/FINISHED"; exit "$rc"' EXIT

{
  echo "[$(date '+%F %T %Z')] pairwise auto-trigger watcher start"
  while true; do
    if [[ ! -s "${DECISION_JSON}" ]]; then
      echo "[$(date '+%F %T %Z')] waiting for decision JSON: ${DECISION_JSON}"
      sleep 1800
      continue
    fi
    status="$("${PYTHON}" - "${DECISION_JSON}" <<'PY'
import json
import sys
from pathlib import Path
obj = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(obj.get("decision", {}).get("status", "missing"))
PY
)"
    echo "[$(date '+%F %T %Z')] decision status=${status}"
    case "${status}" in
      waiting_for_inputs)
        echo "[$(date '+%F %T %Z')] waiting for final posthoc decision; next check in 1800s"
        sleep 1800
        ;;
      launch_pairwise_next)
        if [[ -e "${PAIRWISE_OUT}" && "${FORCE_PAIRWISE_AUTO_TRIGGER:-0}" != "1" ]]; then
          echo "[$(date '+%F %T %Z')] pairwise output already exists; not relaunching: ${PAIRWISE_OUT}"
          exit 0
        fi
        echo "[$(date '+%F %T %Z')] triggering pairwise smoke launcher"
        bash "${PAIRWISE_LAUNCHER}"
        echo "[$(date '+%F %T %Z')] pairwise smoke launcher returned"
        exit 0
        ;;
      fewshot_rescue_candidate|response_geometry_candidate)
        echo "[$(date '+%F %T %Z')] pairwise not needed for status=${status}; exiting"
        exit 0
        ;;
      *)
        echo "[$(date '+%F %T %Z')] unknown decision status=${status}; next check in 1800s"
        sleep 1800
        ;;
    esac
  done
} 2>&1 | tee "${LOG_DIR}/pairwise_auto_trigger.log"
