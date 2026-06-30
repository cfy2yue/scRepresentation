#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_active_decision_20260621
LOG_DIR=${RUN_ROOT}/logs
SYNTH=${ROOT}/ops/synthesize_latentfm_active_decision_20260621.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

mkdir -p "${LOG_DIR}" "${ROOT}/reports"
rm -f "${RUN_ROOT}/EXIT_CODE" "${RUN_ROOT}/FINISHED"
date '+%F %T %Z' > "${RUN_ROOT}/STARTED"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_active_decision_20260621

## Command

\`\`\`bash
bash ${ROOT}/ops/run_latentfm_active_decision_watcher_20260621.sh
\`\`\`

## Runtime classification

Long CPU watcher. It checks every 30 minutes for active posthoc summaries and
bootstrap indices, then writes a final decision report.

## Start time

$(cat "${RUN_ROOT}/STARTED")

## PID / tmux / scheduler ID

tmux session: \`latentfm_active_decision_20260621\`

## Log path

\`${LOG_DIR}/decision_watcher.log\`

## Expected outputs

* \`${ROOT}/reports/LATENTFM_ACTIVE_POSTHOC_DECISION_20260621.md\`
* \`${ROOT}/reports/latentfm_active_posthoc_decision_20260621.json\`

## How to check manually

\`\`\`bash
tmux ls | grep latentfm_active_decision_20260621 || true
tail -n 50 ${LOG_DIR}/decision_watcher.log
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo "still running"
\`\`\`

## Current status

Started.

## Notes

Does not inspect training logs or GPUs. It only watches for final posthoc and
bootstrap artifacts.
EOF

trap 'rc=$?; echo "$rc" > "${RUN_ROOT}/EXIT_CODE"; date "+%F %T %Z" > "${RUN_ROOT}/FINISHED"; exit "$rc"' EXIT

{
  echo "[$(date '+%F %T %Z')] active decision watcher start"
  while true; do
    "${PYTHON}" "${SYNTH}" \
      --out-json "${ROOT}/reports/latentfm_active_posthoc_decision_20260621.json" \
      --out-md "${ROOT}/reports/LATENTFM_ACTIVE_POSTHOC_DECISION_20260621.md"
    status="$("${PYTHON}" - <<'PY'
import json
from pathlib import Path
p = Path("/data/cyx/1030/scLatent/reports/latentfm_active_posthoc_decision_20260621.json")
obj = json.loads(p.read_text(encoding="utf-8"))
print(obj["decision"]["status"])
PY
)"
    echo "[$(date '+%F %T %Z')] decision status=${status}"
    if [[ "${status}" != "waiting_for_inputs" ]]; then
      echo "[$(date '+%F %T %Z')] active decision complete"
      exit 0
    fi
    echo "[$(date '+%F %T %Z')] waiting for posthoc/bootstrap artifacts; next check in 1800s"
    sleep 1800
  done
} 2>&1 | tee "${LOG_DIR}/decision_watcher.log"
