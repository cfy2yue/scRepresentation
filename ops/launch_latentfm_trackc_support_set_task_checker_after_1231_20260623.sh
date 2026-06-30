#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
RUN_NAME="latentfm_trackc_support_set_task_checker_after_1231_20260623"
RUN_DIR="${ROOT}/runs/${RUN_NAME}"
SESSION="lfm_support_set_task_check_1231_20260623"
CHECKER="${ROOT}/ops/check_latentfm_trackc_support_set_task_inputs_after_1231_20260623.sh"
CUTOFF_EPOCH="$(date -d '2026-06-23 12:31:00 +0800' +%s)"
NOW_EPOCH="$(date +%s)"
DELAY=$(( CUTOFF_EPOCH - NOW_EPOCH ))
if (( DELAY < 0 )); then
  DELAY=0
fi

mkdir -p "${RUN_DIR}/logs"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[support-set-task-checker-launch] session already exists: ${SESSION}"
  exit 2
fi

COMMAND="sleep ${DELAY}; ${CHECKER}"

cat > "${RUN_DIR}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_NAME}

## Command

\`\`\`bash
${COMMAND}
\`\`\`

## Runtime classification

Long task. Detached delayed checker; it sleeps until the 30-minute long-job polling window, then runs one guarded check.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux session: \`${SESSION}\`

## Log path

\`${RUN_DIR}/logs/run.log\`

## Expected outputs

* \`${ROOT}/reports/LATENTFM_TRACKC_SUPPORT_SET_TASK_SUMMARY_GATE_20260623.md\` if the input artifact job exited 0 and summary gate runs
* \`${ROOT}/reports/latentfm_trackc_support_set_task_summary_gate_20260623.json\` if the input artifact job exited 0 and summary gate runs
* \`${RUN_DIR}/EXIT_CODE\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${RUN_DIR}/logs/run.log
cat ${RUN_DIR}/EXIT_CODE 2>/dev/null || echo "still running"
\`\`\`

## Current status

Started.

## Notes

This launcher does not inspect the input artifact RUN_STATUS, log, EXIT_CODE, or output directory before the allowed 2026-06-23 12:31 CST window. It only schedules the guarded checker.
EOF

date > "${RUN_DIR}/STARTED"
echo "${SESSION}" > "${RUN_DIR}/SESSION_NAME"

tmux new -d -s "${SESSION}" \
  "set -o pipefail; echo '[support-set-task-checker] start ' \$(date '+%F %T %Z'); echo '[support-set-task-checker] delay_seconds=${DELAY}'; ${COMMAND} > >(tee -a '${RUN_DIR}/logs/run.log') 2> >(tee -a '${RUN_DIR}/logs/run.log' >&2); rc=\$?; echo \$rc > '${RUN_DIR}/EXIT_CODE'; date > '${RUN_DIR}/FINISHED'; if [[ \$rc -eq 0 ]]; then sed -i 's/^Started\\.$/Finished./' '${RUN_DIR}/RUN_STATUS.md'; else sed -i 's/^Started\\.$/Failed./' '${RUN_DIR}/RUN_STATUS.md'; fi; exit \$rc"

echo "[support-set-task-checker-launch] launched ${SESSION}"
echo "[support-set-task-checker-launch] run dir ${RUN_DIR}"
