#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_xverse_stage_summary_watcher_20260621
LOG_ROOT=${ROOT}/logs/latentfm_xverse_stage_summary_watcher_20260621
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi
SUMMARY=${ROOT}/ops/summarize_latentfm_xverse_stage_20260621.py
DECISION=${ROOT}/ops/decide_latentfm_xverse_stage_gate_20260621.py
SLEEP_SECONDS=${SLEEP_SECONDS:-1800}

mkdir -p "${RUN_ROOT}/logs" "${LOG_ROOT}"
date '+%F %T %Z' > "${RUN_ROOT}/STARTED"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_xverse_stage_summary_watcher_20260621

## Command

\`\`\`bash
bash ${ROOT}/ops/watch_latentfm_xverse_stage_summary_20260621.sh
\`\`\`

## Runtime classification

Long low-frequency artifact watcher. It checks expected report files every
${SLEEP_SECONDS} seconds and does not poll GPU logs.

## Start time

$(cat "${RUN_ROOT}/STARTED")

## PID / tmux / scheduler ID

tmux session: latentfm_xverse_stage_summary_watcher_20260621

## Log path

${LOG_ROOT}/watcher.log

## Expected outputs

* /data/cyx/1030/scLatent/reports/LATENTFM_XVERSE_STAGE_SUMMARY_20260621.md
* /data/cyx/1030/scLatent/reports/LATENTFM_XVERSE_STAGE_GATE_DECISION_20260621.md

## How to check manually

\`\`\`bash
tmux ls | grep latentfm_xverse_stage_summary_watcher_20260621 || true
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo "still running"
tail -n 50 ${LOG_ROOT}/watcher.log
\`\`\`

## Current status

Started.

## Notes

The watcher re-runs xverse stage summary and gate decision when expected xverse
2k or seed43 artifacts appear. It exits after both 2k uncapped paired bootstrap
and seed43 uncapped CI artifacts are present.
EOF

expected=(
  "${ROOT}/reports/latentfm_xverse_8k_vs_2k_condition_uncapped_bootstrap_split_20260621.json"
  "${ROOT}/reports/latentfm_xverse_8k_vs_2k_condition_uncapped_bootstrap_family_20260621.json"
  "${ROOT}/reports/LATENTFM_XVERSE_8K_SEED43_condition_uncapped_split_ci_20260621.json"
  "${ROOT}/reports/LATENTFM_XVERSE_8K_SEED43_condition_uncapped_family_ci_20260621.json"
)

last_count=-1
echo "[$(date '+%F %T %Z')] stage summary watcher started; sleep_seconds=${SLEEP_SECONDS}"
while true; do
  count=0
  for path in "${expected[@]}"; do
    [[ -f "${path}" ]] && count=$((count + 1))
  done
  if [[ "${count}" != "${last_count}" ]]; then
    echo "[$(date '+%F %T %Z')] observed ${count}/${#expected[@]} expected artifacts; refreshing summary/gate"
    "${PYTHON}" "${SUMMARY}"
    "${PYTHON}" "${DECISION}"
    last_count="${count}"
  else
    echo "[$(date '+%F %T %Z')] observed ${count}/${#expected[@]} expected artifacts; no refresh needed"
  fi
  if [[ "${count}" -eq "${#expected[@]}" ]]; then
    echo "[$(date '+%F %T %Z')] all expected artifacts present; watcher done"
    echo 0 > "${RUN_ROOT}/EXIT_CODE"
    date '+%F %T %Z' > "${RUN_ROOT}/FINISHED"
    exit 0
  fi
  sleep "${SLEEP_SECONDS}"
done
