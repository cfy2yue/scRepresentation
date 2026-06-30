#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
WATCH_RUN_ROOT=${ROOT}/runs/latentfm_xverse_8k_seed43_posthoc_watcher_20260621
WATCH_LOG_ROOT=${ROOT}/logs/latentfm_xverse_8k_seed43_posthoc_watcher_20260621
TRAIN_RUN_ROOT=${ROOT}/runs/latentfm_xverse_8k_seed_replicate_20260621
TRAIN_RUN_NAME=xverse_comp006_endpoint5_8k_seed43_fulleval
TRAIN_EXIT=${TRAIN_RUN_ROOT}/${TRAIN_RUN_NAME}.EXIT_CODE
RUN_DIR=${ROOT}/CoupledFM/output/latentfm_runs/xverse_8k_seed_replicate_20260621/${TRAIN_RUN_NAME}
POSTHOC_RUN_ROOT=${ROOT}/runs/latentfm_xverse_8k_seed43_uncapped_posthoc_20260621
POSTHOC_LOG_ROOT=${ROOT}/logs/latentfm_xverse_8k_seed43_uncapped_posthoc_20260621
POSTHOC_SESSION=latentfm_xverse_8k_seed43_uncapped_posthoc_20260621
LAUNCHER=${ROOT}/ops/launch_latentfm_xverse_single_uncapped_posthoc_20260621.sh
SLEEP_SECONDS=${SLEEP_SECONDS:-1800}

mkdir -p "${WATCH_RUN_ROOT}/logs" "${WATCH_LOG_ROOT}"

date '+%F %T %Z' > "${WATCH_RUN_ROOT}/STARTED"

cat > "${WATCH_RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_xverse_8k_seed43_posthoc_watcher_20260621

## Command

\`\`\`bash
bash ${ROOT}/ops/watch_latentfm_xverse_seed43_posthoc_20260621.sh
\`\`\`

## Runtime classification

Long low-frequency watcher. It checks marker files every ${SLEEP_SECONDS}
seconds and does not tail training logs.

## Start time

$(cat "${WATCH_RUN_ROOT}/STARTED")

## PID / tmux / scheduler ID

tmux session: latentfm_xverse_8k_seed43_posthoc_watcher_20260621

## Log path

${WATCH_LOG_ROOT}/watcher.log

## Expected outputs

* ${POSTHOC_RUN_ROOT}/RUN_STATUS.md
* /data/cyx/1030/scLatent/reports/LATENTFM_XVERSE_8K_SEED43_CONDITION_UNCAPPED_SPLIT_CI_20260621.md
* /data/cyx/1030/scLatent/reports/LATENTFM_XVERSE_8K_SEED43_CONDITION_UNCAPPED_FAMILY_CI_20260621.md

## How to check manually

\`\`\`bash
tmux ls | grep latentfm_xverse_8k_seed43_posthoc_watcher_20260621 || true
cat ${WATCH_RUN_ROOT}/EXIT_CODE 2>/dev/null || echo "still running"
tail -n 50 ${WATCH_LOG_ROOT}/watcher.log
\`\`\`

## Current status

Started.

## Notes

The watcher launches seed43 condition-uncapped posthoc only after the seed43
training EXIT_CODE is present and zero. The posthoc launcher performs its own
GPU/CPU/RAM resource gate before starting.
EOF

echo "[$(date '+%F %T %Z')] watcher started; sleep_seconds=${SLEEP_SECONDS}"
while true; do
  if [[ -f "${POSTHOC_RUN_ROOT}/EXIT_CODE" ]]; then
    echo "[$(date '+%F %T %Z')] posthoc already has EXIT_CODE; watcher done"
    echo 0 > "${WATCH_RUN_ROOT}/EXIT_CODE"
    date '+%F %T %Z' > "${WATCH_RUN_ROOT}/FINISHED"
    exit 0
  fi
  if tmux has-session -t "${POSTHOC_SESSION}" 2>/dev/null; then
    echo "[$(date '+%F %T %Z')] posthoc session already running; watcher done"
    echo 0 > "${WATCH_RUN_ROOT}/EXIT_CODE"
    date '+%F %T %Z' > "${WATCH_RUN_ROOT}/FINISHED"
    exit 0
  fi
  if [[ -f "${TRAIN_EXIT}" ]]; then
    rc="$(cat "${TRAIN_EXIT}")"
    if [[ "${rc}" != "0" ]]; then
      echo "[$(date '+%F %T %Z')] seed43 training failed with exit ${rc}; not launching posthoc"
      echo 10 > "${WATCH_RUN_ROOT}/EXIT_CODE"
      date '+%F %T %Z' > "${WATCH_RUN_ROOT}/FINISHED"
      exit 10
    fi
    if [[ ! -f "${RUN_DIR}/best.pt" ]]; then
      echo "[$(date '+%F %T %Z')] seed43 exit=0 but missing best.pt: ${RUN_DIR}/best.pt"
      echo 11 > "${WATCH_RUN_ROOT}/EXIT_CODE"
      date '+%F %T %Z' > "${WATCH_RUN_ROOT}/FINISHED"
      exit 11
    fi
    echo "[$(date '+%F %T %Z')] seed43 training complete; launching uncapped posthoc"
    RUN_DIR="${RUN_DIR}" \
      RUN_LABEL="${TRAIN_RUN_NAME}" \
      REPORT_LABEL=LATENTFM_XVERSE_8K_SEED43 \
      REPORT_TITLE="LatentFM xverse 8k seed43 Condition-Uncapped" \
      RUN_ROOT="${POSTHOC_RUN_ROOT}" \
      LOG_ROOT="${POSTHOC_LOG_ROOT}" \
      SESSION="${POSTHOC_SESSION}" \
      bash "${LAUNCHER}"
    echo 0 > "${WATCH_RUN_ROOT}/EXIT_CODE"
    date '+%F %T %Z' > "${WATCH_RUN_ROOT}/FINISHED"
    exit 0
  fi
  echo "[$(date '+%F %T %Z')] seed43 training still running; next marker check in ${SLEEP_SECONDS}s"
  sleep "${SLEEP_SECONDS}"
done
