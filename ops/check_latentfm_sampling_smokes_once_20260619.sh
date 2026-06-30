#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_sampling_smokes_after_metric_gate_20260619
POSTHOC_ROOT=${ROOT}/runs/latentfm_sampling_smokes_posthoc_20260619
NOT_BEFORE=${NOT_BEFORE:-"2026-06-19 19:37:00"}
TZ_NAME=${TZ_NAME:-"Asia/Shanghai"}
REPORT=${REPORT:-"${ROOT}/reports/LATENTFM_SAMPLING_SMOKES_ONE_SHOT_STATUS_$(date +%Y%m%d_%H%M%S).txt"}
MIN_INTERVAL_SECONDS=${MIN_INTERVAL_SECONDS:-1800}
STAMP=${STAMP:-"${RUN_ROOT}/LAST_ONE_SHOT_CHECK_EPOCH"}

if [[ "${CHECK_LOGGING_ACTIVE:-0}" != "1" ]]; then
  mkdir -p "$(dirname "${REPORT}")"
  export CHECK_LOGGING_ACTIVE=1
  exec > >(tee "${REPORT}") 2>&1
fi

now_epoch="$(TZ=${TZ_NAME} date +%s)"
gate_epoch="$(TZ=${TZ_NAME} date -d "${NOT_BEFORE}" +%s)"

if (( now_epoch < gate_epoch )); then
  echo "Too early for the 30-minute cadence check."
  echo "now=$(TZ=${TZ_NAME} date '+%F %T %Z')"
  echo "not_before=${NOT_BEFORE} ${TZ_NAME}"
  echo "No training logs, EXIT_CODE files, or tmux process states were checked."
  exit 10
fi

if [[ -f "${STAMP}" && "${FORCE_ONE_SHOT_CHECK:-0}" != "1" ]]; then
  last_epoch="$(cat "${STAMP}")"
  if [[ "${last_epoch}" =~ ^[0-9]+$ ]]; then
    next_epoch=$((last_epoch + MIN_INTERVAL_SECONDS))
    if (( now_epoch < next_epoch )); then
      echo "Too early for another 30-minute cadence check."
      echo "now=$(TZ=${TZ_NAME} date '+%F %T %Z')"
      echo "last_check_epoch=${last_epoch}"
      echo "next_allowed=$(TZ=${TZ_NAME} date -d "@${next_epoch}" '+%F %T %Z')"
      echo "No training logs, EXIT_CODE files, or tmux process states were checked."
      exit 11
    fi
  fi
fi

echo "[$(date '+%F %T %Z')] one-shot LatentFM sampling-smoke status check"
echo "report=${REPORT}"
echo
echo "RUN_ROOT=${RUN_ROOT}"
echo "POSTHOC_ROOT=${POSTHOC_ROOT}"
echo

echo "tmux sessions:"
tmux ls | grep -E 'lfm_scf_prior010_inject|latentfm_sampling_smokes_(wait1800_once|posthoc_20260619)' || true
echo

for run in \
  scf_prior010_inject_visitcap8_power05_floor32_4k \
  scf_prior010_inject_visitcap8_power05_floor32_dsloss05_4k
do
  echo "### ${run}"
  if [[ -f "${RUN_ROOT}/${run}.EXIT_CODE" ]]; then
    echo "EXIT_CODE=$(cat "${RUN_ROOT}/${run}.EXIT_CODE")"
    [[ -f "${RUN_ROOT}/${run}.FINISHED" ]] && echo "FINISHED=$(cat "${RUN_ROOT}/${run}.FINISHED")"
  else
    echo "still-running"
  fi
done

echo "${now_epoch}" > "${STAMP}"
echo

echo "### posthoc watcher"
if [[ -f "${POSTHOC_ROOT}/EXIT_CODE" ]]; then
  echo "EXIT_CODE=$(cat "${POSTHOC_ROOT}/EXIT_CODE")"
  [[ -f "${POSTHOC_ROOT}/FINISHED" ]] && echo "FINISHED=$(cat "${POSTHOC_ROOT}/FINISHED")"
else
  echo "still-running-or-waiting"
fi
if [[ -f "${POSTHOC_ROOT}/logs/run.log" ]]; then
  echo "posthoc log head/tail:"
  sed -n '1,5p' "${POSTHOC_ROOT}/logs/run.log"
  tail -n 10 "${POSTHOC_ROOT}/logs/run.log"
fi
echo

echo "Expected summary artifacts:"
for p in \
  "${ROOT}/reports/latentfm_sampling_smokes_stablecaps_summary_20260619.csv" \
  "${ROOT}/reports/latentfm_sampling_smokes_stablecaps_summary_20260619_per_dataset.csv" \
  "${ROOT}/reports/latentfm_sampling_smokes_stablecaps_summary_20260619_gate.json" \
  "${ROOT}/reports/LATENTFM_SAMPLING_SMOKES_STABLECAPS_SUMMARY_20260619.md"
do
  if [[ -f "${p}" ]]; then
    echo "present ${p}"
  else
    echo "missing ${p}"
  fi
done
