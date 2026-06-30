#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_priorcorr_scf_inject_eval_20260620
LOG=${RUN_ROOT}/logs/run.log
STAMP=${RUN_ROOT}/LAST_ONE_SHOT_CHECK_EPOCH
NOW=$(date +%s)
MIN_INTERVAL=${MIN_INTERVAL_SECONDS:-1800}

if [[ -f "${STAMP}" ]]; then
  LAST=$(cat "${STAMP}")
  if [[ $((NOW - LAST)) -lt ${MIN_INTERVAL} ]]; then
    echo "Too early for another 30-minute cadence check."
    echo "now=$(date '+%F %T %Z')"
    echo "next_allowed=$(date -d "@$((LAST + MIN_INTERVAL))" '+%F %T %Z')"
    exit 11
  fi
fi
echo "${NOW}" > "${STAMP}"

REPORT=${ROOT}/reports/LATENTFM_PRIORCORR_SCF_INJECT_EVAL_ONE_SHOT_STATUS_$(date '+%Y%m%d_%H%M%S').txt
{
  echo "[$(date '+%F %T %Z')] one-shot prior-correction scFoundation inject eval status"
  echo "report=${REPORT}"
  echo
  tmux ls 2>/dev/null | grep latentfm_priorcorr_scf_inject_eval_20260620 || true
  echo
  if [[ -f "${RUN_ROOT}/EXIT_CODE" ]]; then
    echo "EXIT_CODE=$(cat "${RUN_ROOT}/EXIT_CODE")"
  else
    echo "still-running-or-not-started"
  fi
  if [[ -f "${RUN_ROOT}/FINISHED" ]]; then
    echo "FINISHED=$(cat "${RUN_ROOT}/FINISHED")"
  fi
  echo
  for p in \
    "${ROOT}/reports/LATENTFM_PRIOR_CORRECTION_EVAL_SCF_INJECT_20260620.md" \
    "${ROOT}/reports/latentfm_prior_correction_eval_scf_inject_20260620.csv" \
    "${ROOT}/reports/latentfm_prior_correction_eval_scf_inject_20260620.json"; do
    if [[ -f "${p}" ]]; then
      echo "present ${p}"
    else
      echo "missing ${p}"
    fi
  done
  echo
  echo "tail ${LOG}"
  tail -n 80 "${LOG}" 2>/dev/null || true
} | tee "${REPORT}"
