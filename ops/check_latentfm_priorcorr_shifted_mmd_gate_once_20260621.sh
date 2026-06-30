#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_priorcorr_shifted_mmd_gate_20260621
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

REPORT=${ROOT}/reports/LATENTFM_PRIORCORR_SHIFTED_MMD_GATE_ONE_SHOT_STATUS_$(date '+%Y%m%d_%H%M%S').txt
{
  echo "[$(date '+%F %T %Z')] one-shot prior-correction shifted-MMD gate status"
  echo "report=${REPORT}"
  echo
  tmux ls 2>/dev/null | grep latentfm_priorcorr_shifted_mmd_gate_20260621 || true
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
    "${ROOT}/reports/LATENTFM_PRIOR_CORRECTION_SHIFTED_MMD_GATE_SCF_INJECT_20260621.md" \
    "${ROOT}/reports/latentfm_prior_correction_shifted_mmd_gate_scf_inject_20260621.csv" \
    "${ROOT}/reports/latentfm_prior_correction_shifted_mmd_gate_scf_inject_20260621.json"; do
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
