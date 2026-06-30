#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_NAME=xverse_trackc_route_condprior_w05_replay1_2k_seed42
TRACKC_RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_routed_distill_20260622/${RUN_NAME}
CHECK_RUN_ROOT=${ROOT}/runs/latentfm_trackc_routed_distill_1800s_check_20260622
WAIT_SECONDS=${WAIT_SECONDS:-1800}
REPORT=${ROOT}/reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_${RUN_NAME}.md
TRAIN_LOG=${TRACKC_RUN_ROOT}/logs/${RUN_NAME}.train.log
POSTHOC_LOG=${TRACKC_RUN_ROOT}/logs/${RUN_NAME}.posthoc.log

mkdir -p "${CHECK_RUN_ROOT}/logs"
echo "[check] scheduled at $(date '+%F %T %Z')"
echo "[check] sleeping ${WAIT_SECONDS}s before reading Track C status"
sleep "${WAIT_SECONDS}"
echo "[check] woke at $(date '+%F %T %Z')"

echo
echo "## tmux"
tmux ls 2>&1 | grep -E 'trackc_route|1800s' || true

echo
echo "## exit codes"
printf 'train_exit_code='
cat "${TRACKC_RUN_ROOT}/${RUN_NAME}.EXIT_CODE" 2>/dev/null || echo "still_running"
printf 'posthoc_exit_code='
cat "${TRACKC_RUN_ROOT}/${RUN_NAME}.POSTHOC_EXIT_CODE" 2>/dev/null || echo "posthoc_not_finished"

echo
echo "## expected files"
for path in \
  "${TRACKC_RUN_ROOT}/posthoc.FINISHED" \
  "${TRACKC_RUN_ROOT}/posthoc_eval/support_candidate_split_ode20.json" \
  "${TRACKC_RUN_ROOT}/posthoc_eval/canonical_candidate_split_ode20_stablecaps.json" \
  "${REPORT}"; do
  if [[ -e "${path}" ]]; then
    ls -lh "${path}"
  else
    echo "missing ${path}"
  fi
done

echo
echo "## decision report head"
if [[ -f "${REPORT}" ]]; then
  sed -n '1,120p' "${REPORT}"
else
  echo "decision report not available yet"
fi

echo
echo "## train log tail"
tail -n 120 "${TRAIN_LOG}" 2>/dev/null || true

echo
echo "## posthoc log tail"
tail -n 120 "${POSTHOC_LOG}" 2>/dev/null || true

echo
echo "## gpu snapshot"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv || true

date > "${CHECK_RUN_ROOT}/FINISHED"
