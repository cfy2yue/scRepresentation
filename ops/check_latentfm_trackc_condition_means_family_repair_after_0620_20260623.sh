#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT="${ROOT}/runs/latentfm_trackc_anchor_gated_support_teacher_artifacts_20260623/xverse_support_film_retry1_condition_means_family_repair"
NOT_BEFORE_EPOCH=1782166800  # 2026-06-23 06:20:00 CST
NOW_EPOCH=$(date +%s)

if (( NOW_EPOCH < NOT_BEFORE_EPOCH )); then
  echo "Refusing to check before 2026-06-23 06:20:00 CST" >&2
  exit 3
fi

echo "[$(date '+%F %T %Z')] Track C condition-means family repair check"
echo "run_root=${RUN_ROOT}"

EXIT_CODE=$(cat "${RUN_ROOT}/EXIT_CODE" 2>/dev/null || true)
if [[ -z "${EXIT_CODE}" ]]; then
  echo "repair job still running or EXIT_CODE absent"
  {
    echo
    echo "Guarded check at $(date '+%F %T %Z'): repair job still running or EXIT_CODE absent."
  } >> "${RUN_ROOT}/RUN_STATUS.md"
  exit 0
fi
echo "${EXIT_CODE}"

if [[ "${EXIT_CODE}" != "0" ]]; then
  echo "repair job failed with exit ${EXIT_CODE}" >&2
  tail -n 80 "${RUN_ROOT}/logs/run.log" >&2 || true
  {
    echo
    echo "Guarded check at $(date '+%F %T %Z'): repair job failed with exit ${EXIT_CODE}."
  } >> "${RUN_ROOT}/RUN_STATUS.md"
  exit 1
fi

REPORT="${ROOT}/reports/LATENTFM_TRACKC_ANCHOR_GATED_SUPPORT_TEACHER_CPU_GATE_20260623.md"
if [[ ! -f "${REPORT}" ]]; then
  echo "repair exit 0 but CPU gate report missing: ${REPORT}" >&2
  exit 1
fi

sed -n '1,120p' "${REPORT}"
{
  echo
  echo "Guarded check at $(date '+%F %T %Z'): repair job exit \`0\`; CPU gate report present."
  echo
  echo "Decision report:"
  echo "\`${REPORT}\`"
} >> "${RUN_ROOT}/RUN_STATUS.md"
