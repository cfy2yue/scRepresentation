#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT="${ROOT}/runs/latentfm_trackc_anchor_gated_support_teacher_artifacts_20260623/xverse_support_film_retry1_condition_means_artifacts"
NOT_BEFORE_EPOCH=1782164880  # 2026-06-23 05:48:00 CST
NOW_EPOCH=$(date +%s)

if (( NOW_EPOCH < NOT_BEFORE_EPOCH )); then
  echo "Refusing to check before 2026-06-23 05:48:00 CST" >&2
  exit 3
fi

echo "[$(date '+%F %T %Z')] Track C condition-means artifact check"
echo "run_root=${RUN_ROOT}"

EXIT_CODE=$(cat "${RUN_ROOT}/EXIT_CODE" 2>/dev/null || true)
if [[ -z "${EXIT_CODE}" ]]; then
  echo "artifact job still running or EXIT_CODE absent"
  {
    echo
    echo "Guarded check at $(date '+%F %T %Z'): artifact job still running or EXIT_CODE absent."
  } >> "${RUN_ROOT}/RUN_STATUS.md"
  exit 0
fi
echo "${EXIT_CODE}"

if [[ "${EXIT_CODE}" != "0" ]]; then
  echo "artifact job failed with exit ${EXIT_CODE}" >&2
  tail -n 80 "${RUN_ROOT}/logs/run.log" >&2 || true
  {
    echo
    echo "Guarded check at $(date '+%F %T %Z'): artifact job failed with exit ${EXIT_CODE}."
  } >> "${RUN_ROOT}/RUN_STATUS.md"
  exit 1
fi

PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
export PYTHONPATH="${ROOT}/CoupledFM${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" "${ROOT}/ops/summarize_latentfm_trackc_anchor_gated_support_teacher_cpu_gate_20260623.py"

{
  echo
  echo "Guarded check at $(date '+%F %T %Z'): artifact job exit \`0\`; CPU gate summarizer completed."
  echo
  echo "Decision report:"
  echo "\`${ROOT}/reports/LATENTFM_TRACKC_ANCHOR_GATED_SUPPORT_TEACHER_CPU_GATE_20260623.md\`"
} >> "${RUN_ROOT}/RUN_STATUS.md"
