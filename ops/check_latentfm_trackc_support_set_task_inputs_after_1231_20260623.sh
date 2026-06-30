#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
RUN_ROOT="${ROOT}/runs/latentfm_trackc_support_set_task_input_artifacts_20260623/xverse_support_film_retry1_trainmulti_condition_means"
SESSION="lfm_support_set_task_inputs_20260623"
CUTOFF_EPOCH="$(date -d '2026-06-23 12:31:00 +0800' +%s)"
NOW_EPOCH="$(date +%s)"

if (( NOW_EPOCH < CUTOFF_EPOCH )); then
  echo "[support-set-task-input-check] before 2026-06-23 12:31:00 +0800; refusing to poll long job"
  exit 3
fi

EXIT_CODE_FILE="${RUN_ROOT}/EXIT_CODE"
RUN_STATUS="${RUN_ROOT}/RUN_STATUS.md"
ANCHOR_JSON="${RUN_ROOT}/condition_means/trainselect_anchor_train_support_multi_condition_means_ode20.json"
CANDIDATE_JSON="${RUN_ROOT}/condition_means/trainselect_candidate_train_support_multi_condition_means_ode20.json"
PYTHON="/data/cyx/software/miniconda3/envs/scdfm/bin/python"
SUMMARY="${ROOT}/ops/summarize_latentfm_trackc_support_set_task_summary_gate_20260623.py"
OUT_MD="${ROOT}/reports/LATENTFM_TRACKC_SUPPORT_SET_TASK_SUMMARY_GATE_20260623.md"
OUT_JSON="${ROOT}/reports/latentfm_trackc_support_set_task_summary_gate_20260623.json"

if [[ ! -f "${EXIT_CODE_FILE}" ]]; then
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[support-set-task-input-check] still running after allowed polling window"
    exit 4
  fi
  echo "[support-set-task-input-check] no EXIT_CODE and no tmux session; inspect ${RUN_STATUS}"
  exit 5
fi

EXIT_CODE="$(tr -d '[:space:]' < "${EXIT_CODE_FILE}")"
if [[ "${EXIT_CODE}" != "0" ]]; then
  echo "[support-set-task-input-check] input artifact job failed with exit code ${EXIT_CODE}; inspect ${RUN_STATUS}"
  exit 6
fi

for path in "${ANCHOR_JSON}" "${CANDIDATE_JSON}"; do
  if [[ ! -s "${path}" ]]; then
    echo "[support-set-task-input-check] missing expected artifact: ${path}"
    exit 7
  fi
done

"${PYTHON}" "${SUMMARY}" \
  --run-root "${RUN_ROOT}" \
  --out-md "${OUT_MD}" \
  --out-json "${OUT_JSON}"

echo "[support-set-task-input-check] summary gate written:"
echo "  ${OUT_MD}"
echo "  ${OUT_JSON}"
