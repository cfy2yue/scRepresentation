#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
WINDOW_EPOCH=$(TZ=Asia/Shanghai date -d '2026-06-23 10:56:00' +%s)
NOW_EPOCH=$(TZ=Asia/Shanghai date +%s)
if (( NOW_EPOCH < WINDOW_EPOCH )); then
  echo "Refusing to check before 2026-06-23 10:56:00 CST; v2 residual uncapped no-harm posthoc is a long GPU task." >&2
  exit 3
fi

RUN_NAME=xverse_trackc_support_context_v2_residual_ep050_replay2_2k_seed42
LABEL=latentfm_trackc_support_context_v2_uncapped_noharm_${RUN_NAME}_20260623
RUN_ROOT=${ROOT}/runs/${LABEL}
OUT_DIR=${ROOT}/reports/${LABEL}
EXIT_CODE=${RUN_ROOT}/EXIT_CODE
SUMMARY=${ROOT}/ops/summarize_latentfm_trackc_support_context_v2_uncapped_noharm_20260623.sh
OUT_JSON=${ROOT}/reports/${LABEL}_decision.json
OUT_MD=${ROOT}/reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_UNCAPPED_NOHARM_${RUN_NAME}_DECISION_20260623.md
BOOT_DIR=${ROOT}/reports/${LABEL}_bootstrap

if [[ ! -f "${EXIT_CODE}" ]]; then
  echo "uncapped posthoc still running or exit code missing: ${RUN_ROOT}" >&2
  exit 3
fi

rc=$(cat "${EXIT_CODE}")
if [[ "${rc}" != "0" ]]; then
  echo "uncapped posthoc failed with exit code ${rc}" >&2
  echo "Inspect: ${RUN_ROOT}/logs/uncapped_posthoc.log" >&2
  exit 2
fi

if [[ ! -f "${OUT_DIR}/uncapped_posthoc_index.json" ]]; then
  echo "uncapped posthoc exit 0 but index missing: ${OUT_DIR}/uncapped_posthoc_index.json" >&2
  exit 2
fi

LATENTFM_TRACKC_V2_UNCAPPED_LABEL="${LABEL}" \
LATENTFM_TRACKC_V2_UNCAPPED_INDEX_JSON="${OUT_DIR}/uncapped_posthoc_index.json" \
LATENTFM_TRACKC_V2_UNCAPPED_OUT_JSON="${OUT_JSON}" \
LATENTFM_TRACKC_V2_UNCAPPED_OUT_MD="${OUT_MD}" \
LATENTFM_TRACKC_V2_UNCAPPED_BOOT_DIR="${BOOT_DIR}" \
LATENTFM_TRACKC_V2_UNCAPPED_REPORT_TITLE="LatentFM Track C Support-Context V2 Uncapped Canonical No-Harm Decision: ${RUN_NAME}" \
bash "${SUMMARY}"

echo "uncapped_exit=${rc}"
echo "decision_json=${OUT_JSON}"
echo "decision_md=${OUT_MD}"
