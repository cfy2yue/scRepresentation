#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_ROOT=${ROOT}/runs/latentfm_modality_pathway_mmd_preservation_smoke_20260624
RUN_NAME=xverse_scaling_pathway_mmdpreserve_3k_seed42
RUN_DIR=${RUN_ROOT}/${RUN_NAME}
SUMMARY=${ROOT}/ops/summarize_latentfm_scaling_highthroughput_smokes_20260624.py

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "missing run dir: ${RUN_DIR}" >&2
  exit 2
fi
if [[ ! -f "${RUN_DIR}/${RUN_NAME}.EXIT_CODE" ]]; then
  echo "pending: training still running for ${RUN_NAME}"
  exit 0
fi
train_rc=$(cat "${RUN_DIR}/${RUN_NAME}.EXIT_CODE")
if [[ "${train_rc}" != "0" ]]; then
  echo "failed: training exit ${train_rc} for ${RUN_NAME}" >&2
  exit 3
fi
if [[ ! -f "${RUN_DIR}/POSTHOC_EXIT_CODE" ]]; then
  echo "pending: posthoc not complete for ${RUN_NAME}"
  exit 0
fi
posthoc_rc=$(cat "${RUN_DIR}/POSTHOC_EXIT_CODE")
if [[ "${posthoc_rc}" != "0" ]]; then
  echo "failed: posthoc exit ${posthoc_rc} for ${RUN_NAME}" >&2
  exit 3
fi

LATENTFM_SCALING_HT_RUN_ROOT="${RUN_ROOT}" \
LATENTFM_SCALING_HT_RUNS="${RUN_NAME}" \
LATENTFM_SCALING_HT_DECISION_JSON="${ROOT}/reports/latentfm_modality_pathway_mmd_preservation_smoke_decision_20260624.json" \
LATENTFM_SCALING_HT_DECISION_MD="${ROOT}/reports/LATENTFM_MODALITY_PATHWAY_MMD_PRESERVATION_SMOKE_DECISION_20260624.md" \
"${PYTHON}" "${SUMMARY}"
