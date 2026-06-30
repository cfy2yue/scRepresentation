#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
fi

echo "## Time"
date '+%F %T %Z'

echo
echo "## tmux"
tmux ls 2>/dev/null || true

echo
echo "## GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits

echo
echo "## Markers"
for p in \
  "${ROOT}/runs/latentfm_trackc_support_present_ablation_controls_20260624/EXIT_CODE" \
  "${ROOT}/runs/latentfm_trackc_support_present_ablation_controls_20260624/FINISHED" \
  "${ROOT}/runs/latentfm_modality_pathway_sampling_smoke_20260624/xverse_scaling_pathway_quota12_3k_seed42/xverse_scaling_pathway_quota12_3k_seed42.EXIT_CODE" \
  "${ROOT}/runs/latentfm_modality_pathway_sampling_smoke_20260624/xverse_scaling_pathway_quota12_3k_seed42/POSTHOC_EXIT_CODE" \
  "${ROOT}/runs/latentfm_modality_pathway_randomcount_control_smoke_20260624/xverse_scaling_pathway_randomcount_3k_seed42/xverse_scaling_pathway_randomcount_3k_seed42.EXIT_CODE" \
  "${ROOT}/runs/latentfm_modality_pathway_randomcount_control_smoke_20260624/xverse_scaling_pathway_randomcount_3k_seed42/POSTHOC_EXIT_CODE"; do
  echo "--- ${p}"
  cat "${p}" 2>/dev/null || echo pending
done

echo
echo "## Pathway Decision"
LATENTFM_SCALING_HT_RUN_ROOT="${ROOT}/runs/latentfm_modality_pathway_sampling_smoke_20260624" \
LATENTFM_SCALING_HT_RUNS=xverse_scaling_pathway_quota12_3k_seed42 \
LATENTFM_SCALING_HT_DECISION_JSON="${ROOT}/reports/latentfm_modality_pathway_sampling_smoke_decision_20260624.json" \
LATENTFM_SCALING_HT_DECISION_MD="${ROOT}/reports/LATENTFM_MODALITY_PATHWAY_SAMPLING_SMOKE_DECISION_20260624.md" \
"${PYTHON}" "${ROOT}/ops/summarize_latentfm_scaling_highthroughput_smokes_20260624.py"

echo
echo "## Pathway Random-Count Control Decision"
LATENTFM_SCALING_HT_RUN_ROOT="${ROOT}/runs/latentfm_modality_pathway_randomcount_control_smoke_20260624" \
LATENTFM_SCALING_HT_RUNS=xverse_scaling_pathway_randomcount_3k_seed42 \
LATENTFM_SCALING_HT_DECISION_JSON="${ROOT}/reports/latentfm_modality_pathway_randomcount_control_smoke_decision_20260624.json" \
LATENTFM_SCALING_HT_DECISION_MD="${ROOT}/reports/LATENTFM_MODALITY_PATHWAY_RANDOMCOUNT_CONTROL_SMOKE_DECISION_20260624.md" \
"${PYTHON}" "${ROOT}/ops/summarize_latentfm_scaling_highthroughput_smokes_20260624.py"

echo
echo "## Track C Support-Control Gate"
"${PYTHON}" "${ROOT}/ops/audit_latentfm_trackc_support_present_ablation_reproducibility_gate_20260624.py"
