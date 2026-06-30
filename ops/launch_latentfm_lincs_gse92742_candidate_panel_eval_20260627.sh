#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <physical_gpu_id> <panel: mechanism|response>" >&2
  exit 2
fi

GPU_ID="$1"
PANEL="$2"
ROOT="/data/cyx/1030/scLatent"
RUN_DIR="${ROOT}/runs/latentfm_lincs_gse92742_candidate_panel_eval_20260627"
ENV_PY="/data/cyx/software/miniconda3/envs/scdfm/bin/python"
SPLIT="${ROOT}/dataset/biFlow_data/split_seed42_lincs_gse92742_train_gene_eval_20260627.json"
ANCHOR_JSON="${ROOT}/runs/latentfm_lincs_gse92742_train_gene_outcome_eval_20260627/anchor_eval/split_group_eval_anchor_lincs_gse92742_train_gene_ode20.json"
REPORT_DIR="${ROOT}/reports/lincs_gse92742_train_gene_candidate_panel_20260627"

mkdir -p "${RUN_DIR}/candidate_eval/${PANEL}" "${RUN_DIR}/logs" "${REPORT_DIR}"

declare -a LABELS=()
declare -a CKPTS=()

case "${PANEL}" in
  mechanism)
    LABELS+=("xverse_risk_row_cvar_allrisk_w020_2k_seed42")
    CKPTS+=("${ROOT}/CoupledFM/output/latentfm_runs/risk_row_cvar_trainonly_20260624/xverse_risk_row_cvar_allrisk_w020_2k_seed42/latest.pt")
    LABELS+=("xverse_scaling_cap60_6k_seed42")
    CKPTS+=("${ROOT}/CoupledFM/output/latentfm_runs/scaling_highthroughput_smokes_20260624/xverse_scaling_cap60_6k_seed42/best.pt")
    ;;
  response)
    LABELS+=("xverse_scaling_cap60_resp010_replay05_4k_seed42")
    CKPTS+=("${ROOT}/CoupledFM/output/latentfm_runs/scaling_cap60_response_repair_20260624/xverse_scaling_cap60_resp010_replay05_4k_seed42/best.pt")
    LABELS+=("xverse_scaling_cap60_resp025_replay05_4k_seed42")
    CKPTS+=("${ROOT}/CoupledFM/output/latentfm_runs/scaling_cap60_response_repair_20260624/xverse_scaling_cap60_resp025_replay05_4k_seed42/best.pt")
    ;;
  *)
    echo "Unknown panel: ${PANEL}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${ANCHOR_JSON}" ]]; then
  echo "Missing anchor JSON: ${ANCHOR_JSON}" >&2
  exit 3
fi

echo "[panel] start $(date '+%F %T %Z')"
echo "[panel] panel ${PANEL}"
echo "[panel] physical GPU ${GPU_ID}"
echo "[panel] split ${SPLIT}"
echo "[panel] anchor json ${ANCHOR_JSON}"

cd "${ROOT}/CoupledFM"

for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"
  ckpt="${CKPTS[$i]}"
  out_json="${RUN_DIR}/candidate_eval/${PANEL}/${label}_lincs_gse92742_train_gene_ode20.json"
  summary_prefix="${REPORT_DIR}/${label}_vs_anchor"

  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing checkpoint for ${label}: ${ckpt}" >&2
    exit 4
  fi

  echo "[panel] evaluating ${label}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  "${ENV_PY}" -m model.latent.eval_split_groups \
    --checkpoint "${ckpt}" \
    --split-file "${SPLIT}" \
    --groups test \
    --out "${out_json}" \
    --device cuda:0 \
    --gpu 0 \
    --ode-steps 20 \
    --eval-max-mse-cells 2048 \
    --eval-max-mmd-cells 2048 \
    --eval-seed 42

  "${ENV_PY}" "${ROOT}/ops/summarize_latentfm_lincs_gse92742_train_gene_outcome_eval_20260627.py" \
    --anchor-json "${ANCHOR_JSON}" \
    --candidate-json "${out_json}" \
    --candidate-label "${label}" \
    --out-prefix "${summary_prefix}"
done

echo "[panel] finished $(date '+%F %T %Z')"
