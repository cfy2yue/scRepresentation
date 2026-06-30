#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <physical_gpu_id>" >&2
  exit 2
fi

GPU_ID="$1"
ROOT="/data/cyx/1030/scLatent"
RUN_DIR="${ROOT}/runs/latentfm_lincs_gse92742_train_gene_outcome_eval_20260627"
ENV_PY="/data/cyx/software/miniconda3/envs/scdfm/bin/python"
SPLIT="${ROOT}/dataset/biFlow_data/split_seed42_lincs_gse92742_train_gene_eval_20260627.json"
ANCHOR_CKPT="${ROOT}/CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
CANDIDATE_CKPT="${ROOT}/CoupledFM/output/latentfm_runs/true_cell_count_budget128_tail_stability_6k_20260625/xverse_truecell_nested_budget128_tailstable_seed42_6000/best.pt"
ANCHOR_OUT="${RUN_DIR}/anchor_eval/split_group_eval_anchor_lincs_gse92742_train_gene_ode20.json"
CANDIDATE_OUT="${RUN_DIR}/candidate_eval/split_group_eval_truecell_budget128_lincs_gse92742_train_gene_ode20.json"
SUMMARY_PREFIX="${ROOT}/reports/lincs_gse92742_train_gene_outcome_eval_20260627/xverse_truecell_budget128_vs_anchor"

mkdir -p "${RUN_DIR}/anchor_eval" "${RUN_DIR}/candidate_eval" "${RUN_DIR}/logs" "$(dirname "${SUMMARY_PREFIX}")"

echo "[launcher] start $(date '+%F %T %Z')"
echo "[launcher] physical GPU ${GPU_ID}"
echo "[launcher] split ${SPLIT}"
echo "[launcher] anchor ${ANCHOR_CKPT}"
echo "[launcher] candidate ${CANDIDATE_CKPT}"

cd "${ROOT}/CoupledFM"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
"${ENV_PY}" -m model.latent.eval_split_groups \
  --checkpoint "${ANCHOR_CKPT}" \
  --split-file "${SPLIT}" \
  --groups test \
  --out "${ANCHOR_OUT}" \
  --device cuda:0 \
  --gpu 0 \
  --ode-steps 20 \
  --eval-max-mse-cells 2048 \
  --eval-max-mmd-cells 2048 \
  --eval-seed 42

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
"${ENV_PY}" -m model.latent.eval_split_groups \
  --checkpoint "${CANDIDATE_CKPT}" \
  --split-file "${SPLIT}" \
  --groups test \
  --out "${CANDIDATE_OUT}" \
  --device cuda:0 \
  --gpu 0 \
  --ode-steps 20 \
  --eval-max-mse-cells 2048 \
  --eval-max-mmd-cells 2048 \
  --eval-seed 42

"${ENV_PY}" "${ROOT}/ops/summarize_latentfm_lincs_gse92742_train_gene_outcome_eval_20260627.py" \
  --anchor-json "${ANCHOR_OUT}" \
  --candidate-json "${CANDIDATE_OUT}" \
  --candidate-label "xverse_truecell_nested_budget128_tailstable_seed42_6000" \
  --out-prefix "${SUMMARY_PREFIX}"

echo "[launcher] finished $(date '+%F %T %Z')"
