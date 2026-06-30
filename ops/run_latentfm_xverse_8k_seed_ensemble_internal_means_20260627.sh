#!/usr/bin/env bash
set -euo pipefail

source /data/cyx/1030/scLatent/init-scdfm.sh >/dev/null
cd /data/cyx/1030/scLatent/CoupledFM

export CUDA_VISIBLE_DEVICES="${LATENTFM_GPU:-2}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export PYTHONPATH=/data/cyx/1030/scLatent/CoupledFM:${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene

OUT_DIR=/data/cyx/1030/scLatent/reports/latentfm_xverse_8k_seed_ensemble_internal_means_20260627
mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --data-dir /data/cyx/1030/dataset/latentfm_full/xverse
  --biflow-dir /data/cyx/1030/dataset/biFlow_data
  --split-file /data/cyx/1030/dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json
  --groups internal_val_cross_background_seen_gene_proxy internal_val_family_gene_proxy
  --gpu 0
  --ode-steps 20
  --max-chunk 512
  --eval-max-conditions 0
  --eval-max-conditions-per-dataset 0
  --eval-max-mse-cells 2048
  --eval-max-mmd-cells 2048
  --eval-seed 42
  --save-condition-means
)

/data/cyx/software/miniconda3/envs/scdfm/bin/python -m model.latent.eval_split_groups \
  --checkpoint /data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt \
  --out "${OUT_DIR}/seed42_internal_split_group_means_evalseed42.json" \
  "${COMMON_ARGS[@]}"

/data/cyx/software/miniconda3/envs/scdfm/bin/python -m model.latent.eval_split_groups \
  --checkpoint /data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/xverse_8k_seed_replicate_20260621/xverse_comp006_endpoint5_8k_seed43_fulleval/best.pt \
  --out "${OUT_DIR}/seed43_internal_split_group_means_evalseed42.json" \
  "${COMMON_ARGS[@]}"
