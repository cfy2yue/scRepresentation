#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
RUN_ROOT="${ROOT}/runs/latentfm_trackc_support_set_task_input_artifacts_20260623/xverse_support_film_retry1_trainmulti_condition_means"
OUT_DIR="${RUN_ROOT}/condition_means"
GPU_ID="${LATENTFM_SUPPORT_SET_TASK_INPUT_GPU:-0}"
DATA_DIR="${ROOT}/dataset/latentfm_full/xverse"
SUPPORT_SPLIT="${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
ANCHOR_CKPT="${ROOT}/CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
CANDIDATE_CKPT="${ROOT}/CoupledFM/output/latentfm_runs/xverse_trackc_support_film_20260623/xverse_trackc_support_film_absroute_2k_seed42_retry1/best.pt"
PERT_MEANS="${DATA_DIR}/pert_means.npz"

export PYTHONPATH="${ROOT}/CoupledFM${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS="${LATENTFM_SUPPORT_SET_TASK_INPUT_THREADS:-4}"
export MKL_NUM_THREADS="${LATENTFM_SUPPORT_SET_TASK_INPUT_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${LATENTFM_SUPPORT_SET_TASK_INPUT_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${LATENTFM_SUPPORT_SET_TASK_INPUT_THREADS:-4}"

mkdir -p "${OUT_DIR}"

common=(--data-dir "${DATA_DIR}" --gpu 0 --device cuda:0 --ode-steps 20 --max-chunk 512 --eval-max-mmd-cells 2048 --pert-means-file "${PERT_MEANS}" --save-condition-means)

echo "[support-set-task-input] start $(date '+%F %T %Z')"
echo "[support-set-task-input] gpu=${GPU_ID} split=${SUPPORT_SPLIT}"

"${PYTHON}" -m model.latent.eval_split_groups \
  --checkpoint "${ANCHOR_CKPT}" \
  --split-file "${SUPPORT_SPLIT}" \
  --groups train_multi support_val_multi \
  --out "${OUT_DIR}/trainselect_anchor_train_support_multi_condition_means_ode20.json" \
  "${common[@]}"

"${PYTHON}" -m model.latent.eval_split_groups \
  --checkpoint "${CANDIDATE_CKPT}" \
  --split-file "${SUPPORT_SPLIT}" \
  --groups train_multi support_val_multi \
  --out "${OUT_DIR}/trainselect_candidate_train_support_multi_condition_means_ode20.json" \
  "${common[@]}"

echo "[support-set-task-input] finished $(date '+%F %T %Z')"
