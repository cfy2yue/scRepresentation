#!/usr/bin/env bash
set -euo pipefail

# First-pass LatentFM baseline for the selected top3 encoders.
#
# Assumptions:
# - The dataset bundle exists under /data/cyx/1030/dataset/latentfm_full/<model>.
# - Run this only after checking GPU availability with nvidia-smi.
# - One model is launched per GPU; logs and checkpoints are resumable.

ROOT="/data/cyx/1030/scLatent"
COUPLEDFM="${ROOT}/CoupledFM"
DATA_ROOT="${DATA_ROOT:-${ROOT}/dataset/latentfm_full}"
BIFLOW_DIR="${ROOT}/dataset/biFlow_data"
OUT_ROOT="${COUPLEDFM}/output/latentfm_runs/top3_pertcond_baseline"
LOG_ROOT="${ROOT}/logs/latentfm_top3_baseline"
GENE_CACHE="${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene"

MODELS_CSV="${MODELS:-stack,scldm,scfoundation}"
GPUS_CSV="${GPUS:-1,2,3}"
TOTAL_STEPS="${TOTAL_STEPS:-8000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
WAIT_FOR_JOBS="${WAIT_FOR_JOBS:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export PYTHONPATH="${COUPLEDFM}:${PYTHONPATH:-}"
export PERT_EMBED_SOURCE="${PERT_EMBED_SOURCE:-scgpt_embed_gene}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

IFS=',' read -r -a models <<< "${MODELS_CSV}"
IFS=',' read -r -a gpus <<< "${GPUS_CSV}"

if (( ${#gpus[@]} < ${#models[@]} )); then
  echo "Need at least as many GPUS as MODELS: MODELS=${MODELS_CSV} GPUS=${GPUS_CSV}" >&2
  exit 2
fi

emb_dim() {
  case "$1" in
    stack) echo 1600 ;;
    scldm) echo 4096 ;;
    scfoundation) echo 3072 ;;
    state) echo 2058 ;;
    *) echo "Unknown model: $1" >&2; return 2 ;;
  esac
}

echo "[$(date '+%F %T')] launch top3 LatentFM baseline"
echo "models=${models[*]}"
echo "gpus=${gpus[*]}"
echo "total_steps=${TOTAL_STEPS} batch_size=${BATCH_SIZE} run_tag=${RUN_TAG}"
echo "pert_embed_source=${PERT_EMBED_SOURCE}"
echo "wait_for_jobs=${WAIT_FOR_JOBS}"
echo "out_root=${OUT_ROOT}"
echo "log_root=${LOG_ROOT}"

pids=()
for i in "${!models[@]}"; do
  model="${models[$i]}"
  gpu="${gpus[$i]}"
  dim="$(emb_dim "${model}")"
  save_dir="${OUT_ROOT}/${RUN_TAG}/${model}"
  log_file="${LOG_ROOT}/${RUN_TAG}_${model}.log"
  mkdir -p "${save_dir}"

  echo "[$(date '+%F %T')] start ${model} gpu=${gpu} dim=${dim} log=${log_file}"
  (
    cd "${COUPLEDFM}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    python -m model.latent.train \
      --data-dir "${DATA_ROOT}/${model}" \
      --biflow-dir "${BIFLOW_DIR}" \
      --save-dir "${save_dir}" \
      --latent-backbone "${model}" \
      --model-type control_mlp \
      --emb-dim "${dim}" \
      --gpu 0 \
      --batch-size "${BATCH_SIZE}" \
      --min-cells 16 \
      --scale-noise 0.01 \
      --lr 1e-4 \
      --weight-decay 1e-4 \
      --warmup-steps 300 \
      --total-steps "${TOTAL_STEPS}" \
      --lr-decay-steps "${TOTAL_STEPS}" \
      --print-every 100 \
      --selection-metric test_mmd \
      --ot-method torch_sinkhorn \
      --ot-sinkhorn-reg 0.05 \
      --ot-sinkhorn-iter 30 \
      --use-mmd \
      --gamma 0.03 \
      --gamma-warmup-start 500 \
      --gamma-warmup-end 2500 \
      --mmd-every 4 \
      --mmd-estimator unbiased \
      --use-ema \
      --ema-update-after 500 \
      --ema-decay 0.999 \
      --amp-dtype bf16 \
      --use-pert-condition \
      --pert-gene-emb-cache-dir "${GENE_CACHE}" \
      --pert-chem-enabled \
      --pert-chem-emb-dim 512 \
      --chem-fallback-embed-dim 512 \
      --pert-to-c-init-mode xavier_small \
      --use-pert-in-fusion \
      --patience 8
  ) > "${log_file}" 2>&1 &
  pids+=("$!")
  echo "${pids[-1]}" > "${LOG_ROOT}/${RUN_TAG}_${model}.pid"
  sleep 20
done

printf '%s\n' "${pids[@]}" > "${LOG_ROOT}/${RUN_TAG}.pids"
echo "[$(date '+%F %T')] launched pids=${pids[*]}"
echo "pid_file=${LOG_ROOT}/${RUN_TAG}.pids"

if [[ "${WAIT_FOR_JOBS}" == "1" ]]; then
  status=0
  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    model="${models[$i]}"
    echo "[$(date '+%F %T')] waiting ${model} pid=${pid}"
    if wait "${pid}"; then
      echo "[$(date '+%F %T')] finished ${model} pid=${pid} status=0"
    else
      code="$?"
      echo "[$(date '+%F %T')] failed ${model} pid=${pid} status=${code}" >&2
      status="${code}"
    fi
  done
  echo "[$(date '+%F %T')] all jobs completed status=${status}"
  exit "${status}"
fi
