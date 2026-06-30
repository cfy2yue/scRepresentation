#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
COUPLEDFM="${ROOT}/CoupledFM"
RUN_TAG="${RUN_TAG:-20260616_stack_gpu_efficiency_sweep}"
DATA_DIR="${ROOT}/dataset/latentfm_full/stack"
BIFLOW_DIR="${ROOT}/dataset/biFlow_data"
GENE_CACHE="${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene"
OUT_ROOT="${COUPLEDFM}/output/latentfm_runs/stack_efficiency_sweep/${RUN_TAG}"
LOG_ROOT="${ROOT}/logs/latentfm_efficiency_sweep/${RUN_TAG}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

if [[ -f "${ROOT}/init-scdfm.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/init-scdfm.sh" >/dev/null
fi

export PYTHONPATH="${COUPLEDFM}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

cd "${COUPLEDFM}"

# name:gpu:gamma:mmd_estimator:grad_accum_steps:mmd_every
configs=(
  "g001_unbiased_acc1:2:0.01:unbiased:1:4"
  "g003_biased_acc1:3:0.03:biased:1:4"
  "g003_unbiased_acc4:4:0.03:unbiased:4:4"
)

echo "[$(date '+%F %T')] stack efficiency sweep run_tag=${RUN_TAG}"
echo "out_root=${OUT_ROOT}"
echo "log_root=${LOG_ROOT}"
echo "configs=${configs[*]}"

pids=()
for cfg in "${configs[@]}"; do
  IFS=':' read -r name gpu gamma estimator grad_accum mmd_every <<< "${cfg}"
  save_dir="${OUT_ROOT}/${name}"
  log_file="${LOG_ROOT}/${name}.log"
  mkdir -p "${save_dir}"
  echo "[$(date '+%F %T')] start ${name} gpu=${gpu} gamma=${gamma} estimator=${estimator} grad_accum=${grad_accum}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    python -m model.latent.train \
      --data-dir "${DATA_DIR}" \
      --biflow-dir "${BIFLOW_DIR}" \
      --save-dir "${save_dir}" \
      --latent-backbone stack \
      --model-type control_mlp \
      --emb-dim 1600 \
      --gpu 0 \
      --batch-size 64 \
      --grad-accum-steps "${grad_accum}" \
      --min-cells 16 \
      --scale-noise 0.01 \
      --lr 1e-4 \
      --weight-decay 1e-4 \
      --warmup-steps 300 \
      --total-steps 3000 \
      --lr-decay-steps 3000 \
      --print-every 100 \
      --eval-max-conditions 64 \
      --eval-max-conditions-per-dataset 4 \
      --eval-max-mse-cells 512 \
      --eval-max-mmd-cells 512 \
      --selection-metric test_mmd \
      --ot-method torch_sinkhorn \
      --ot-sinkhorn-reg 0.05 \
      --ot-sinkhorn-iter 30 \
      --prefetch 16 \
      --use-mmd \
      --gamma "${gamma}" \
      --gamma-warmup-start 500 \
      --gamma-warmup-end 2500 \
      --mmd-every "${mmd_every}" \
      --mmd-estimator "${estimator}" \
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
      --patience 4
  ) > "${log_file}" 2>&1 &
  pids+=("$!")
  echo "${pids[-1]}" > "${LOG_ROOT}/${name}.pid"
  sleep 20
done

printf '%s\n' "${pids[@]}" > "${LOG_ROOT}/pids.txt"
status=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if wait "${pid}"; then
    echo "[$(date '+%F %T')] pid=${pid} done"
  else
    code="$?"
    echo "[$(date '+%F %T')] pid=${pid} failed status=${code}" >&2
    status="${code}"
  fi
done
echo "[$(date '+%F %T')] sweep finished status=${status}"
exit "${status}"
