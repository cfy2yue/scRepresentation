#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
COUPLEDFM="${ROOT}/CoupledFM"
RUN_DIR="${COUPLEDFM}/output/latentfm_runs/full_stack/20260616_full_stack_20k_g003_unbiased_full_eval"
CKPT="${RUN_DIR}/best.pt"
DATA_DIR="${ROOT}/dataset/latentfm_full/stack"
BIFLOW_DIR="${ROOT}/dataset/biFlow_data"
LOG_ROOT="${ROOT}/logs/latentfm_full_eval"
TAG="${TAG:-20260616_stack_best_eval}"

mkdir -p "${LOG_ROOT}" "${RUN_DIR}/posthoc_eval"

if [[ ! -f "${CKPT}" ]]; then
  echo "missing checkpoint: ${CKPT}" >&2
  exit 2
fi

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

run_split_groups() {
  local gpu="${1:-2}"
  local log="${LOG_ROOT}/${TAG}_split_groups.log"
  local out="${RUN_DIR}/posthoc_eval/split_group_eval_best_ode20_mse2048_mmd2048.json"
  echo "[$(date '+%F %T')] split-group eval gpu=${gpu} out=${out}" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m model.latent.eval_split_groups \
    --checkpoint "${CKPT}" \
    --data-dir "${DATA_DIR}" \
    --biflow-dir "${BIFLOW_DIR}" \
    --out "${out}" \
    --gpu 0 \
    --ode-steps 20 \
    --max-chunk 512 \
    --eval-max-mse-cells 2048 \
    --eval-max-mmd-cells 2048 \
    2>&1 | tee -a "${log}"
  echo "[$(date '+%F %T')] split-group eval done" | tee -a "${log}"
}

run_ode_sweep() {
  local gpu="${1:-3}"
  local log="${LOG_ROOT}/${TAG}_ode_sweep.log"
  local out_dir="${RUN_DIR}/posthoc_eval/ode_step_sweep"
  mkdir -p "${out_dir}"
  echo "[$(date '+%F %T')] ODE step sweep gpu=${gpu} out_dir=${out_dir}" | tee -a "${log}"
  for steps in 10 20 50 100; do
    local out="${out_dir}/test_ode${steps}_max128_mse1024_mmd1024.json"
    echo "[$(date '+%F %T')] ode_steps=${steps} out=${out}" | tee -a "${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" python -m model.latent.eval_split_groups \
      --checkpoint "${CKPT}" \
      --data-dir "${DATA_DIR}" \
      --biflow-dir "${BIFLOW_DIR}" \
      --groups test \
      --out "${out}" \
      --gpu 0 \
      --ode-steps "${steps}" \
      --max-chunk 512 \
      --eval-max-conditions 128 \
      --eval-max-mse-cells 1024 \
      --eval-max-mmd-cells 1024 \
      2>&1 | tee -a "${log}"
  done
  echo "[$(date '+%F %T')] ODE step sweep done" | tee -a "${log}"
}

case "${1:-all}" in
  split)
    run_split_groups "${2:-2}"
    ;;
  ode)
    run_ode_sweep "${2:-3}"
    ;;
  all)
    run_split_groups "${SPLIT_GPU:-2}"
    run_ode_sweep "${ODE_GPU:-3}"
    ;;
  *)
    echo "usage: $0 [split GPU|ode GPU|all]" >&2
    exit 2
    ;;
esac
