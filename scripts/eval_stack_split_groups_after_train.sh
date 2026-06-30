#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
COUPLEDFM="${ROOT}/CoupledFM"
RUN_TAG="${RUN_TAG:-20260616_full_stack_20k_g003_unbiased_full_eval}"
RUN_DIR="${RUN_DIR:-${COUPLEDFM}/output/latentfm_runs/full_stack/${RUN_TAG}}"
TRAIN_SESSION="${TRAIN_SESSION:-latentfm_stack_full20k_g003u}"
LOG_ROOT="${ROOT}/logs/latentfm_full_train"
WATCH_LOG="${LOG_ROOT}/${RUN_TAG}_split_group_eval_watcher.log"
GPU="${GPU:-0}"

mkdir -p "${LOG_ROOT}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${WATCH_LOG}"
}

log "watcher started; waiting for tmux session ${TRAIN_SESSION}"
while tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; do
  sleep 300
  log "still waiting for train session ${TRAIN_SESSION}"
done

CKPT="${RUN_DIR}/best.pt"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="${RUN_DIR}/latest.pt"
fi
if [[ ! -f "${CKPT}" ]]; then
  log "no checkpoint found under ${RUN_DIR}; skip split-group eval"
  exit 2
fi

if [[ -f "${ROOT}/init-scdfm.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/init-scdfm.sh" >/dev/null
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export PYTHONPATH="${COUPLEDFM}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"

cd "${COUPLEDFM}"
OUT="${RUN_DIR}/split_group_eval_results.json"
log "running split-group eval ckpt=${CKPT} out=${OUT} gpu=${GPU}"
python -m model.latent.eval_split_groups \
  --checkpoint "${CKPT}" \
  --data-dir "${ROOT}/dataset/latentfm_full/stack" \
  --biflow-dir "${ROOT}/dataset/biFlow_data" \
  --out "${OUT}" \
  --gpu 0 \
  2>&1 | tee -a "${WATCH_LOG}"
log "split-group eval done"
