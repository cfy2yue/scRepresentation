#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
COUPLEDFM="${ROOT}/CoupledFM"
LOG_ROOT="${ROOT}/logs/latentfm_full_prepare"
WATCH_LOG="${LOG_ROOT}/top3_after_sync_20260616.log"
SYNC_SESSION="${SYNC_SESSION:-sync_top3_training_data}"

mkdir -p "${LOG_ROOT}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${WATCH_LOG}"
}

log "watcher started; waiting for tmux session ${SYNC_SESSION}"
while tmux has-session -t "${SYNC_SESSION}" 2>/dev/null; do
  sleep 300
  log "still waiting for rsync session ${SYNC_SESSION}"
done

log "rsync session finished; begin packing top3 full bundles"

if [[ -f "${ROOT}/init-scdfm.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/init-scdfm.sh" >/dev/null
fi

export COUPLEDFM_ROOT="${COUPLEDFM}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export PYTHONPATH="${COUPLEDFM}:${PYTHONPATH:-}"

cd "${COUPLEDFM}"

for model in scldm scfoundation; do
  export COUPLEDFM_BIFLOW_CTRL="${ROOT}/dataset/Training_data/${model}/control_${model}"
  export COUPLEDFM_BIFLOW_GT="${ROOT}/dataset/Training_data/${model}/gt_${model}"
  export COUPLEDFM_FM_DATA="${ROOT}/dataset/latentfm_full/${model}"
  MODEL_LOG="${LOG_ROOT}/${model}_20260616_full_prepare.log"

  if [[ ! -d "${COUPLEDFM_BIFLOW_CTRL}" || ! -d "${COUPLEDFM_BIFLOW_GT}" ]]; then
    log "missing source directories for ${model}: ctrl=${COUPLEDFM_BIFLOW_CTRL} gt=${COUPLEDFM_BIFLOW_GT}"
    exit 2
  fi

  log "pack ${model}: ctrl=${COUPLEDFM_BIFLOW_CTRL} gt=${COUPLEDFM_BIFLOW_GT} out=${COUPLEDFM_FM_DATA}"
  python model/latent/prepare_fm_data.py --n-workers 2 2>&1 | tee -a "${MODEL_LOG}"
  log "done pack ${model}"
done

log "all top3 full bundle packing done"
