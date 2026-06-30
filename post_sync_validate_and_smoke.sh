#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/cyx/1030/scLatent}"
LOG_DIR="${ROOT}/logs/post_sync_validate"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG="${LOG_DIR}/${RUN_ID}.log"
TRANSFER_PID_FILE="${TRANSFER_PID_FILE:-${ROOT}/logs/transfer_from_lilab.pid}"
TRANSFER_STATUS_FILE="${TRANSFER_STATUS_FILE:-${ROOT}/logs/transfer_from_lilab.status}"
WAIT_FOR_SYNC="${WAIT_FOR_SYNC:-1}"
MIN_MEM_AVAILABLE_GIB="${MIN_MEM_AVAILABLE_GIB:-16}"
RUN_NICE="${RUN_NICE:-3}"
RUN_IONICE_CLASS="${RUN_IONICE_CLASS:-2}"
RUN_IONICE_LEVEL="${RUN_IONICE_LEVEL:-4}"

mkdir -p "$LOG_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$RUN_LOG"
}

on_exit() {
  local rc="$1"
  if [[ "$rc" -ne 0 ]]; then
    log "POST-SYNC VALIDATION FAILED rc=${rc}"
    "${ROOT}/ops/generate_validation_report.sh" "$RUN_LOG" >/dev/null 2>&1 || true
  fi
}

trap 'on_exit "$?"' EXIT

run_logged() {
  log "RUN: $*"
  if command -v ionice >/dev/null 2>&1; then
    nice -n "$RUN_NICE" ionice -c "$RUN_IONICE_CLASS" -n "$RUN_IONICE_LEVEL" "$@" 2>&1 | tee -a "$RUN_LOG"
  else
    nice -n "$RUN_NICE" "$@" 2>&1 | tee -a "$RUN_LOG"
  fi
}

check_memory_headroom() {
  local label="$1"
  local avail_gib load_avg
  avail_gib="$(awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)"
  load_avg="$(awk '{print $1 "," $2 "," $3}' /proc/loadavg)"
  log "resource check before ${label}: MemAvailable=${avail_gib}GiB min=${MIN_MEM_AVAILABLE_GIB}GiB load=${load_avg}"
  awk -v avail="$avail_gib" -v min="$MIN_MEM_AVAILABLE_GIB" 'BEGIN { exit(avail >= min ? 0 : 1) }' || {
    log "not enough memory headroom for ${label}; aborting without starting heavy work"
    exit 20
  }
}

if [[ "$WAIT_FOR_SYNC" == "1" && -f "$TRANSFER_PID_FILE" ]]; then
  pid="$(cat "$TRANSFER_PID_FILE")"
  log "waiting for transfer pid=${pid}"
  while kill -0 "$pid" 2>/dev/null; do
    if [[ -f "$TRANSFER_STATUS_FILE" ]]; then
      log "transfer status: $(cat "$TRANSFER_STATUS_FILE")"
    fi
    sleep "${WAIT_POLL_SECONDS:-3600}"
  done
  log "transfer pid=${pid} exited"
fi

if [[ -f "$TRANSFER_STATUS_FILE" ]]; then
  log "final transfer status: $(cat "$TRANSFER_STATUS_FILE")"
fi
if ! grep -q $'\tALL DONE' "$TRANSFER_STATUS_FILE" 2>/dev/null && \
   ! grep -q 'ALL DONE' "${ROOT}/logs/transfer_from_lilab.log" 2>/dev/null; then
  log "transfer did not report ALL DONE; aborting validation"
  exit 10
fi

source "${ROOT}/init-scdfm.sh" > >(tee -a "$RUN_LOG") 2>&1
export CUDA_VISIBLE_DEVICES="${SMOKE_CUDA_VISIBLE_DEVICES:-0}"
log "safety caps: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} nice=${RUN_NICE} ionice=${RUN_IONICE_CLASS}/${RUN_IONICE_LEVEL}"

log "disk usage"
du -sh \
  "${ROOT}/dataset" \
  "${ROOT}/dataset/biFlow_data" \
  "${ROOT}/dataset/cellgene_census" \
  "${ROOT}/dataset/scFM_data" \
  "${ROOT}/dataset/raw" \
  "${ROOT}/scFM_pretrained" \
  "${ROOT}/scFM_third_party" \
  "${ROOT}/pretrainckpt" 2>&1 | tee -a "$RUN_LOG"

log "dataset inventory"
run_logged "${ROOT}/ops/generate_dataset_inventory.sh"

log "CoupledFM resource validation"
(
  cd "${ROOT}/CoupledFM"
  export PYTHONPATH="${ROOT}/CoupledFM:${PYTHONPATH:-}"
  run_logged python -m model.tools.validate_resources --mode local-smoke --datasets Adamson
  run_logged python -m model.tools.validate_resources
  run_logged python -m pytest \
    model/tests/test_plan_guards.py \
    model/tests/test_multi_pool_aggregation.py \
    model/tests/test_unified_condition_embedding.py \
    -q
  run_logged python model/tools/smoke_test.py
  run_logged bash model/scripts/submit_pert_embed_compare_8gpu.sh \
    --dry-run \
    --gpus 0,1,2,3 \
    --out-root "${ROOT}/CoupledFM/output/post_sync_compare_dry" \
    --log-dir "${ROOT}/CoupledFM/output/post_sync_compare_dry/logs"
  check_memory_headroom "CoupledFM local single-GPU smoke"
  coupled_smoke_out="${ROOT}/CoupledFM/output/post_sync_local_single_gpu_smoke"
  run_logged env GPU=0 RUN_SCGPT="${RUN_SCGPT_SMOKE:-1}" \
    PRETRAIN_EPOCHS=1 PRETRAIN_STEPS=1 PRETRAIN_BATCH=2 PRETRAIN_MICRO=1 \
    COUPLED_EPOCHS=1 COUPLED_BATCH=2 COUPLED_MICRO=1 \
    COUPLED_MAX_TRAIN_STEPS=1 COUPLED_VAL_EVERY=50 COUPLED_TEST_EVERY_EPOCH=99 \
    OUT_ROOT="$coupled_smoke_out" \
    bash model/tests/local_single_gpu_smoke.sh
  ckpt="$(find "$coupled_smoke_out/coupled_adamson_cellnavi" -type f -name 'last.pt' 2>/dev/null | sort | tail -n 1)"
  if [[ -z "$ckpt" ]]; then
    log "no cellnavi last.pt found under $coupled_smoke_out"
    exit 30
  fi
  run_logged python -m model.inference \
    --ckpt "$ckpt" \
    --dataset Adamson \
    --mode ot \
    --method euler \
    --n_steps 2 \
    --device cuda \
    --max-cells-per-cond 4 \
    --micro-batch 2 \
    --max-conditions 2 \
    --output_dir "${ROOT}/CoupledFM/output/post_sync_inference_smoke"
)

log "scFMBench resource validation and preflight"
(
  cd "${ROOT}/scFMBench"
  export PYTHONPATH="${ROOT}/scFMBench/fm:${PYTHONPATH:-}"
  run_logged python fm/tools/validate_resources.py --models scgpt cellnavi stack
  run_logged python fm/tools/preflight_embedding.py --models scgpt cellnavi stack --require-materialized-x
)

log "scFMBench baseline smoke"
(
  cd "${ROOT}/scFMBench"
  export PYTHONPATH="${ROOT}/scFMBench/fm:${PYTHONPATH:-}"
  run_logged python fm/smoke/test_pca_baseline.py
  run_logged python benchmark/smoke/test_metrics_pipeline.py
  run_logged python benchmark/smoke/test_metrics_cli.py
)

log "scFMBench one-model embedding smoke"
(
  cd "${ROOT}/scFMBench"
  export PYTHONPATH="${ROOT}/scFMBench/fm:${PYTHONPATH:-}"
  check_memory_headroom "scFMBench CellNavi embedding smoke"
  manifest="${SCFM_OUTPUT_ROOT}/embedding_runs/manifest_with_X.jsonl"
  adata="$(python - "$manifest" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
with p.open() as f:
    for line in f:
        row = json.loads(line)
        path = row.get("path")
        if path:
            print(path)
            raise SystemExit(0)
raise SystemExit(1)
PY
)"
  log "embedding smoke adata=${adata}"
  run_logged python fm/tools/export_embedding_one.py \
    --model cellnavi \
    --adata "$adata" \
    --out-dir "${SCFM_OUTPUT_ROOT}/smoke/cellnavi_first_with_x" \
    --device cuda \
    --batch-size 4 \
    --max-cells 8
  run_logged python benchmark/cli/run_metrics_one.py \
    --emb-dir "${SCFM_OUTPUT_ROOT}/smoke/cellnavi_first_with_x" \
    --out-dir "${SCFM_OUTPUT_ROOT}/smoke/cellnavi_first_with_x_metrics_raw" \
    --latent-space raw \
    --skip atlas
)

log "POST-SYNC VALIDATION PASSED"
run_logged "${ROOT}/ops/generate_validation_report.sh" "$RUN_LOG"
