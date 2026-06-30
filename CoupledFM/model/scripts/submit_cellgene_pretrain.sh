#!/usr/bin/env bash
# Single DDP job: cellgene_census pairwise raw FM pretrain (default 4 GPUs).
#
# **Environment namespace:** ``PRETRAIN_*`` vars here are **only** read by
# ``python -m model.raw_pretrain.train``. They are **not** shared with
# CoupledFM main training — set ``RAW_*`` separately for that.
#
# Usage:
#   bash model/scripts/submit_cellgene_pretrain.sh
#   PRETRAIN_GPUS=0,1,2,3 PRETRAIN_EPOCHS=5 bash model/scripts/submit_cellgene_pretrain.sh
#   bash model/scripts/submit_cellgene_pretrain.sh --out-root /path/to/run --log-dir /path/to/logs
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DATASET_ROOT="${SCDFM_DATASET_ROOT:-$ROOT/dataset}"
PRETRAIN_ROOT="${SCDFM_PRETRAIN_ROOT:-$ROOT/pretrainckpt}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

TORCHRUN="${TORCHRUN:-torchrun}"

CLI_OUT_ROOT=""
CLI_LOG_DIR=""
CLI_GPUS=""
CLI_PROCESSED_DIR=""
CLI_EXTRA_ARGS=""

usage() {
  sed -n '1,12p' "$0"
  cat <<'USAGE'

Options:
  --out-root PATH        Training output directory (overrides PRETRAIN_OUT_DIR)
  --log-dir PATH         Log directory (default: OUT_ROOT/logs)
  --gpus CSV            CUDA_VISIBLE_DEVICES list (overrides PRETRAIN_GPUS)
  --processed-dir PATH   Processed cellgene-census root (overrides PRETRAIN_PROCESSED_DIR)
  --extra-args STRING    Extra args passed to model.raw_pretrain.train
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root) CLI_OUT_ROOT="${2:?--out-root needs PATH}"; shift 2 ;;
    --log-dir) CLI_LOG_DIR="${2:?--log-dir needs PATH}"; shift 2 ;;
    --gpus) CLI_GPUS="${2:?--gpus needs CSV}"; shift 2 ;;
    --processed-dir) CLI_PROCESSED_DIR="${2:?--processed-dir needs PATH}"; shift 2 ;;
    --extra-args) CLI_EXTRA_ARGS="${2:?--extra-args needs STRING}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[fatal] unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PRETRAIN_GPUS="${PRETRAIN_GPUS:-0,1,2,3}"
[[ -n "$CLI_GPUS" ]] && PRETRAIN_GPUS="$CLI_GPUS"
OUT_ROOT="${CLI_OUT_ROOT:-${PRETRAIN_OUT_DIR:-$ROOT/output/cellgene_pretrain_$(date +%Y%m%d_%H%M%S)}}"
LOG_DIR="${CLI_LOG_DIR:-${PRETRAIN_LOG_DIR:-$OUT_ROOT/logs}}"
[[ -n "$CLI_PROCESSED_DIR" ]] && PRETRAIN_PROCESSED_DIR="$CLI_PROCESSED_DIR"
[[ -n "$CLI_EXTRA_ARGS" ]] && PRETRAIN_EXTRA_ARGS="$CLI_EXTRA_ARGS"

NPROC="$(echo "$PRETRAIN_GPUS" | tr ',' '\n' | sed '/^$/d' | wc -l | tr -d ' ')"

mkdir -p "$OUT_ROOT" "$LOG_DIR"
export PRETRAIN_OUT_DIR="$OUT_ROOT"

echo "[pretrain] ROOT=$ROOT  OUT=$OUT_ROOT  LOG_DIR=$LOG_DIR  nproc=$NPROC  GPUs=$PRETRAIN_GPUS" | tee "$LOG_DIR/submit.log"
echo "[pretrain] DATASET_ROOT=$DATASET_ROOT  PRETRAIN_ROOT=$PRETRAIN_ROOT" | tee -a "$LOG_DIR/submit.log"
echo "[pretrain] data summary will be written to $OUT_ROOT/data_summary.json" | tee -a "$LOG_DIR/submit.log"

"${PYTHON:-python}" -m model.tools.validate_resources \
  --mode pretrain \
  --datasets Adamson \
  2>&1 | tee "$LOG_DIR/validate_resources.log"

(
  export CUDA_VISIBLE_DEVICES="$PRETRAIN_GPUS"
  export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
  "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
    -m model.raw_pretrain.train \
    --output-dir "$OUT_ROOT" \
    --processed-dir "${PRETRAIN_PROCESSED_DIR:-$DATASET_ROOT/cellgene_census/processed}" \
    --gene-name-path "${PRETRAIN_GENE_NAME_PATH:-$PRETRAIN_ROOT/cellnavi/data/gene_name.txt}" \
    --nichenet-node2idx-path "${PRETRAIN_NICHENET_NODE2IDX_PATH:-$PRETRAIN_ROOT/cellnavi/data/Nichenet/node2idx.json}" \
    --pretrained-ckpt "${PRETRAIN_CKPT:-$PRETRAIN_ROOT/cellnavi/data/pretrain/pretrain_weights.pth}" \
    ${PRETRAIN_EXTRA_ARGS:-}
) 2>&1 | tee "$LOG_DIR/train.log"

echo "[pretrain] done -> $OUT_ROOT"
