#!/usr/bin/env bash
# Local single-GPU smoke runs for environment and batch-size probing.
#
# What it runs, sequentially on one visible GPU:
#   1. raw pretrain on one kidney cellgene-census shard
#   2. CoupledFM/raw Adamson with CellNavi perturb gene cache
#   3. CoupledFM/raw Adamson with scGPT perturb gene cache
#
# Typical use on a 4090/A6000-class local box:
#   cd /data2/cfy/FM/CoupledFM
#   DRY_RUN=1 bash model/tests/local_single_gpu_smoke.sh
#   GPU=0 bash model/tests/local_single_gpu_smoke.sh
#
# Useful knobs:
#   RUN_PRETRAIN=0/1 RUN_CELLNAVI=0/1 RUN_SCGPT=0/1
#   PRETRAIN_BATCH=16 PRETRAIN_MICRO=2 PRETRAIN_STEPS=10
#   COUPLED_BATCH=4 COUPLED_MICRO=1 COUPLED_EPOCHS=1

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATASET_ROOT="${SCDFM_DATASET_ROOT:-$ROOT/dataset}"
PRETRAIN_ROOT="${SCDFM_PRETRAIN_ROOT:-$ROOT/pretrainckpt}"
GENE_CACHE_ROOT="${SCDFM_GENE_CACHE_ROOT:-$PRETRAIN_ROOT/genepert_cache}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_local_single_gpu_smoke}"
OUT_ROOT="${OUT_ROOT:-$ROOT/output/local_single_gpu_smoke/$RUN_ID}"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
RUN_CELLNAVI="${RUN_CELLNAVI:-1}"
RUN_SCGPT="${RUN_SCGPT:-1}"

# One-shard pretrain probe.
KIDNEY_H5AD="${KIDNEY_H5AD:-$DATASET_ROOT/cellgene_census/processed/kidney/kidney_top6000var.h5ad}"
CELLGENE_PROCESSED_ROOT="${CELLGENE_PROCESSED_ROOT:-$DATASET_ROOT/cellgene_census/processed}"
PRETRAIN_BATCH="${PRETRAIN_BATCH:-16}"
PRETRAIN_MICRO="${PRETRAIN_MICRO:-2}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-10}"
PRETRAIN_LR="${PRETRAIN_LR:-5e-5}"
PRETRAIN_LOG_EVERY="${PRETRAIN_LOG_EVERY:-1}"
PRETRAIN_CKPT_EVERY="${PRETRAIN_CKPT_EVERY:-1000000}"
PRETRAIN_MAX_PERT_GENES="${PRETRAIN_MAX_PERT_GENES:-24}"
PRETRAIN_MIN_GENE_HIT_RATE="${PRETRAIN_MIN_GENE_HIT_RATE:-0.80}"

# Adamson coupled/raw probe.
BIFLOW_DIR="${BIFLOW_DIR:-$DATASET_ROOT/biFlow_data}"
BACKBONE="${BACKBONE:-stack}"
SPLIT_SEED="${SPLIT_SEED:-42}"
COUPLED_BATCH="${COUPLED_BATCH:-4}"
COUPLED_MICRO="${COUPLED_MICRO:-1}"
COUPLED_EPOCHS="${COUPLED_EPOCHS:-1}"
COUPLED_MAX_TRAIN_STEPS="${COUPLED_MAX_TRAIN_STEPS:-2}"
COUPLED_LR="${COUPLED_LR:-5e-5}"
COUPLED_VAL_EVERY="${COUPLED_VAL_EVERY:-50}"
COUPLED_TEST_EVERY_EPOCH="${COUPLED_TEST_EVERY_EPOCH:-1}"
COUPLED_EARLY_STOP_PATIENCE="${COUPLED_EARLY_STOP_PATIENCE:-2}"
COUPLED_VAL_ODE_STEPS="${COUPLED_VAL_ODE_STEPS:-4}"
COUPLED_EVAL_ODE_STEPS="${COUPLED_EVAL_ODE_STEPS:-4}"
COUPLED_AMP_DTYPE="${COUPLED_AMP_DTYPE:-bfloat16}"
COUPLED_MAX_PERT_GENES="${COUPLED_MAX_PERT_GENES:-16}"
COUPLED_CFG_DROP_PROB="${COUPLED_CFG_DROP_PROB:-0.0}"
COUPLED_SELECTION_METRIC="${COUPLED_SELECTION_METRIC:-pearson_delta_ctrl}"
PERT_POOL_AGGREGATIONS="${PERT_POOL_AGGREGATIONS:-mean,max,min}"
PERT_POOL_SCALE_INIT="${PERT_POOL_SCALE_INIT:-1.0,0.5,0.5}"

CELLNAVI_CACHE="${CELLNAVI_CACHE:-$GENE_CACHE_ROOT/cellnavi_embed_gene}"
SCGPT_CACHE="${SCGPT_CACHE:-$GENE_CACHE_ROOT/scgpt_embed_gene}"

is_dry_run() {
  [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[fatal] missing file: $path" >&2
    exit 2
  fi
}

run_cmd() {
  local log="$1"
  shift
  if is_dry_run; then
    printf '[dry-run] ' | tee "$log"
    printf '%q ' "$@" | tee -a "$log"
    printf '\n' | tee -a "$log"
  else
    "$@" 2>&1 | tee "$log"
  fi
}

write_kidney_metainfo() {
  local meta="$1"
  mkdir -p "$(dirname "$meta")"
  {
    echo "tissue,path"
    echo "kidney,kidney/kidney_top6000var.h5ad"
  } >"$meta"
}

echo "============================================================" | tee "$LOG_DIR/submit.log"
echo "  Local single-GPU smoke" | tee -a "$LOG_DIR/submit.log"
echo "  ROOT=$ROOT" | tee -a "$LOG_DIR/submit.log"
echo "  DATASET_ROOT=$DATASET_ROOT" | tee -a "$LOG_DIR/submit.log"
echo "  PRETRAIN_ROOT=$PRETRAIN_ROOT" | tee -a "$LOG_DIR/submit.log"
echo "  OUT_ROOT=$OUT_ROOT" | tee -a "$LOG_DIR/submit.log"
echo "  GPU=$GPU DRY_RUN=$DRY_RUN" | tee -a "$LOG_DIR/submit.log"
echo "  RUN_PRETRAIN=$RUN_PRETRAIN RUN_CELLNAVI=$RUN_CELLNAVI RUN_SCGPT=$RUN_SCGPT" | tee -a "$LOG_DIR/submit.log"
echo "============================================================" | tee -a "$LOG_DIR/submit.log"

if ! is_dry_run; then
  require_file "$KIDNEY_H5AD"
  require_file "$CELLNAVI_CACHE/gene_embeddings.npy"
  require_file "$SCGPT_CACHE/gene_embeddings.npy"
  require_file "$BIFLOW_DIR/control_${BACKBONE}/Adamson.h5ad"
  require_file "$BIFLOW_DIR/gt_${BACKBONE}/Adamson.h5ad"
fi

if is_dry_run; then
  "$PYTHON" -m model.tools.validate_resources --mode local-smoke --print-only \
    2>&1 | tee "$LOG_DIR/validate_resources.log"
else
  "$PYTHON" -m model.tools.validate_resources --mode local-smoke --datasets Adamson \
    2>&1 | tee "$LOG_DIR/validate_resources.log"
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PRETRAIN_NUM_WORKERS="${PRETRAIN_NUM_WORKERS:-0}"
export CELLNAVI_CACHE SCGPT_CACHE

echo "[check] perturb caches" | tee -a "$LOG_DIR/submit.log"
run_cmd "$LOG_DIR/check_caches.log" "$PYTHON" - <<'PY'
import os
from model.condition_emb.genepert.gene_cache import GeneEmbeddingCache
for name, expected_dim, path in [
    ("cellnavi", 256, os.environ["CELLNAVI_CACHE"]),
    ("scgpt", 512, os.environ["SCGPT_CACHE"]),
]:
    c = GeneEmbeddingCache(path)
    c.validate_index_bounds()
    print(name, "dim", c.embed_dim, "rows", c.num_embeddings, "TP53", c.lookup("TP53"))
    assert c.embed_dim == expected_dim
PY

if [[ "$RUN_PRETRAIN" == "1" ]]; then
  PRETRAIN_OUT="$OUT_ROOT/pretrain_kidney"
  PRETRAIN_META="$OUT_ROOT/pretrain_kidney_metainfo.csv"
  write_kidney_metainfo "$PRETRAIN_META"
  echo "[pretrain] kidney -> $PRETRAIN_OUT" | tee -a "$LOG_DIR/submit.log"
  run_cmd "$LOG_DIR/pretrain_kidney.log" \
    "$PYTHON" -m model.raw_pretrain.train \
      --processed-dir "$CELLGENE_PROCESSED_ROOT" \
      --tissue-metainfo-path "$PRETRAIN_META" \
      --output-dir "$PRETRAIN_OUT" \
      --epochs "$PRETRAIN_EPOCHS" \
      --steps-per-epoch "$PRETRAIN_STEPS" \
      --batch-size "$PRETRAIN_BATCH" \
      --micro-batch "$PRETRAIN_MICRO" \
      --lr "$PRETRAIN_LR" \
      --log-every-steps "$PRETRAIN_LOG_EVERY" \
      --ckpt-every-steps "$PRETRAIN_CKPT_EVERY" \
      --max-pert-genes "$PRETRAIN_MAX_PERT_GENES" \
      --min-gene-hit-rate "$PRETRAIN_MIN_GENE_HIT_RATE"
fi

SPLIT_PATH="$BIFLOW_DIR/split_seed${SPLIT_SEED}.json"
if [[ ! -f "$SPLIT_PATH" || "${FORCE_SPLIT:-0}" == "1" ]]; then
  echo "[split] build Adamson split -> $SPLIT_PATH" | tee -a "$LOG_DIR/submit.log"
  run_cmd "$LOG_DIR/build_split_adamson.log" \
    "$PYTHON" -m model.tools.build_split \
      --biflow-dir "$BIFLOW_DIR" \
      --latent-backbone "$BACKBONE" \
      --seed "$SPLIT_SEED" \
      --coupling-mode ot \
      --ot-feature raw \
      --datasets Adamson \
      --force
else
  echo "[split] reuse existing $SPLIT_PATH" | tee -a "$LOG_DIR/submit.log"
fi

run_coupled() {
  local name="$1"
  local cache="$2"
  local source="$3"
  local out="$OUT_ROOT/coupled_adamson_${name}"
  local log="$LOG_DIR/coupled_adamson_${name}.log"
  echo "[coupled] Adamson $name -> $out" | tee -a "$LOG_DIR/submit.log"
  run_cmd "$log" \
    "$PYTHON" -m model.tools.launch_stack_train \
      --variant model \
      --data-kind gene \
      --biflow-dir "$BIFLOW_DIR" \
      --latent-backbone "$BACKBONE" \
      --split-seed "$SPLIT_SEED" \
      --datasets Adamson \
      --output-dir "$out" \
      --mode ot \
      --ot-feature raw \
      --epochs "$COUPLED_EPOCHS" \
      --batch-size "$COUPLED_BATCH" \
      --micro-batch "$COUPLED_MICRO" \
      --grad-accum-steps 1 \
      --lr "$COUPLED_LR" \
      --val-every-steps "$COUPLED_VAL_EVERY" \
      --max-train-steps-per-epoch "$COUPLED_MAX_TRAIN_STEPS" \
      --test-every-epoch "$COUPLED_TEST_EVERY_EPOCH" \
      --val-ode-steps "$COUPLED_VAL_ODE_STEPS" \
      --eval-ode-steps "$COUPLED_EVAL_ODE_STEPS" \
      --early-stop-patience "$COUPLED_EARLY_STOP_PATIENCE" \
      --amp-dtype "$COUPLED_AMP_DTYPE" \
      --selection-metric "$COUPLED_SELECTION_METRIC" \
      --pert-embed-mode pretrained_frozen \
      --pert-embed-cache-dir "$cache" \
      --pert-embed-source "$source" \
      --max-pert-genes "$COUPLED_MAX_PERT_GENES" \
      --pert-pool-aggregations "$PERT_POOL_AGGREGATIONS" \
      --pert-pool-scale-init "$PERT_POOL_SCALE_INIT" \
      --cfg-drop-prob "$COUPLED_CFG_DROP_PROB" \
      --latent-z-mode interp
}

if [[ "$RUN_CELLNAVI" == "1" ]]; then
  run_coupled cellnavi "$CELLNAVI_CACHE" cellnavi_embed_gene
fi
if [[ "$RUN_SCGPT" == "1" ]]; then
  run_coupled scgpt "$SCGPT_CACHE" scgpt_embed_gene
fi

echo "============================================================" | tee -a "$LOG_DIR/submit.log"
echo "Local single-GPU smoke finished" | tee -a "$LOG_DIR/submit.log"
echo "Outputs: $OUT_ROOT" | tee -a "$LOG_DIR/submit.log"
echo "Logs: $LOG_DIR" | tee -a "$LOG_DIR/submit.log"
echo "============================================================" | tee -a "$LOG_DIR/submit.log"
