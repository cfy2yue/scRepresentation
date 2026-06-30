#!/usr/bin/env bash
# CellNavi vs scGPT raw/coupled comparison (same hyperparams, different gene embedding caches).
#
# Default (PARALLEL=0): sequential on one local 4-GPU group (COMPARE_GPUS).
# 32-GPU rack (PARALLEL=1): CellNavi on CELLNAVI_GPUS + scGPT on SCGPT_GPUS concurrently
#   (e.g. 8+8 while sweep uses GPUs 0–15).
#
# Runs the same stack raw/coupled recipe twice:
#   1. CellNavi embed_gene cache
#   2. scGPT encoder.embedding cache
#
# Usage:
#   bash model/scripts/submit_pert_embed_compare_8gpu.sh
#   DRY_RUN=1 bash model/scripts/submit_pert_embed_compare_8gpu.sh
#   COMPARE_GPUS=0,1,2,3 WAIT=1 bash model/scripts/submit_pert_embed_compare_8gpu.sh
#   PARALLEL=1 bash model/scripts/submit_pert_embed_compare_8gpu.sh
#   bash model/scripts/submit_pert_embed_compare_8gpu.sh --out-root /path/to/out --log-dir /path/to/logs

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
TORCHRUN="${TORCHRUN:-torchrun}"

CLI_RUN_ID=""
CLI_OUT_ROOT=""
CLI_LOG_DIR=""
CLI_GPUS=""

csv_field_count() {
  # Count comma-separated GPUs (no spaces expected in CSV).
  local s="${1:-}"
  [[ -z "$s" ]] && echo 0 && return
  awk -F',' '{print NF}' <<<"$s"
}

usage() {
sed -n '1,20p' "$0"
  cat <<'USAGE'

Options:
  --run-id NAME      Run id used when --out-root is not set
  --out-root PATH    Output root containing {cellnavi,scgpt}
  --log-dir PATH     Log directory (default: OUT_ROOT/logs)
  --gpus CSV         CUDA_VISIBLE_DEVICES for sequential mode (PARALLEL=0); overrides COMPARE_GPUS
  --dry-run          Print commands only (skips cache/build_split checks)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) CLI_RUN_ID="${2:?--run-id needs NAME}"; shift 2 ;;
    --out-root) CLI_OUT_ROOT="${2:?--out-root needs PATH}"; shift 2 ;;
    --log-dir) CLI_LOG_DIR="${2:?--log-dir needs PATH}"; shift 2 ;;
    --gpus) CLI_GPUS="${2:?--gpus needs CSV}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[fatal] unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

BACKBONE="${BACKBONE:-stack}"
STACK_ROOT="${STACK_ROOT:-$DATASET_ROOT/biFlow_data}"
SPLIT_SEED="${SPLIT_SEED:-42}"
COMPARE_GPUS="${COMPARE_GPUS:-0,1,2,3}"
[[ -n "$CLI_GPUS" ]] && COMPARE_GPUS="$CLI_GPUS"
NPROC="${NPROC:-$(csv_field_count "$COMPARE_GPUS")}"
DRY_RUN="${DRY_RUN:-0}"
WAIT="${WAIT:-1}"

# Optional concurrent mode: run CellNavi and scGPT on disjoint GPU groups.
PARALLEL="${PARALLEL:-0}"
CELLNAVI_GPUS="${CELLNAVI_GPUS:-0,1}"
SCGPT_GPUS="${SCGPT_GPUS:-2,3}"

if [[ "$PARALLEL" == "1" ]]; then
  NPROC_CELLNAVI="${NPROC_CELLNAVI:-$(csv_field_count "$CELLNAVI_GPUS")}"
  NPROC_SCGPT="${NPROC_SCGPT:-$(csv_field_count "$SCGPT_GPUS")}"
fi

RUN_ID="${CLI_RUN_ID:-${RUN_ID:-$(date +%Y%m%d_%H%M%S)_pert_embed_compare_8gpu}}"
OUT_ROOT="${CLI_OUT_ROOT:-${OUT_ROOT:-$ROOT/output/pert_embed_compare_8gpu/$RUN_ID}}"
LOG_DIR="${CLI_LOG_DIR:-${LOG_DIR:-$OUT_ROOT/logs}}"
mkdir -p "$LOG_DIR"

EPOCHS="${EPOCHS:-80}"
RAW_BATCH="${RAW_BATCH:-32}"
RAW_MICRO="${RAW_MICRO:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LR="${LR:-5e-5}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-500}"
TEST_EVERY_EPOCH="${TEST_EVERY_EPOCH:-1}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-4}"
VAL_ODE_STEPS="${VAL_ODE_STEPS:-10}"
EVAL_ODE_STEPS="${EVAL_ODE_STEPS:-10}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
SELECTION_METRIC="${SELECTION_METRIC:-pearson_delta_ctrl}"

PERT_POOL_AGGREGATIONS="${PERT_POOL_AGGREGATIONS:-mean,max,min}"
PERT_POOL_SCALE_INIT="${PERT_POOL_SCALE_INIT:-1.0,0.5,0.5}"
PERT_POOL_FUSION_MODE="${PERT_POOL_FUSION_MODE:-sum}"
PERT_TYPE_ADAPTER_MODE="${PERT_TYPE_ADAPTER_MODE:-scalar}"
MAX_PERT_GENES="${MAX_PERT_GENES:-16}"
CFG_DROP_PROB="${CFG_DROP_PROB:-0.0}"
LATENT_Z_MODE="${LATENT_Z_MODE:-interp}"

CELLNAVI_CACHE="${CELLNAVI_CACHE:-$GENE_CACHE_ROOT/cellnavi_embed_gene}"
SCGPT_CACHE="${SCGPT_CACHE:-$GENE_CACHE_ROOT/scgpt_embed_gene}"

GENE_DATASETS=(
  Adamson
  DixitRegev2016_K562_TFs_High_MOI
  Frangieh
  GasperiniShendure2019_lowMOI
  Jiang_IFNB
  Jiang_IFNG
  Jiang_INS
  Jiang_TGFB
  Jiang_TNFA
  Nadig_hepg2
  Nadig_jurket
  NormanWeissman2019_filtered
  Papalexi
  ReplogleWeissman2022_K562_gwps
  Replogle_RPE1essential
  Schmidt
  TianActivation
  TianInhibition
  Wessels
)

if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
  read -r -a GENE_DATASETS <<< "$DATASETS_OVERRIDE"
fi

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[fatal] missing file: $path" >&2
    exit 2
  fi
}

if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" && "$DRY_RUN" != "TRUE" ]]; then
  "$PYTHON" -m model.tools.validate_resources \
    --mode coupled \
    --datasets "${GENE_DATASETS[@]}" \
    2>&1 | tee "$LOG_DIR/validate_resources.log"
  if [[ ! -d "$STACK_ROOT" ]]; then
    echo "[fatal] STACK_ROOT not found: $STACK_ROOT" >&2
    exit 2
  fi
  require_file "$CELLNAVI_CACHE/gene_embeddings.npy"
  require_file "$CELLNAVI_CACHE/gene_index.tsv"
  require_file "$CELLNAVI_CACHE/manifest.json"
  require_file "$SCGPT_CACHE/gene_embeddings.npy"
  require_file "$SCGPT_CACHE/gene_index.tsv"
  require_file "$SCGPT_CACHE/manifest.json"
fi

echo "============================================================" | tee "$LOG_DIR/submit.log"
echo "  Pert embedding compare: CellNavi vs scGPT" | tee -a "$LOG_DIR/submit.log"
echo "  ROOT=$ROOT" | tee -a "$LOG_DIR/submit.log"
echo "  STACK_ROOT=$STACK_ROOT" | tee -a "$LOG_DIR/submit.log"
echo "  OUT_ROOT=$OUT_ROOT" | tee -a "$LOG_DIR/submit.log"
if [[ "$PARALLEL" == "1" ]]; then
  echo "  PARALLEL=1 CELLNAVI_GPUS=$CELLNAVI_GPUS NPROC_CELLNAVI=$NPROC_CELLNAVI" | tee -a "$LOG_DIR/submit.log"
  echo "              SCGPT_GPUS=$SCGPT_GPUS NPROC_SCGPT=$NPROC_SCGPT" | tee -a "$LOG_DIR/submit.log"
else
  echo "  GPUS=$COMPARE_GPUS nproc=$NPROC (sequential CellNavi then scGPT)" | tee -a "$LOG_DIR/submit.log"
fi
echo "  datasets=${GENE_DATASETS[*]}" | tee -a "$LOG_DIR/submit.log"
echo "============================================================" | tee -a "$LOG_DIR/submit.log"

if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" && "$DRY_RUN" != "TRUE" ]]; then
  SPLIT_PATH="$STACK_ROOT/split_seed${SPLIT_SEED}.json"
  if [[ ! -f "$SPLIT_PATH" || "${FORCE_SPLIT:-0}" == "1" ]]; then
    echo "[1/3] Build canonical split -> $SPLIT_PATH" | tee -a "$LOG_DIR/submit.log"
    "$PYTHON" -m model.tools.build_split \
      --biflow-dir "$STACK_ROOT" \
      --latent-backbone "$BACKBONE" \
      --seed "$SPLIT_SEED" \
      --coupling-mode coupled \
      --datasets "${GENE_DATASETS[@]}" \
      --force \
      2>&1 | tee "$LOG_DIR/build_split.log"
  else
    echo "[1/3] Reuse existing split $SPLIT_PATH (set FORCE_SPLIT=1 to rebuild)" | tee -a "$LOG_DIR/submit.log"
  fi
else
  "$PYTHON" -m model.tools.validate_resources --mode coupled --print-only \
    2>&1 | tee "$LOG_DIR/validate_resources.log"
  echo "[dry-run] skip build_split" | tee -a "$LOG_DIR/submit.log"
fi

run_one() {
  local name="$1"
  local cache="$2"
  local source="$3"
  local gpus="${4:-$COMPARE_GPUS}"
  local nproc="${5:-$NPROC}"
  local out_dir="$OUT_ROOT/$name"
  local log="$LOG_DIR/${name}.log"
  mkdir -p "$out_dir"

  local cmd=(
    "$TORCHRUN" --standalone --nproc_per_node="$nproc"
    -m model.tools.launch_stack_train
    --variant model
    --data-kind gene
    --biflow-dir "$STACK_ROOT"
    --latent-backbone "$BACKBONE"
    --split-seed "$SPLIT_SEED"
    --datasets "${GENE_DATASETS[@]}"
    --output-dir "$out_dir"
    --mode ot
    --ot-feature raw
    --epochs "$EPOCHS"
    --batch-size "$RAW_BATCH"
    --micro-batch "$RAW_MICRO"
    --grad-accum-steps "$GRAD_ACCUM_STEPS"
    --lr "$LR"
    --val-every-steps "$VAL_EVERY_STEPS"
    --test-every-epoch "$TEST_EVERY_EPOCH"
    --val-ode-steps "$VAL_ODE_STEPS"
    --eval-ode-steps "$EVAL_ODE_STEPS"
    --early-stop-patience "$EARLY_STOP_PATIENCE"
    --amp-dtype "$AMP_DTYPE"
    --selection-metric "$SELECTION_METRIC"
    --pert-embed-mode pretrained_frozen
    --pert-embed-cache-dir "$cache"
    --pert-embed-source "$source"
    --max-pert-genes "$MAX_PERT_GENES"
    --pert-pool-aggregations "$PERT_POOL_AGGREGATIONS"
    --pert-pool-scale-init "$PERT_POOL_SCALE_INIT"
    --pert-pool-fusion-mode "$PERT_POOL_FUSION_MODE"
    --pert-type-adapter-mode "$PERT_TYPE_ADAPTER_MODE"
    --cfg-drop-prob "$CFG_DROP_PROB"
    --latent-z-mode "$LATENT_Z_MODE"
  )

  {
    echo "name=$name"
    echo "cache=$cache"
    echo "source=$source"
    echo "cuda_visible_devices=$gpus"
    echo "cmd=${cmd[*]}"
  } >"$out_dir/_run_config.txt"

  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]; then
    echo "[dry-run] $name" | tee "$log"
    printf 'CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q ' "$gpus" "$ROOT:${PYTHONPATH:-}" | tee -a "$log"
    printf '%q ' "${cmd[@]}" | tee -a "$log"
    printf '\n' | tee -a "$log"
    return 0
  fi

  echo "[run] $name -> $out_dir (gpus=$gpus nproc=$nproc)" | tee -a "$LOG_DIR/submit.log"

  if [[ "${RUN_ONE_BACKGROUND:-0}" == "1" ]]; then
    (
      export CUDA_VISIBLE_DEVICES="$gpus"
      export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
      "${cmd[@]}"
    ) >"$log" 2>&1 &
    return 0
  fi
  (
    export CUDA_VISIBLE_DEVICES="$gpus"
    export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
    "${cmd[@]}"
  ) >"$log" 2>&1
}

if [[ "$PARALLEL" == "1" ]]; then
  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]; then
    cmp_mode="dry-run"
  else
    cmp_mode="live_parallel"
  fi
  echo "[2/3] CellNavi + scGPT (${cmp_mode})" | tee -a "$LOG_DIR/submit.log"
  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]; then
    run_one cellnavi "$CELLNAVI_CACHE" cellnavi_embed_gene "$CELLNAVI_GPUS" "$NPROC_CELLNAVI"
    run_one scgpt "$SCGPT_CACHE" scgpt_embed_gene "$SCGPT_GPUS" "$NPROC_SCGPT"
  else
    RUN_ONE_BACKGROUND=1 run_one cellnavi "$CELLNAVI_CACHE" cellnavi_embed_gene "$CELLNAVI_GPUS" "$NPROC_CELLNAVI"
    RUN_ONE_BACKGROUND=1 run_one scgpt "$SCGPT_CACHE" scgpt_embed_gene "$SCGPT_GPUS" "$NPROC_SCGPT"
    wait || true
  fi
else
  echo "[2/3] CellNavi cache run" | tee -a "$LOG_DIR/submit.log"
  run_one cellnavi "$CELLNAVI_CACHE" cellnavi_embed_gene

  echo "[3/3] scGPT cache run" | tee -a "$LOG_DIR/submit.log"
  run_one scgpt "$SCGPT_CACHE" scgpt_embed_gene
fi

echo "============================================================" | tee -a "$LOG_DIR/submit.log"
echo "Finished submit_pert_embed_compare_8gpu.sh" | tee -a "$LOG_DIR/submit.log"
echo "Outputs: $OUT_ROOT/{cellnavi,scgpt}" | tee -a "$LOG_DIR/submit.log"
echo "Logs: $LOG_DIR" | tee -a "$LOG_DIR/submit.log"
if [[ "$PARALLEL" == "1" ]]; then
  echo "PARALLEL=1 concurrent groups; WAIT=$WAIT ignored for training fan-out" | tee -a "$LOG_DIR/submit.log"
else
  echo "WAIT=$WAIT (sequential CellNavi then scGPT)" | tee -a "$LOG_DIR/submit.log"
fi
echo "============================================================" | tee -a "$LOG_DIR/submit.log"
