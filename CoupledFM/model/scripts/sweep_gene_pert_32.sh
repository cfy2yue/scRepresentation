#!/usr/bin/env bash
# 40-run grid: gene perturbation raw FM via stack launcher.
# Local default: 4 GPUs -> 2 concurrent jobs x 2 GPU DDP each; 40 runs -> 20 waves total.
#
# Dimensions: 5×LR × 2×max_pert_genes × 2×pool × 2×cfg_drop × latent_z_mode=interp × pearson_delta_ctrl.
#
# Usage:
#   bash model/scripts/sweep_gene_pert_32.sh
#   EPOCHS_SWEEP=20 STACK_ROOT=/path/to/biFlow bash model/scripts/sweep_gene_pert_32.sh
#   LATENT_FM_CKPT=/path/to/latent_fm.pt bash model/scripts/sweep_gene_pert_32.sh
#   bash model/scripts/sweep_gene_pert_32.sh --out-base /path/to/out --log-dir /path/to/logs
#
# To include latent-z-mode=curriculum, edit LZM_LIST / SELECTION_METRIC_LIST below; curriculum
# requires LATENT_FM_CKPT unless DRY_RUN=1.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TORCHRUN="${TORCHRUN:-torchrun}"
PYTHON="${PYTHON:-python}"
PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH}"
DATASET_ROOT="${SCDFM_DATASET_ROOT:-$ROOT/dataset}"
PRETRAIN_ROOT="${SCDFM_PRETRAIN_ROOT:-$ROOT/pretrainckpt}"
GENE_CACHE_ROOT="${SCDFM_GENE_CACHE_ROOT:-$PRETRAIN_ROOT/genepert_cache}"

CLI_OUT_BASE=""
CLI_LOG_DIR=""

usage() {
  sed -n '1,12p' "$0"
  cat <<'USAGE'

Options:
  --out-base PATH    Sweep output root containing sweep_* run dirs
  --log-dir PATH     Log directory (default: OUT_BASE/logs)
  --dry-run          Print jobs only
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-base) CLI_OUT_BASE="${2:?--out-base needs PATH}"; shift 2 ;;
    --log-dir) CLI_LOG_DIR="${2:?--log-dir needs PATH}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[fatal] unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

EPOCHS_SWEEP="${EPOCHS_SWEEP:-30}"
STACK_ROOT="${STACK_ROOT:-$DATASET_ROOT/biFlow_data}"
DRY_RUN="${DRY_RUN:-0}"
OUT_BASE="${CLI_OUT_BASE:-${OUT_BASE:-$ROOT/output/sweep_gene_pert_32_$(date +%Y%m%d_%H%M%S)}}"
LOG_DIR="${CLI_LOG_DIR:-${LOG_DIR:-$OUT_BASE/logs}}"
mkdir -p "$OUT_BASE" "$LOG_DIR"

# Same gene dataset list as submit_stack_gene_training.sh (gene mode).
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

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]; then
  "$PYTHON" -m model.tools.validate_resources --mode coupled --print-only \
    2>&1 | tee "$LOG_DIR/validate_resources.log"
else
  "$PYTHON" -m model.tools.validate_resources --mode coupled --datasets "${GENE_DATASETS[@]}" \
    2>&1 | tee "$LOG_DIR/validate_resources.log"
fi

# 5 × 2 × 2 × 2 = 40 (fixed interp + pearson_delta_ctrl; see header to extend grid).
LR_LIST=(3e-5 5e-5 1e-4 2e-4 3e-4)
MPG_LIST=(8 16)
POOL_MEAN_TAG="mean"
POOL_MULTI_TAG="mean_max_min"
CFG_LIST=(0.0 0.1)
LZM_LIST=(interp)
SELECTION_METRIC_LIST=(pearson_delta_ctrl)

# Space-separated CSV groups. Override for a larger machine, for example:
#   SWEEP_SLOT_GPUS_CSV="0,1 2,3 4,5 6,7"
read -r -a SLOT_GPUS <<< "${SWEEP_SLOT_GPUS_CSV:-0,1 2,3}"
N_SLOTS="${#SLOT_GPUS[@]}"
if [[ "$N_SLOTS" -lt 1 ]]; then
  echo "[fatal] no GPU slots configured in SWEEP_SLOT_GPUS_CSV" >&2
  exit 2
fi
SLOT_PORT_BASE="${SLOT_PORT_BASE:-29500}"
SLOT_PORTS=()
for s in $(seq 0 $((N_SLOTS - 1))); do
  SLOT_PORTS+=("$((SLOT_PORT_BASE + s))")
done
declare -A SLOT_PID=()

launch_one() {
  local slot=$1 idx=$2 lr=$3 mpg=$4 pool_tag=$5 cfg=$6 lzm=$7 selection_metric=$8
  local pool_cli scale_cli stack_mode="ot"
  local latent_extra=()
  local pert_extra=()
  if [[ "$lzm" == "curriculum" ]]; then
    stack_mode="coupled"
    if [[ -z "${LATENT_FM_CKPT:-}" && "$DRY_RUN" != "1" && "$DRY_RUN" != "true" && "$DRY_RUN" != "TRUE" ]]; then
      echo "[sweep_gene_pert_32] LATENT_FM_CKPT must be set for latent-z-mode=curriculum" >&2
      exit 1
    fi
    if [[ -n "${LATENT_FM_CKPT:-}" ]]; then
      latent_extra=(--latent-fm-ckpt "$LATENT_FM_CKPT")
    fi
  fi
  if [[ -n "${PERT_EMBED_CACHE_DIR:-}" ]]; then
    pert_extra+=(--pert-embed-cache-dir "$PERT_EMBED_CACHE_DIR")
  fi
  if [[ -n "${PERT_EMBED_SOURCE:-}" ]]; then
    pert_extra+=(--pert-embed-source "$PERT_EMBED_SOURCE")
  fi
  if [[ "$pool_tag" == "$POOL_MEAN_TAG" ]]; then
    pool_cli="mean"
    scale_cli="1.0"
  else
    pool_cli="mean,max,min"
    scale_cli="1.0,0.5,0.5"
  fi
  local tag="lr${lr}_mpg${mpg}_pool${pool_tag}_cfg${cfg}_lz${lzm}_sel${selection_metric}"
  local run_id
  run_id="$(printf 'sweep_%02d_%s' "$idx" "$tag")"
  local out_dir="$OUT_BASE/$run_id"
  local log="$LOG_DIR/$run_id.log"
  mkdir -p "$out_dir"
  {
    echo "run_id=$run_id"
    echo "slot=$slot"
    echo "cuda_visible_devices=${SLOT_GPUS[$slot]}"
    echo "lr=$lr"
    echo "max_pert_genes=$mpg"
    echo "pool=$pool_cli"
    echo "pool_scale=$scale_cli"
    echo "cfg_drop_prob=$cfg"
    echo "latent_z_mode=$lzm"
    echo "selection_metric=$selection_metric"
    echo "datasets=${GENE_DATASETS[*]}"
  } >"$out_dir/_grid_line.txt"
  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]; then
    echo "[sweep_gene_pert_32] DRY_RUN run_id=$run_id slot=$slot gpus=${SLOT_GPUS[$slot]}" | tee "$log"
    return 0
  fi
  (
    export CUDA_VISIBLE_DEVICES="${SLOT_GPUS[$slot]}"
    export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
    "$TORCHRUN" --standalone --nproc_per_node=2 \
      --rdzv-endpoint="localhost:${SLOT_PORTS[$slot]}" \
      -m model.tools.launch_stack_train \
      --variant model \
      --data-kind gene \
      --biflow-dir "$STACK_ROOT" \
      --latent-backbone stack \
      --split-seed "${SPLIT_SEED:-42}" \
      --datasets "${GENE_DATASETS[@]}" \
      --output-dir "$out_dir" \
      --mode "$stack_mode" \
      --ot-feature raw \
      --epochs "$EPOCHS_SWEEP" \
      --batch-size "${RAW_BATCH:-32}" \
      --micro-batch "${RAW_MICRO:-8}" \
      --grad-accum-steps "${GRAD_ACCUM_STEPS:-1}" \
      --lr "$lr" \
      --pert-embed-mode pretrained_frozen \
      --max-pert-genes "$mpg" \
      --pert-pool-aggregations "$pool_cli" \
      --pert-pool-scale-init "$scale_cli" \
      --cfg-drop-prob "$cfg" \
      --latent-z-mode "$lzm" \
      --selection-metric "$selection_metric" \
      --val-every-steps "${VAL_EVERY_STEPS:-500}" \
      --test-every-epoch "${TEST_EVERY_EPOCH:-1}" \
      --early-stop-patience "${EARLY_STOP_PATIENCE:-4}" \
      --amp-dtype "${AMP_DTYPE:-bfloat16}" \
      "${pert_extra[@]}" \
      "${latent_extra[@]}"
  ) >"$log" 2>&1 &
  SLOT_PID[$slot]=$!
}

slot_idle() {
  local s=$1 pid
  pid="${SLOT_PID[$s]:-}"
  [[ -z "$pid" ]] && return 0
  ! kill -0 "$pid" 2>/dev/null
}

idx=0
for lr in "${LR_LIST[@]}"; do
  for mpg in "${MPG_LIST[@]}"; do
    for pool_tag in "$POOL_MEAN_TAG" "$POOL_MULTI_TAG"; do
      for cfg in "${CFG_LIST[@]}"; do
        for lzm in "${LZM_LIST[@]}"; do
          for selection_metric in "${SELECTION_METRIC_LIST[@]}"; do
            while true; do
              placed=false
              if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]; then
                s=$((idx % N_SLOTS))
                launch_one "$s" "$idx" "$lr" "$mpg" "$pool_tag" "$cfg" "$lzm" "$selection_metric"
                idx=$((idx + 1))
                placed=true
                break
              fi
              for s in $(seq 0 $((N_SLOTS - 1))); do
                if slot_idle "$s"; then
                  launch_one "$s" "$idx" "$lr" "$mpg" "$pool_tag" "$cfg" "$lzm" "$selection_metric"
                  idx=$((idx + 1))
                  placed=true
                  break
                fi
              done
              if [[ "$placed" == true ]]; then
                break
              fi
              wait -n
            done
          done
        done
      done
    done
  done
done

wait || true
echo "[sweep_gene_pert_32] finished ${idx} jobs → $OUT_BASE"
