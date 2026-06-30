#!/usr/bin/env bash
# Submit stack-backbone training jobs.
#
# Defaults:
#   - DATA_KIND=gene (no drug datasets, no chemical condition slots)
#   - latent: single DDP job on LATENT_GPUS (default 1,2) via torchrun, one model under OUT_ROOT/latent/
#   - coupled (raw): DDP on COUPLED_GPUS (default 3,4,5,6), mode=ot, ot_feature=raw
#   - coupled_independent: OFF unless RUN_INDEPENDENT=1 (uses INDEPENDENT_GPUS, default 5,6)
#   - logs and checkpoints under output/stack_runs/<RUN_ID>/
#
# Usage:
#   bash scripts/submit_stack_gene_training.sh
#   RUN_INDEPENDENT=1 INDEPENDENT_GPUS=7,8 bash scripts/submit_stack_gene_training.sh
#   DATA_KIND=drug bash scripts/submit_stack_gene_training.sh
#   bash model/scripts/submit_stack_gene_training.sh --out-root /path/to/run --log-dir /path/to/logs

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
CLI_PREP_DIR=""

usage() {
  sed -n '1,14p' "$0"
  cat <<'USAGE'

Options:
  --run-id NAME      Run id used when --out-root is not set
  --out-root PATH    Output root for checkpoints/artifacts
  --log-dir PATH     Log directory (default: OUT_ROOT/logs)
  --prep-dir PATH    Prepared latent HDF5 directory (default: OUT_ROOT/prepared_latent)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) CLI_RUN_ID="${2:?--run-id needs NAME}"; shift 2 ;;
    --out-root) CLI_OUT_ROOT="${2:?--out-root needs PATH}"; shift 2 ;;
    --log-dir) CLI_LOG_DIR="${2:?--log-dir needs PATH}"; shift 2 ;;
    --prep-dir) CLI_PREP_DIR="${2:?--prep-dir needs PATH}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[fatal] unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

DATA_KIND="${DATA_KIND:-gene}"      # gene | drug | all
BACKBONE="${BACKBONE:-stack}"
STACK_ROOT="${STACK_ROOT:-$DATASET_ROOT/biFlow_data}"
SPLIT_SEED="${SPLIT_SEED:-42}"

RUN_ID="${CLI_RUN_ID:-${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${BACKBONE}_${DATA_KIND}}}"
OUT_ROOT="${CLI_OUT_ROOT:-${OUT_ROOT:-$ROOT/output/stack_runs/$RUN_ID}}"
LOG_DIR="${CLI_LOG_DIR:-${LOG_DIR:-$OUT_ROOT/logs}}"
PREP_DIR="${CLI_PREP_DIR:-${PREP_DIR:-$OUT_ROOT/prepared_latent}}"
mkdir -p "$LOG_DIR" "$PREP_DIR"

EPOCHS="${EPOCHS:-80}"
LATENT_STEPS="${LATENT_STEPS:-200000}"
LATENT_BATCH="${LATENT_BATCH:-128}"
RAW_BATCH="${RAW_BATCH:-32}"
RAW_MICRO="${RAW_MICRO:-8}"
LR="${LR:-5e-5}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-500}"
TEST_EVERY_EPOCH="${TEST_EVERY_EPOCH:-1}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-4}"
VAL_ODE_STEPS="${VAL_ODE_STEPS:-10}"
EVAL_ODE_STEPS="${EVAL_ODE_STEPS:-10}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"

# Latent FM γ schedule (kept reasonable relative to LATENT_STEPS).
# Override these for short Adamson latent runs via LATENT_GAMMA / LATENT_* env vars in this script or your shell.
LATENT_GAMMA="${LATENT_GAMMA:-0.03}"
LATENT_GAMMA_WARMUP_START="${LATENT_GAMMA_WARMUP_START:-5000}"
LATENT_GAMMA_WARMUP_END="${LATENT_GAMMA_WARMUP_END:-50000}"
LATENT_LR="${LATENT_LR:-1e-4}"
LATENT_PATIENCE="${LATENT_PATIENCE:-10}"
LATENT_SELECTION_METRIC="${LATENT_SELECTION_METRIC:-test_mse}"
LATENT_DS_ALPHA="${LATENT_DS_ALPHA:-1.0}"
LATENT_TIME_SAMPLING="${LATENT_TIME_SAMPLING:-uniform}"
LATENT_MMD_ODE_STEPS="${LATENT_MMD_ODE_STEPS:-10}"
LATENT_MMD_ESTIMATOR="${LATENT_MMD_ESTIMATOR:-biased}"
LATENT_USE_PARAM_GROUPS="${LATENT_USE_PARAM_GROUPS:-1}"
LATENT_LR_NEW_MODULE_MULT="${LATENT_LR_NEW_MODULE_MULT:-3.0}"
LATENT_WEIGHT_DECAY_BACKBONE="${LATENT_WEIGHT_DECAY_BACKBONE:-0.001}"
LATENT_WEIGHT_DECAY_NEW="${LATENT_WEIGHT_DECAY_NEW:-0.01}"

# v3 ablation knobs (latent only); empty defaults => use train.py / config.py defaults
LATENT_PERT_TO_C_INIT_MODE="${LATENT_PERT_TO_C_INIT_MODE:-}"   # zero | xavier_small
LATENT_USE_PERT_IN_FUSION="${LATENT_USE_PERT_IN_FUSION:-}"     # 1 | 0
LATENT_D_MODEL="${LATENT_D_MODEL:-}"                           # mlp_d_model
LATENT_PERT_COND_DIM="${LATENT_PERT_COND_DIM:-}"               # pert_cond_dim
LATENT_N_LAYERS="${LATENT_N_LAYERS:-}"                         # mlp_n_layers
LATENT_LR_NEW_MODULE_MULT_OVERRIDE="${LATENT_LR_NEW_MODULE_MULT_OVERRIDE:-}"  # if set, overrides LR_NEW_MODULE_MULT

# Raw/coupled MMD + selection knobs (Adamson runs may override).
RAW_MMD_GAMMA_MAX="${RAW_MMD_GAMMA_MAX:-}"
RAW_MMD_EVERY="${RAW_MMD_EVERY:-}"
RAW_MMD_EPOCH_START="${RAW_MMD_EPOCH_START:-}"
RAW_MMD_MICRO_CHUNK="${RAW_MMD_MICRO_CHUNK:-}"
RAW_SELECTION_METRIC="${RAW_SELECTION_METRIC:-}"

# Raw/coupled audit-fix knobs (Adamson runs may override; empty = use train.py defaults).
# Optimizer param groups
RAW_USE_PARAM_GROUPS="${RAW_USE_PARAM_GROUPS:-}"
RAW_LR_NEW_MODULE_MULT="${RAW_LR_NEW_MODULE_MULT:-}"
RAW_WEIGHT_DECAY_BACKBONE="${RAW_WEIGHT_DECAY_BACKBONE:-}"
RAW_WEIGHT_DECAY_NEW="${RAW_WEIGHT_DECAY_NEW:-}"
# Conditioning channel + sampling
RAW_USE_PERT_TOKEN="${RAW_USE_PERT_TOKEN:-}"
RAW_PERT_IDX_MODE="${RAW_PERT_IDX_MODE:-}"
RAW_DS_ALPHA="${RAW_DS_ALPHA:-}"
RAW_CFG_DROP_PROB="${RAW_CFG_DROP_PROB:-}"
# Flow-matching time / loss weighting
RAW_TIME_SAMPLING="${RAW_TIME_SAMPLING:-}"
RAW_LOSS_WEIGHTING="${RAW_LOSS_WEIGHTING:-}"
RAW_MIN_SNR_GAMMA="${RAW_MIN_SNR_GAMMA:-}"

RUN_LATENT="${RUN_LATENT:-1}"
RUN_COUPLED="${RUN_COUPLED:-1}"
RUN_INDEPENDENT="${RUN_INDEPENDENT:-0}"
PREPARE_LATENT="${PREPARE_LATENT:-1}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
N_PREP_WORKERS="${N_PREP_WORKERS:-1}"

LATENT_GPUS="${LATENT_GPUS:-1,2}"
COUPLED_GPUS="${COUPLED_GPUS:-3,4,5,6}"
INDEPENDENT_GPUS="${INDEPENDENT_GPUS:-5,6}"

# When both coupled and independent run, set RAW_PARALLEL=1 to launch both at once.
RAW_PARALLEL="${RAW_PARALLEL:-1}"

num_csv() {
  awk -F',' '{print NF}' <<< "$1"
}

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

DRUG_DATASETS=(
  sciplex3_A549
  sciplex3_K562
  sciplex3_MCF7
)

case "$DATA_KIND" in
  gene)
    DATASETS=("${GENE_DATASETS[@]}")
    ;;
  drug)
    DATASETS=("${DRUG_DATASETS[@]}")
    ;;
  all)
    DATASETS=("${GENE_DATASETS[@]}" "${DRUG_DATASETS[@]}")
    ;;
  *)
    echo "[fatal] DATA_KIND must be gene | drug | all, got: $DATA_KIND" >&2
    exit 2
    ;;
esac

if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
  read -r -a DATASETS <<< "$DATASETS_OVERRIDE"
fi

if [[ ! -d "$STACK_ROOT" ]]; then
  echo "[fatal] STACK_ROOT not found: $STACK_ROOT" >&2
  exit 2
fi

"$PYTHON" -m model.tools.validate_resources \
  --mode coupled \
  --datasets "${DATASETS[@]}" \
  2>&1 | tee "$LOG_DIR/validate_resources.log"

echo "============================================================" | tee "$LOG_DIR/submit.log"
echo "  Stack training submit" | tee -a "$LOG_DIR/submit.log"
echo "  ROOT=$ROOT" | tee -a "$LOG_DIR/submit.log"
echo "  DATA_KIND=$DATA_KIND" | tee -a "$LOG_DIR/submit.log"
echo "  STACK_ROOT=$STACK_ROOT" | tee -a "$LOG_DIR/submit.log"
echo "  OUT_ROOT=$OUT_ROOT" | tee -a "$LOG_DIR/submit.log"
echo "  DATASETS=${DATASETS[*]}" | tee -a "$LOG_DIR/submit.log"
echo "  latent DDP GPUs=$LATENT_GPUS (n=$(num_csv "$LATENT_GPUS"))" | tee -a "$LOG_DIR/submit.log"
echo "  coupled DDP GPUs=$COUPLED_GPUS (n=$(num_csv "$COUPLED_GPUS"))" | tee -a "$LOG_DIR/submit.log"
echo "  RUN_INDEPENDENT=$RUN_INDEPENDENT  INDEPENDENT_GPUS=$INDEPENDENT_GPUS" | tee -a "$LOG_DIR/submit.log"
echo "============================================================" | tee -a "$LOG_DIR/submit.log"

echo "[1/4] Build/rebuild shared canonical split" | tee -a "$LOG_DIR/submit.log"
"$PYTHON" -m model.tools.build_split \
  --biflow-dir "$STACK_ROOT" \
  --latent-backbone "$BACKBONE" \
  --seed "$SPLIT_SEED" \
  --coupling-mode coupled \
  --datasets "${DATASETS[@]}" \
  --force \
  2>&1 | tee "$LOG_DIR/build_split.log"

LATENT_ALL="$PREP_DIR/latent_all"

prepare_latent_all() {
  mkdir -p "$LATENT_ALL"
  if [[ "$FORCE_PREPARE" == "1" ]]; then
    rm -f "$LATENT_ALL"/*.h5 "$LATENT_ALL"/manifest.json
  fi
  if [[ "$PREPARE_LATENT" != "1" ]]; then
    echo "[latent] PREPARE_LATENT=0, expect $LATENT_ALL/manifest.json" | tee "$LOG_DIR/prepare_latent_all.log"
    return 0
  fi
  echo "[2/4] Prepare latent HDF5 (all datasets) -> $LATENT_ALL" | tee "$LOG_DIR/prepare_latent_all.log"
  COUPLEDFM_BIFLOW_CTRL="$STACK_ROOT/control_${BACKBONE}" \
  COUPLEDFM_BIFLOW_GT="$STACK_ROOT/gt_${BACKBONE}" \
  COUPLEDFM_FM_DATA="$LATENT_ALL" \
  "$PYTHON" model/latent/prepare_fm_data.py \
    --datasets "${DATASETS[@]}" \
    --n-workers "$N_PREP_WORKERS" \
    2>&1 | tee -a "$LOG_DIR/prepare_latent_all.log"
}

run_latent_ddp() {
  local gpus="$1"
  local nproc
  nproc="$(num_csv "$gpus")"
  local log="$LOG_DIR/latent_ddp.log"
  echo "[3/4] Launch latent on GPUs $gpus (nproc=$nproc)" | tee -a "$LOG_DIR/submit.log" >&2
  # Single-GPU: skip torchrun (it triggers concurrent CUDA-init races when many
  # standalone instances are spawned at once). Use plain python.
  local launcher
  if [[ "$nproc" -eq 1 ]]; then
    launcher=("$PYTHON")
  else
    launcher=("$TORCHRUN" --standalone --nproc_per_node="$nproc")
  fi
  (
    export CUDA_VISIBLE_DEVICES="$gpus"
    export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
    "${launcher[@]}" -m model.latent.train \
      --data-dir "$LATENT_ALL" \
      --biflow-dir "$STACK_ROOT" \
      --latent-backbone "$BACKBONE" \
      --split-file "$STACK_ROOT/split_seed${SPLIT_SEED}.json" \
      --gpu 0 \
      --save-dir "$OUT_ROOT/latent" \
      --model-type control_mlp \
      --emb-dim 1600 \
      --total-steps "$LATENT_STEPS" \
      --batch-size "$LATENT_BATCH" \
      --lr "$LATENT_LR" \
      --gamma "$LATENT_GAMMA" \
      --gamma-warmup-start "$LATENT_GAMMA_WARMUP_START" \
      --gamma-warmup-end "$LATENT_GAMMA_WARMUP_END" \
      --patience "$LATENT_PATIENCE" \
      --selection-metric "$LATENT_SELECTION_METRIC" \
      --ds-alpha "$LATENT_DS_ALPHA" \
      --time-sampling "$LATENT_TIME_SAMPLING" \
      --mmd-ode-steps "$LATENT_MMD_ODE_STEPS" \
      --mmd-estimator "$LATENT_MMD_ESTIMATOR" \
      --lr-new-module-mult "${LATENT_LR_NEW_MODULE_MULT_OVERRIDE:-$LATENT_LR_NEW_MODULE_MULT}" \
      --weight-decay-backbone "$LATENT_WEIGHT_DECAY_BACKBONE" \
      --weight-decay-new "$LATENT_WEIGHT_DECAY_NEW" \
      $(case "$LATENT_USE_PARAM_GROUPS" in 1|true|TRUE|yes|YES) echo -n "--use-param-groups" ;; esac) \
      $([[ -n "$LATENT_PERT_TO_C_INIT_MODE" ]] && echo -n "--pert-to-c-init-mode $LATENT_PERT_TO_C_INIT_MODE") \
      $(case "$LATENT_USE_PERT_IN_FUSION" in 1|true|TRUE|yes|YES) echo -n "--use-pert-in-fusion" ;; esac) \
      $([[ -n "$LATENT_D_MODEL"      ]] && echo -n "--mlp-d-model $LATENT_D_MODEL") \
      $([[ -n "$LATENT_PERT_COND_DIM" ]] && echo -n "--pert-cond-dim $LATENT_PERT_COND_DIM") \
      $([[ -n "$LATENT_N_LAYERS"     ]] && echo -n "--mlp-n-layers $LATENT_N_LAYERS") \
      --use-pert-condition \
      --pert-embed-mode pretrained_frozen \
      --use-h5ad-pert-metadata \
      --pert-pool-aggregations mean max min \
      --pert-pool-scale-init 1.0 0.5 0.5 \
      $([[ -n "${PERT_POOL_FUSION_MODE:-}" ]] && echo --pert-pool-fusion-mode "$PERT_POOL_FUSION_MODE") \
      $([[ -n "${PERT_TYPE_ADAPTER_MODE:-}" ]] && echo --pert-type-adapter-mode "$PERT_TYPE_ADAPTER_MODE") \
      $([[ -n "${PERT_EMBED_CACHE_DIR:-}" ]] && echo --pert-gene-emb-cache-dir "$PERT_EMBED_CACHE_DIR") \
      $([[ -n "${PERT_EMBED_SOURCE:-}" ]] && echo --pert-embed-source "$PERT_EMBED_SOURCE") \
      $( [[ "$DATA_KIND" == "drug" || "$DATA_KIND" == "all" ]] && echo "--pert-chem-enabled" || echo "--no-pert-chem-enabled" )
  ) >"$log" 2>&1 &
  LAUNCHED_PID=$!
}

run_coupled_job() {
  local variant="$1"
  local gpus="$2"
  local log="$LOG_DIR/${variant}.log"
  local nproc
  nproc="$(num_csv "$gpus")"
  echo "[4/4] Launch $variant on physical GPUs $gpus (nproc=$nproc)" | tee -a "$LOG_DIR/submit.log" >&2
  (
    export CUDA_VISIBLE_DEVICES="$gpus"
    export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
    local extra=()
    [[ -n "$RAW_MMD_GAMMA_MAX"     ]] && extra+=(--mmd-gamma-max     "$RAW_MMD_GAMMA_MAX")
    [[ -n "$RAW_MMD_EVERY"         ]] && extra+=(--mmd-every         "$RAW_MMD_EVERY")
    [[ -n "$RAW_MMD_EPOCH_START"   ]] && extra+=(--mmd-epoch-start   "$RAW_MMD_EPOCH_START")
    [[ -n "$RAW_MMD_MICRO_CHUNK"   ]] && extra+=(--mmd-micro-chunk   "$RAW_MMD_MICRO_CHUNK")
    [[ -n "$RAW_SELECTION_METRIC"  ]] && extra+=(--selection-metric  "$RAW_SELECTION_METRIC")

    # Optimizer param-group fixes
    case "$RAW_USE_PARAM_GROUPS" in
      1|true|TRUE|yes|YES) extra+=(--use-param-groups) ;;
      0|false|FALSE|no|NO) extra+=(--no-param-groups) ;;
    esac
    [[ -n "$RAW_LR_NEW_MODULE_MULT"    ]] && extra+=(--lr-new-module-mult    "$RAW_LR_NEW_MODULE_MULT")
    [[ -n "$RAW_WEIGHT_DECAY_BACKBONE" ]] && extra+=(--weight-decay-backbone "$RAW_WEIGHT_DECAY_BACKBONE")
    [[ -n "$RAW_WEIGHT_DECAY_NEW"      ]] && extra+=(--weight-decay-new      "$RAW_WEIGHT_DECAY_NEW")

    # Conditioning channel + sampling fixes
    case "$RAW_USE_PERT_TOKEN" in
      1|true|TRUE|yes|YES) extra+=(--use-pert-token) ;;
      0|false|FALSE|no|NO) extra+=(--no-pert-token) ;;
    esac
    [[ -n "$RAW_PERT_IDX_MODE"  ]] && extra+=(--pert-idx-mode  "$RAW_PERT_IDX_MODE")
    [[ -n "$RAW_DS_ALPHA"       ]] && extra+=(--ds-alpha       "$RAW_DS_ALPHA")
    [[ -n "$RAW_CFG_DROP_PROB"  ]] && extra+=(--cfg-drop-prob  "$RAW_CFG_DROP_PROB")

    # Flow-matching time/loss weighting
    [[ -n "$RAW_TIME_SAMPLING"  ]] && extra+=(--time-sampling  "$RAW_TIME_SAMPLING")
    [[ -n "$RAW_LOSS_WEIGHTING" ]] && extra+=(--loss-weighting "$RAW_LOSS_WEIGHTING")
    [[ -n "$RAW_MIN_SNR_GAMMA"  ]] && extra+=(--min-snr-gamma  "$RAW_MIN_SNR_GAMMA")

    RAW_EXTRA_PERT_ARGS=()
    [[ -n "${PERT_POOL_FUSION_MODE:-}" ]] && RAW_EXTRA_PERT_ARGS+=(--pert-pool-fusion-mode "$PERT_POOL_FUSION_MODE")
    [[ -n "${PERT_TYPE_ADAPTER_MODE:-}" ]] && RAW_EXTRA_PERT_ARGS+=(--pert-type-adapter-mode "$PERT_TYPE_ADAPTER_MODE")
    [[ -n "${PERT_EMBED_CACHE_DIR:-}" ]] && RAW_EXTRA_PERT_ARGS+=(--pert-embed-cache-dir "$PERT_EMBED_CACHE_DIR")
    [[ -n "${PERT_EMBED_SOURCE:-}" ]] && RAW_EXTRA_PERT_ARGS+=(--pert-embed-source "$PERT_EMBED_SOURCE")

    "$TORCHRUN" --standalone --nproc_per_node="$nproc" -m model.tools.launch_stack_train \
      --variant "$variant" \
      --data-kind "$DATA_KIND" \
      --biflow-dir "$STACK_ROOT" \
      --latent-backbone "$BACKBONE" \
      --split-seed "$SPLIT_SEED" \
      --datasets "${DATASETS[@]}" \
      --output-dir "$OUT_ROOT/$variant" \
      --mode ot \
      --ot-feature raw \
      --epochs "$EPOCHS" \
      --batch-size "$RAW_BATCH" \
      --micro-batch "$RAW_MICRO" \
      --grad-accum-steps "$GRAD_ACCUM_STEPS" \
      --lr "$LR" \
      --val-every-steps "$VAL_EVERY_STEPS" \
      --test-every-epoch "$TEST_EVERY_EPOCH" \
      --val-ode-steps "$VAL_ODE_STEPS" \
      --eval-ode-steps "$EVAL_ODE_STEPS" \
      --early-stop-patience "$EARLY_STOP_PATIENCE" \
      --amp-dtype "$AMP_DTYPE" \
      "${extra[@]}" \
      "${RAW_EXTRA_PERT_ARGS[@]}"
  ) >"$log" 2>&1 &
  LAUNCHED_PID=$!
}

LATENT_PIDS=()
if [[ "$RUN_LATENT" == "1" ]]; then
  prepare_latent_all
  run_latent_ddp "$LATENT_GPUS"
  LATENT_PIDS+=("$LAUNCHED_PID")
fi

RAW_PIDS=()
if [[ "$RAW_PARALLEL" == "1" ]]; then
  if [[ "$RUN_COUPLED" == "1" ]]; then
    run_coupled_job coupled "$COUPLED_GPUS"
    RAW_PIDS+=("$LAUNCHED_PID")
  fi
  if [[ "$RUN_INDEPENDENT" == "1" ]]; then
    run_coupled_job coupled_independent "$INDEPENDENT_GPUS"
    RAW_PIDS+=("$LAUNCHED_PID")
  fi
else
  (
    set -euo pipefail
    if [[ "$RUN_COUPLED" == "1" ]]; then
      run_coupled_job coupled "$COUPLED_GPUS"
      wait "$LAUNCHED_PID"
    fi
    if [[ "$RUN_INDEPENDENT" == "1" ]]; then
      run_coupled_job coupled_independent "$INDEPENDENT_GPUS"
      wait "$LAUNCHED_PID"
    fi
  ) &
  RAW_PIDS+=("$!")
fi

echo "============================================================" | tee -a "$LOG_DIR/submit.log"
echo "Submitted jobs:" | tee -a "$LOG_DIR/submit.log"
echo "  LATENT_PIDS=${LATENT_PIDS[*]:-(none)}" | tee -a "$LOG_DIR/submit.log"
echo "  RAW_PIDS=${RAW_PIDS[*]:-(none)}  (RAW_PARALLEL=$RAW_PARALLEL)" | tee -a "$LOG_DIR/submit.log"
echo "Logs:" | tee -a "$LOG_DIR/submit.log"
ls -1 "$LOG_DIR" | sed "s#^#  $LOG_DIR/#" | tee -a "$LOG_DIR/submit.log"
echo "============================================================" | tee -a "$LOG_DIR/submit.log"

if [[ "${WAIT:-0}" == "1" ]]; then
  status=0
  for pid in "${LATENT_PIDS[@]}"; do
    wait "$pid" || status=$?
  done
  for pid in "${RAW_PIDS[@]}"; do
    wait "$pid" || status=$?
  done
  exit "$status"
fi
