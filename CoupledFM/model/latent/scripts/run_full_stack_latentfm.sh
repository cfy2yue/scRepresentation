#!/usr/bin/env bash
set -euo pipefail

# Full-data LatentFM launcher for Stack latent embeddings.
#
# Data source is /data/cyx/1030/dataset/latentfm_full/stack, built from
# /data/cyx/1030/dataset/biFlow_data/control_stack + gt_stack.  This script is
# intentionally separate from the scFMBench/benchmark launcher so formal
# LatentFM training does not accidentally use the small benchmark staging set.

ROOT="/data/cyx/1030/scLatent"
COUPLEDFM="${ROOT}/CoupledFM"
LATENT_BACKBONE="${LATENT_BACKBONE:-stack}"
DATA_DIR="${DATA_DIR:-${ROOT}/dataset/latentfm_full/${LATENT_BACKBONE}}"
BIFLOW_DIR="${ROOT}/dataset/biFlow_data"
SPLIT_FILE="${SPLIT_FILE:-}"
PERT_MEANS_FILE="${PERT_MEANS_FILE:-}"
OUT_ROOT="${OUT_ROOT:-${COUPLEDFM}/output/latentfm_runs/full_${LATENT_BACKBONE}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/latentfm_full_train}"
GENE_CACHE="${GENE_CACHE:-${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene}"

if [[ -f "${ROOT}/init-scdfm.sh" ]]; then
  # Make the launcher self-contained for tmux/nohup/background execution.
  # shellcheck disable=SC1091
  source "${ROOT}/init-scdfm.sh" >/dev/null
fi
PYTHON_BIN="${PYTHON_BIN:-${SCDFM_CONDA_ENV:-/data/cyx/software/miniconda3/envs/scdfm}/bin/python}"

GPU="${GPU:-0}"
TOTAL_STEPS="${TOTAL_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
LR="${LR:-1e-4}"
USE_PARAM_GROUPS="${USE_PARAM_GROUPS:-0}"
LR_NEW_MODULE_MULT="${LR_NEW_MODULE_MULT:-3.0}"
GAMMA="${GAMMA:-0.03}"
GAMMA_WARMUP_START="${GAMMA_WARMUP_START:-500}"
GAMMA_WARMUP_END="${GAMMA_WARMUP_END:-2500}"
MMD_EVERY="${MMD_EVERY:-4}"
MMD_ESTIMATOR="${MMD_ESTIMATOR:-unbiased}"
MMD_ODE_STEPS="${MMD_ODE_STEPS:-0}"
MMD_DATASET_FILTER="${MMD_DATASET_FILTER:-}"
RISK_ROW_CVAR_LOSS_WEIGHT="${RISK_ROW_CVAR_LOSS_WEIGHT:-0.0}"
RISK_ROW_CVAR_LOSS_WARMUP_START="${RISK_ROW_CVAR_LOSS_WARMUP_START:-0}"
RISK_ROW_CVAR_LOSS_WARMUP_END="${RISK_ROW_CVAR_LOSS_WARMUP_END:-0}"
RISK_ROW_CVAR_DATASET_FILTER="${RISK_ROW_CVAR_DATASET_FILTER:-}"
RISK_ROW_CVAR_HISTORY_SIZE="${RISK_ROW_CVAR_HISTORY_SIZE:-256}"
RISK_ROW_CVAR_MIN_HISTORY="${RISK_ROW_CVAR_MIN_HISTORY:-8}"
RISK_ROW_CVAR_TOP_FRAC="${RISK_ROW_CVAR_TOP_FRAC:-0.20}"
RISK_ROW_CVAR_MMD_THRESHOLD="${RISK_ROW_CVAR_MMD_THRESHOLD:-0.005}"
OT_PAIR_MODE="${OT_PAIR_MODE:-multinomial}"
OT_THREADS="${OT_THREADS:-4}"
PREFETCH="${PREFETCH:-8}"
N_OT_WORKERS="${N_OT_WORKERS:-6}"
SELECTION_METRIC="${SELECTION_METRIC:-test_mmd}"
SELECTION_MMD_LAMBDA="${SELECTION_MMD_LAMBDA:-1.0}"
PERT_TO_C_INIT_MODE="${PERT_TO_C_INIT_MODE:-xavier_small}"
USE_PERT_IN_FUSION="${USE_PERT_IN_FUSION:-1}"
USE_H5AD_PERT_METADATA="${USE_H5AD_PERT_METADATA:-0}"
PERT_POOL_AGGREGATIONS="${PERT_POOL_AGGREGATIONS:-mean max min}"
PERT_POOL_SCALE_INIT="${PERT_POOL_SCALE_INIT:-1.0 1.0 1.0}"
PERT_POOL_FUSION_MODE="${PERT_POOL_FUSION_MODE:-sum}"
PERT_TYPE_ADAPTER_MODE="${PERT_TYPE_ADAPTER_MODE:-scalar}"
PERT_PAIRWISE_MODE="${PERT_PAIRWISE_MODE:-off}"
PERT_GENE_PROJECTOR_HIDDEN="${PERT_GENE_PROJECTOR_HIDDEN:-1024}"
PERT_CHEM_PROJECTOR_HIDDEN="${PERT_CHEM_PROJECTOR_HIDDEN:-1024}"
PERT_CHEM_EMB_DIM="${PERT_CHEM_EMB_DIM:-512}"
CHEM_FALLBACK_EMBED_DIM="${CHEM_FALLBACK_EMBED_DIM:-${PERT_CHEM_EMB_DIM}}"
DS_LOSS_ALPHA="${DS_LOSS_ALPHA:-0.0}"
DS_LOSS_WARMUP_START="${DS_LOSS_WARMUP_START:-0}"
DS_ALPHA="${DS_ALPHA:-0.7}"
MIN_SELECTED_CONDITIONS_PER_DATASET="${MIN_SELECTED_CONDITIONS_PER_DATASET:-0}"
CONDITION_VISIT_POWER="${CONDITION_VISIT_POWER:-1.0}"
CONDITION_VISIT_CAP="${CONDITION_VISIT_CAP:-0}"
CONDITION_LOSS_WEIGHT_FILE="${CONDITION_LOSS_WEIGHT_FILE:-}"
CONDITION_LOSS_WEIGHT_COLUMN="${CONDITION_LOSS_WEIGHT_COLUMN:-weight}"
CONDITION_LOSS_WEIGHT_NORMALIZE_MEAN="${CONDITION_LOSS_WEIGHT_NORMALIZE_MEAN:-1}"
COMPOSITION_DELTA_LOSS_WEIGHT="${COMPOSITION_DELTA_LOSS_WEIGHT:-0.0}"
COMPOSITION_DELTA_LOSS_WARMUP_START="${COMPOSITION_DELTA_LOSS_WARMUP_START:-500}"
COMPOSITION_DELTA_LOSS_WARMUP_END="${COMPOSITION_DELTA_LOSS_WARMUP_END:-2500}"
COMPOSITION_DELTA_LOSS_EVERY="${COMPOSITION_DELTA_LOSS_EVERY:-1}"
COMPOSITION_DELTA_BANK_SIZE="${COMPOSITION_DELTA_BANK_SIZE:-512}"
ENDPOINT_DELTA_LOSS_WEIGHT="${ENDPOINT_DELTA_LOSS_WEIGHT:-0.0}"
ENDPOINT_DELTA_LOSS_WARMUP_START="${ENDPOINT_DELTA_LOSS_WARMUP_START:-500}"
ENDPOINT_DELTA_LOSS_WARMUP_END="${ENDPOINT_DELTA_LOSS_WARMUP_END:-2500}"
RESPONSE_GEOMETRY_LOSS_WEIGHT="${RESPONSE_GEOMETRY_LOSS_WEIGHT:-0.0}"
RESPONSE_GEOMETRY_LOSS_WARMUP_START="${RESPONSE_GEOMETRY_LOSS_WARMUP_START:-0}"
RESPONSE_GEOMETRY_LOSS_WARMUP_END="${RESPONSE_GEOMETRY_LOSS_WARMUP_END:-0}"
RESPONSE_NORMALIZATION_MODE="${RESPONSE_NORMALIZATION_MODE:-off}"
RESPONSE_NORMALIZATION_ARTIFACT="${RESPONSE_NORMALIZATION_ARTIFACT:-}"
RESPONSE_GEOMETRY_CONDITION_FILTER="${RESPONSE_GEOMETRY_CONDITION_FILTER:-all}"
PERT_RESIDUAL_DIRECTION_LOSS_WEIGHT="${PERT_RESIDUAL_DIRECTION_LOSS_WEIGHT:-0.0}"
PERT_RESIDUAL_DIRECTION_LOSS_WARMUP_START="${PERT_RESIDUAL_DIRECTION_LOSS_WARMUP_START:-500}"
PERT_RESIDUAL_DIRECTION_LOSS_WARMUP_END="${PERT_RESIDUAL_DIRECTION_LOSS_WARMUP_END:-2500}"
PERT_RESIDUAL_CONTRASTIVE_LOSS_WEIGHT="${PERT_RESIDUAL_CONTRASTIVE_LOSS_WEIGHT:-0.0}"
PERT_RESIDUAL_CONTRASTIVE_LOSS_WARMUP_START="${PERT_RESIDUAL_CONTRASTIVE_LOSS_WARMUP_START:-500}"
PERT_RESIDUAL_CONTRASTIVE_LOSS_WARMUP_END="${PERT_RESIDUAL_CONTRASTIVE_LOSS_WARMUP_END:-2500}"
PERT_RESIDUAL_CONTRASTIVE_TEMPERATURE="${PERT_RESIDUAL_CONTRASTIVE_TEMPERATURE:-0.10}"
PERT_RESIDUAL_CONTRASTIVE_BANK_SIZE="${PERT_RESIDUAL_CONTRASTIVE_BANK_SIZE:-256}"
PERT_RESIDUAL_CONTRASTIVE_MIN_NORM="${PERT_RESIDUAL_CONTRASTIVE_MIN_NORM:-1e-6}"
PERT_RESIDUAL_RELATIONAL_LOSS_WEIGHT="${PERT_RESIDUAL_RELATIONAL_LOSS_WEIGHT:-0.0}"
PERT_RESIDUAL_RELATIONAL_LOSS_WARMUP_START="${PERT_RESIDUAL_RELATIONAL_LOSS_WARMUP_START:-500}"
PERT_RESIDUAL_RELATIONAL_LOSS_WARMUP_END="${PERT_RESIDUAL_RELATIONAL_LOSS_WARMUP_END:-2500}"
PERT_RESIDUAL_RELATIONAL_TEMPERATURE="${PERT_RESIDUAL_RELATIONAL_TEMPERATURE:-0.10}"
PERT_RESIDUAL_RELATIONAL_TARGET_TEMPERATURE="${PERT_RESIDUAL_RELATIONAL_TARGET_TEMPERATURE:-0.10}"
CONDITION_DELTA_HEAD_LOSS_WEIGHT="${CONDITION_DELTA_HEAD_LOSS_WEIGHT:-0.0}"
CONDITION_DELTA_HEAD_LOSS_WARMUP_START="${CONDITION_DELTA_HEAD_LOSS_WARMUP_START:-500}"
CONDITION_DELTA_HEAD_LOSS_WARMUP_END="${CONDITION_DELTA_HEAD_LOSS_WARMUP_END:-2500}"
CONDITION_DELTA_HEAD_HIDDEN="${CONDITION_DELTA_HEAD_HIDDEN:-1024}"
CONDITION_DELTA_HEAD_TARGET="${CONDITION_DELTA_HEAD_TARGET:-endpoint_delta}"
CONDITION_DELTA_HEAD_USE_IN_MODEL="${CONDITION_DELTA_HEAD_USE_IN_MODEL:-0}"
CONDITION_DELTA_IN_MODEL_FILTER="${CONDITION_DELTA_IN_MODEL_FILTER:-all}"
CONDITION_DELTA_ALLOWLIST_GENE_FILE="${CONDITION_DELTA_ALLOWLIST_GENE_FILE:-}"
ADDITIVE_CONDITION_DELTA_LOSS_WEIGHT="${ADDITIVE_CONDITION_DELTA_LOSS_WEIGHT:-0.0}"
ADDITIVE_CONDITION_DELTA_LOSS_WARMUP_START="${ADDITIVE_CONDITION_DELTA_LOSS_WARMUP_START:-500}"
ADDITIVE_CONDITION_DELTA_LOSS_WARMUP_END="${ADDITIVE_CONDITION_DELTA_LOSS_WARMUP_END:-2500}"
CONDITION_PRIOR_DELTA_LOSS_WEIGHT="${CONDITION_PRIOR_DELTA_LOSS_WEIGHT:-0.0}"
CONDITION_PRIOR_DELTA_LOSS_WARMUP_START="${CONDITION_PRIOR_DELTA_LOSS_WARMUP_START:-500}"
CONDITION_PRIOR_DELTA_LOSS_WARMUP_END="${CONDITION_PRIOR_DELTA_LOSS_WARMUP_END:-2500}"
CONDITION_PRIOR_DELTA_LOSS_EVERY="${CONDITION_PRIOR_DELTA_LOSS_EVERY:-1}"
CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WEIGHT="${CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WEIGHT:-0.0}"
CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WARMUP_START="${CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WARMUP_START:-500}"
CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WARMUP_END="${CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WARMUP_END:-2500}"
CONDITION_PRIOR_BANK_MAX_CELLS="${CONDITION_PRIOR_BANK_MAX_CELLS:-512}"
CONDITION_PRIOR_BANK_MIN_NORM="${CONDITION_PRIOR_BANK_MIN_NORM:-1e-6}"
CONDITION_PRIOR_NUM_GENES="${CONDITION_PRIOR_NUM_GENES:-2}"
CONDITION_PRIOR_BANK_SCOPE="${CONDITION_PRIOR_BANK_SCOPE:-same_dataset}"
CONDITION_PRIOR_BANK_SPLIT_FILE="${CONDITION_PRIOR_BANK_SPLIT_FILE:-}"
CONDITION_PRIOR_BANK_AGGREGATION="${CONDITION_PRIOR_BANK_AGGREGATION:-condition}"
ANCHOR_REPLAY_LOSS_WEIGHT="${ANCHOR_REPLAY_LOSS_WEIGHT:-0.0}"
ANCHOR_REPLAY_LOSS_WARMUP_START="${ANCHOR_REPLAY_LOSS_WARMUP_START:-500}"
ANCHOR_REPLAY_LOSS_WARMUP_END="${ANCHOR_REPLAY_LOSS_WARMUP_END:-2500}"
ANCHOR_REPLAY_CONDITION_FILTER="${ANCHOR_REPLAY_CONDITION_FILTER:-all}"
ANCHOR_REPLAY_DATASET_FILTER="${ANCHOR_REPLAY_DATASET_FILTER:-}"
ANCHOR_REPLAY_CHECKPOINT="${ANCHOR_REPLAY_CHECKPOINT:-}"
TRACKC_ROUTED_DISTILL_LOSS_WEIGHT="${TRACKC_ROUTED_DISTILL_LOSS_WEIGHT:-0.0}"
TRACKC_ROUTED_DISTILL_LOSS_WARMUP_START="${TRACKC_ROUTED_DISTILL_LOSS_WARMUP_START:-500}"
TRACKC_ROUTED_DISTILL_LOSS_WARMUP_END="${TRACKC_ROUTED_DISTILL_LOSS_WARMUP_END:-2500}"
TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT="${TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT:-0.0}"
TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_START="${TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_START:-500}"
TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_END="${TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_END:-2500}"
TRACKC_ROUTED_DISTILL_ROUTE_FILE="${TRACKC_ROUTED_DISTILL_ROUTE_FILE:-}"
TRACKC_ROUTED_DISTILL_BANK_SPLIT_FILE="${TRACKC_ROUTED_DISTILL_BANK_SPLIT_FILE:-}"
TRACKC_ROUTED_DISTILL_TARGET_FRAME="${TRACKC_ROUTED_DISTILL_TARGET_FRAME:-endpoint_delta}"
TRACKC_ROUTED_DISTILL_MEMORY_MODE="${TRACKC_ROUTED_DISTILL_MEMORY_MODE:-off}"
TRACKC_ROUTED_DISTILL_MEMORY_K="${TRACKC_ROUTED_DISTILL_MEMORY_K:-3}"
TRACKC_ROUTED_DISTILL_MEMORY_MIN_SCORE="${TRACKC_ROUTED_DISTILL_MEMORY_MIN_SCORE:-0.25}"
TRACKC_ROUTED_DISTILL_MEMORY_SCOPE="${TRACKC_ROUTED_DISTILL_MEMORY_SCOPE:-same_dataset}"
TRACKC_SUPPORT_CONTEXT_USE_IN_MODEL="${TRACKC_SUPPORT_CONTEXT_USE_IN_MODEL:-0}"
TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL="${TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL:-0}"
TRACKC_SUPPORT_CONTEXT_DIM="${TRACKC_SUPPORT_CONTEXT_DIM:-0}"
TRACKC_SUPPORT_CONTEXT_SOURCE="${TRACKC_SUPPORT_CONTEXT_SOURCE:-off}"
TRACKC_SUPPORT_CONTEXT_PAIR_TYPE_FILTER="${TRACKC_SUPPORT_CONTEXT_PAIR_TYPE_FILTER:-off}"
TRACKC_SUPPORT_SET_TASK_USE_IN_MODEL="${TRACKC_SUPPORT_SET_TASK_USE_IN_MODEL:-0}"
TRACKC_SUPPORT_SET_TASK_DIM="${TRACKC_SUPPORT_SET_TASK_DIM:-0}"
TRACKC_SUPPORT_SET_TASK_SOURCE="${TRACKC_SUPPORT_SET_TASK_SOURCE:-off}"
TRACKC_SUPPORT_SET_TASK_SAFE_SPLIT_FILE="${TRACKC_SUPPORT_SET_TASK_SAFE_SPLIT_FILE:-}"
TRACKC_SUPPORT_SET_TASK_ANCHOR_CONDITION_MEANS="${TRACKC_SUPPORT_SET_TASK_ANCHOR_CONDITION_MEANS:-}"
TRACKC_SUPPORT_SET_TASK_CANDIDATE_CONDITION_MEANS="${TRACKC_SUPPORT_SET_TASK_CANDIDATE_CONDITION_MEANS:-}"
TRACKC_SUPPORT_SET_TASK_SCALE="${TRACKC_SUPPORT_SET_TASK_SCALE:-1.0}"
TRACKC_SUPPORT_SET_TASK_MIN_SUPPORT_COUNT="${TRACKC_SUPPORT_SET_TASK_MIN_SUPPORT_COUNT:-1}"
TRACKC_SUPPORT_SET_TASK_EVAL_CONTROL="${TRACKC_SUPPORT_SET_TASK_EVAL_CONTROL:-actual}"
FINETUNE_TRAINABLE_SCOPE="${FINETUNE_TRAINABLE_SCOPE:-all}"
EVAL_MAX_CONDITIONS="${EVAL_MAX_CONDITIONS:-0}"
EVAL_MAX_CONDITIONS_PER_DATASET="${EVAL_MAX_CONDITIONS_PER_DATASET:-0}"
EVAL_MAX_MSE_CELLS="${EVAL_MAX_MSE_CELLS:-0}"
EVAL_MAX_MMD_CELLS="${EVAL_MAX_MMD_CELLS:-2048}"
EVAL_MAX_CHUNK="${EVAL_MAX_CHUNK:-256}"
TRAIN_EVAL_ENABLED="${TRAIN_EVAL_ENABLED:-1}"
PERTURBATION_FAMILY_FILTER="${PERTURBATION_FAMILY_FILTER:-all}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
read -r -a PERT_POOL_AGGREGATIONS_ARGS <<< "${PERT_POOL_AGGREGATIONS}"
read -r -a PERT_POOL_SCALE_INIT_ARGS <<< "${PERT_POOL_SCALE_INIT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export PYTHONPATH="${COUPLEDFM}:${PYTHONPATH:-}"
export PERT_EMBED_SOURCE="${PERT_EMBED_SOURCE:-scgpt_embed_gene}"

mkdir -p "${OUT_ROOT}/${RUN_TAG}" "${LOG_ROOT}"

if [[ ! -f "${DATA_DIR}/manifest.json" ]]; then
  echo "Missing full ${LATENT_BACKBONE} manifest: ${DATA_DIR}/manifest.json" >&2
  echo "Build it first with model/latent/prepare_fm_data.py from the matching biFlow control/GT folders." >&2
  exit 2
fi

EMB_DIM="${EMB_DIM:-$("${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
p = Path("${DATA_DIR}") / "manifest.json"
print(json.loads(p.read_text()).get("emb_dim", ""))
PY
)}"
if [[ -z "${EMB_DIM}" || "${EMB_DIM}" == "None" ]]; then
  echo "Could not infer emb_dim from ${DATA_DIR}/manifest.json; set EMB_DIM explicitly." >&2
  exit 2
fi

save_dir="${OUT_ROOT}/${RUN_TAG}"
log_file="${LOG_ROOT}/${RUN_TAG}.log"

echo "[$(date '+%F %T')] launch full ${LATENT_BACKBONE} LatentFM"
echo "gpu=${GPU} total_steps=${TOTAL_STEPS} batch_size=${BATCH_SIZE} grad_accum=${GRAD_ACCUM_STEPS} run_tag=${RUN_TAG} emb_dim=${EMB_DIM}"
echo "gamma=${GAMMA} gamma_warmup=${GAMMA_WARMUP_START}-${GAMMA_WARMUP_END} mmd_every=${MMD_EVERY} estimator=${MMD_ESTIMATOR} mmd_ode_steps=${MMD_ODE_STEPS} mmd_dataset_filter=${MMD_DATASET_FILTER:-all}"
echo "risk_row_cvar_weight=${RISK_ROW_CVAR_LOSS_WEIGHT} warmup=${RISK_ROW_CVAR_LOSS_WARMUP_START}-${RISK_ROW_CVAR_LOSS_WARMUP_END} dataset_filter=${RISK_ROW_CVAR_DATASET_FILTER:-none} history=${RISK_ROW_CVAR_HISTORY_SIZE} min_history=${RISK_ROW_CVAR_MIN_HISTORY} top_frac=${RISK_ROW_CVAR_TOP_FRAC} threshold=${RISK_ROW_CVAR_MMD_THRESHOLD}"
echo "ot_pair_mode=${OT_PAIR_MODE}"
echo "ot_threads=${OT_THREADS} prefetch=${PREFETCH} n_ot_workers=${N_OT_WORKERS}"
echo "selection_metric=${SELECTION_METRIC} selection_mmd_lambda=${SELECTION_MMD_LAMBDA}"
echo "seed=${SEED}"
echo "lr=${LR}"
echo "use_param_groups=${USE_PARAM_GROUPS} lr_new_module_mult=${LR_NEW_MODULE_MULT}"
echo "composition_delta_weight=${COMPOSITION_DELTA_LOSS_WEIGHT} warmup=${COMPOSITION_DELTA_LOSS_WARMUP_START}-${COMPOSITION_DELTA_LOSS_WARMUP_END} every=${COMPOSITION_DELTA_LOSS_EVERY}"
echo "endpoint_delta_weight=${ENDPOINT_DELTA_LOSS_WEIGHT} warmup=${ENDPOINT_DELTA_LOSS_WARMUP_START}-${ENDPOINT_DELTA_LOSS_WARMUP_END}"
echo "response_geometry_weight=${RESPONSE_GEOMETRY_LOSS_WEIGHT} warmup=${RESPONSE_GEOMETRY_LOSS_WARMUP_START}-${RESPONSE_GEOMETRY_LOSS_WARMUP_END} mode=${RESPONSE_NORMALIZATION_MODE} artifact=${RESPONSE_NORMALIZATION_ARTIFACT:-none} filter=${RESPONSE_GEOMETRY_CONDITION_FILTER}"
echo "pert_residual_direction_weight=${PERT_RESIDUAL_DIRECTION_LOSS_WEIGHT} warmup=${PERT_RESIDUAL_DIRECTION_LOSS_WARMUP_START}-${PERT_RESIDUAL_DIRECTION_LOSS_WARMUP_END}"
echo "pert_residual_contrastive_weight=${PERT_RESIDUAL_CONTRASTIVE_LOSS_WEIGHT} warmup=${PERT_RESIDUAL_CONTRASTIVE_LOSS_WARMUP_START}-${PERT_RESIDUAL_CONTRASTIVE_LOSS_WARMUP_END} temp=${PERT_RESIDUAL_CONTRASTIVE_TEMPERATURE} bank=${PERT_RESIDUAL_CONTRASTIVE_BANK_SIZE}"
echo "pert_residual_relational_weight=${PERT_RESIDUAL_RELATIONAL_LOSS_WEIGHT} warmup=${PERT_RESIDUAL_RELATIONAL_LOSS_WARMUP_START}-${PERT_RESIDUAL_RELATIONAL_LOSS_WARMUP_END} temp=${PERT_RESIDUAL_RELATIONAL_TEMPERATURE} target_temp=${PERT_RESIDUAL_RELATIONAL_TARGET_TEMPERATURE}"
echo "condition_delta_head_weight=${CONDITION_DELTA_HEAD_LOSS_WEIGHT} warmup=${CONDITION_DELTA_HEAD_LOSS_WARMUP_START}-${CONDITION_DELTA_HEAD_LOSS_WARMUP_END} hidden=${CONDITION_DELTA_HEAD_HIDDEN} target=${CONDITION_DELTA_HEAD_TARGET} use_in_model=${CONDITION_DELTA_HEAD_USE_IN_MODEL} in_model_filter=${CONDITION_DELTA_IN_MODEL_FILTER} allowlist=${CONDITION_DELTA_ALLOWLIST_GENE_FILE:-none}"
echo "additive_condition_delta_weight=${ADDITIVE_CONDITION_DELTA_LOSS_WEIGHT} warmup=${ADDITIVE_CONDITION_DELTA_LOSS_WARMUP_START}-${ADDITIVE_CONDITION_DELTA_LOSS_WARMUP_END}"
echo "condition_prior_delta_weight=${CONDITION_PRIOR_DELTA_LOSS_WEIGHT} warmup=${CONDITION_PRIOR_DELTA_LOSS_WARMUP_START}-${CONDITION_PRIOR_DELTA_LOSS_WARMUP_END} every=${CONDITION_PRIOR_DELTA_LOSS_EVERY}"
echo "condition_prior_additive_delta_weight=${CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WEIGHT} warmup=${CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WARMUP_START}-${CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WARMUP_END} genes=${CONDITION_PRIOR_NUM_GENES} scope=${CONDITION_PRIOR_BANK_SCOPE} aggregation=${CONDITION_PRIOR_BANK_AGGREGATION}"
echo "anchor_replay_weight=${ANCHOR_REPLAY_LOSS_WEIGHT} warmup=${ANCHOR_REPLAY_LOSS_WARMUP_START}-${ANCHOR_REPLAY_LOSS_WARMUP_END} filter=${ANCHOR_REPLAY_CONDITION_FILTER} dataset_filter=${ANCHOR_REPLAY_DATASET_FILTER:-all}"
echo "trackc_routed_distill_weight=${TRACKC_ROUTED_DISTILL_LOSS_WEIGHT} warmup=${TRACKC_ROUTED_DISTILL_LOSS_WARMUP_START}-${TRACKC_ROUTED_DISTILL_LOSS_WARMUP_END} route_file=${TRACKC_ROUTED_DISTILL_ROUTE_FILE:-none} target=${TRACKC_ROUTED_DISTILL_TARGET_FRAME}"
echo "trackc_routed_endpoint_weight=${TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT} warmup=${TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_START}-${TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_END}"
echo "trackc_routed_distill_bank_split_file=${TRACKC_ROUTED_DISTILL_BANK_SPLIT_FILE:-default_training_split}"
echo "trackc_routed_distill_memory mode=${TRACKC_ROUTED_DISTILL_MEMORY_MODE} k=${TRACKC_ROUTED_DISTILL_MEMORY_K} min_score=${TRACKC_ROUTED_DISTILL_MEMORY_MIN_SCORE} scope=${TRACKC_ROUTED_DISTILL_MEMORY_SCOPE}"
echo "trackc_support_context use_in_model=${TRACKC_SUPPORT_CONTEXT_USE_IN_MODEL} residual_use_in_model=${TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL} film_use_in_model=${TRACKC_SUPPORT_FILM_USE_IN_MODEL:-0} dim=${TRACKC_SUPPORT_CONTEXT_DIM} source=${TRACKC_SUPPORT_CONTEXT_SOURCE} pair_type_filter=${TRACKC_SUPPORT_CONTEXT_PAIR_TYPE_FILTER}"
echo "trackc_support_set_task use_in_model=${TRACKC_SUPPORT_SET_TASK_USE_IN_MODEL} dim=${TRACKC_SUPPORT_SET_TASK_DIM} source=${TRACKC_SUPPORT_SET_TASK_SOURCE} safe_split=${TRACKC_SUPPORT_SET_TASK_SAFE_SPLIT_FILE:-none} anchor_means=${TRACKC_SUPPORT_SET_TASK_ANCHOR_CONDITION_MEANS:-none} candidate_means=${TRACKC_SUPPORT_SET_TASK_CANDIDATE_CONDITION_MEANS:-none} scale=${TRACKC_SUPPORT_SET_TASK_SCALE} min_support=${TRACKC_SUPPORT_SET_TASK_MIN_SUPPORT_COUNT} eval_control=${TRACKC_SUPPORT_SET_TASK_EVAL_CONTROL}"
echo "pert_pool_aggregations=${PERT_POOL_AGGREGATIONS} scale_init=${PERT_POOL_SCALE_INIT} fusion_mode=${PERT_POOL_FUSION_MODE}"
echo "pert_chem_emb_dim=${PERT_CHEM_EMB_DIM} chem_fallback_embed_dim=${CHEM_FALLBACK_EMBED_DIM} raw_drug_emb_cache_dir=${RAW_DRUG_EMB_CACHE_DIR:-none} latent_drug_emb_cache_dir=${LATENT_DRUG_EMB_CACHE_DIR:-none}"
echo "pert_type_adapter_mode=${PERT_TYPE_ADAPTER_MODE}"
echo "pert_pairwise_mode=${PERT_PAIRWISE_MODE}"
echo "sampling ds_alpha=${DS_ALPHA} ds_loss_alpha=${DS_LOSS_ALPHA} ds_loss_warmup_start=${DS_LOSS_WARMUP_START} min_selected_conditions_per_dataset=${MIN_SELECTED_CONDITIONS_PER_DATASET} condition_visit_power=${CONDITION_VISIT_POWER} condition_visit_cap=${CONDITION_VISIT_CAP} condition_loss_weight_file=${CONDITION_LOSS_WEIGHT_FILE:-none} condition_loss_weight_column=${CONDITION_LOSS_WEIGHT_COLUMN}"
echo "pert_embed_source=${PERT_EMBED_SOURCE}"
echo "gene_cache=${GENE_CACHE}"
echo "perturbation_family_filter=${PERTURBATION_FAMILY_FILTER}"
echo "init_checkpoint=${INIT_CHECKPOINT:-none}"
echo "init_checkpoint_use_ema=${INIT_CHECKPOINT_USE_EMA:-0}"
echo "anchor_replay_checkpoint=${ANCHOR_REPLAY_CHECKPOINT:-default}"
echo "anchor_replay_checkpoint_use_ema=${ANCHOR_REPLAY_CHECKPOINT_USE_EMA:-0}"
echo "finetune_trainable_scope=${FINETUNE_TRAINABLE_SCOPE}"
echo "eval_caps=max_conditions=${EVAL_MAX_CONDITIONS} max_conditions_per_dataset=${EVAL_MAX_CONDITIONS_PER_DATASET} max_mse_cells=${EVAL_MAX_MSE_CELLS} max_mmd_cells=${EVAL_MAX_MMD_CELLS} max_chunk=${EVAL_MAX_CHUNK}"
echo "train_eval_enabled=${TRAIN_EVAL_ENABLED}"
echo "python_bin=${PYTHON_BIN}"
echo "data_dir=${DATA_DIR}"
echo "split_file=${SPLIT_FILE:-default}"
echo "pert_means_file=${PERT_MEANS_FILE:-default}"
echo "save_dir=${save_dir}"
echo "log_file=${log_file}"

cd "${COUPLEDFM}"
export CUDA_VISIBLE_DEVICES="${GPU}"

use_pert_in_fusion_args=()
if [[ "${USE_PERT_IN_FUSION}" == "1" || "${USE_PERT_IN_FUSION}" == "true" ]]; then
  use_pert_in_fusion_args+=(--use-pert-in-fusion)
fi

condition_delta_head_use_args=()
if [[ "${CONDITION_DELTA_HEAD_USE_IN_MODEL}" == "1" || "${CONDITION_DELTA_HEAD_USE_IN_MODEL}" == "true" ]]; then
  condition_delta_head_use_args+=(--condition-delta-head-use-in-model)
fi

param_group_args=()
if [[ "${USE_PARAM_GROUPS}" == "1" || "${USE_PARAM_GROUPS}" == "true" ]]; then
  param_group_args+=(--use-param-groups)
fi

init_checkpoint_args=()
if [[ -n "${INIT_CHECKPOINT}" ]]; then
  init_checkpoint_args+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi
if [[ "${INIT_CHECKPOINT_USE_EMA:-0}" == "1" || "${INIT_CHECKPOINT_USE_EMA:-0}" == "true" ]]; then
  init_checkpoint_args+=(--init-checkpoint-use-ema)
fi
anchor_replay_checkpoint_args=()
if [[ -n "${ANCHOR_REPLAY_CHECKPOINT}" ]]; then
  anchor_replay_checkpoint_args+=(--anchor-replay-checkpoint "${ANCHOR_REPLAY_CHECKPOINT}")
fi
if [[ "${ANCHOR_REPLAY_CHECKPOINT_USE_EMA:-0}" == "1" || "${ANCHOR_REPLAY_CHECKPOINT_USE_EMA:-0}" == "true" ]]; then
  anchor_replay_checkpoint_args+=(--anchor-replay-checkpoint-use-ema)
fi
trackc_routed_distill_route_args=()
if [[ -n "${TRACKC_ROUTED_DISTILL_ROUTE_FILE}" ]]; then
  trackc_routed_distill_route_args+=(--trackc-routed-distill-route-file "${TRACKC_ROUTED_DISTILL_ROUTE_FILE}")
fi
trackc_routed_distill_bank_split_args=()
if [[ -n "${TRACKC_ROUTED_DISTILL_BANK_SPLIT_FILE}" ]]; then
  trackc_routed_distill_bank_split_args+=(--trackc-routed-distill-bank-split-file "${TRACKC_ROUTED_DISTILL_BANK_SPLIT_FILE}")
fi
trackc_support_context_args=()
if [[ "${TRACKC_SUPPORT_CONTEXT_USE_IN_MODEL}" == "1" || "${TRACKC_SUPPORT_CONTEXT_USE_IN_MODEL}" == "true" ]]; then
  trackc_support_context_args+=(--trackc-support-context-use-in-model)
fi
if [[ "${TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL}" == "1" || "${TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL}" == "true" ]]; then
  trackc_support_context_args+=(--trackc-support-residual-use-in-model)
fi
if [[ "${TRACKC_SUPPORT_FILM_USE_IN_MODEL:-0}" == "1" || "${TRACKC_SUPPORT_FILM_USE_IN_MODEL:-0}" == "true" ]]; then
  trackc_support_context_args+=(--trackc-support-film-use-in-model)
fi
trackc_support_set_task_args=()
if [[ "${TRACKC_SUPPORT_SET_TASK_USE_IN_MODEL}" == "1" || "${TRACKC_SUPPORT_SET_TASK_USE_IN_MODEL}" == "true" ]]; then
  trackc_support_set_task_args+=(--trackc-support-set-task-use-in-model)
fi
split_file_args=()
if [[ -n "${SPLIT_FILE}" ]]; then
  split_file_args+=(--split-file "${SPLIT_FILE}")
fi
condition_prior_bank_split_args=()
if [[ -n "${CONDITION_PRIOR_BANK_SPLIT_FILE}" ]]; then
  condition_prior_bank_split_args+=(--condition-prior-bank-split-file "${CONDITION_PRIOR_BANK_SPLIT_FILE}")
fi
condition_delta_allowlist_args=()
if [[ -n "${CONDITION_DELTA_ALLOWLIST_GENE_FILE}" ]]; then
  condition_delta_allowlist_args+=(--condition-delta-allowlist-gene-file "${CONDITION_DELTA_ALLOWLIST_GENE_FILE}")
fi
pert_means_file_args=()
if [[ -n "${PERT_MEANS_FILE}" ]]; then
  pert_means_file_args+=(--pert-means-file "${PERT_MEANS_FILE}")
fi

h5ad_metadata_args=()
if [[ "${USE_H5AD_PERT_METADATA}" == "1" || "${USE_H5AD_PERT_METADATA}" == "true" ]]; then
  h5ad_metadata_args+=(--use-h5ad-pert-metadata)
fi

train_eval_args=()
if [[ "${TRAIN_EVAL_ENABLED}" == "0" || "${TRAIN_EVAL_ENABLED}" == "false" ]]; then
  train_eval_args+=(--no-train-eval-enabled)
fi

"${PYTHON_BIN}" -m model.latent.train \
  --data-dir "${DATA_DIR}" \
  --biflow-dir "${BIFLOW_DIR}" \
  "${split_file_args[@]}" \
  "${pert_means_file_args[@]}" \
  --save-dir "${save_dir}" \
  --latent-backbone "${LATENT_BACKBONE}" \
  --model-type control_mlp \
  --emb-dim "${EMB_DIM}" \
  --gpu 0 \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  "${param_group_args[@]}" \
  --lr-new-module-mult "${LR_NEW_MODULE_MULT}" \
  --min-cells 16 \
  --ds-alpha "${DS_ALPHA}" \
  --ds-loss-alpha "${DS_LOSS_ALPHA}" \
  --ds-loss-warmup-start "${DS_LOSS_WARMUP_START}" \
  --min-selected-conditions-per-dataset "${MIN_SELECTED_CONDITIONS_PER_DATASET}" \
  --condition-visit-power "${CONDITION_VISIT_POWER}" \
  --condition-visit-cap "${CONDITION_VISIT_CAP}" \
  --condition-loss-weight-file "${CONDITION_LOSS_WEIGHT_FILE}" \
  --condition-loss-weight-column "${CONDITION_LOSS_WEIGHT_COLUMN}" \
  "$(if [[ "${CONDITION_LOSS_WEIGHT_NORMALIZE_MEAN}" == "0" || "${CONDITION_LOSS_WEIGHT_NORMALIZE_MEAN}" == "false" ]]; then echo "--no-condition-loss-weight-normalize-mean"; else echo "--condition-loss-weight-normalize-mean"; fi)" \
  --perturbation-family-filter "${PERTURBATION_FAMILY_FILTER}" \
  "${init_checkpoint_args[@]}" \
  --finetune-trainable-scope "${FINETUNE_TRAINABLE_SCOPE}" \
  --scale-noise 0.01 \
  --lr "${LR}" \
  --weight-decay 1e-4 \
  --warmup-steps 300 \
  --total-steps "${TOTAL_STEPS}" \
  --lr-decay-steps "${TOTAL_STEPS}" \
  --print-every 100 \
  --eval-max-conditions "${EVAL_MAX_CONDITIONS}" \
  --eval-max-conditions-per-dataset "${EVAL_MAX_CONDITIONS_PER_DATASET}" \
  --eval-max-mse-cells "${EVAL_MAX_MSE_CELLS}" \
  --eval-max-mmd-cells "${EVAL_MAX_MMD_CELLS}" \
  --eval-max-chunk "${EVAL_MAX_CHUNK}" \
  --selection-metric "${SELECTION_METRIC}" \
  --selection-mmd-lambda "${SELECTION_MMD_LAMBDA}" \
  --seed "${SEED}" \
  --ot-method torch_sinkhorn \
  --ot-pair-mode "${OT_PAIR_MODE}" \
  --ot-threads "${OT_THREADS}" \
  --ot-sinkhorn-reg 0.05 \
  --ot-sinkhorn-iter 30 \
  --prefetch "${PREFETCH}" \
  --n-ot-workers "${N_OT_WORKERS}" \
  --use-mmd \
  --gamma "${GAMMA}" \
  --gamma-warmup-start "${GAMMA_WARMUP_START}" \
  --gamma-warmup-end "${GAMMA_WARMUP_END}" \
  --mmd-ode-steps "${MMD_ODE_STEPS}" \
  --mmd-every "${MMD_EVERY}" \
  --mmd-estimator "${MMD_ESTIMATOR}" \
  --mmd-dataset-filter "${MMD_DATASET_FILTER}" \
  --risk-row-cvar-loss-weight "${RISK_ROW_CVAR_LOSS_WEIGHT}" \
  --risk-row-cvar-loss-warmup-start "${RISK_ROW_CVAR_LOSS_WARMUP_START}" \
  --risk-row-cvar-loss-warmup-end "${RISK_ROW_CVAR_LOSS_WARMUP_END}" \
  --risk-row-cvar-dataset-filter "${RISK_ROW_CVAR_DATASET_FILTER}" \
  --risk-row-cvar-history-size "${RISK_ROW_CVAR_HISTORY_SIZE}" \
  --risk-row-cvar-min-history "${RISK_ROW_CVAR_MIN_HISTORY}" \
  --risk-row-cvar-top-frac "${RISK_ROW_CVAR_TOP_FRAC}" \
  --risk-row-cvar-mmd-threshold "${RISK_ROW_CVAR_MMD_THRESHOLD}" \
  --composition-delta-loss-weight "${COMPOSITION_DELTA_LOSS_WEIGHT}" \
  --composition-delta-loss-warmup-start "${COMPOSITION_DELTA_LOSS_WARMUP_START}" \
  --composition-delta-loss-warmup-end "${COMPOSITION_DELTA_LOSS_WARMUP_END}" \
  --composition-delta-loss-every "${COMPOSITION_DELTA_LOSS_EVERY}" \
  --composition-delta-bank-size "${COMPOSITION_DELTA_BANK_SIZE}" \
  --endpoint-delta-loss-weight "${ENDPOINT_DELTA_LOSS_WEIGHT}" \
  --endpoint-delta-loss-warmup-start "${ENDPOINT_DELTA_LOSS_WARMUP_START}" \
  --endpoint-delta-loss-warmup-end "${ENDPOINT_DELTA_LOSS_WARMUP_END}" \
  --response-geometry-loss-weight "${RESPONSE_GEOMETRY_LOSS_WEIGHT}" \
  --response-geometry-loss-warmup-start "${RESPONSE_GEOMETRY_LOSS_WARMUP_START}" \
  --response-geometry-loss-warmup-end "${RESPONSE_GEOMETRY_LOSS_WARMUP_END}" \
  --response-normalization-mode "${RESPONSE_NORMALIZATION_MODE}" \
  --response-normalization-artifact "${RESPONSE_NORMALIZATION_ARTIFACT}" \
  --response-geometry-condition-filter "${RESPONSE_GEOMETRY_CONDITION_FILTER}" \
  --pert-residual-direction-loss-weight "${PERT_RESIDUAL_DIRECTION_LOSS_WEIGHT}" \
  --pert-residual-direction-loss-warmup-start "${PERT_RESIDUAL_DIRECTION_LOSS_WARMUP_START}" \
  --pert-residual-direction-loss-warmup-end "${PERT_RESIDUAL_DIRECTION_LOSS_WARMUP_END}" \
  --pert-residual-contrastive-loss-weight "${PERT_RESIDUAL_CONTRASTIVE_LOSS_WEIGHT}" \
  --pert-residual-contrastive-loss-warmup-start "${PERT_RESIDUAL_CONTRASTIVE_LOSS_WARMUP_START}" \
  --pert-residual-contrastive-loss-warmup-end "${PERT_RESIDUAL_CONTRASTIVE_LOSS_WARMUP_END}" \
  --pert-residual-contrastive-temperature "${PERT_RESIDUAL_CONTRASTIVE_TEMPERATURE}" \
  --pert-residual-contrastive-bank-size "${PERT_RESIDUAL_CONTRASTIVE_BANK_SIZE}" \
  --pert-residual-contrastive-min-norm "${PERT_RESIDUAL_CONTRASTIVE_MIN_NORM}" \
  --pert-residual-relational-loss-weight "${PERT_RESIDUAL_RELATIONAL_LOSS_WEIGHT}" \
  --pert-residual-relational-loss-warmup-start "${PERT_RESIDUAL_RELATIONAL_LOSS_WARMUP_START}" \
  --pert-residual-relational-loss-warmup-end "${PERT_RESIDUAL_RELATIONAL_LOSS_WARMUP_END}" \
  --pert-residual-relational-temperature "${PERT_RESIDUAL_RELATIONAL_TEMPERATURE}" \
  --pert-residual-relational-target-temperature "${PERT_RESIDUAL_RELATIONAL_TARGET_TEMPERATURE}" \
  --condition-delta-head-loss-weight "${CONDITION_DELTA_HEAD_LOSS_WEIGHT}" \
  --condition-delta-head-loss-warmup-start "${CONDITION_DELTA_HEAD_LOSS_WARMUP_START}" \
  --condition-delta-head-loss-warmup-end "${CONDITION_DELTA_HEAD_LOSS_WARMUP_END}" \
  --condition-delta-head-hidden "${CONDITION_DELTA_HEAD_HIDDEN}" \
  --condition-delta-head-target "${CONDITION_DELTA_HEAD_TARGET}" \
  --condition-delta-in-model-filter "${CONDITION_DELTA_IN_MODEL_FILTER}" \
  "${condition_delta_allowlist_args[@]}" \
  --additive-condition-delta-loss-weight "${ADDITIVE_CONDITION_DELTA_LOSS_WEIGHT}" \
  --additive-condition-delta-loss-warmup-start "${ADDITIVE_CONDITION_DELTA_LOSS_WARMUP_START}" \
  --additive-condition-delta-loss-warmup-end "${ADDITIVE_CONDITION_DELTA_LOSS_WARMUP_END}" \
  --condition-prior-delta-loss-weight "${CONDITION_PRIOR_DELTA_LOSS_WEIGHT}" \
  --condition-prior-delta-loss-warmup-start "${CONDITION_PRIOR_DELTA_LOSS_WARMUP_START}" \
  --condition-prior-delta-loss-warmup-end "${CONDITION_PRIOR_DELTA_LOSS_WARMUP_END}" \
  --condition-prior-delta-loss-every "${CONDITION_PRIOR_DELTA_LOSS_EVERY}" \
  --condition-prior-additive-delta-loss-weight "${CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WEIGHT}" \
  --condition-prior-additive-delta-loss-warmup-start "${CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WARMUP_START}" \
  --condition-prior-additive-delta-loss-warmup-end "${CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WARMUP_END}" \
  --condition-prior-bank-max-cells "${CONDITION_PRIOR_BANK_MAX_CELLS}" \
  --condition-prior-bank-min-norm "${CONDITION_PRIOR_BANK_MIN_NORM}" \
  --condition-prior-num-genes "${CONDITION_PRIOR_NUM_GENES}" \
  --condition-prior-bank-scope "${CONDITION_PRIOR_BANK_SCOPE}" \
  --condition-prior-bank-aggregation "${CONDITION_PRIOR_BANK_AGGREGATION}" \
  "${condition_prior_bank_split_args[@]}" \
  --anchor-replay-loss-weight "${ANCHOR_REPLAY_LOSS_WEIGHT}" \
  --anchor-replay-loss-warmup-start "${ANCHOR_REPLAY_LOSS_WARMUP_START}" \
  --anchor-replay-loss-warmup-end "${ANCHOR_REPLAY_LOSS_WARMUP_END}" \
  --anchor-replay-condition-filter "${ANCHOR_REPLAY_CONDITION_FILTER}" \
  --anchor-replay-dataset-filter "${ANCHOR_REPLAY_DATASET_FILTER}" \
  "${anchor_replay_checkpoint_args[@]}" \
  --trackc-routed-distill-loss-weight "${TRACKC_ROUTED_DISTILL_LOSS_WEIGHT}" \
  --trackc-routed-distill-loss-warmup-start "${TRACKC_ROUTED_DISTILL_LOSS_WARMUP_START}" \
  --trackc-routed-distill-loss-warmup-end "${TRACKC_ROUTED_DISTILL_LOSS_WARMUP_END}" \
  --trackc-routed-endpoint-loss-weight "${TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT}" \
  --trackc-routed-endpoint-loss-warmup-start "${TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_START}" \
  --trackc-routed-endpoint-loss-warmup-end "${TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_END}" \
  --trackc-routed-distill-target-frame "${TRACKC_ROUTED_DISTILL_TARGET_FRAME}" \
  --trackc-routed-distill-memory-mode "${TRACKC_ROUTED_DISTILL_MEMORY_MODE}" \
  --trackc-routed-distill-memory-k "${TRACKC_ROUTED_DISTILL_MEMORY_K}" \
  --trackc-routed-distill-memory-min-score "${TRACKC_ROUTED_DISTILL_MEMORY_MIN_SCORE}" \
  --trackc-routed-distill-memory-scope "${TRACKC_ROUTED_DISTILL_MEMORY_SCOPE}" \
  "${trackc_routed_distill_route_args[@]}" \
  "${trackc_routed_distill_bank_split_args[@]}" \
  "${trackc_support_context_args[@]}" \
  --trackc-support-context-dim "${TRACKC_SUPPORT_CONTEXT_DIM}" \
  --trackc-support-context-source "${TRACKC_SUPPORT_CONTEXT_SOURCE}" \
  --trackc-support-context-pair-type-filter "${TRACKC_SUPPORT_CONTEXT_PAIR_TYPE_FILTER}" \
  "${trackc_support_set_task_args[@]}" \
  --trackc-support-set-task-dim "${TRACKC_SUPPORT_SET_TASK_DIM}" \
  --trackc-support-set-task-source "${TRACKC_SUPPORT_SET_TASK_SOURCE}" \
  --trackc-support-set-task-safe-split-file "${TRACKC_SUPPORT_SET_TASK_SAFE_SPLIT_FILE}" \
  --trackc-support-set-task-anchor-condition-means "${TRACKC_SUPPORT_SET_TASK_ANCHOR_CONDITION_MEANS}" \
  --trackc-support-set-task-candidate-condition-means "${TRACKC_SUPPORT_SET_TASK_CANDIDATE_CONDITION_MEANS}" \
  --trackc-support-set-task-scale "${TRACKC_SUPPORT_SET_TASK_SCALE}" \
  --trackc-support-set-task-min-support-count "${TRACKC_SUPPORT_SET_TASK_MIN_SUPPORT_COUNT}" \
  --trackc-support-set-task-eval-control "${TRACKC_SUPPORT_SET_TASK_EVAL_CONTROL}" \
  "${condition_delta_head_use_args[@]}" \
  --use-ema \
  --ema-update-after 500 \
  --ema-decay 0.999 \
  --amp-dtype bf16 \
  "${train_eval_args[@]}" \
  --use-pert-condition \
  "${h5ad_metadata_args[@]}" \
  --pert-gene-emb-cache-dir "${GENE_CACHE}" \
  --pert-pool-aggregations "${PERT_POOL_AGGREGATIONS_ARGS[@]}" \
  --pert-pool-scale-init "${PERT_POOL_SCALE_INIT_ARGS[@]}" \
  --pert-pool-fusion-mode "${PERT_POOL_FUSION_MODE}" \
  --pert-type-adapter-mode "${PERT_TYPE_ADAPTER_MODE}" \
  --pert-pairwise-mode "${PERT_PAIRWISE_MODE}" \
  --pert-gene-projector-hidden "${PERT_GENE_PROJECTOR_HIDDEN}" \
  --pert-chem-enabled \
  --pert-chem-emb-dim "${PERT_CHEM_EMB_DIM}" \
  --pert-chem-projector-hidden "${PERT_CHEM_PROJECTOR_HIDDEN}" \
  --chem-fallback-embed-dim "${CHEM_FALLBACK_EMBED_DIM}" \
  --pert-to-c-init-mode "${PERT_TO_C_INIT_MODE}" \
  "${use_pert_in_fusion_args[@]}" \
  --patience 8 \
  2>&1 | tee "${log_file}"
