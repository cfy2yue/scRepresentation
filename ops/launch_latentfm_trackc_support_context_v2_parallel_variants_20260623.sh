#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

GATE_JSON=${ROOT}/reports/latentfm_trackc_support_context_v2_parallel_extension_gate_20260623.json
GENERIC_LAUNCHER=${ROOT}/ops/launch_latentfm_xverse_trackc_routed_distill_smoke_20260622.sh

"${PYTHON}" - "${GATE_JSON}" <<'PY'
import json
import sys
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = "trackc_support_context_v2_parallel_extension_gate_pass_two_capped_smokes_allowed"
if gate.get("status") != expected:
    raise SystemExit(f"parallel extension gate is not pass: {gate.get('status')}")
if gate.get("gpu_authorization") != "two_capped_v2_variants_after_fresh_resource_audit":
    raise SystemExit("parallel extension gate did not authorize two capped variants")
PY

COMMON_RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_support_context_v2_parallel_20260623
COMMON_OUT_ROOT=${ROOT}/CoupledFM/output/latentfm_runs/xverse_trackc_support_context_v2_parallel_20260623
COMMON_LOG_ROOT=${ROOT}/logs/latentfm_xverse_trackc_support_context_v2_parallel_20260623
SAFE_SPLIT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json
ROUTE_FILE=${ROOT}/reports/latentfm_trackc_residual_operator_route_teacher_20260623.json
ANCHOR_CKPT=${ROOT}/CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt

launch_variant() {
  local run_name="$1"
  local scope="$2"
  local use_context="$3"
  local use_residual="$4"
  local use_film="$5"
  local hypothesis="$6"

  export LATENTFM_TRACKC_RUN_NAME="${run_name}"
  export LATENTFM_TRACKC_RUN_ROOT="${COMMON_RUN_ROOT}"
  export LATENTFM_TRACKC_OUT_ROOT="${COMMON_OUT_ROOT}"
  export LATENTFM_TRACKC_LOG_ROOT="${COMMON_LOG_ROOT}"
  export LATENTFM_TRACKC_TRAINSELECT_SPLIT="${SAFE_SPLIT}"
  export LATENTFM_TRACKC_BANK_SPLIT_FILE="${SAFE_SPLIT}"
  export LATENTFM_TRACKC_ROUTE_FILE="${ROUTE_FILE}"
  export LATENTFM_TRACKC_ANCHOR_CKPT="${ANCHOR_CKPT}"

  export LATENTFM_TRACKC_FINETUNE_TRAINABLE_SCOPE="${scope}"
  export LATENTFM_TRACKC_SUPPORT_CONTEXT_USE_IN_MODEL="${use_context}"
  export LATENTFM_TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL="${use_residual}"
  export LATENTFM_TRACKC_SUPPORT_FILM_USE_IN_MODEL="${use_film}"
  export LATENTFM_TRACKC_SUPPORT_CONTEXT_DIM=384
  export LATENTFM_TRACKC_SUPPORT_CONTEXT_SOURCE=routed_distill_target

  export LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WEIGHT=0.0
  export LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT=0.50
  export LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_START=0
  export LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_END=500
  export LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_MODE=jaccard
  export LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_K=3
  export LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_MIN_SCORE=0.25
  export LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_SCOPE=all_dataset
  export LATENTFM_TRACKC_CONDITION_PRIOR_BANK_MAX_CELLS=512

  export LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WEIGHT=2.0
  export LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WARMUP_START=0
  export LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WARMUP_END=500
  export LATENTFM_TRACKC_ANCHOR_REPLAY_CONDITION_FILTER=all
  export LATENTFM_TRACKC_INIT_CHECKPOINT_USE_EMA=1
  export LATENTFM_TRACKC_ANCHOR_REPLAY_CHECKPOINT_USE_EMA=1
  export LATENTFM_TRACKC_CONDITION_DELTA_HEAD_USE_IN_MODEL=0
  export LATENTFM_TRACKC_TOTAL_STEPS=2000
  export LATENTFM_TRACKC_SMOKE_HYPOTHESIS="${hypothesis}"

  bash "${GENERIC_LAUNCHER}"
}

launch_variant \
  xverse_trackc_support_context_v2_residual_ep050_replay2_2k_seed42 \
  support_residual_adapter \
  0 \
  1 \
  0 \
  "Track C support-context v2 residual variant: explicit support context present on support-val and forced absent on canonical Track A; direct velocity residual may absorb support teacher signal with less canonical harm than FiLM scaling."

launch_variant \
  xverse_trackc_support_context_v2_contextc_ep050_replay2_2k_seed42 \
  support_context_adapter \
  1 \
  0 \
  0 \
  "Track C support-context v2 context-c variant: explicit support context present on support-val and forced absent on canonical Track A; conditioning-vector support injection tests whether support signal is better absorbed through c instead of output residual/FiLM."
