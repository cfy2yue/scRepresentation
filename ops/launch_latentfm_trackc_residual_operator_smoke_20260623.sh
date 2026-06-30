#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
LAUNCHER=${ROOT}/ops/launch_latentfm_xverse_trackc_routed_distill_smoke_20260622.sh
RUN_NAME=${LATENTFM_TRACKC_RESIDUAL_RUN_NAME:-xverse_trackc_residual_operator_memall_resid_ep050_replay2_2k_seed42}
RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_residual_operator_20260623
OUT_ROOT=${ROOT}/CoupledFM/output/latentfm_runs/xverse_trackc_residual_operator_20260623
LOG_ROOT=${ROOT}/logs/latentfm_xverse_trackc_residual_operator_20260623
TRAINSELECT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json
BANK_SPLIT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json
ROUTE_FILE=${ROOT}/reports/latentfm_trackc_residual_operator_route_teacher_20260623.json
CPU_GATE_JSON=${ROOT}/reports/latentfm_trackc_residual_operator_cpu_gate_20260623.json
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

mkdir -p "${RUN_ROOT}" "${OUT_ROOT}" "${LOG_ROOT}" "${ROOT}/reports"

"${PYTHON}" - "${CPU_GATE_JSON}" "${ROUTE_FILE}" <<'PY'
import json
import sys
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
route = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
decision = gate.get("decision") or {}
if decision.get("status") != "residual_operator_cpu_gate_pass_authorize_one_capped_gpu_smoke":
    raise SystemExit(f"residual operator CPU gate did not pass: {decision.get('status')}")
if decision.get("gpu_authorization") != "one_capped_trackc_support_only_smoke":
    raise SystemExit(f"unexpected gpu authorization: {decision.get('gpu_authorization')}")
if route.get("route") != {
    "NormanWeissman2019_filtered": "train_multi_memory",
    "Wessels": "train_multi_memory",
}:
    raise SystemExit("route file is not the frozen train_multi_memory route")
rule = route.get("memory_rule") or {}
if (rule.get("mode"), int(rule.get("k")), rule.get("scope"), float(rule.get("min_score"))) != (
    "jaccard",
    3,
    "all_dataset",
    0.25,
):
    raise SystemExit(f"unexpected memory rule: {rule}")
PY

LATENTFM_TRACKC_RUN_NAME="${RUN_NAME}" \
LATENTFM_TRACKC_RUN_ROOT="${RUN_ROOT}" \
LATENTFM_TRACKC_OUT_ROOT="${OUT_ROOT}" \
LATENTFM_TRACKC_LOG_ROOT="${LOG_ROOT}" \
LATENTFM_TRACKC_TRAINSELECT_SPLIT="${TRAINSELECT}" \
LATENTFM_TRACKC_BANK_SPLIT_FILE="${BANK_SPLIT}" \
LATENTFM_TRACKC_ROUTE_FILE="${ROUTE_FILE}" \
LATENTFM_TRACKC_INIT_CHECKPOINT_USE_EMA=1 \
LATENTFM_TRACKC_ANCHOR_REPLAY_CHECKPOINT_USE_EMA=1 \
LATENTFM_TRACKC_FINETUNE_TRAINABLE_SCOPE=support_residual_adapter \
LATENTFM_TRACKC_TOTAL_STEPS=2000 \
LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WEIGHT=0.0 \
LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT=0.50 \
LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_START=0 \
LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_END=500 \
LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_MODE=jaccard \
LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_K=3 \
LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_MIN_SCORE=0.25 \
LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_SCOPE=all_dataset \
LATENTFM_TRACKC_SUPPORT_CONTEXT_USE_IN_MODEL=0 \
LATENTFM_TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL=1 \
LATENTFM_TRACKC_SUPPORT_CONTEXT_DIM=384 \
LATENTFM_TRACKC_SUPPORT_CONTEXT_SOURCE=routed_distill_target \
LATENTFM_TRACKC_CONDITION_PRIOR_BANK_MAX_CELLS=256 \
LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WEIGHT=2.0 \
LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WARMUP_START=0 \
LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WARMUP_END=500 \
LATENTFM_TRACKC_ANCHOR_REPLAY_CONDITION_FILTER=all \
LATENTFM_TRACKC_CONDITION_DELTA_HEAD_USE_IN_MODEL=0 \
LATENTFM_TRACKC_PERT_PAIRWISE_MODE=off \
LATENTFM_TRACKC_RELAXED_GPU_SELECTION=1 \
LATENTFM_TRACKC_RELAXED_GPU_MIN_FREE_MIB=8192 \
LATENTFM_TRACKC_SMOKE_HYPOTHESIS="CPU gate passed for a support-conditioned residual operator; train only support_context_to_v on safe trainselect with train_multi_memory all-dataset support context, then require support-val pass before any query." \
bash "${LAUNCHER}"

echo "residual_operator_smoke_launched"
echo "run_status=${RUN_ROOT}/${RUN_NAME}/RUN_STATUS.md"
