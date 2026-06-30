#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
LAUNCHER=${ROOT}/ops/launch_latentfm_xverse_trackc_routed_distill_smoke_20260622.sh
RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_head_noharm_parallel_d_20260622
OUT_ROOT=${ROOT}/CoupledFM/output/latentfm_runs/xverse_trackc_head_noharm_parallel_d_20260622
LOG_ROOT=${ROOT}/logs/latentfm_xverse_trackc_head_noharm_parallel_d_20260622
TRAINSELECT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_route_datasets_trainselect.json
BANK_SPLIT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json
MANIFEST=${ROOT}/reports/latentfm_trackc_head_noharm_parallel_d_manifest_20260622.jsonl
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

mkdir -p "${RUN_ROOT}" "${OUT_ROOT}" "${LOG_ROOT}" "${ROOT}/reports"
if [[ -e "${MANIFEST}" ]]; then
  echo "Manifest already exists: ${MANIFEST}" >&2
  exit 2
fi

launch_one() {
  local run_name="$1"
  local scope="$2"
  local pairwise_mode="$3"
  local head_weight="$4"
  local replay_weight="$5"
  local replay_filter="$6"
  local hypothesis="$7"

  echo "[$(date '+%F %T %Z')] launching ${run_name} on GPU0"
  LATENTFM_TRACKC_RUN_NAME="${run_name}" \
  LATENTFM_TRACKC_RUN_ROOT="${RUN_ROOT}" \
  LATENTFM_TRACKC_OUT_ROOT="${OUT_ROOT}" \
  LATENTFM_TRACKC_LOG_ROOT="${LOG_ROOT}" \
  LATENTFM_TRACKC_TRAINSELECT_SPLIT="${TRAINSELECT}" \
  LATENTFM_TRACKC_BANK_SPLIT_FILE="${BANK_SPLIT}" \
  LATENTFM_TRACKC_INIT_CHECKPOINT_USE_EMA=1 \
  LATENTFM_TRACKC_ANCHOR_REPLAY_CHECKPOINT_USE_EMA=1 \
  LATENTFM_TRACKC_FINETUNE_TRAINABLE_SCOPE="${scope}" \
  LATENTFM_TRACKC_PERT_PAIRWISE_MODE="${pairwise_mode}" \
  LATENTFM_TRACKC_FORCE_GPU=0 \
  LATENTFM_TRACKC_RELAXED_GPU_SELECTION=1 \
  LATENTFM_TRACKC_RELAXED_GPU_MIN_FREE_MIB=8192 \
  LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WEIGHT="${head_weight}" \
  LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WARMUP_START=0 \
  LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WARMUP_END=500 \
  LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT=0.0 \
  LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_START=0 \
  LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_END=500 \
  LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WEIGHT="${replay_weight}" \
  LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WARMUP_START=0 \
  LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WARMUP_END=500 \
  LATENTFM_TRACKC_ANCHOR_REPLAY_CONDITION_FILTER="${replay_filter}" \
  LATENTFM_TRACKC_SMOKE_HYPOTHESIS="${hypothesis}" \
  bash "${LAUNCHER}"

  "${PYTHON}" - "${MANIFEST}" "${run_name}" "${scope}" "${pairwise_mode}" \
    "${head_weight}" "${replay_weight}" "${replay_filter}" "${hypothesis}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

manifest = Path(sys.argv[1])
row = {
    "launched_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
    "run_name": sys.argv[2],
    "forced_gpu": 0,
    "finetune_trainable_scope": sys.argv[3],
    "pert_pairwise_mode": sys.argv[4],
    "head_distill_weight": float(sys.argv[5]),
    "endpoint_weight": 0.0,
    "anchor_replay_weight": float(sys.argv[6]),
    "anchor_replay_filter": sys.argv[7],
    "hypothesis": sys.argv[8],
    "run_status": f"/data/cyx/1030/scLatent/runs/latentfm_xverse_trackc_head_noharm_parallel_d_20260622/{sys.argv[2]}/RUN_STATUS.md",
    "decision_json": f"/data/cyx/1030/scLatent/reports/latentfm_trackc_routed_distill_smoke_decision_{sys.argv[2]}.json",
    "decision_md": f"/data/cyx/1030/scLatent/reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_{sys.argv[2]}.md",
}
with manifest.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

launch_one xverse_trackc_head_cp_w025_replay2_all_2k_seed42 \
  condition_prior_adapter off 0.25 2.0 all \
  "Condition-prior adapter head-distill w0.25 with replay2/all tests route-head supervision without endpoint cell MSE under direct canonical replay."
launch_one xverse_trackc_head_cp_w050_replay4_nongm_2k_seed42 \
  condition_prior_adapter off 0.50 4.0 non_gene_multi \
  "Condition-prior adapter head-distill w0.5 with replay4/non_gene_multi tests whether stronger head route pressure works when canonical non-multi replay is stronger."
launch_one xverse_trackc_head_pc_w025_replay2_all_2k_seed42 \
  pairwise_condition_adapter hadamard_mean 0.25 2.0 all \
  "Pairwise-condition adapter head-distill w0.25 with replay2/all tests whether interaction capacity improves routed head learning without endpoint MSE."
launch_one xverse_trackc_head_pc_w050_replay4_nongm_2k_seed42 \
  pairwise_condition_adapter hadamard_mean 0.50 4.0 non_gene_multi \
  "Pairwise-condition adapter head-distill w0.5 with replay4/non_gene_multi tests the high head-route pressure and canonical replay corner."

echo "head_noharm_parallel_d_launched"
echo "manifest=${MANIFEST}"
