#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
LAUNCHER=${ROOT}/ops/launch_latentfm_xverse_trackc_routed_distill_smoke_20260622.sh
RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_noharm_adapter_parallel_c_20260622
OUT_ROOT=${ROOT}/CoupledFM/output/latentfm_runs/xverse_trackc_noharm_adapter_parallel_c_20260622
LOG_ROOT=${ROOT}/logs/latentfm_xverse_trackc_noharm_adapter_parallel_c_20260622
TRAINSELECT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_route_datasets_trainselect.json
BANK_SPLIT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json
MANIFEST=${ROOT}/reports/latentfm_trackc_noharm_adapter_parallel_c_manifest_20260622.jsonl
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
  local forced_gpu="$2"
  local scope="$3"
  local pairwise_mode="$4"
  local endpoint_weight="$5"
  local endpoint_start="$6"
  local endpoint_end="$7"
  local replay_weight="$8"
  local replay_filter="$9"
  local hypothesis="${10}"

  echo "[$(date '+%F %T %Z')] launching ${run_name} on GPU${forced_gpu}"
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
  LATENTFM_TRACKC_FORCE_GPU="${forced_gpu}" \
  LATENTFM_TRACKC_RELAXED_GPU_SELECTION=1 \
  LATENTFM_TRACKC_RELAXED_GPU_MIN_FREE_MIB=8192 \
  LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WEIGHT=0.0 \
  LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WARMUP_START=0 \
  LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WARMUP_END=500 \
  LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT="${endpoint_weight}" \
  LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_START="${endpoint_start}" \
  LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_END="${endpoint_end}" \
  LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WEIGHT="${replay_weight}" \
  LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WARMUP_START=0 \
  LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WARMUP_END=500 \
  LATENTFM_TRACKC_ANCHOR_REPLAY_CONDITION_FILTER="${replay_filter}" \
  LATENTFM_TRACKC_SMOKE_HYPOTHESIS="${hypothesis}" \
  bash "${LAUNCHER}"

  "${PYTHON}" - "${MANIFEST}" "${run_name}" "${forced_gpu}" "${scope}" "${pairwise_mode}" \
    "${endpoint_weight}" "${endpoint_start}" "${endpoint_end}" "${replay_weight}" "${replay_filter}" "${hypothesis}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

manifest = Path(sys.argv[1])
row = {
    "launched_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
    "run_name": sys.argv[2],
    "forced_gpu": int(sys.argv[3]),
    "finetune_trainable_scope": sys.argv[4],
    "pert_pairwise_mode": sys.argv[5],
    "endpoint_weight": float(sys.argv[6]),
    "endpoint_warmup_start": int(sys.argv[7]),
    "endpoint_warmup_end": int(sys.argv[8]),
    "anchor_replay_weight": float(sys.argv[9]),
    "anchor_replay_filter": sys.argv[10],
    "hypothesis": sys.argv[11],
    "run_status": f"/data/cyx/1030/scLatent/runs/latentfm_xverse_trackc_noharm_adapter_parallel_c_20260622/{sys.argv[2]}/RUN_STATUS.md",
    "decision_json": f"/data/cyx/1030/scLatent/reports/latentfm_trackc_routed_distill_smoke_decision_{sys.argv[2]}.json",
    "decision_md": f"/data/cyx/1030/scLatent/reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_{sys.argv[2]}.md",
}
with manifest.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

# Condition-prior adapter: stronger no-harm replay tests without changing the
# small adapter family used by the previous route-focused smokes.
launch_one xverse_trackc_noharm_cp_ep050_replay2_all_2k_seed42 0 \
  condition_prior_adapter off 0.50 0 500 2.0 all \
  "Condition-prior adapter with endpoint w0.5 and replay2/all tests whether direct canonical replay veto reduces family_gene harm while retaining support signal."
launch_one xverse_trackc_noharm_cp_ep100_replay2_all_2k_seed42 0 \
  condition_prior_adapter off 1.00 0 500 2.0 all \
  "Condition-prior adapter with endpoint w1.0 and replay2/all tests whether stronger support pressure can pass only if broader replay prevents canonical harm."
launch_one xverse_trackc_noharm_cp_ep050del_replay2_all_2k_seed42 0 \
  condition_prior_adapter off 0.50 500 1500 2.0 all \
  "Condition-prior adapter with delayed endpoint w0.5 and replay2/all tests whether delayed support pressure plus all-condition replay improves no-harm over the failed delayed baseline."
launch_one xverse_trackc_noharm_cp_ep100del_replay4_all_2k_seed42 0 \
  condition_prior_adapter off 1.00 500 1500 4.0 all \
  "Condition-prior adapter with delayed endpoint w1.0 and replay4/all tests a high no-harm veto under stronger routed teacher pressure."

# Pairwise-condition adapter: opens the interaction bridge while keeping base
# flow frozen, testing whether endpoint supervision was under-parameterized.
launch_one xverse_trackc_noharm_pc_ep050_replay2_all_2k_seed42 2 \
  pairwise_condition_adapter hadamard_mean 0.50 0 500 2.0 all \
  "Pairwise-condition adapter with endpoint w0.5 and replay2/all tests whether a small interaction bridge captures support routes without canonical drift."
launch_one xverse_trackc_noharm_pc_ep100_replay2_all_2k_seed42 2 \
  pairwise_condition_adapter hadamard_mean 1.00 0 500 2.0 all \
  "Pairwise-condition adapter with endpoint w1.0 and replay2/all tests whether wider condition capacity can turn routed teacher pressure into material support gain."
launch_one xverse_trackc_noharm_pc_ep050del_replay2_all_2k_seed42 2 \
  pairwise_condition_adapter hadamard_mean 0.50 500 1500 2.0 all \
  "Pairwise-condition adapter with delayed endpoint w0.5 and replay2/all tests whether warm anchor stabilization helps interaction capacity avoid no-harm violations."
launch_one xverse_trackc_noharm_pc_ep100del_replay4_all_2k_seed42 2 \
  pairwise_condition_adapter hadamard_mean 1.00 500 1500 4.0 all \
  "Pairwise-condition adapter with delayed endpoint w1.0 and replay4/all tests the strongest no-harm veto among interaction-bridge variants."

# Replay-filter contrast: keep support multi less constrained while increasing
# canonical preservation elsewhere; these use a third strict-empty GPU.
launch_one xverse_trackc_noharm_cp_ep050_replay4_nongm_2k_seed42 3 \
  condition_prior_adapter off 0.50 0 500 4.0 non_gene_multi \
  "Condition-prior adapter replay4/non_gene_multi tests whether stronger non-multi replay protects canonical family strata without directly damping support multi."
launch_one xverse_trackc_noharm_cp_ep100del_replay4_nongm_2k_seed42 3 \
  condition_prior_adapter off 1.00 500 1500 4.0 non_gene_multi \
  "Condition-prior adapter delayed endpoint w1.0 with replay4/non_gene_multi tests a high-pressure support route with canonical-only replay protection."
launch_one xverse_trackc_noharm_pc_ep050_replay4_nongm_2k_seed42 3 \
  pairwise_condition_adapter hadamard_mean 0.50 0 500 4.0 non_gene_multi \
  "Pairwise-condition adapter replay4/non_gene_multi tests whether interaction capacity plus canonical replay can improve support while preserving family_gene."
launch_one xverse_trackc_noharm_pc_ep100del_replay4_nongm_2k_seed42 3 \
  pairwise_condition_adapter hadamard_mean 1.00 500 1500 4.0 non_gene_multi \
  "Pairwise-condition adapter delayed endpoint w1.0 with replay4/non_gene_multi tests the high-support/high-no-harm corner without using held-out query."

echo "noharm_adapter_parallel_c_launched"
echo "manifest=${MANIFEST}"
