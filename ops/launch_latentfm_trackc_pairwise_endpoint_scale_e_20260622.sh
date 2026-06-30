#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
LAUNCHER=${ROOT}/ops/launch_latentfm_xverse_trackc_routed_distill_smoke_20260622.sh
RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_pairwise_endpoint_scale_e_20260622
OUT_ROOT=${ROOT}/CoupledFM/output/latentfm_runs/xverse_trackc_pairwise_endpoint_scale_e_20260622
LOG_ROOT=${ROOT}/logs/latentfm_xverse_trackc_pairwise_endpoint_scale_e_20260622
TRAINSELECT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_route_datasets_trainselect.json
BANK_SPLIT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json
MANIFEST=${ROOT}/reports/latentfm_trackc_pairwise_endpoint_scale_e_manifest_20260622.jsonl
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
  local endpoint_weight="$3"
  local endpoint_start="$4"
  local endpoint_end="$5"
  local replay_weight="$6"
  local replay_filter="$7"
  local hypothesis="$8"

  echo "[$(date '+%F %T %Z')] launching ${run_name} on GPU${forced_gpu}"
  LATENTFM_TRACKC_RUN_NAME="${run_name}" \
  LATENTFM_TRACKC_RUN_ROOT="${RUN_ROOT}" \
  LATENTFM_TRACKC_OUT_ROOT="${OUT_ROOT}" \
  LATENTFM_TRACKC_LOG_ROOT="${LOG_ROOT}" \
  LATENTFM_TRACKC_TRAINSELECT_SPLIT="${TRAINSELECT}" \
  LATENTFM_TRACKC_BANK_SPLIT_FILE="${BANK_SPLIT}" \
  LATENTFM_TRACKC_INIT_CHECKPOINT_USE_EMA=1 \
  LATENTFM_TRACKC_ANCHOR_REPLAY_CHECKPOINT_USE_EMA=1 \
  LATENTFM_TRACKC_FINETUNE_TRAINABLE_SCOPE=pairwise_condition_adapter \
  LATENTFM_TRACKC_PERT_PAIRWISE_MODE=hadamard_mean \
  LATENTFM_TRACKC_FORCE_GPU="${forced_gpu}" \
  LATENTFM_TRACKC_RELAXED_GPU_SELECTION=1 \
  LATENTFM_TRACKC_RELAXED_GPU_MIN_FREE_MIB=8192 \
  LATENTFM_TRACKC_TOTAL_STEPS=4000 \
  LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WEIGHT=0.0 \
  LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WARMUP_START=0 \
  LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WARMUP_END=1000 \
  LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT="${endpoint_weight}" \
  LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_START="${endpoint_start}" \
  LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_END="${endpoint_end}" \
  LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WEIGHT="${replay_weight}" \
  LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WARMUP_START=0 \
  LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WARMUP_END=1000 \
  LATENTFM_TRACKC_ANCHOR_REPLAY_CONDITION_FILTER="${replay_filter}" \
  LATENTFM_TRACKC_SMOKE_HYPOTHESIS="${hypothesis}" \
  bash "${LAUNCHER}"

  "${PYTHON}" - "${MANIFEST}" "${run_name}" "${forced_gpu}" "${endpoint_weight}" \
    "${endpoint_start}" "${endpoint_end}" "${replay_weight}" "${replay_filter}" "${hypothesis}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

manifest = Path(sys.argv[1])
row = {
    "launched_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
    "run_name": sys.argv[2],
    "forced_gpu": int(sys.argv[3]),
    "finetune_trainable_scope": "pairwise_condition_adapter",
    "pert_pairwise_mode": "hadamard_mean",
    "total_steps": 4000,
    "endpoint_weight": float(sys.argv[4]),
    "endpoint_warmup_start": int(sys.argv[5]),
    "endpoint_warmup_end": int(sys.argv[6]),
    "head_distill_weight": 0.0,
    "anchor_replay_weight": float(sys.argv[7]),
    "anchor_replay_filter": sys.argv[8],
    "hypothesis": sys.argv[9],
    "run_status": f"/data/cyx/1030/scLatent/runs/latentfm_xverse_trackc_pairwise_endpoint_scale_e_20260622/{sys.argv[2]}/RUN_STATUS.md",
    "decision_json": f"/data/cyx/1030/scLatent/reports/latentfm_trackc_routed_distill_smoke_decision_{sys.argv[2]}.json",
    "decision_md": f"/data/cyx/1030/scLatent/reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_{sys.argv[2]}.md",
}
with manifest.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

# Immediate endpoint pressure with replay2/all. This extends the best C-block
# pairwise-condition endpoint pattern from 2k to 4k and raises teacher pressure.
launch_one xverse_trackc_pcscale_ep100_replay2_all_4k_seed42 0 1.00 0 1000 2.0 all \
  "Pairwise-condition endpoint scale-up: 4k steps, endpoint w1.0, replay2/all tests whether the C-block partial +0.0106 support signal grows with training length while no-harm remains clean."
launch_one xverse_trackc_pcscale_ep150_replay2_all_4k_seed42 0 1.50 0 1000 2.0 all \
  "Pairwise-condition endpoint scale-up: endpoint w1.5 tests whether stronger routed endpoint pressure can reach the +0.02 support gate under replay2/all."
launch_one xverse_trackc_pcscale_ep200_replay2_all_4k_seed42 0 2.00 0 1000 2.0 all \
  "Pairwise-condition endpoint scale-up: endpoint w2.0 tests support-gain dose response with all-condition replay."
launch_one xverse_trackc_pcscale_ep300_replay2_all_4k_seed42 0 3.00 0 1000 2.0 all \
  "Pairwise-condition endpoint scale-up: endpoint w3.0 is an aggressive support-pressure probe with canonical no-harm as veto."

# Delayed endpoint pressure with stronger all-condition replay. This preserves
# anchor behavior for longer before exposing support teacher pressure.
launch_one xverse_trackc_pcscale_ep100del_replay4_all_4k_seed42 1 1.00 1000 3000 4.0 all \
  "Delayed pairwise-condition endpoint scale-up: w1.0 with replay4/all tests whether longer anchor stabilization improves support gain without harm."
launch_one xverse_trackc_pcscale_ep150del_replay4_all_4k_seed42 1 1.50 1000 3000 4.0 all \
  "Delayed pairwise-condition endpoint scale-up: w1.5 with replay4/all tests stronger support pressure after anchor stabilization."
launch_one xverse_trackc_pcscale_ep200del_replay4_all_4k_seed42 1 2.00 1000 3000 4.0 all \
  "Delayed pairwise-condition endpoint scale-up: w2.0 with replay4/all tests dose response under the strongest all-condition replay used in C."
launch_one xverse_trackc_pcscale_ep300del_replay4_all_4k_seed42 1 3.00 1000 3000 4.0 all \
  "Delayed pairwise-condition endpoint scale-up: w3.0 with replay4/all tests the high-pressure/high-replay corner."

# Non-gene-multi replay contrast: protect canonical strata without directly
# constraining gene-multi support batches.
launch_one xverse_trackc_pcscale_ep100_replay4_nongm_4k_seed42 2 1.00 0 1000 4.0 non_gene_multi \
  "Pairwise-condition endpoint scale-up with replay4/non_gene_multi tests whether support-gene-multi freedom helps exceed +0.02 while preserving canonical strata."
launch_one xverse_trackc_pcscale_ep150_replay4_nongm_4k_seed42 2 1.50 0 1000 4.0 non_gene_multi \
  "Pairwise-condition endpoint scale-up with w1.5 and replay4/non_gene_multi tests support-pressure dose with canonical-only replay."
launch_one xverse_trackc_pcscale_ep200_replay4_nongm_4k_seed42 2 2.00 0 1000 4.0 non_gene_multi \
  "Pairwise-condition endpoint scale-up with w2.0 and replay4/non_gene_multi tests the most plausible path to material support gain without family harm."
launch_one xverse_trackc_pcscale_ep300_replay4_nongm_4k_seed42 2 3.00 0 1000 4.0 non_gene_multi \
  "Pairwise-condition endpoint scale-up with w3.0 and replay4/non_gene_multi tests aggressive support pressure, closed immediately if no-harm fails."

echo "pairwise_endpoint_scale_e_launched"
echo "manifest=${MANIFEST}"
