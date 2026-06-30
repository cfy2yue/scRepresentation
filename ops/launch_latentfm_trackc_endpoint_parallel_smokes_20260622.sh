#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
LAUNCHER=${ROOT}/ops/launch_latentfm_xverse_trackc_routed_distill_smoke_20260622.sh
RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_endpoint_parallel_20260622
OUT_ROOT=${ROOT}/CoupledFM/output/latentfm_runs/xverse_trackc_endpoint_parallel_20260622
LOG_ROOT=${ROOT}/logs/latentfm_xverse_trackc_endpoint_parallel_20260622
TRAINSELECT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_route_datasets_trainselect.json
BANK_SPLIT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json
MANIFEST=${ROOT}/reports/latentfm_trackc_endpoint_parallel_smokes_manifest_20260622.jsonl
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
  local endpoint_weight="$2"
  local endpoint_start="$3"
  local endpoint_end="$4"
  local head_weight="$5"
  local hypothesis="$6"

  echo "[$(date '+%F %T %Z')] launching ${run_name}"
  LATENTFM_TRACKC_RUN_NAME="${run_name}" \
  LATENTFM_TRACKC_RUN_ROOT="${RUN_ROOT}" \
  LATENTFM_TRACKC_OUT_ROOT="${OUT_ROOT}" \
  LATENTFM_TRACKC_LOG_ROOT="${LOG_ROOT}" \
  LATENTFM_TRACKC_TRAINSELECT_SPLIT="${TRAINSELECT}" \
  LATENTFM_TRACKC_BANK_SPLIT_FILE="${BANK_SPLIT}" \
  LATENTFM_TRACKC_INIT_CHECKPOINT_USE_EMA=1 \
  LATENTFM_TRACKC_ANCHOR_REPLAY_CHECKPOINT_USE_EMA=1 \
  LATENTFM_TRACKC_FINETUNE_TRAINABLE_SCOPE=condition_prior_adapter \
  LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WEIGHT="${head_weight}" \
  LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WARMUP_START=0 \
  LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WARMUP_END=500 \
  LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT="${endpoint_weight}" \
  LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_START="${endpoint_start}" \
  LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_END="${endpoint_end}" \
  LATENTFM_TRACKC_SMOKE_HYPOTHESIS="${hypothesis}" \
  bash "${LAUNCHER}"

  "${PYTHON}" - "${MANIFEST}" "${run_name}" "${endpoint_weight}" "${endpoint_start}" "${endpoint_end}" "${head_weight}" "${hypothesis}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
row = {
    "launched_at": __import__("datetime").datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
    "run_name": sys.argv[2],
    "endpoint_weight": float(sys.argv[3]),
    "endpoint_warmup_start": int(sys.argv[4]),
    "endpoint_warmup_end": int(sys.argv[5]),
    "head_distill_weight": float(sys.argv[6]),
    "hypothesis": sys.argv[7],
    "run_status": f"/data/cyx/1030/scLatent/runs/latentfm_xverse_trackc_endpoint_parallel_20260622/{sys.argv[2]}/RUN_STATUS.md",
    "decision_json": f"/data/cyx/1030/scLatent/reports/latentfm_trackc_routed_distill_smoke_decision_{sys.argv[2]}.json",
    "decision_md": f"/data/cyx/1030/scLatent/reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_{sys.argv[2]}.md",
}
with manifest.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

launch_one \
  xverse_trackc_endpoint_w025_replay1_2k_seed42 \
  0.25 0 500 0.0 \
  "Lower endpoint-routed teacher weight tests whether conservative endpoint supervision can retain canonical no-harm while still producing material support-val gain."

sleep 20

launch_one \
  xverse_trackc_endpoint_w100_replay1_2k_seed42 \
  1.0 0 500 0.0 \
  "Higher endpoint-routed teacher weight tests whether the pending w0.5 branch is underpowered on support-val route capture."

sleep 20

launch_one \
  xverse_trackc_endpoint_w050_head010_replay1_2k_seed42 \
  0.5 0 500 0.1 \
  "Mixed endpoint and small head distillation tests whether a weak condition-delta head signal stabilizes routed teacher learning without returning to the failed head-only objective."

sleep 20

launch_one \
  xverse_trackc_endpoint_w050_delayed500_1500_replay1_2k_seed42 \
  0.5 500 1500 0.0 \
  "Delayed endpoint-routed supervision tests whether giving EMA anchor replay the first 500 steps reduces canonical no-harm damage while preserving support-val gain."

echo "parallel_endpoint_smokes_launched"
echo "manifest=${MANIFEST}"
