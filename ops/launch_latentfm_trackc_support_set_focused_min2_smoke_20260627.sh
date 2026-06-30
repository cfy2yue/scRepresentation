#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

FOOTPRINT_JSON=${ROOT}/reports/latentfm_trackc_support_set_footprint_prevalence_gate_20260627.json
FOCUSED_JSON=${ROOT}/reports/latentfm_trackc_support_set_focused_split_20260627.json
FOCUSED_SPLIT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect_supportset_min2_focused.json
SAFE_SPLIT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json
BASE_LAUNCHER=${ROOT}/ops/launch_latentfm_trackc_support_set_smoke_20260627.sh

"${PYTHON}" - "${FOOTPRINT_JSON}" "${FOCUSED_JSON}" "${FOCUSED_SPLIT}" "${SAFE_SPLIT}" <<'PY'
import json
import sys
from pathlib import Path

footprint = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
focused = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if footprint.get("status") != "trackc_support_set_footprint_prevalence_gate_pass_focused_split_next_no_gpu":
    raise SystemExit(f"footprint gate did not pass: {footprint.get('status')}")
if focused.get("status") != "trackc_support_set_focused_split_ready_no_gpu":
    raise SystemExit(f"focused split gate did not pass: {focused.get('status')}")
for path_s in sys.argv[3:]:
    path = Path(path_s)
    if not path.is_file():
        raise SystemExit(f"missing split: {path}")
counts = focused.get("counts") or {}
if int(counts.get("train") or 0) != 32 or int(counts.get("support_val_multi") or 0) != 24:
    raise SystemExit(f"unexpected focused counts: {counts}")
boundary = focused.get("boundary") or {}
if boundary.get("heldout_trackc_query_used_for_training_or_selection") or boundary.get("canonical_multi_selection_used"):
    raise SystemExit(f"unsafe focused boundary: {boundary}")
print("focused_support_set_gate_ok")
PY

export LATENTFM_TRACKC_SUPPORT_SET_RUN_NAME=${LATENTFM_TRACKC_SUPPORT_SET_RUN_NAME:-xverse_trackc_support_set_focused_min2_adapter_2k_seed42}
export LATENTFM_TRACKC_SUPPORT_SET_RUN_ROOT=${LATENTFM_TRACKC_SUPPORT_SET_RUN_ROOT:-${ROOT}/runs/latentfm_trackc_support_set_focused_20260627}
export LATENTFM_TRACKC_SUPPORT_SET_OUT_ROOT=${LATENTFM_TRACKC_SUPPORT_SET_OUT_ROOT:-${ROOT}/CoupledFM/output/latentfm_runs/trackc_support_set_focused_20260627}
export LATENTFM_TRACKC_SUPPORT_SET_LOG_ROOT=${LATENTFM_TRACKC_SUPPORT_SET_LOG_ROOT:-${ROOT}/logs/latentfm_trackc_support_set_focused_20260627}
export LATENTFM_TRACKC_SUPPORT_SET_SEED=${LATENTFM_TRACKC_SUPPORT_SET_SEED:-42}
export LATENTFM_TRACKC_SUPPORT_SET_TOTAL_STEPS=${LATENTFM_TRACKC_SUPPORT_SET_TOTAL_STEPS:-2000}
export LATENTFM_TRACKC_SUPPORT_SET_MIN_SUPPORT_COUNT=2
export LATENTFM_TRACKC_SUPPORT_SET_TRAIN_SPLIT=${FOCUSED_SPLIT}
export LATENTFM_TRACKC_SUPPORT_SET_SAFE_SPLIT=${SAFE_SPLIT}
export LATENTFM_TRACKC_SUPPORT_SET_HYPOTHESIS="Focused Track C support-set min2 smoke tests whether the previous near-inert result was caused by 0.1877% token-present exposure in full train. Train split contains only safe-trainselect token-present Norman/Wessels train_multi rows; eval remains safe support-val multi with zero/shuffle/absent controls."

bash "${BASE_LAUNCHER}"
