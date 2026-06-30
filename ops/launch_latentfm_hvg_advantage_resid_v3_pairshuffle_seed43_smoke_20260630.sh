#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

ACK=${LATENTFM_HVGADV_V3_PLACEBO_ACK:-}
if [[ "${ACK}" != "real_v3_internal_pass_pairshuffle_control" ]]; then
  cat >&2 <<'EOF'
Refusing to launch HVG-advantage residual v3 pair-shuffle placebo smoke.

Set:
  LATENTFM_HVGADV_V3_PLACEBO_ACK=real_v3_internal_pass_pairshuffle_control

Required boundary:
  - real v3 high/low smoke must first pass internal decision gate
  - placebo seed must pass CPU split-integrity and axis-collapse gate
  - this is a control only, not promotion or canonical checkpoint selection
EOF
  exit 4
fi

GATE_JSON=${LATENTFM_HVGADV_V3_GATE_JSON:-${ROOT}/reports/hvg_advantage_resid_v3_pair_pool_20260630/hvg_advantage_resid_v3_packet_audit_20260630.json}
REAL_DECISION_JSON=${LATENTFM_HVGADV_V3_REAL_DECISION_JSON:-${ROOT}/reports/hvg_advantage_resid_v3_highlow_smoke_20260630/latentfm_hvg_advantage_resid_v3_highlow_decision_20260630.json}
PLACEBO_MANIFEST=${LATENTFM_HVGADV_V3_PLACEBO_MANIFEST:-${ROOT}/reports/hvg_advantage_resid_v3_pairshuffle_placebo_splits_20260630/latentfm_hvg_advantage_resid_v3_pairshuffle_placebo_splits_20260630.json}
PLACEBO_SEED=${LATENTFM_HVGADV_V3_PLACEBO_SEED:-43}
TRAIN_SEED=${LATENTFM_HVGADV_V3_PLACEBO_TRAIN_SEED:-42}
BASE_LAUNCHER=${ROOT}/ops/launch_latentfm_scaling_v2_condition_information_highlow_smoke_20260628.sh

HIGH_SPLIT=${ROOT}/reports/hvg_advantage_resid_v3_pairshuffle_placebo_splits_20260630/split_seed42_xverse_hvg_advantage_resid_v3_pairshuffle_seed${PLACEBO_SEED}_high_from_cap120_all_v2.json
LOW_SPLIT=${ROOT}/reports/hvg_advantage_resid_v3_pairshuffle_placebo_splits_20260630/split_seed42_xverse_hvg_advantage_resid_v3_pairshuffle_seed${PLACEBO_SEED}_low_from_cap120_all_v2.json

for required in "${GATE_JSON}" "${REAL_DECISION_JSON}" "${PLACEBO_MANIFEST}" "${HIGH_SPLIT}" "${LOW_SPLIT}" "${BASE_LAUNCHER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

"${PYTHON}" - "${GATE_JSON}" "${REAL_DECISION_JSON}" "${PLACEBO_MANIFEST}" "${PLACEBO_SEED}" <<'PY'
import json
import sys
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
decision = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
manifest = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
seed = int(sys.argv[4])

if gate.get("status") != "hvg_advantage_resid_v3_pair_pool_pass_prepare_gpu_smoke":
    raise SystemExit(f"unexpected v3 gate status: {gate.get('status')!r}")
decision_status = (decision.get("decision") or {}).get("status")
if decision_status != "scaling_v2_condition_information_highlow_internal_pass_needs_placebo_noharm":
    raise SystemExit(f"real v3 high/low did not pass placebo gate: {decision_status!r}")
if manifest.get("status") not in {
    "hvg_advantage_resid_v3_pairshuffle_placebo_ready_no_gpu",
    "hvg_advantage_resid_v3_pairshuffle_placebo_partial_ready_no_gpu",
}:
    raise SystemExit(f"placebo manifest not ready: {manifest.get('status')!r}")
seed_rows = {int(row["seed"]): row for row in manifest.get("seeds", [])}
if seed not in seed_rows or seed_rows[seed].get("status") != "pass":
    raise SystemExit(f"placebo seed {seed} not pass in manifest")
PY

export LATENTFM_SCALING_V2_INFO_PACKET_JSON="${GATE_JSON}"
export LATENTFM_SCALING_V2_INFO_ALLOWED_PACKET_STATUS="hvg_advantage_resid_v3_pair_pool_pass_prepare_gpu_smoke"
export LATENTFM_SCALING_V2_INFO_HIGH_SPLIT="${HIGH_SPLIT}"
export LATENTFM_SCALING_V2_INFO_LOW_SPLIT="${LOW_SPLIT}"
export LATENTFM_SCALING_V2_INFO_RUN_ROOT="${ROOT}/runs/latentfm_hvg_advantage_resid_v3_pairshuffle_seed${PLACEBO_SEED}_smoke_20260630"
export LATENTFM_SCALING_V2_INFO_OUT_ROOT="${ROOT}/CoupledFM/output/latentfm_runs/hvg_advantage_resid_v3_pairshuffle_seed${PLACEBO_SEED}_smoke_20260630"
export LATENTFM_SCALING_V2_INFO_LOG_ROOT="${ROOT}/logs/latentfm_hvg_advantage_resid_v3_pairshuffle_seed${PLACEBO_SEED}_smoke_20260630"
export LATENTFM_SCALING_V2_INFO_REPORT_DIR="${ROOT}/reports/hvg_advantage_resid_v3_pairshuffle_seed${PLACEBO_SEED}_smoke_20260630"
export LATENTFM_SCALING_V2_INFO_RUN_PREFIX="xverse_hvgadv_resid_v3_pairshuffle_seed${PLACEBO_SEED}"
export LATENTFM_SCALING_V2_INFO_SESSION_PREFIX="lfm_hvgadvv3ps${PLACEBO_SEED}"
export LATENTFM_SCALING_V2_INFO_RUN_STATUS_TITLE="latentfm_hvg_advantage_resid_v3_pairshuffle_seed${PLACEBO_SEED}_smoke_20260630"
export LATENTFM_SCALING_V2_INFO_LAUNCH_COMMAND="LATENTFM_HVGADV_V3_PLACEBO_ACK=real_v3_internal_pass_pairshuffle_control LATENTFM_HVGADV_V3_PLACEBO_SEED=${PLACEBO_SEED} bash ${ROOT}/ops/launch_latentfm_hvg_advantage_resid_v3_pairshuffle_seed43_smoke_20260630.sh"
export LATENTFM_SCALING_V2_INFO_STEPS="${LATENTFM_HVGADV_V3_PLACEBO_STEPS:-2000}"
export LATENTFM_SCALING_V2_INFO_SEED="${TRAIN_SEED}"

bash "${BASE_LAUNCHER}"
