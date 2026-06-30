#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

ACK=${LATENTFM_CNH_RESPONSE_RESID_PLACEBO_ACK:-}
if [[ "${ACK}" != "real_internal_pass_pairshuffle_control" ]]; then
  cat >&2 <<'EOF'
Refusing to launch response-residualized support pair-shuffle placebo smoke.

Set:
  LATENTFM_CNH_RESPONSE_RESID_PLACEBO_ACK=real_internal_pass_pairshuffle_control

Required boundary:
  - real response-residualized high/low smoke must pass internal summary gate
  - pair-shuffle placebo split package must be CPU-ready
  - this is a control only, not promotion or canonical checkpoint selection
EOF
  exit 4
fi

GATE_JSON=${LATENTFM_CNH_RESPONSE_RESID_GATE_JSON:-${ROOT}/reports/condition_neighborhood_response_residualized_support_gate_20260629/latentfm_condition_neighborhood_response_residualized_support_gate_20260629.json}
REAL_DECISION_JSON=${LATENTFM_CNH_RESPONSE_RESID_REAL_DECISION_JSON:-${ROOT}/reports/condition_neighborhood_response_resid_highlow_smoke_20260629/latentfm_condition_neighborhood_response_resid_highlow_decision_20260629.json}
PLACEBO_MANIFEST=${LATENTFM_CNH_RESPONSE_RESID_PLACEBO_MANIFEST:-${ROOT}/reports/condition_neighborhood_response_resid_pairshuffle_placebo_splits_20260629/latentfm_condition_neighborhood_response_resid_pairshuffle_placebo_splits_20260629.json}
PLACEBO_SEED=${LATENTFM_CNH_RESPONSE_RESID_PLACEBO_SEED:-43}
TRAIN_SEED=${LATENTFM_CNH_RESPONSE_RESID_PLACEBO_TRAIN_SEED:-42}
BASE_LAUNCHER=${ROOT}/ops/launch_latentfm_scaling_v2_condition_information_highlow_smoke_20260628.sh

HIGH_SPLIT=${ROOT}/dataset/biFlow_data/xverse_condition_neighborhood_support_splits_20260629/split_seed42_xverse_condition_neighborhood_response_resid_pairshuffle_seed${PLACEBO_SEED}_high_320pair.json
LOW_SPLIT=${ROOT}/dataset/biFlow_data/xverse_condition_neighborhood_support_splits_20260629/split_seed42_xverse_condition_neighborhood_response_resid_pairshuffle_seed${PLACEBO_SEED}_low_320pair.json

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

if gate.get("status") != "condition_neighborhood_response_residualized_support_pass_external_audit_no_gpu":
    raise SystemExit(f"unexpected gate status: {gate.get('status')!r}")
if (decision.get("decision") or {}).get("status") != "scaling_v2_condition_information_highlow_internal_pass_needs_placebo_noharm":
    raise SystemExit(f"real high/low did not pass placebo gate: {(decision.get('decision') or {}).get('status')!r}")
if manifest.get("status") != "condition_neighborhood_response_resid_pairshuffle_placebo_ready_no_gpu":
    raise SystemExit(f"placebo manifest not ready: {manifest.get('status')!r}")
seed_rows = {int(row["seed"]): row for row in manifest.get("seeds", [])}
if seed not in seed_rows or seed_rows[seed].get("status") != "pass":
    raise SystemExit(f"placebo seed {seed} not pass in manifest")
PY

export LATENTFM_SCALING_V2_INFO_PACKET_JSON="${GATE_JSON}"
export LATENTFM_SCALING_V2_INFO_ALLOWED_PACKET_STATUS="condition_neighborhood_response_residualized_support_pass_external_audit_no_gpu"
export LATENTFM_SCALING_V2_INFO_HIGH_SPLIT="${HIGH_SPLIT}"
export LATENTFM_SCALING_V2_INFO_LOW_SPLIT="${LOW_SPLIT}"
export LATENTFM_SCALING_V2_INFO_RUN_ROOT="${ROOT}/runs/latentfm_condition_neighborhood_response_resid_pairshuffle_seed${PLACEBO_SEED}_smoke_20260629"
export LATENTFM_SCALING_V2_INFO_OUT_ROOT="${ROOT}/CoupledFM/output/latentfm_runs/condition_neighborhood_response_resid_pairshuffle_seed${PLACEBO_SEED}_smoke_20260629"
export LATENTFM_SCALING_V2_INFO_LOG_ROOT="${ROOT}/logs/latentfm_condition_neighborhood_response_resid_pairshuffle_seed${PLACEBO_SEED}_smoke_20260629"
export LATENTFM_SCALING_V2_INFO_REPORT_DIR="${ROOT}/reports/condition_neighborhood_response_resid_pairshuffle_seed${PLACEBO_SEED}_smoke_20260629"
export LATENTFM_SCALING_V2_INFO_RUN_PREFIX="xverse_condition_neighborhood_response_resid_pairshuffle_seed${PLACEBO_SEED}"
export LATENTFM_SCALING_V2_INFO_SESSION_PREFIX="lfm_cnhrrps${PLACEBO_SEED}"
export LATENTFM_SCALING_V2_INFO_RUN_STATUS_TITLE="latentfm_condition_neighborhood_response_resid_pairshuffle_seed${PLACEBO_SEED}_smoke_20260629"
export LATENTFM_SCALING_V2_INFO_LAUNCH_COMMAND="LATENTFM_CNH_RESPONSE_RESID_PLACEBO_ACK=real_internal_pass_pairshuffle_control bash ${ROOT}/ops/launch_latentfm_condition_neighborhood_response_resid_pairshuffle_seed43_smoke_20260629.sh"
export LATENTFM_SCALING_V2_INFO_STEPS="${LATENTFM_CNH_RESPONSE_RESID_PLACEBO_STEPS:-2000}"
export LATENTFM_SCALING_V2_INFO_SEED="${TRAIN_SEED}"

bash "${BASE_LAUNCHER}"
