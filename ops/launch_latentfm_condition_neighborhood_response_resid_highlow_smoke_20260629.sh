#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

ACK=${LATENTFM_CNH_RESPONSE_RESID_ACK:-}
if [[ "${ACK}" != "external_audit_passed_bounded_smoke" ]]; then
  cat >&2 <<'EOF'
Refusing to launch condition-neighborhood response-residualized high/low smoke.

Set:
  LATENTFM_CNH_RESPONSE_RESID_ACK=external_audit_passed_bounded_smoke

Required boundary:
  - response-residualized support CPU gate must pass
  - external audit must approve bounded GPU smoke
  - no canonical multi or Track C query selection
  - train/internal selection only; frozen canonical no-harm comes after internal pass
EOF
  exit 4
fi

GATE_JSON=${LATENTFM_CNH_RESPONSE_RESID_GATE_JSON:-${ROOT}/reports/condition_neighborhood_response_residualized_support_gate_20260629/latentfm_condition_neighborhood_response_residualized_support_gate_20260629.json}
HIGH_SPLIT=${LATENTFM_CNH_RESPONSE_RESID_HIGH_SPLIT:-${ROOT}/dataset/biFlow_data/xverse_condition_neighborhood_support_splits_20260629/split_seed42_xverse_condition_neighborhood_high_support_response_resid_320pair_q30_resp0.35_cell0.75_ds1.json}
LOW_SPLIT=${LATENTFM_CNH_RESPONSE_RESID_LOW_SPLIT:-${ROOT}/dataset/biFlow_data/xverse_condition_neighborhood_support_splits_20260629/split_seed42_xverse_condition_neighborhood_low_support_response_resid_320pair_q30_resp0.35_cell0.75_ds1.json}
AUDIT_OK_JSON=${LATENTFM_CNH_RESPONSE_RESID_AUDIT_OK_JSON:-${ROOT}/reports/condition_neighborhood_response_residualized_support_gate_20260629/EXTERNAL_AUDIT_DECISION.json}
BASE_LAUNCHER=${ROOT}/ops/launch_latentfm_scaling_v2_condition_information_highlow_smoke_20260628.sh

for required in "${GATE_JSON}" "${HIGH_SPLIT}" "${LOW_SPLIT}" "${BASE_LAUNCHER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

"${PYTHON}" - "${GATE_JSON}" "${AUDIT_OK_JSON}" <<'PY'
import json
import sys
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if gate.get("status") != "condition_neighborhood_response_residualized_support_pass_external_audit_no_gpu":
    raise SystemExit(f"gate status does not authorize audit-stage launcher: {gate.get('status')!r}")
audit_path = Path(sys.argv[2])
if not audit_path.exists():
    raise SystemExit(f"external audit decision JSON is required before launch: {audit_path}")
audit = json.loads(audit_path.read_text(encoding="utf-8"))
if audit.get("status") != "external_audit_pass_bounded_gpu_smoke":
    raise SystemExit(f"external audit did not approve bounded GPU smoke: {audit.get('status')!r}")
PY

export LATENTFM_SCALING_V2_INFO_PACKET_JSON="${GATE_JSON}"
export LATENTFM_SCALING_V2_INFO_ALLOWED_PACKET_STATUS="condition_neighborhood_response_residualized_support_pass_external_audit_no_gpu"
export LATENTFM_SCALING_V2_INFO_HIGH_SPLIT="${HIGH_SPLIT}"
export LATENTFM_SCALING_V2_INFO_LOW_SPLIT="${LOW_SPLIT}"
export LATENTFM_SCALING_V2_INFO_RUN_ROOT="${ROOT}/runs/latentfm_condition_neighborhood_response_resid_highlow_smoke_20260629"
export LATENTFM_SCALING_V2_INFO_OUT_ROOT="${ROOT}/CoupledFM/output/latentfm_runs/condition_neighborhood_response_resid_highlow_smoke_20260629"
export LATENTFM_SCALING_V2_INFO_LOG_ROOT="${ROOT}/logs/latentfm_condition_neighborhood_response_resid_highlow_smoke_20260629"
export LATENTFM_SCALING_V2_INFO_REPORT_DIR="${ROOT}/reports/condition_neighborhood_response_resid_highlow_smoke_20260629"
export LATENTFM_SCALING_V2_INFO_RUN_PREFIX="xverse_condition_neighborhood_response_resid"
export LATENTFM_SCALING_V2_INFO_SESSION_PREFIX="lfm_cnhrr"
export LATENTFM_SCALING_V2_INFO_RUN_STATUS_TITLE="latentfm_condition_neighborhood_response_resid_highlow_smoke_20260629"
export LATENTFM_SCALING_V2_INFO_LAUNCH_COMMAND="LATENTFM_CNH_RESPONSE_RESID_ACK=external_audit_passed_bounded_smoke bash ${ROOT}/ops/launch_latentfm_condition_neighborhood_response_resid_highlow_smoke_20260629.sh"
export LATENTFM_SCALING_V2_INFO_STEPS="${LATENTFM_CNH_RESPONSE_RESID_STEPS:-2000}"
export LATENTFM_SCALING_V2_INFO_SEED="${LATENTFM_CNH_RESPONSE_RESID_SEED:-42}"

bash "${BASE_LAUNCHER}"
