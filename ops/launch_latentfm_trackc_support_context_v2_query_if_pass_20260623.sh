#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM

RUN_NAME=${LATENTFM_TRACKC_V2_RUN_NAME:-}
if [[ -z "${RUN_NAME}" ]]; then
  echo "Set LATENTFM_TRACKC_V2_RUN_NAME to the frozen support-context v2 run name." >&2
  echo "This wrapper intentionally has no default because query is one-shot." >&2
  exit 2
fi

case "${RUN_NAME}" in
  xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42)
    DEFAULT_OUT_ROOT=${COUPLED}/output/latentfm_runs/xverse_trackc_support_context_v2_20260623
    ;;
  xverse_trackc_support_context_v2_residual_ep050_replay2_2k_seed42|\
  xverse_trackc_support_context_v2_contextc_ep050_replay2_2k_seed42)
    DEFAULT_OUT_ROOT=${COUPLED}/output/latentfm_runs/xverse_trackc_support_context_v2_parallel_20260623
    ;;
  *)
    echo "Unsupported v2 query run name: ${RUN_NAME}" >&2
    exit 2
    ;;
esac

SAFE_RUN_ID=$(printf '%s' "${RUN_NAME}" | tr -c 'A-Za-z0-9_' '_')
UNCAPPED_LABEL=${LATENTFM_TRACKC_V2_UNCAPPED_LABEL:-latentfm_trackc_support_context_v2_uncapped_noharm_${SAFE_RUN_ID}_20260623}
OUT_ROOT=${LATENTFM_TRACKC_V2_OUT_ROOT:-${DEFAULT_OUT_ROOT}}
FREEZE_JSON=${LATENTFM_TRACKC_V2_QUERY_FREEZE_JSON:-${ROOT}/reports/latentfm_trackc_support_context_v2_query_freeze_${SAFE_RUN_ID}_20260623.json}

PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ ! -f "${FREEZE_JSON}" ]]; then
  echo "Missing required v2 query freeze artifact: ${FREEZE_JSON}" >&2
  echo "Run ops/audit_latentfm_trackc_support_context_v2_query_freeze_gate_20260623.py after uncapped no-harm passes." >&2
  exit 2
fi

"${PYTHON}" - "${FREEZE_JSON}" "${RUN_NAME}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
run_name = sys.argv[2]
expected = "trackc_support_context_v2_query_freeze_pass_query_allowed_once"
if payload.get("status") != expected:
    raise SystemExit(f"v2 query freeze status is {payload.get('status')!r}, not {expected!r}")
if payload.get("run_name") != run_name:
    raise SystemExit(f"v2 query freeze run_name mismatch: {payload.get('run_name')!r} != {run_name!r}")
if payload.get("query_authorization") != "one_shot_query_allowed":
    raise SystemExit("v2 query freeze did not authorize one-shot query")
PY

export LATENTFM_TRACKC_QUERY_RUN_NAME=${RUN_NAME}
export LATENTFM_TRACKC_QUERY_LABEL=${LATENTFM_TRACKC_QUERY_LABEL:-latentfm_trackc_support_context_v2_query_once_${SAFE_RUN_ID}_20260623}
export LATENTFM_TRACKC_QUERY_CANDIDATE_CKPT=${LATENTFM_TRACKC_QUERY_CANDIDATE_CKPT:-${OUT_ROOT}/${RUN_NAME}/best.pt}
export LATENTFM_TRACKC_QUERY_SMOKE_DECISION=${LATENTFM_TRACKC_QUERY_SMOKE_DECISION:-${ROOT}/reports/latentfm_trackc_routed_distill_smoke_decision_${RUN_NAME}.json}
export LATENTFM_TRACKC_QUERY_UNCAPPED_DECISION=${LATENTFM_TRACKC_QUERY_UNCAPPED_DECISION:-${ROOT}/reports/${UNCAPPED_LABEL}_decision.json}
export LATENTFM_TRACKC_QUERY_DECISION_JSON=${LATENTFM_TRACKC_QUERY_DECISION_JSON:-${ROOT}/reports/latentfm_trackc_support_context_v2_query_once_decision_${SAFE_RUN_ID}_20260623.json}
export LATENTFM_TRACKC_QUERY_DECISION_MD=${LATENTFM_TRACKC_QUERY_DECISION_MD:-${ROOT}/reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_QUERY_ONCE_DECISION_${SAFE_RUN_ID}_20260623.md}
export LATENTFM_TRACKC_QUERY_BOOT_DIR=${LATENTFM_TRACKC_QUERY_BOOT_DIR:-${ROOT}/reports/latentfm_trackc_support_context_v2_query_once_bootstrap_${SAFE_RUN_ID}_20260623}
export LATENTFM_TRACKC_QUERY_REPORT_TITLE=${LATENTFM_TRACKC_QUERY_REPORT_TITLE:-LatentFM Track C Support-Context V2 One-Shot Query Decision: ${RUN_NAME}}

bash ${ROOT}/ops/launch_latentfm_trackc_routefocus_query_if_pass_20260622.sh
