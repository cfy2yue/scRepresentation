#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM

RUN_NAME=${LATENTFM_TRACKC_SUPPORT_CONTEXT_QUERY_RUN_NAME:-}
if [[ -z "${RUN_NAME}" ]]; then
  echo "Set LATENTFM_TRACKC_SUPPORT_CONTEXT_QUERY_RUN_NAME to the frozen support-context run name." >&2
  echo "This wrapper intentionally has no default because query is one-shot and must target a frozen checkpoint." >&2
  exit 2
fi

case "${RUN_NAME}" in
  xverse_trackc_ctx_bridge_fm_2k_seed42|\
  xverse_trackc_ctx_bridge_ep025_2k_seed42|\
  xverse_trackc_ctx_bridge_ep050_2k_seed42)
    ;;
  *)
    echo "Unsupported support-context query run name: ${RUN_NAME}" >&2
    exit 2
    ;;
esac

export LATENTFM_TRACKC_QUERY_RUN_NAME=${RUN_NAME}
export LATENTFM_TRACKC_QUERY_LABEL=latentfm_trackc_support_context_query_once_${RUN_NAME}_20260622
export LATENTFM_TRACKC_QUERY_CANDIDATE_CKPT=${COUPLED}/output/latentfm_runs/xverse_trackc_support_context_20260622/${RUN_NAME}/best.pt
export LATENTFM_TRACKC_QUERY_SMOKE_DECISION=${ROOT}/reports/latentfm_trackc_routed_distill_smoke_decision_${RUN_NAME}.json
export LATENTFM_TRACKC_QUERY_UNCAPPED_DECISION=${ROOT}/reports/latentfm_trackc_support_context_uncapped_noharm_decision_20260622.json
export LATENTFM_TRACKC_QUERY_DECISION_JSON=${ROOT}/reports/latentfm_trackc_support_context_query_once_decision_${RUN_NAME}_20260622.json
export LATENTFM_TRACKC_QUERY_DECISION_MD=${ROOT}/reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_QUERY_ONCE_DECISION_${RUN_NAME}_20260622.md
export LATENTFM_TRACKC_QUERY_BOOT_DIR=${ROOT}/reports/latentfm_trackc_support_context_query_once_bootstrap_${RUN_NAME}_20260622
export LATENTFM_TRACKC_QUERY_REPORT_TITLE="LatentFM Track C Support-Context One-Shot Query Decision: ${RUN_NAME}"

bash ${ROOT}/ops/launch_latentfm_trackc_routefocus_query_if_pass_20260622.sh
