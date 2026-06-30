#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PY=/data/cyx/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PY}" ]]; then
  PY=${ROOT}/software/miniconda3/envs/scdfm/bin/python
fi

now_epoch=$(date +%s)
boundary_epoch=$(date -d '2026-06-22 22:55:00' +%s)
if (( now_epoch < boundary_epoch )); then
  echo "Refusing to check Track C support-context smokes before 2026-06-22 22:55:00 CST" >&2
  echo "These are long GPU jobs; AGENTS.md requires a >=30 minute check interval." >&2
  exit 3
fi

echo "# Track C support-context capped smoke status"
echo "checked_at=$(date '+%F %T %Z')"
echo

RUN_ROOT="${ROOT}/runs/latentfm_xverse_trackc_support_context_20260622"
runs=(
  xverse_trackc_ctx_bridge_fm_2k_seed42
  xverse_trackc_ctx_bridge_ep025_2k_seed42
  xverse_trackc_ctx_bridge_ep050_2k_seed42
)

for run in "${runs[@]}"; do
  posthoc="${RUN_ROOT}/${run}/posthoc_eval"
  posthoc_exit="${RUN_ROOT}/${run}/${run}.POSTHOC_EXIT_CODE"
  if [[ -f "${posthoc_exit}" && "$(<"${posthoc_exit}")" == "0" && -f "${posthoc}/support_anchor_split_ode20.json" && -f "${posthoc}/support_candidate_split_ode20.json" ]]; then
    if ! "${PY}" "${ROOT}/ops/evaluate_latentfm_trackc_support_context_route_gap_gate_20260622.py" \
      --run-name "${run}"; then
      echo "route_gap_sidecar_failed_fail_closed=${run}" >&2
    fi
  else
    echo "route_gap_sidecar_pending=${run}"
  fi
done

"${PY}" "${ROOT}/ops/summarize_latentfm_trackc_support_context_block_20260622.py"

echo
echo "## Manual follow-up if still pending"
echo
cat <<'EOF'
Do not keep polling. If the summary is still pending, wait until the next
30-minute window before checking again. If a run passes the capped gate, the
next allowed action is uncapped canonical no-harm only; held-out query remains
forbidden until route/checkpoint is frozen after no-harm.
EOF
