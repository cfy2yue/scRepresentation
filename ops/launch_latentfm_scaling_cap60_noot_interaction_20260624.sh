#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_CAP60_NOOT_ACK:-}" != "ot_random_not_worse_scaling_interesting" && "${LATENTFM_CAP60_NOOT_ACK:-}" != "active_exploratory_before_ot_final" ]]; then
  cat >&2 <<'EOF'
Refusing cap60 no-OT interaction launch.

Set:
  LATENTFM_CAP60_NOOT_ACK=ot_random_not_worse_scaling_interesting
or, for the active compute-first exploratory portfolio:
  LATENTFM_CAP60_NOOT_ACK=active_exploratory_before_ot_final

Required trigger:
  - OT/no-OT random rerun decision is pass/near-miss, not pending/fail.
  - Scaling refill decision has at least one internal-passed arm.

Exploratory override:
  - allowed only as a bounded cap60 x no-OT interaction smoke while OT final
    posthoc is pending and scaling internal signal is already real.
  - not a promotion claim and not permission to read canonical multi/Track C.
EOF
  exit 4
fi

OT_JSON=${LATENTFM_CAP60_NOOT_OT_JSON:-${ROOT}/reports/latentfm_xverse_ot_pairmode_random_rerun_decision_20260624.json}
SCALING_JSON=${LATENTFM_CAP60_NOOT_SCALING_JSON:-${ROOT}/reports/latentfm_scaling_highthroughput_smokes_refill_decision_20260624.json}

for required in "${OT_JSON}" "${SCALING_JSON}" "${ROOT}/ops/launch_latentfm_scaling_highthroughput_smokes_20260624.sh"; do
  [[ -e "${required}" ]] || { echo "Missing required artifact: ${required}" >&2; exit 2; }
done

"${PYTHON}" - "${OT_JSON}" "${SCALING_JSON}" "${LATENTFM_CAP60_NOOT_ACK}" <<'PY'
import json
import sys
from pathlib import Path

ot = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
scaling = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
ack = str(sys.argv[3])

ot_status = (ot.get("decision") or {}).get("status")
if ack != "active_exploratory_before_ot_final" and ot_status not in {"pass_candidate_available", "near_miss_candidate_available"}:
    raise SystemExit(f"OT random/no-OT trigger not satisfied: {ot_status!r}")
if ack == "active_exploratory_before_ot_final" and ot_status not in {None, "pending", "pass_candidate_available", "near_miss_candidate_available"}:
    raise SystemExit(f"Exploratory override only allowed before OT final fail/close, got {ot_status!r}")

scaling_decision = scaling.get("decision") or {}
passed = scaling_decision.get("passed") or []
if not passed:
    raise SystemExit("Scaling refill has no internal-passed arm; no no-OT interaction launch")
if scaling_decision.get("status") not in {"internal_partial_pass", "internal_pass_seed_and_replay"}:
    raise SystemExit(f"Scaling refill status not launchable: {scaling_decision.get('status')!r}")
PY

exec env \
  LATENTFM_SCALING_HT_ACK=bounded_exploratory_smokes \
  LATENTFM_SCALING_HT_ONLY_ARM=cap60_noot_3k_seed42 \
  LATENTFM_SCALING_HT_RUN_ROOT=${LATENTFM_CAP60_NOOT_RUN_ROOT:-${ROOT}/runs/latentfm_scaling_cap60_noot_interaction_20260624} \
  LATENTFM_SCALING_HT_OUT_ROOT=${LATENTFM_CAP60_NOOT_OUT_ROOT:-${ROOT}/CoupledFM/output/latentfm_runs/scaling_cap60_noot_interaction_20260624} \
  LATENTFM_SCALING_HT_LOG_ROOT=${LATENTFM_CAP60_NOOT_LOG_ROOT:-${ROOT}/logs/latentfm_scaling_cap60_noot_interaction_20260624} \
  bash "${ROOT}/ops/launch_latentfm_scaling_highthroughput_smokes_20260624.sh"
