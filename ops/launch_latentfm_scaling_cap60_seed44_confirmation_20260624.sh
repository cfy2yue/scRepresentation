#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_CAP60_SEED44_ACK:-}" != "active_seed42_seed43_internal_pass" ]]; then
  cat >&2 <<'EOF'
Refusing cap60 seed44 confirmation launch.

Set:
  LATENTFM_CAP60_SEED44_ACK=active_seed42_seed43_internal_pass

Required trigger:
  - Scaling refill decision is internal pass with both
    xverse_scaling_cap60_6k_seed42 and xverse_scaling_cap60_6k_seed43 passed.
EOF
  exit 4
fi

SCALING_JSON=${LATENTFM_CAP60_SEED44_SCALING_JSON:-${ROOT}/reports/latentfm_scaling_highthroughput_smokes_refill_decision_20260624.json}

for required in "${SCALING_JSON}" "${ROOT}/ops/launch_latentfm_scaling_highthroughput_smokes_20260624.sh"; do
  [[ -e "${required}" ]] || { echo "Missing required artifact: ${required}" >&2; exit 2; }
done

"${PYTHON}" - "${SCALING_JSON}" <<'PY'
import json
import sys
from pathlib import Path

scaling = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
decision = scaling.get("decision") or {}
passed = set(decision.get("passed") or [])
required = {"xverse_scaling_cap60_6k_seed42", "xverse_scaling_cap60_6k_seed43"}
missing = sorted(required - passed)
if missing:
    raise SystemExit(f"Seed44 trigger not satisfied; missing internal passes: {missing}")
if decision.get("status") not in {"internal_pass_seed_and_replay", "internal_partial_pass"}:
    raise SystemExit(f"Scaling refill status not launchable: {decision.get('status')!r}")
PY

exec env \
  LATENTFM_SCALING_HT_ACK=bounded_exploratory_smokes \
  LATENTFM_SCALING_HT_ONLY_ARM=cap60_6k_seed44 \
  LATENTFM_SCALING_HT_RUN_ROOT=${LATENTFM_CAP60_SEED44_RUN_ROOT:-${ROOT}/runs/latentfm_scaling_cap60_seed44_confirmation_20260624} \
  LATENTFM_SCALING_HT_OUT_ROOT=${LATENTFM_CAP60_SEED44_OUT_ROOT:-${ROOT}/CoupledFM/output/latentfm_runs/scaling_cap60_seed44_confirmation_20260624} \
  LATENTFM_SCALING_HT_LOG_ROOT=${LATENTFM_CAP60_SEED44_LOG_ROOT:-${ROOT}/logs/latentfm_scaling_cap60_seed44_confirmation_20260624} \
  bash "${ROOT}/ops/launch_latentfm_scaling_highthroughput_smokes_20260624.sh"
