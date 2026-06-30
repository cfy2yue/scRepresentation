#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_PATHWAY_CANONICAL_ACK:-}" != "pathway_internal_pass_frozen_noharm" ]]; then
  cat >&2 <<'EOF'
Refusing modality/pathway canonical no-harm launch.

Set:
  LATENTFM_PATHWAY_CANONICAL_ACK=pathway_internal_pass_frozen_noharm

Boundary:
  - only after pathway-quota smoke passes train-only internal gate;
  - canonical split is a frozen single/family no-harm veto only;
  - canonical multi is not selected or evaluated;
  - held-out Track C query is not read.
EOF
  exit 4
fi

INTERNAL_JSON=${ROOT}/reports/latentfm_modality_pathway_sampling_smoke_decision_20260624.json
RUN_NAME=xverse_scaling_pathway_quota12_3k_seed42

for required in \
  "${INTERNAL_JSON}" \
  "${ROOT}/ops/launch_latentfm_scaling_highthroughput_canonical_noharm_20260624.sh"; do
  [[ -e "${required}" ]] || { echo "Missing required artifact: ${required}" >&2; exit 2; }
done

"${PYTHON}" - "${INTERNAL_JSON}" "${RUN_NAME}" <<'PY'
import json
import sys
from pathlib import Path

obj = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
run_name = sys.argv[2]
decision = obj.get("decision") or {}
passed = set(map(str, decision.get("passed") or []))
status = decision.get("status")
if run_name not in passed:
    raise SystemExit(f"{run_name} not in internal passed list; status={status!r}; passed={sorted(passed)}")
if status not in {"internal_partial_pass", "internal_pass_seed_and_replay"}:
    raise SystemExit(f"internal decision status not launchable: {status!r}")
PY

exec env \
  LATENTFM_SCALING_HT_CANONICAL_ACK=internal_pass_frozen_noharm \
  LATENTFM_SCALING_HT_INTERNAL_JSON="${INTERNAL_JSON}" \
  LATENTFM_SCALING_HT_TRAIN_OUT_ROOT="${ROOT}/CoupledFM/output/latentfm_runs/modality_pathway_sampling_smoke_20260624" \
  LATENTFM_SCALING_HT_CANONICAL_RUN_ROOT="${ROOT}/runs/latentfm_modality_pathway_sampling_canonical_noharm_20260624" \
  LATENTFM_SCALING_HT_CANONICAL_LOG_ROOT="${ROOT}/logs/latentfm_modality_pathway_sampling_canonical_noharm_20260624" \
  LATENTFM_SCALING_HT_CANONICAL_DECISION_JSON="${ROOT}/reports/latentfm_modality_pathway_sampling_canonical_noharm_decision_20260624.json" \
  LATENTFM_SCALING_HT_CANONICAL_DECISION_MD="${ROOT}/reports/LATENTFM_MODALITY_PATHWAY_SAMPLING_CANONICAL_NOHARM_DECISION_20260624.md" \
  bash "${ROOT}/ops/launch_latentfm_scaling_highthroughput_canonical_noharm_20260624.sh"
