#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_CONDRES_SCALE_SEED44_ACK:-}" != "seed43_pair_pass_integrated" ]]; then
  cat >&2 <<'EOF'
Refusing to launch condition-residual scaling seed44 extension.

Set:
  LATENTFM_CONDRES_SCALE_SEED44_ACK=seed43_pair_pass_integrated

Required preread:
  reports/LATENTFM_CONDITION_RESIDUAL_SCALING_SLATE_DECISION_20260628.md
EOF
  exit 4
fi

SUMMARIZER=${ROOT}/ops/summarize_latentfm_condition_residual_scaling_slate_20260628.py
DECISION_JSON=${ROOT}/reports/latentfm_condition_residual_scaling_slate_decision_20260628.json
LAUNCHER=${ROOT}/ops/launch_latentfm_condition_residual_scaling_slate_20260628.sh

for required in "${SUMMARIZER}" "${LAUNCHER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

"${PYTHON}" "${SUMMARIZER}" >/dev/null

mapfile -t PASS_PAIRS < <("${PYTHON}" - "${DECISION_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing decision json: {path}")
payload = json.loads(path.read_text())
if payload.get("status") == "pending":
    raise SystemExit("seed43 slate decision is pending; do not launch seed44")
decision = payload.get("decision") or {}
for pair in ["response_strength_vs_breadth", "perturbation_type_breadth"]:
    obj = decision.get(pair) or {}
    if obj.get("status") == "pass_extend_seed44":
        print(pair)
PY
)

if [[ "${#PASS_PAIRS[@]}" -eq 0 ]]; then
  echo "No pair has status pass_extend_seed44; refusing seed44 launch." >&2
  exit 5
fi

for pair in "${PASS_PAIRS[@]}"; do
  echo "Launching seed44 extension for pair=${pair}"
  LATENTFM_CONDRES_SCALE_ACK=condition_residual_scaling_robust_pass \
  LATENTFM_CONDRES_SCALE_SEED=44 \
  LATENTFM_CONDRES_SCALE_ONLY_PAIR="${pair}" \
  bash "${LAUNCHER}"
done
