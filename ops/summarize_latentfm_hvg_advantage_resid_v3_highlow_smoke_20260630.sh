#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent

python "${ROOT}/ops/summarize_latentfm_scaling_v2_condition_information_highlow_smoke_20260628.py" \
  --run-root "${ROOT}/runs/latentfm_hvg_advantage_resid_v3_highlow_smoke_20260630" \
  --report-dir "${ROOT}/reports/hvg_advantage_resid_v3_highlow_smoke_20260630" \
  --run-prefix xverse_hvgadv_resid_v3 \
  --steps 2000 \
  --seed 42 \
  --title "LatentFM HVG-Advantage Residual V3 High/Low Decision" \
  --stem latentfm_hvg_advantage_resid_v3_highlow_decision_20260630
