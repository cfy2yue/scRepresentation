#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

PREFLIGHT_JSON=${ROOT}/reports/latentfm_trackc_support_set_abstention_launcher_preflight_20260627.json
"${PYTHON}" - "${PREFLIGHT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing launcher preflight JSON: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "trackc_support_set_abstention_launcher_preflight_pass_no_gpu":
    raise SystemExit(f"launcher preflight did not pass: {payload.get('status')}")
if payload.get("gpu_authorized") is not False:
    raise SystemExit("preflight must not directly authorize GPU")
spec = payload.get("frozen_router_spec") or {}
expected = {"policy": "shared_gene_component", "beta": 1.0, "min_support": 2, "min_confidence": -1.0}
for key, value in expected.items():
    if spec.get(key) != value:
        raise SystemExit(f"router spec mismatch for {key}: {spec.get(key)!r} != {value!r}")
PY

export LATENTFM_TRACKC_SUPPORT_SET_RUN_NAME="${LATENTFM_TRACKC_SUPPORT_SET_RUN_NAME:-xverse_trackc_support_set_abstention_min2_adapter_2k_seed42}"
export LATENTFM_TRACKC_SUPPORT_SET_RUN_ROOT="${LATENTFM_TRACKC_SUPPORT_SET_RUN_ROOT:-${ROOT}/runs/latentfm_trackc_support_set_abstention_20260627}"
export LATENTFM_TRACKC_SUPPORT_SET_OUT_ROOT="${LATENTFM_TRACKC_SUPPORT_SET_OUT_ROOT:-${ROOT}/CoupledFM/output/latentfm_runs/trackc_support_set_abstention_20260627}"
export LATENTFM_TRACKC_SUPPORT_SET_LOG_ROOT="${LATENTFM_TRACKC_SUPPORT_SET_LOG_ROOT:-${ROOT}/logs/latentfm_trackc_support_set_abstention_20260627}"
export LATENTFM_TRACKC_SUPPORT_SET_SEED="${LATENTFM_TRACKC_SUPPORT_SET_SEED:-42}"
export LATENTFM_TRACKC_SUPPORT_SET_TOTAL_STEPS="${LATENTFM_TRACKC_SUPPORT_SET_TOTAL_STEPS:-2000}"
export LATENTFM_TRACKC_SUPPORT_SET_MIN_SUPPORT_COUNT=2
export LATENTFM_TRACKC_SUPPORT_SET_HYPOTHESIS="${LATENTFM_TRACKC_SUPPORT_SET_HYPOTHESIS:-Track C abstention router freezes shared-gene support-set task tokens to require at least two same-dataset shared-gene train_multi supports; unsupported or weakly supported rows should be exact anchor/no-op, while actual support-val beats zero/shuffle/absent controls on safe trainselect only.}"

bash "${ROOT}/ops/launch_latentfm_trackc_support_set_smoke_20260627.sh"
