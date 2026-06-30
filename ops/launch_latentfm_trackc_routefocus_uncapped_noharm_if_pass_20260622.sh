#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_NAME=xverse_trackc_routefocus_condprior_w05_replay1_2k_seed42
RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_routefocused_distill_20260622/${RUN_NAME}
OUT_DIR=${COUPLED}/output/latentfm_runs/xverse_trackc_routefocused_distill_20260622/${RUN_NAME}
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
CANDIDATE_CKPT=${OUT_DIR}/best.pt
DECISION_JSON=${ROOT}/reports/latentfm_trackc_routed_distill_smoke_decision_${RUN_NAME}.json
MANIFEST=${ROOT}/reports/latentfm_trackc_routefocus_uncapped_noharm_manifest_20260622.json
LABEL=latentfm_trackc_routefocus_uncapped_noharm_20260622
UNCAPPED_OUT=${ROOT}/reports/${LABEL}
LAUNCHER=${ROOT}/ops/launch_latentfm_uncapped_posthoc_from_manifest_20260621.sh

for required in \
  "${DECISION_JSON}" \
  "${ANCHOR_CKPT}" \
  "${CANDIDATE_CKPT}" \
  "${ROOT}/dataset/latentfm_full/xverse/manifest.json" \
  "${ROOT}/dataset/biFlow_data/split_seed42.json" \
  "${LAUNCHER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required pass-only artifact: ${required}" >&2
    exit 2
  fi
done

status="$("${PYTHON}" - "${DECISION_JSON}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print((payload.get("decision") or {}).get("status", ""))
PY
)"

if [[ "${status}" != "trackc_smoke_support_pass_needs_uncapped_noharm_before_query" ]]; then
  echo "Refusing uncapped no-harm: decision status is '${status}', not pass." >&2
  exit 5
fi

"${PYTHON}" - "${MANIFEST}" "${ANCHOR_CKPT}" "${CANDIDATE_CKPT}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
payload = {
    "purpose": "Track C route-focused canonical no-harm only; no held-out query",
    "split_file": "/data/cyx/1030/dataset/biFlow_data/split_seed42.json",
    "data_dir": "/data/cyx/1030/dataset/latentfm_full/xverse",
    "biflow_dir": "/data/cyx/1030/dataset/biFlow_data",
    "anchor_checkpoint": sys.argv[2],
    "launched_runs": [
        {
            "run_name": "xverse_trackc_routefocus_condprior_w05_replay1_2k_seed42",
            "candidate_checkpoint": sys.argv[3],
            "split_file": "/data/cyx/1030/dataset/biFlow_data/split_seed42.json",
            "data_dir": "/data/cyx/1030/dataset/latentfm_full/xverse",
            "biflow_dir": "/data/cyx/1030/dataset/biFlow_data",
        }
    ],
    "heldout_query_used": False,
    "selection_weight_canonical_multi": 0,
}
manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(manifest)
PY

MANIFEST="${MANIFEST}" \
LABEL="${LABEL}" \
OUT_DIR="${UNCAPPED_OUT}" \
ONLY_RUN_NAME="${RUN_NAME}" \
EVAL_MAX_MSE_CELLS="${EVAL_MAX_MSE_CELLS:-2048}" \
EVAL_MAX_MMD_CELLS="${EVAL_MAX_MMD_CELLS:-2048}" \
bash "${LAUNCHER}"
