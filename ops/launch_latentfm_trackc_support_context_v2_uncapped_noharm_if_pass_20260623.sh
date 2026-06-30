#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_NAME=xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42
RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_support_context_v2_20260623/${RUN_NAME}
DECISION_JSON=${ROOT}/reports/latentfm_trackc_routed_distill_smoke_decision_${RUN_NAME}.json
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
CANDIDATE_CKPT=${COUPLED}/output/latentfm_runs/xverse_trackc_support_context_v2_20260623/${RUN_NAME}/best.pt
MANIFEST=${ROOT}/reports/latentfm_trackc_support_context_v2_uncapped_noharm_manifest_20260623.json
LABEL=latentfm_trackc_support_context_v2_uncapped_noharm_20260623
UNCAPPED_OUT=${ROOT}/reports/${LABEL}
LAUNCHER=${ROOT}/ops/launch_latentfm_uncapped_posthoc_from_manifest_20260621.sh
SUMMARY_WRAPPER=${ROOT}/ops/summarize_latentfm_trackc_support_context_v2_uncapped_noharm_20260623.sh

for required in \
  "${DECISION_JSON}" \
  "${ANCHOR_CKPT}" \
  "${CANDIDATE_CKPT}" \
  "${ROOT}/dataset/latentfm_full/xverse/manifest.json" \
  "${ROOT}/dataset/biFlow_data/split_seed42.json" \
  "${LAUNCHER}" \
  "${SUMMARY_WRAPPER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required pass-only artifact: ${required}" >&2
    exit 2
  fi
done

"${PYTHON}" - "${DECISION_JSON}" "${MANIFEST}" "${ANCHOR_CKPT}" "${CANDIDATE_CKPT}" "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

decision_json = Path(sys.argv[1])
manifest = Path(sys.argv[2])
anchor_ckpt = Path(sys.argv[3])
candidate_ckpt = Path(sys.argv[4])
run_root = Path(sys.argv[5])
payload = json.loads(decision_json.read_text(encoding="utf-8"))
pass_status = "trackc_smoke_support_pass_needs_uncapped_noharm_before_query"
decision_status = payload.get("status") or (payload.get("decision") or {}).get("status")
if decision_status != pass_status:
    print(json.dumps({"status": "smoke_not_passed", "decision_status": decision_status}, indent=2), file=sys.stderr)
    raise SystemExit(5)
if manifest.exists():
    raise FileExistsError(f"refusing to overwrite existing manifest: {manifest}")

posthoc = run_root / "posthoc_eval"
required_canonical = [
    posthoc / "canonical_anchor_split_ode20_stablecaps.json",
    posthoc / "canonical_candidate_split_ode20_stablecaps.json",
    posthoc / "canonical_anchor_family_ode20_stablecaps.json",
    posthoc / "canonical_candidate_family_ode20_stablecaps.json",
]
for path in required_canonical:
    if not path.exists():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("support_context_forced_absent") is not True:
        raise RuntimeError(f"canonical capped posthoc did not force support absent: {path}")

out = {
    "purpose": "Track C support-context v2 uncapped canonical no-harm only; no held-out query",
    "split_file": "/data/cyx/1030/dataset/biFlow_data/split_seed42.json",
    "data_dir": "/data/cyx/1030/dataset/latentfm_full/xverse",
    "biflow_dir": "/data/cyx/1030/dataset/biFlow_data",
    "anchor_checkpoint": str(anchor_ckpt),
    "force_support_context_absent": True,
    "heldout_query_used": False,
    "selection_weight_canonical_multi": 0,
    "launched_runs": [
        {
            "run_name": "xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42",
            "candidate_checkpoint": str(candidate_ckpt),
            "split_file": "/data/cyx/1030/dataset/biFlow_data/split_seed42.json",
            "data_dir": "/data/cyx/1030/dataset/latentfm_full/xverse",
            "biflow_dir": "/data/cyx/1030/dataset/biFlow_data",
            "force_support_context_absent": True,
            "source_smoke_decision_json": str(decision_json),
        }
    ],
}
manifest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"status": "manifest_written", "manifest": str(manifest)}, indent=2))
PY

MANIFEST="${MANIFEST}" \
LABEL="${LABEL}" \
OUT_DIR="${UNCAPPED_OUT}" \
EVAL_MAX_MSE_CELLS="${EVAL_MAX_MSE_CELLS:-0}" \
EVAL_MAX_MMD_CELLS="${EVAL_MAX_MMD_CELLS:-0}" \
bash "${LAUNCHER}"

echo
echo "After uncapped posthoc finishes, summarize no-harm with:"
echo "bash ${SUMMARY_WRAPPER}"
