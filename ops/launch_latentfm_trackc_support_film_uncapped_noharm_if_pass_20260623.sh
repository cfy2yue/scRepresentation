#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_NAME=xverse_trackc_support_film_absroute_2k_seed42_retry1
RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_support_film_20260623/${RUN_NAME}
OUT_DIR=${COUPLED}/output/latentfm_runs/xverse_trackc_support_film_20260623/${RUN_NAME}
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
CANDIDATE_CKPT=${OUT_DIR}/best.pt
DECISION_JSON=${ROOT}/reports/latentfm_trackc_routed_distill_smoke_decision_${RUN_NAME}.json
ROUTE_GAP_JSON=${ROOT}/reports/latentfm_trackc_support_film_route_gap_gate_${RUN_NAME}.json
MANIFEST=${ROOT}/reports/latentfm_trackc_support_film_uncapped_noharm_manifest_20260623.json
LABEL=latentfm_trackc_support_film_uncapped_noharm_20260623
UNCAPPED_OUT=${ROOT}/reports/${LABEL}
LAUNCHER=${ROOT}/ops/launch_latentfm_uncapped_posthoc_from_manifest_20260621.sh
SUMMARY_WRAPPER=${ROOT}/ops/summarize_latentfm_trackc_support_film_uncapped_noharm_20260623.sh

for required in \
  "${DECISION_JSON}" \
  "${ROUTE_GAP_JSON}" \
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

"${PYTHON}" - "${DECISION_JSON}" "${ROUTE_GAP_JSON}" "${RUN_NAME}" "${MANIFEST}" "${ANCHOR_CKPT}" "${CANDIDATE_CKPT}" <<'PY'
import json
import sys
from pathlib import Path

decision_json = Path(sys.argv[1])
route_gap_json = Path(sys.argv[2])
run_name = sys.argv[3]
manifest = Path(sys.argv[4])
anchor_ckpt = sys.argv[5]
candidate_ckpt = sys.argv[6]

pass_status = "trackc_smoke_support_pass_needs_uncapped_noharm_before_query"
decision = json.loads(decision_json.read_text(encoding="utf-8"))
decision_status = (decision.get("decision") or {}).get("status")
if decision_status != pass_status:
    print(
        json.dumps(
            {
                "status": "support_film_smoke_not_pass",
                "decision_status": decision_status,
                "decision_json": str(decision_json),
            },
            indent=2,
        ),
        file=sys.stderr,
    )
    raise SystemExit(5)

route = json.loads(route_gap_json.read_text(encoding="utf-8"))
if route.get("run_name") != run_name:
    raise SystemExit(f"route-gap run mismatch: {route.get('run_name')} vs {run_name}")
if route.get("heldout_query_used") is not False:
    raise SystemExit("route-gap payload unexpectedly used held-out query")
if route.get("canonical_outputs_used") is not False:
    raise SystemExit("route-gap payload unexpectedly used canonical outputs")
if (route.get("decision") or {}).get("status") != "support_film_route_gap_gate_pass":
    print(
        json.dumps(
            {
                "status": "support_film_route_gap_not_pass",
                "route_gap_status": (route.get("decision") or {}).get("status"),
                "route_gap_json": str(route_gap_json),
            },
            indent=2,
        ),
        file=sys.stderr,
    )
    raise SystemExit(5)

if manifest.exists():
    raise FileExistsError(f"refusing to overwrite existing manifest: {manifest}")
payload = {
    "purpose": "Track C support-FiLM canonical no-harm only; no held-out query",
    "split_file": "/data/cyx/1030/dataset/biFlow_data/split_seed42.json",
    "data_dir": "/data/cyx/1030/dataset/latentfm_full/xverse",
    "biflow_dir": "/data/cyx/1030/dataset/biFlow_data",
    "anchor_checkpoint": anchor_ckpt,
    "heldout_query_used": False,
    "selection_weight_canonical_multi": 0,
    "source_smoke_decision_json": str(decision_json),
    "source_route_gap_json": str(route_gap_json),
    "launched_runs": [
        {
            "run_name": run_name,
            "candidate_checkpoint": candidate_ckpt,
            "split_file": "/data/cyx/1030/dataset/biFlow_data/split_seed42.json",
            "data_dir": "/data/cyx/1030/dataset/latentfm_full/xverse",
            "biflow_dir": "/data/cyx/1030/dataset/biFlow_data",
            "source_smoke_decision_json": str(decision_json),
            "source_route_gap_json": str(route_gap_json),
        }
    ],
}
manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"status": "manifest_written", "manifest": str(manifest)}, indent=2))
PY

MANIFEST="${MANIFEST}" \
LABEL="${LABEL}" \
OUT_DIR="${UNCAPPED_OUT}" \
ONLY_RUN_NAME="${RUN_NAME}" \
EVAL_MAX_MSE_CELLS="${EVAL_MAX_MSE_CELLS:-2048}" \
EVAL_MAX_MMD_CELLS="${EVAL_MAX_MMD_CELLS:-2048}" \
bash "${LAUNCHER}"

echo
echo "After uncapped posthoc finishes, summarize no-harm with:"
echo "bash ${SUMMARY_WRAPPER}"
