#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

SUMMARY_JSON=${ROOT}/reports/latentfm_trackc_support_context_smoke_summary_20260622.json
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
MANIFEST=${ROOT}/reports/latentfm_trackc_support_context_uncapped_noharm_manifest_20260622.json
LABEL=latentfm_trackc_support_context_uncapped_noharm_20260622
UNCAPPED_OUT=${ROOT}/reports/${LABEL}
LAUNCHER=${ROOT}/ops/launch_latentfm_uncapped_posthoc_from_manifest_20260621.sh
SUMMARY_WRAPPER=${ROOT}/ops/summarize_latentfm_trackc_support_context_uncapped_noharm_20260622.sh

for required in \
  "${SUMMARY_JSON}" \
  "${ANCHOR_CKPT}" \
  "${ROOT}/dataset/latentfm_full/xverse/manifest.json" \
  "${ROOT}/dataset/biFlow_data/split_seed42.json" \
  "${LAUNCHER}" \
  "${SUMMARY_WRAPPER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required pass-only artifact: ${required}" >&2
    exit 2
  fi
done

"${PYTHON}" - \
  "${SUMMARY_JSON}" \
  "${MANIFEST}" \
  "${ANCHOR_CKPT}" <<'PY'
import json
import sys
from pathlib import Path

summary_json = Path(sys.argv[1])
manifest = Path(sys.argv[2])
anchor_ckpt = sys.argv[3]
pass_status = "trackc_smoke_support_pass_needs_uncapped_noharm_before_query"
summary = json.loads(summary_json.read_text(encoding="utf-8"))
if summary.get("query_read") is not False:
    print(json.dumps({"status": "unsafe_summary_query_flag", "summary": str(summary_json)}, indent=2), file=sys.stderr)
    raise SystemExit(4)
if summary.get("route_gap_gate_required") is not True:
    print(json.dumps({"status": "route_gap_gate_not_enforced", "summary": str(summary_json)}, indent=2), file=sys.stderr)
    raise SystemExit(4)
passed = []
blocked = []
for row in summary.get("runs") or []:
    run = str(row.get("run") or "")
    status = str(row.get("status") or "")
    if status != pass_status:
        continue
    if row.get("route_gap_status") != "route_gap_gate_pass":
        blocked.append(f"passed run missing route-gap pass: {run}")
        continue
    route_gap_json = Path(str(row.get("route_gap_json") or ""))
    if not route_gap_json.is_file():
        blocked.append(f"passed run missing route_gap_json: {route_gap_json}")
        continue
    route_payload = json.loads(route_gap_json.read_text(encoding="utf-8"))
    if route_payload.get("heldout_query_used") is not False:
        blocked.append(f"passed run route_gap_json unsafe query flag: {route_gap_json}")
        continue
    if ((route_payload.get("decision") or {}).get("status")) != "route_gap_gate_pass":
        blocked.append(f"passed run route_gap_json decision not pass: {route_gap_json}")
        continue
    if str(route_payload.get("run_name") or "") != run:
        blocked.append(f"passed run route_gap_json run mismatch: {route_gap_json}")
        continue
    ckpt = (
        Path("/data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/xverse_trackc_support_context_20260622")
        / run
        / "best.pt"
    )
    decision_json = Path(str(row.get("decision_json") or ""))
    if not ckpt.is_file():
        blocked.append(f"passed run missing best.pt: {ckpt}")
        continue
    if not decision_json.is_file():
        blocked.append(f"passed run missing decision_json: {decision_json}")
        continue
    passed.append(
        {
            "run_name": run,
            "candidate_checkpoint": str(ckpt),
            "split_file": "/data/cyx/1030/dataset/biFlow_data/split_seed42.json",
            "data_dir": "/data/cyx/1030/dataset/latentfm_full/xverse",
            "biflow_dir": "/data/cyx/1030/dataset/biFlow_data",
            "source_smoke_decision_json": str(decision_json),
            "source_smoke_summary_json": str(summary_json),
        }
    )
if blocked:
    print(json.dumps({"status": "blocked", "reasons": blocked}, indent=2), file=sys.stderr)
    raise SystemExit(4)
if not passed:
    if summary.get("overall_status") == "support_context_smokes_pending":
        print(json.dumps({"status": "pending_no_pass_yet", "summary": str(summary_json)}, indent=2), file=sys.stderr)
        raise SystemExit(3)
    print(
        json.dumps(
            {
                "status": "no_support_context_smoke_pass",
                "summary_status": summary.get("overall_status"),
                "summary": str(summary_json),
            },
            indent=2,
        ),
        file=sys.stderr,
    )
    raise SystemExit(5)
if manifest.exists():
    raise FileExistsError(f"refusing to overwrite existing manifest: {manifest}")
payload = {
    "purpose": "Track C support-context canonical no-harm only; no held-out query",
    "split_file": "/data/cyx/1030/dataset/biFlow_data/split_seed42.json",
    "data_dir": "/data/cyx/1030/dataset/latentfm_full/xverse",
    "biflow_dir": "/data/cyx/1030/dataset/biFlow_data",
    "anchor_checkpoint": anchor_ckpt,
    "launched_runs": passed,
    "heldout_query_used": False,
    "selection_weight_canonical_multi": 0,
    "source_smoke_summary_json": str(summary_json),
}
manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"status": "manifest_written", "manifest": str(manifest), "passed_runs": len(passed)}, indent=2))
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
