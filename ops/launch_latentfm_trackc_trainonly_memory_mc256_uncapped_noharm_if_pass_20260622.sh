#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

BASE_MANIFEST=${ROOT}/reports/latentfm_trackc_trainonly_memory_parallel_mc256_manifest_20260622.jsonl
EXT_MANIFEST=${ROOT}/reports/latentfm_trackc_trainonly_memory_parallel_mc256_ext_manifest_20260622.jsonl
BASE_OUT_ROOT=${COUPLED}/output/latentfm_runs/xverse_trackc_trainonly_memory_parallel_mc256_20260622
EXT_OUT_ROOT=${COUPLED}/output/latentfm_runs/xverse_trackc_trainonly_memory_parallel_mc256_ext_20260622
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
MANIFEST=${ROOT}/reports/latentfm_trackc_trainonly_memory_mc256_uncapped_noharm_manifest_20260622.json
LABEL=latentfm_trackc_trainonly_memory_mc256_uncapped_noharm_20260622
UNCAPPED_OUT=${ROOT}/reports/${LABEL}
LAUNCHER=${ROOT}/ops/launch_latentfm_uncapped_posthoc_from_manifest_20260621.sh

for required in \
  "${BASE_MANIFEST}" \
  "${EXT_MANIFEST}" \
  "${ANCHOR_CKPT}" \
  "${ROOT}/dataset/latentfm_full/xverse/manifest.json" \
  "${ROOT}/dataset/biFlow_data/split_seed42.json" \
  "${LAUNCHER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required pass-only artifact: ${required}" >&2
    exit 2
  fi
done

"${PYTHON}" - \
  "${MANIFEST}" \
  "${ANCHOR_CKPT}" \
  "${BASE_MANIFEST}" \
  "${BASE_OUT_ROOT}" \
  "${EXT_MANIFEST}" \
  "${EXT_OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

manifest_out = Path(sys.argv[1])
anchor_ckpt = sys.argv[2]
manifest_pairs = [
    (Path(sys.argv[3]), Path(sys.argv[4])),
    (Path(sys.argv[5]), Path(sys.argv[6])),
]
pass_status = "trackc_smoke_support_pass_needs_uncapped_noharm_before_query"
missing = []
blocked = []
passed = []
for manifest_path, out_root in manifest_pairs:
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        run = row["run_name"]
        decision_path = Path(
            row.get("decision_json")
            or f"/data/cyx/1030/scLatent/reports/latentfm_trackc_routed_distill_smoke_decision_{run}.json"
        )
        if not decision_path.is_file():
            missing.append(str(decision_path))
            continue
        payload = json.loads(decision_path.read_text(encoding="utf-8"))
        status = (payload.get("decision") or {}).get("status", "")
        if status == pass_status:
            candidate = out_root / run / "best.pt"
            if not candidate.is_file():
                blocked.append(f"passed run missing best.pt: {candidate}")
                continue
            passed.append(
                {
                    "run_name": run,
                    "candidate_checkpoint": str(candidate),
                    "split_file": "/data/cyx/1030/dataset/biFlow_data/split_seed42.json",
                    "data_dir": "/data/cyx/1030/dataset/latentfm_full/xverse",
                    "biflow_dir": "/data/cyx/1030/dataset/biFlow_data",
                    "source_smoke_decision_json": str(decision_path),
                    "source_smoke_manifest": str(manifest_path),
                }
            )
        elif not status.startswith("trackc_smoke_fail_"):
            blocked.append(f"{run} has nonterminal smoke status: {status}")
if missing:
    print(json.dumps({"status": "decision_missing", "missing": missing}, indent=2), file=sys.stderr)
    raise SystemExit(3)
if blocked:
    print(json.dumps({"status": "blocked", "reasons": blocked}, indent=2), file=sys.stderr)
    raise SystemExit(4)
if not passed:
    print(json.dumps({"status": "no_smoke_pass", "message": "No memory mc256 smoke passed uncapped no-harm gate."}, indent=2), file=sys.stderr)
    raise SystemExit(5)
if manifest_out.exists():
    raise FileExistsError(f"refusing to overwrite existing manifest: {manifest_out}")
payload = {
    "purpose": "Track C train-only memory mc256 canonical no-harm only; no held-out query",
    "split_file": "/data/cyx/1030/dataset/biFlow_data/split_seed42.json",
    "data_dir": "/data/cyx/1030/dataset/latentfm_full/xverse",
    "biflow_dir": "/data/cyx/1030/dataset/biFlow_data",
    "anchor_checkpoint": anchor_ckpt,
    "launched_runs": passed,
    "heldout_query_used": False,
    "selection_weight_canonical_multi": 0,
    "source_smoke_manifests": [str(x[0]) for x in manifest_pairs],
}
manifest_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"status": "manifest_written", "manifest": str(manifest_out), "passed_runs": len(passed)}, indent=2))
PY

MANIFEST="${MANIFEST}" \
LABEL="${LABEL}" \
OUT_DIR="${UNCAPPED_OUT}" \
EVAL_MAX_MSE_CELLS="${EVAL_MAX_MSE_CELLS:-2048}" \
EVAL_MAX_MMD_CELLS="${EVAL_MAX_MMD_CELLS:-2048}" \
bash "${LAUNCHER}"
