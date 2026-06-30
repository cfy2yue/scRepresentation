#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
SCFMBENCH=${ROOT}/scFMBench
COUPLEDFM=${ROOT}/CoupledFM
RUN_ROOT=${ROOT}/runs/xverse_full_de5000_export_pack_20260620
LOG_DIR=${RUN_ROOT}/logs
REPORT_DIR=${ROOT}/reports
MANIFEST=${ROOT}/scFM_output/embedding_runs/manifest_latentfm_full_de5000.jsonl
EXPORT_ROOT=${ROOT}/scFM_output/embeddings
STATUS_JSONL=${ROOT}/scFM_output/embedding_runs/xverse_full_de5000_20260620_status.jsonl
QUEUE_LOG=${ROOT}/scFM_output/logs/xverse_full_de5000_20260620.log
PACK_LOG=${LOG_DIR}/pack_xverse_full_de5000.log
VALIDATE_JSON=${REPORT_DIR}/xverse_full_de5000_bundle_validation_20260620.json
SUMMARY_MD=${REPORT_DIR}/LATENTFM_XVERSE_FULL_DE5000_EXPORT_PACK_20260620.md

GPU_ID=${GPU_ID:-2}
BATCH_SIZE=${BATCH_SIZE:-8}
PY=${PY:-/data/cyx/software/miniconda3/envs/scdfm/bin/python}
OUT_DIR=${ROOT}/dataset/latentfm_full/xverse

mkdir -p "${RUN_ROOT}" "${LOG_DIR}" "${REPORT_DIR}" "$(dirname "${STATUS_JSONL}")" "$(dirname "${QUEUE_LOG}")"
rm -f "${RUN_ROOT}/EXIT_CODE" "${RUN_ROOT}/FINISHED"

log() {
  echo "[$(date '+%F %T %Z')] $*" | tee -a "${LOG_DIR}/run.log"
}

finish() {
  code="$?"
  echo "${code}" > "${RUN_ROOT}/EXIT_CODE"
  date '+%F %T %Z' > "${RUN_ROOT}/FINISHED"
  exit "${code}"
}
trap finish EXIT

date '+%F %T %Z' > "${RUN_ROOT}/STARTED"
cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: xverse_full_de5000_export_pack_20260620

Started: $(cat "${RUN_ROOT}/STARTED")

Runtime classification: long GPU embedding export followed by CPU/HDF5 pack and validation.

GPU: ${GPU_ID}
Batch size: ${BATCH_SIZE}

Command:
\`\`\`bash
GPU_ID=${GPU_ID} BATCH_SIZE=${BATCH_SIZE} bash ${ROOT}/ops/launch_xverse_full_de5000_export_pack_20260620.sh
\`\`\`

Expected outputs:
- ${EXPORT_ROOT}/xverse/<dataset>/raw/{latent.npy,obs.parquet|obs.csv.gz,meta.json}
- ${OUT_DIR}/manifest.json
- ${OUT_DIR}/condition_metadata.json
- ${OUT_DIR}/ctrl_means.npz
- ${OUT_DIR}/pert_means.npz
- ${VALIDATE_JSON}
- ${SUMMARY_MD}

Status checks:
\`\`\`bash
bash ${ROOT}/ops/check_xverse_full_de5000_export_pack_once_20260620.sh
\`\`\`
EOF

log "start xverse full DE5000 export+pack"
log "manifest=${MANIFEST}"
log "gpu=${GPU_ID} batch_size=${BATCH_SIZE} py=${PY}"
log "export_root=${EXPORT_ROOT}"
log "out_dir=${OUT_DIR}"

if [[ ! -x "${PY}" ]]; then
  log "missing python: ${PY}"
  exit 2
fi
if [[ ! -s "${MANIFEST}" ]]; then
  log "missing manifest: ${MANIFEST}"
  exit 2
fi

if [[ -s "${STATUS_JSONL}" ]]; then
  mv "${STATUS_JSONL}" "${STATUS_JSONL}.prev_$(date +%Y%m%d_%H%M%S)"
fi

export SCFM_XVERSE_PYTHON="${PY}"
export SCFM_ENVS_ROOT=/data/cyx/software/miniconda3/envs
export SCFM_OUTPUT_ROOT=${ROOT}/scFM_output
export SCFM_PRETRAINED_ROOT=${ROOT}/scFM_pretrained
export SCFM_THIRD_PARTY_ROOT=${ROOT}/scFM_third_party
export SCFM_CACHE_ROOT=${ROOT}/scFM_cache
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TORCHINDUCTOR_COMPILE_THREADS=2
export MAX_JOBS=2

log "preflight xverse python/weights"
"${PY}" - <<'PY' 2>&1 | tee -a "${LOG_DIR}/run.log"
import sys
sys.path.insert(0, "/data/cyx/1030/scLatent/scFMBench/fm/tools")
import model_registry
print("python_for_xverse", model_registry.python_for_model("xverse"))
print("weights", model_registry.check_weights("xverse"))
PY

log "embedding export queue start"
"${PY}" "${SCFMBENCH}/fm/tools/submit_embedding_queue.py" \
  --manifest "${MANIFEST}" \
  --export-root "${EXPORT_ROOT}" \
  --status-jsonl "${STATUS_JSONL}" \
  --log-file "${QUEUE_LOG}" \
  --gpus "${GPU_ID}" \
  --models xverse \
  --batch-size "${BATCH_SIZE}" \
  --skip-existing \
  --abort-after-consecutive-fails 1 2>&1 | tee -a "${LOG_DIR}/run.log"

log "embedding export queue finished"

"${PY}" - "${MANIFEST}" "${STATUS_JSONL}" "${EXPORT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
status = Path(sys.argv[2])
export_root = Path(sys.argv[3])
rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
events = [json.loads(line) for line in status.read_text().splitlines() if line.strip()] if status.is_file() else []
failed = [e for e in events if e.get("event") == "failed"]
complete = []
missing = []
for row in rows:
    raw = export_root / "xverse" / row["dataset_id"] / "raw"
    ok = (
        (raw / "latent.npy").is_file()
        and (raw / "meta.json").is_file()
        and ((raw / "obs.parquet").is_file() or (raw / "obs.csv.gz").is_file())
    )
    (complete if ok else missing).append(row["dataset_id"])
summary = {
    "manifest_rows": len(rows),
    "complete_raw_artifacts": len(complete),
    "missing_raw_artifacts": missing,
    "failed_events": failed,
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
if failed or missing:
    raise SystemExit(8)
PY

DATASETS="$("${PY}" - <<'PY'
import json
from pathlib import Path
manifest = Path("/data/cyx/1030/scLatent/scFM_output/embedding_runs/manifest_latentfm_full_de5000.jsonl")
ids = []
for line in manifest.read_text().splitlines():
    if line.strip():
        ids.append(json.loads(line)["dataset_id"])
print(" ".join(ids))
PY
)"

log "pack xverse LatentFM bundle datasets=${DATASETS}"
(
  cd "${COUPLEDFM}"
  export PYTHONPATH="${COUPLEDFM}:${PYTHONPATH:-}"
  "${PY}" -m model.latent.prepare_scfm_fm_data \
    --embeddings-root "${EXPORT_ROOT}" \
    --model xverse \
    --datasets ${DATASETS} \
    --out-dir "${OUT_DIR}" \
    --force
) 2>&1 | tee "${PACK_LOG}" | tee -a "${LOG_DIR}/run.log"

log "validate xverse bundle"
(
  cd "${COUPLEDFM}"
  export PYTHONPATH="${COUPLEDFM}:${PYTHONPATH:-}"
  "${PY}" -m model.latent.validate_fm_bundle \
    --data-dir "${OUT_DIR}" \
    --require-metadata \
    --out "${VALIDATE_JSON}"
) 2>&1 | tee "${LOG_DIR}/validate_xverse_full_de5000.log" | tee -a "${LOG_DIR}/run.log"

log "write summary"
"${PY}" - <<'PY'
import json
from pathlib import Path

root = Path("/data/cyx/1030/scLatent")
manifest_path = root / "scFM_output/embedding_runs/manifest_latentfm_full_de5000.jsonl"
status_path = root / "scFM_output/embedding_runs/xverse_full_de5000_20260620_status.jsonl"
validate_path = root / "reports/xverse_full_de5000_bundle_validation_20260620.json"
summary_path = root / "reports/LATENTFM_XVERSE_FULL_DE5000_EXPORT_PACK_20260620.md"
export_root = root / "scFM_output/embeddings/xverse"
bundle = root / "dataset/latentfm_full/xverse"

rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
events = [json.loads(line) for line in status_path.read_text().splitlines() if line.strip()] if status_path.is_file() else []
done = [e for e in events if e.get("event") == "done"]
failed = [e for e in events if e.get("event") == "failed"]
artifact_ok = []
artifact_missing = []
for row in rows:
    raw = export_root / row["dataset_id"] / "raw"
    ok = (raw / "latent.npy").is_file() and (raw / "meta.json").is_file() and ((raw / "obs.parquet").is_file() or (raw / "obs.csv.gz").is_file())
    (artifact_ok if ok else artifact_missing).append(row["dataset_id"])
validation = json.loads(validate_path.read_text()) if validate_path.is_file() else {}
summary = validation.get("summary", {})
lines = [
    "# LatentFM xverse Full DE5000 Export/Pack",
    "",
    f"- manifest rows: {len(rows)}",
    f"- export done events: {len(done)}",
    f"- export failed events: {len(failed)}",
    f"- complete raw artifacts: {len(artifact_ok)} / {len(rows)}",
    f"- bundle: `{bundle}`",
    f"- validation ok: {validation.get('ok')}",
    f"- validation datasets: {summary.get('datasets')}",
    f"- validation conditions: {summary.get('conditions')}",
    f"- validation ctrl rows: {summary.get('ctrl_rows')}",
    f"- validation gt rows: {summary.get('gt_rows')}",
    f"- validation emb_dim: {summary.get('emb_dim')}",
    "",
    "## Missing Raw Artifacts",
    "",
]
if artifact_missing:
    lines.extend(f"- {x}" for x in artifact_missing)
else:
    lines.append("- none")
lines.extend(["", "## Validation Errors", ""])
errors = validation.get("errors") or []
if errors:
    lines.extend(f"- {e}" for e in errors)
else:
    lines.append("- none")
summary_path.write_text("\n".join(lines) + "\n")
print(summary_path)
PY

log "completed xverse full DE5000 export+pack"
