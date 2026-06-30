#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLEDFM=${ROOT}/CoupledFM
MANIFEST=${ROOT}/scFM_output/embedding_runs/manifest_latentfm_full_de5000.jsonl
STATUS_JSONL=${ROOT}/scFM_output/embedding_runs/xverse_full_de5000_20260620_status.jsonl
EXPORT_ROOT=${ROOT}/scFM_output/embeddings
OUT_DIR=${ROOT}/dataset/latentfm_full/xverse
RUN_ROOT=${ROOT}/runs/xverse_full_de5000_pack_validate_recovery_20260620
LOG_DIR=${RUN_ROOT}/logs
PACK_LOG=${LOG_DIR}/pack_xverse_full_de5000.log
VALIDATE_JSON=${ROOT}/reports/xverse_full_de5000_bundle_validation_20260620.json
SUMMARY_MD=${ROOT}/reports/LATENTFM_XVERSE_FULL_DE5000_EXPORT_PACK_20260620.md
PY=${ROOT}/software/miniconda3/envs/scdfm/bin/python

if [[ ! -x "${PY}" ]]; then
  PY=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

mkdir -p "${LOG_DIR}" "${ROOT}/reports"
rm -f "${RUN_ROOT}/EXIT_CODE" "${RUN_ROOT}/FINISHED"
date "+%F %T %Z" > "${RUN_ROOT}/STARTED"
cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: xverse_full_de5000_pack_validate_recovery_20260620

## Command

\`\`\`bash
bash ${ROOT}/ops/recover_xverse_full_de5000_pack_validate_20260620.sh
\`\`\`

## Runtime classification

CPU/IO recovery task. No GPU is used; it consumes existing 22/22 xverse raw
embedding artifacts and performs pack + validation only.

## Start time

$(cat "${RUN_ROOT}/STARTED")

## Log path

\`${LOG_DIR}/run.log\`

## Expected outputs

* \`${SUMMARY_MD}\`
* \`${VALIDATE_JSON}\`
* \`${OUT_DIR}/manifest.json\`

## How to check manually

\`\`\`bash
bash ${ROOT}/ops/check_xverse_full_de5000_pack_validate_recovery_once_20260620.sh
\`\`\`

## Current status

Started.
EOF

log() {
  echo "[$(date '+%F %T %Z')] $*" | tee -a "${LOG_DIR}/run.log"
}

trap 'rc=$?; echo "$rc" > "${RUN_ROOT}/EXIT_CODE"; date "+%F %T %Z" > "${RUN_ROOT}/FINISHED"; exit "$rc"' EXIT

log "start xverse full DE5000 pack/validate recovery"
log "manifest=${MANIFEST}"
log "export_root=${EXPORT_ROOT}"
log "out_dir=${OUT_DIR}"

"${PY}" - "${MANIFEST}" "${STATUS_JSONL}" "${EXPORT_ROOT}" <<'PY' 2>&1 | tee -a "${LOG_DIR}/run.log"
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
status = Path(sys.argv[2])
export_root = Path(sys.argv[3])
rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
events = [json.loads(line) for line in status.read_text().splitlines() if line.strip()] if status.is_file() else []
failed = [e for e in events if e.get("event") == "failed"]
done = [e for e in events if e.get("event") == "done"]
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
    "done_events": len(done),
    "failed_events": len(failed),
    "complete_raw_artifacts": len(complete),
    "missing_raw_artifacts": missing,
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
if failed or missing or len(done) != len(rows):
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
    f"- recovery run: `xverse_full_de5000_pack_validate_recovery_20260620`",
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
lines.extend([
    "",
    "## Failed Export Events",
    "",
])
if failed:
    lines.extend(f"- {e.get('dataset_id')}: rc={e.get('returncode')}" for e in failed)
else:
    lines.append("- none")
summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_path)
PY

log "done"
