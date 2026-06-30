#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/xverse_full_de5000_export_pack_20260620
REPORT=${ROOT}/reports/XVERSE_FULL_DE5000_EXPORT_PACK_ONE_SHOT_STATUS_$(date +%Y%m%d_%H%M%S).txt
STATUS_JSONL=${ROOT}/scFM_output/embedding_runs/xverse_full_de5000_20260620_status.jsonl
QUEUE_LOG=${ROOT}/scFM_output/logs/xverse_full_de5000_20260620.log
SUMMARY=${ROOT}/reports/LATENTFM_XVERSE_FULL_DE5000_EXPORT_PACK_20260620.md
VALIDATE_JSON=${ROOT}/reports/xverse_full_de5000_bundle_validation_20260620.json

mkdir -p "${ROOT}/reports"

{
  echo "[$(date '+%F %T %Z')] one-shot xverse full DE5000 export/pack status check"
  echo "report=${REPORT}"
  echo
  tmux ls 2>/dev/null | grep 'xverse_full_de5000_export_pack_20260620' || true
  echo
  if [[ -f "${RUN_ROOT}/EXIT_CODE" ]]; then
    echo "EXIT_CODE=$(cat "${RUN_ROOT}/EXIT_CODE")"
    echo "FINISHED=$(cat "${RUN_ROOT}/FINISHED" 2>/dev/null || true)"
  else
    echo "still-running-or-not-started"
    echo "STARTED=$(cat "${RUN_ROOT}/STARTED" 2>/dev/null || true)"
  fi
  echo
  if [[ -f "${STATUS_JSONL}" ]]; then
    python - <<'PY'
import json
from pathlib import Path
p = Path("/data/cyx/1030/scLatent/scFM_output/embedding_runs/xverse_full_de5000_20260620_status.jsonl")
events = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
starts = [e for e in events if e.get("event") == "start"]
dones = [e for e in events if e.get("event") == "done"]
fails = [e for e in events if e.get("event") == "failed"]
print(f"status_events={len(events)} starts={len(starts)} dones={len(dones)} fails={len(fails)}")
if events:
    last = events[-1]
    print("last_event=" + json.dumps(last, ensure_ascii=False, sort_keys=True))
PY
  else
    echo "missing ${STATUS_JSONL}"
  fi
  echo
  python - <<'PY'
import json
from pathlib import Path
root = Path("/data/cyx/1030/scLatent")
manifest = root / "scFM_output/embedding_runs/manifest_latentfm_full_de5000.jsonl"
export_root = root / "scFM_output/embeddings/xverse"
rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
complete = 0
missing = []
for row in rows:
    raw = export_root / row["dataset_id"] / "raw"
    ok = (raw / "latent.npy").is_file() and (raw / "meta.json").is_file() and ((raw / "obs.parquet").is_file() or (raw / "obs.csv.gz").is_file())
    if ok:
        complete += 1
    else:
        missing.append(row["dataset_id"])
print(f"raw_artifacts_complete={complete}/{len(rows)}")
if missing:
    print("missing_raw=" + ",".join(missing[:20]))
PY
  echo
  for path in "${SUMMARY}" "${VALIDATE_JSON}" "${ROOT}/dataset/latentfm_full/xverse/manifest.json"; do
    if [[ -s "${path}" ]]; then
      echo "present ${path}"
    else
      echo "missing ${path}"
    fi
  done
  echo
  if [[ -f "${RUN_ROOT}/logs/run.log" ]]; then
    echo "tail ${RUN_ROOT}/logs/run.log"
    tail -n 120 "${RUN_ROOT}/logs/run.log"
  else
    echo "missing ${RUN_ROOT}/logs/run.log"
  fi
  echo
  if [[ -f "${QUEUE_LOG}" ]]; then
    echo "tail ${QUEUE_LOG}"
    tail -n 80 "${QUEUE_LOG}"
  else
    echo "missing ${QUEUE_LOG}"
  fi
} | tee "${REPORT}"
