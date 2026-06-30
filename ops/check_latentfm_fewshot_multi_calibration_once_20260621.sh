#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_fewshot_multi_calibration_20260621
REPORT_DIR=${ROOT}/reports
MIN_INTERVAL_SECONDS=${MIN_INTERVAL_SECONDS:-1800}
FORCE=${FORCE:-0}

mkdir -p "${REPORT_DIR}" "${RUN_ROOT}/logs"
STAMP=${RUN_ROOT}/logs/last_manual_check_epoch.txt
now=$(date +%s)
if [[ "${FORCE}" != "1" && -s "${STAMP}" ]]; then
  last=$(cat "${STAMP}")
  if [[ "${last}" =~ ^[0-9]+$ ]]; then
    elapsed=$((now - last))
    if (( elapsed < MIN_INTERVAL_SECONDS )); then
      echo "Last manual check was ${elapsed}s ago; minimum is ${MIN_INTERVAL_SECONDS}s. Set FORCE=1 to override." >&2
      exit 8
    fi
  fi
fi
echo "${now}" > "${STAMP}"

OUT=${REPORT_DIR}/LATENTFM_FEWSHOT_MULTI_CALIBRATION_ONE_SHOT_STATUS_$(date +%Y%m%d_%H%M%S).txt
MANIFEST=${RUN_ROOT}/launch_manifest.json

{
  date '+%F %T %Z'
  echo "## tmux"
  tmux ls | rg 'latentfm_fewshot|lfm_scf_prior010_inject_fewshot' || true
  echo "## nvidia-smi"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  echo "## run statuses"
  python - "${RUN_ROOT}" "${MANIFEST}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
for row in manifest.get("launched_runs", []):
    name = row["run_name"]
    exitp = root / f"{name}.EXIT_CODE"
    finp = root / f"{name}.FINISHED"
    outdir = pathlib.Path(row["out_dir"])
    print(name)
    print("  exit:", exitp.read_text().strip() if exitp.exists() else "RUNNING")
    print("  finished:", finp.read_text().strip() if finp.exists() else "-")
    print(
        "  best:",
        (outdir / "best.pt").exists(),
        "latest:",
        (outdir / "latest.pt").exists(),
        "config:",
        (outdir / "config.json").exists(),
        "iid:",
        (outdir / "iid_metrics.json").exists(),
    )
PY
  echo "## posthoc"
  cat "${RUN_ROOT}/POSTHOC_EXIT_CODE" 2>/dev/null || echo "POSTHOC_RUNNING_OR_WAITING"
  cat "${RUN_ROOT}/POSTHOC_FINISHED" 2>/dev/null || true
  echo "## log tails"
  python - "${MANIFEST}" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for row in manifest.get("launched_runs", []):
    log = pathlib.Path(row["log"])
    print(f"### {row['run_name']}")
    if not log.exists():
        print("NO_LOG")
        continue
    lines = log.read_text(errors="replace").splitlines()
    for line in lines[-8:]:
        print(line)
PY
} | tee "${OUT}"

echo "STATUS_REPORT=${OUT}"
