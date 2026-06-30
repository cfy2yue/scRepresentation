#!/usr/bin/env bash
set -u -o pipefail

ROOT="/data/cyx/1030/scLatent"
TARGET_TIME="2026-06-27 04:20:00"
DOWNLOAD_RUN="${ROOT}/runs/latentfm_external_replogle_bulk_sources_download_20260627"
INSPECT_RUN="${ROOT}/runs/latentfm_external_replogle_bulk_source_inspect_20260627"
INSPECT_SCRIPT="${ROOT}/ops/inspect_latentfm_replogle_bulk_sources_20260627.py"
PYTHON_BIN="/data/cyx/software/miniconda3/envs/scdfm/bin/python"

now_epoch="$(date +%s)"
target_epoch="$(date -d "${TARGET_TIME}" +%s)"
if (( now_epoch < target_epoch )); then
  echo "[check] refusing before ${TARGET_TIME}; now=$(date '+%F %T %Z')"
  exit 3
fi

cat "${DOWNLOAD_RUN}/EXIT_CODE" 2>/dev/null || {
  echo "[check] download still running or marker missing"
  exit 4
}
download_rc="$(tr -d '[:space:]' < "${DOWNLOAD_RUN}/EXIT_CODE")"
if [[ "${download_rc}" != "0" ]]; then
  echo "[check] download failed rc=${download_rc}"
  exit 1
fi

"${PYTHON_BIN}" "${INSPECT_SCRIPT}"
inspect_rc="$?"
echo "${inspect_rc}" > "${INSPECT_RUN}/INSPECT_EXIT_CODE"
exit "${inspect_rc}"
