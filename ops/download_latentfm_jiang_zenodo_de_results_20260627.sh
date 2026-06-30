#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
OUT_DIR="${ROOT}/reports/external_artifact_sources_20260627/jiang_zenodo_v2_1"
URL="https://zenodo.org/api/records/14518762/files/DE_results_all_pathway.zip/content"
OUT="${OUT_DIR}/DE_results_all_pathway.zip"
EXPECTED_MD5="f077cba680a1affc599f5153d99b0e45"

mkdir -p "${OUT_DIR}"
echo "[download] start=$(date '+%F %T %Z')"
echo "[download] url=${URL}"
echo "[download] out=${OUT}"

curl -L --fail --retry 3 --retry-delay 10 -C - -o "${OUT}" "${URL}"

actual_md5="$(md5sum "${OUT}" | awk '{print $1}')"
echo "[download] md5=${actual_md5}"
if [[ "${actual_md5}" != "${EXPECTED_MD5}" ]]; then
  echo "[download] md5 mismatch expected=${EXPECTED_MD5} actual=${actual_md5}" >&2
  exit 2
fi

unzip -l "${OUT}" > "${OUT_DIR}/DE_results_all_pathway_zip_listing.txt"
echo "[download] listing=${OUT_DIR}/DE_results_all_pathway_zip_listing.txt"
echo "[download] finished=$(date '+%F %T %Z')"
