#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
OUT_DIR="${ROOT}/reports/external_artifact_sources_20260627/replogle_figshare_bulk"
mkdir -p "${OUT_DIR}"

download_one() {
  local name="$1"
  local url="$2"
  local expected_md5="$3"
  local out="${OUT_DIR}/${name}"
  echo "[download] ${name} start=$(date '+%F %T %Z')"
  if [[ -s "${out}" ]]; then
    local existing_md5
    existing_md5="$(md5sum "${out}" | awk '{print $1}')"
    if [[ "${existing_md5}" == "${expected_md5}" ]]; then
      echo "[download] ${name} already present with matching md5; skipping"
      return 0
    fi
  fi
  curl -L --fail --retry 20 --retry-delay 10 --retry-all-errors \
    --connect-timeout 30 --speed-time 180 --speed-limit 1024 \
    --continue-at - -A 'Mozilla/5.0' -o "${out}" "${url}"
  local actual_md5
  actual_md5="$(md5sum "${out}" | awk '{print $1}')"
  echo "[download] ${name} md5=${actual_md5}"
  if [[ "${actual_md5}" != "${expected_md5}" ]]; then
    echo "[download] ${name} md5 mismatch expected=${expected_md5} actual=${actual_md5}" >&2
    exit 2
  fi
}

echo "[download] Replogle et al. 2022 Figshare+ bulk source acquisition"
echo "[download] article=https://api.figshare.com/v2/articles/20029387"
echo "[download] start=$(date '+%F %T %Z')"
download_one "K562_essential_normalized_bulk_01.h5ad" "https://ndownloader.figshare.com/files/35780870" "30496767641cd2e660ee6ecb5baee132"
download_one "K562_gwps_normalized_bulk_01.h5ad" "https://ndownloader.figshare.com/files/35773217" "a3dfaa94ea8724217f5ecb1e14a5f0c8"
download_one "rpe1_normalized_bulk_01.h5ad" "https://ndownloader.figshare.com/files/35775512" "6f1e7d6a09e2f869759e3c4526b7f171"
echo "[download] finished=$(date '+%F %T %Z')"
