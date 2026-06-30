#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
OUT_DIR="${ROOT}/reports/external_artifact_sources_20260627/depmap_24q4_figshare"
mkdir -p "${OUT_DIR}"

download_one() {
  local name="$1"
  local url="$2"
  local expected_md5="$3"
  local out="${OUT_DIR}/${name}"
  local signed_url
  echo "[download] ${name} start=$(date '+%F %T %Z')"
  if [[ -s "${out}" ]]; then
    local existing_md5
    existing_md5="$(md5sum "${out}" | awk '{print $1}')"
    if [[ "${existing_md5}" == "${expected_md5}" ]]; then
      echo "[download] ${name} already present with matching md5; skipping"
      return 0
    fi
  fi
  for attempt in $(seq 1 30); do
    echo "[download] ${name} attempt=${attempt}"
    signed_url="$(curl -sSI -A 'Mozilla/5.0' "${url}" | awk 'tolower($1)=="location:" {sub(/\r$/, "", $2); print $2; exit}')"
    if [[ -z "${signed_url}" ]]; then
      echo "[download] ${name} failed to resolve signed URL from ${url}" >&2
      sleep 5
      continue
    fi
    if curl -L --fail --connect-timeout 8 --speed-time 60 --speed-limit 1024 -C - -A 'Mozilla/5.0' -o "${out}" "${signed_url}"; then
      break
    fi
    sleep 5
  done
  local actual_md5
  actual_md5="$(md5sum "${out}" | awk '{print $1}')"
  echo "[download] ${name} md5=${actual_md5}"
  if [[ "${actual_md5}" != "${expected_md5}" ]]; then
    echo "[download] ${name} md5 mismatch expected=${expected_md5} actual=${actual_md5}" >&2
    exit 2
  fi
}

echo "[download] DepMap 24Q4 Public source acquisition"
echo "[download] start=$(date '+%F %T %Z')"
download_one "Model.csv" "https://ndownloader.figshare.com/files/51065297" "675210d17675f3517b0ce39a3c274f16"
download_one "CRISPRScreenMap.csv" "https://ndownloader.figshare.com/files/51065159" "dda4409d030d486e8a66915990731a62"
download_one "CRISPRGeneEffect.csv" "https://ndownloader.figshare.com/files/51064667" "6edf7ade09b9b34199210b559d4745d3"
echo "[download] finished=$(date '+%F %T %Z')"
