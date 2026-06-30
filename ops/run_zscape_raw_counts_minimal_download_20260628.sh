#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?run dir required}"
DATA_DIR="${2:?data dir required}"
OUT_DIR="${RUN_DIR}/outputs"

mkdir -p "${DATA_DIR}" "${OUT_DIR}"

BASE_URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE202nnn/GSE202639/suppl"
FILE="GSE202639_zperturb_full_raw_counts.RDS.gz"
URL="${BASE_URL}/${FILE}"
DEST="${DATA_DIR}/${FILE}"

date > "${RUN_DIR}/DOWNLOAD_STARTED"
{
  echo "[download] ${URL}"
  echo "[dest] ${DEST}"
  echo "[boundary] zperturb raw counts only; no CDS/reference matrix/model/GPU"
} > "${OUT_DIR}/download_manifest.txt"

curl \
  -L \
  --fail \
  --silent \
  --show-error \
  --retry 12 \
  --retry-delay 30 \
  --retry-all-errors \
  --retry-connrefused \
  --connect-timeout 60 \
  --speed-time 240 \
  --speed-limit 1024 \
  -C - \
  -o "${DEST}" \
  "${URL}"

date > "${RUN_DIR}/DOWNLOAD_FINISHED"
sha256sum "${DEST}" > "${RUN_DIR}/SHA256SUMS"
gzip -t "${DEST}"
stat -c '%n	%s bytes' "${DEST}" > "${OUT_DIR}/downloaded_file_size.txt"

cat > "${OUT_DIR}/LATENTFM_ZSCAPE_RAW_COUNTS_MINIMAL_DOWNLOAD_20260628.md" <<EOF
# LatentFM ZSCAPE Minimal Raw Counts Download

Status: \`zscape_raw_counts_minimal_download_complete_no_gpu\`

Boundary:
- Downloaded only \`${FILE}\`.
- Did not download CDS/reference expression objects.
- Did not train, infer, embed, read canonical multi, or read Track C query.
- Did not use GPU.

Output:
- \`${DEST}\`
- SHA256: \`${RUN_DIR}/SHA256SUMS\`
EOF
