#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?run dir required}"
DATA_DIR="${2:?data dir required}"
OUT_DIR="${RUN_DIR}/outputs"

mkdir -p "${DATA_DIR}" "${OUT_DIR}"

BASE_URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE202nnn/GSE202639/suppl"
FILES=(
  "GSE202639_zperturb_full_cell_metadata.csv.gz"
  "GSE202639_zperturb_full_gene_metadata.csv.gz"
  "GSE202639_reference_cell_metadata.csv.gz"
  "GSE202639_reference_gene_metadata.csv.gz"
)

date > "${RUN_DIR}/DOWNLOAD_STARTED"
for file in "${FILES[@]}"; do
  url="${BASE_URL}/${file}"
  dest="${DATA_DIR}/${file}"
  echo "[download] ${url} -> ${dest}"
  curl \
    -L \
    --fail \
    --retry 10 \
    --retry-delay 20 \
    --retry-all-errors \
    --retry-connrefused \
    --connect-timeout 45 \
    --speed-time 180 \
    --speed-limit 1024 \
    -C - \
    -o "${dest}" \
    "${url}"
done
date > "${RUN_DIR}/DOWNLOAD_FINISHED"

sha256sum "${FILES[@]/#/${DATA_DIR}/}" > "${RUN_DIR}/SHA256SUMS"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/data/cyx/software/miniconda3/envs/scdfm/bin/python \
  /data/cyx/1030/scLatent/ops/audit_zscape_metadata_coverage_20260628.py \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --run-name "$(basename "${RUN_DIR}")"
