#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-LiLab}"
REMOTE_ROOT="${REMOTE_ROOT:-/data2/cfy/FM/CoupledFM}"
LOCAL_ROOT="${LOCAL_ROOT:-/data/cyx/1030/scLatent}"
LOG_DIR="${LOCAL_ROOT}/logs"

mkdir -p \
  "${LOCAL_ROOT}/dataset" \
  "${LOCAL_ROOT}/pretrainckpt/genepert_cache" \
  "${LOCAL_ROOT}/scFM_pretrained" \
  "${LOCAL_ROOT}/scFM_third_party" \
  "${LOCAL_ROOT}/scFM_output" \
  "${LOG_DIR}"

RSYNC_BASE=(
  rsync -a
  --partial
  --partial-dir=.rsync-partial
  --timeout=600
  --human-readable
  --info=progress2,stats2
)

run_rsync() {
  local label="$1"
  local src="$2"
  local dst="$3"
  local status_file="${LOG_DIR}/transfer_from_lilab.status"
  local rc=0
  shift 3
  echo "===== $(date '+%F %T') START ${label} ====="
  printf '%s\tSTART\t%s\n' "$(date '+%F %T')" "$label" > "$status_file"
  mkdir -p "$dst"
  set +e
  "${RSYNC_BASE[@]}" "$@" "${REMOTE}:${src%/}/" "${dst%/}/"
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "===== $(date '+%F %T') FAIL  ${label} rc=${rc} ====="
    printf '%s\tFAIL\t%s\trc=%s\n' "$(date '+%F %T')" "$label" "$rc" > "$status_file"
    exit "$rc"
  fi
  echo "===== $(date '+%F %T') DONE  ${label} ====="
  printf '%s\tDONE\t%s\n' "$(date '+%F %T')" "$label" > "$status_file"
}

# CoupledFM training and raw-pretrain inputs. These land in one standalone
# dataset folder so the data can be backed up or published independently.
run_rsync "CoupledFM biFlow stack data" \
  "${REMOTE_ROOT}/model/local_biFlow_data/biFlow_data" \
  "${LOCAL_ROOT}/dataset/biFlow_data"

run_rsync "CoupledFM cellgene census processed data" \
  "${REMOTE_ROOT}/model/local_biFlow_data/cellgene_census" \
  "${LOCAL_ROOT}/dataset/cellgene_census"

# scFMBench h5ad inputs.
run_rsync "scFMBench staging data" \
  "${REMOTE_ROOT}/scFM/data" \
  "${LOCAL_ROOT}/dataset/scFM_data" \
  --exclude='*.tmp.h5ad' \
  --exclude='*.bak*' \
  --exclude='*.before_*'

# Raw sources are not needed for immediate training, but are useful for
# rebuilding/Zenodo-style data releases.
run_rsync "raw source h5ad data" \
  "${REMOTE_ROOT}/data/raw" \
  "${LOCAL_ROOT}/dataset/raw"

# Pretrained model assets for scFMBench and shared CellNavi resources.
run_rsync "scFM pretrained assets" \
  "${REMOTE_ROOT}/scFM/pretrained" \
  "${LOCAL_ROOT}/scFM_pretrained"

run_rsync "CoupledFM CellNavi gene cache" \
  "${REMOTE_ROOT}/model/condition_emb/genepert/cache/cellnavi_embed_gene" \
  "${LOCAL_ROOT}/pretrainckpt/genepert_cache/cellnavi_embed_gene"

run_rsync "CoupledFM scGPT gene cache" \
  "${REMOTE_ROOT}/model/condition_emb/genepert/cache/scgpt_embed_gene" \
  "${LOCAL_ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene"

# Share CellNavi resources with CoupledFM without duplicating the 223M tree.
if [[ ! -e "${LOCAL_ROOT}/pretrainckpt/cellnavi" ]]; then
  ln -s ../scFM_pretrained/cellnavi "${LOCAL_ROOT}/pretrainckpt/cellnavi"
fi

# Third-party source mirrors used by scFMBench adapters.
run_rsync "scFM third-party sources" \
  "${REMOTE_ROOT}/scFM/fm/third_party" \
  "${LOCAL_ROOT}/scFM_third_party"

# Lightweight output metadata; old embeddings are optional because they are
# result artifacts, not required inputs.
run_rsync "scFM output run metadata" \
  "${REMOTE_ROOT}/scFM/output/embedding_runs" \
  "${LOCAL_ROOT}/scFM_output/embedding_runs"

run_rsync "scFM benchmark inventory" \
  "${REMOTE_ROOT}/scFM/output/benchmark_inventory" \
  "${LOCAL_ROOT}/scFM_output/benchmark_inventory"

if [[ "${SYNC_OUTPUT_EMBEDDINGS:-0}" == "1" ]]; then
  run_rsync "scFM output embeddings" \
    "${REMOTE_ROOT}/scFM/output/embeddings" \
    "${LOCAL_ROOT}/scFM_output/embeddings"
fi

echo "===== $(date '+%F %T') ALL DONE ====="
printf '%s\tALL DONE\t%s\n' "$(date '+%F %T')" "LiLab resource sync complete" > "${LOG_DIR}/transfer_from_lilab.status"
