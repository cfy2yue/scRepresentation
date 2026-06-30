#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="/data2/cfy/FM/CoupledFM/scFM/output/embeddings"
LOCAL_ROOT="/data/cyx/1030/dataset/latentfm_staging/scfm_embeddings"
LOG_DIR="/data/cyx/1030/scLatent/logs"
mkdir -p "${LOCAL_ROOT}" "${LOG_DIR}"

models=(stack scfoundation state)
datasets=(
  Adamson
  NormanWeissman2019_filtered__single
  ReplogleWeissman2022_K562_gwps
  Replogle_RPE1essential
  TianActivation
  TianInhibition
  sciplex3_A549
  sciplex3_K562
  sciplex3_MCF7
  sciplex3_xCellLine
)

echo "[$(date '+%F %T')] sync start"
echo "remote=${REMOTE_ROOT}"
echo "local=${LOCAL_ROOT}"
echo "models=${models[*]}"
echo "datasets=${datasets[*]}"

rsync_opts=(
  -a
  --partial
  --append-verify
  --timeout=120
  --info=stats2
)

for model in "${models[@]}"; do
  for ds in "${datasets[@]}"; do
    src="LiLab:${REMOTE_ROOT}/${model}/${ds}/raw/"
    dst="${LOCAL_ROOT}/${model}/${ds}/raw/"
    mkdir -p "${dst}"
    echo "[$(date '+%F %T')] rsync ${model}/${ds}"
    ok=0
    for attempt in 1 2 3; do
      if nice -n 10 rsync "${rsync_opts[@]}" \
        --include='latent.npy' \
        --include='meta.json' \
        --include='obs.parquet' \
        --include='obs.csv.gz' \
        --include='obs.csv' \
        --exclude='*' \
        "${src}" "${dst}"; then
        ok=1
        break
      fi
      echo "[$(date '+%F %T')] WARN retry ${attempt}/3 failed for ${model}/${ds}"
      sleep 5
    done
    if [[ "${ok}" != "1" ]]; then
      echo "[$(date '+%F %T')] ERROR giving up ${model}/${ds}" >&2
      exit 1
    fi
  done
done

echo "[$(date '+%F %T')] sync complete"
find "${LOCAL_ROOT}" -path '*/raw/latent.npy' -printf '%p\t%s\n' | sort
