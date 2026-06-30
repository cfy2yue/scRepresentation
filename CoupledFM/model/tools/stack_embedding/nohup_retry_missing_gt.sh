#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${STACK_PYTHON:-python}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOGROOT="${ROOT}/logs/stack_embedding/retry_missing_gt_${RUN_ID}"
TMPDIR_STACK="${ROOT}/tmp/stack_embedding"
OUTROOT="${ROOT}/data/latent_data/stack"

mkdir -p "${LOGROOT}" "${TMPDIR_STACK}" "${OUTROOT}"

export TMPDIR="${TMPDIR_STACK}"
export TEMP="${TMPDIR_STACK}"
export TMP="${TMPDIR_STACK}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "${ROOT}"

nohup "${PY}" -u model/tools/stack_embedding/step3_gt_embedding.py \
  --biflow-dir "${OUTROOT}" \
  --tmp-dir "${TMPDIR_STACK}" \
  --log-dir "${LOGROOT}" \
  --datasets ReplogleWeissman2022_K562_gwps sciplex3_K562 \
  --gpus 0,1 -j 2 --overwrite \
  >"${LOGROOT}/nohup.out" 2>&1 &

echo "${!}" > "${LOGROOT}/pid.txt"
