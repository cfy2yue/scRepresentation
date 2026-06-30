#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PY=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PY}" ]]; then
  PY=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

TZ_NAME=Asia/Shanghai
NOT_BEFORE="2026-06-23 00:56:00"
now_epoch="$(TZ=${TZ_NAME} date +%s)"
gate_epoch="$(TZ=${TZ_NAME} date -d "${NOT_BEFORE}" +%s)"
if (( now_epoch < gate_epoch )); then
  echo "Refusing to check before ${NOT_BEFORE} CST; long GPU tasks should not be polled early." >&2
  exit 3
fi

exec "${PY}" "${ROOT}/ops/summarize_latentfm_crosslatent_tracka_gene_reliability_adapter_block_20260623.py"
