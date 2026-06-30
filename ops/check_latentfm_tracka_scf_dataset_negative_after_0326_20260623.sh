#!/usr/bin/env bash
set -euo pipefail

WINDOW="2026-06-23 03:26:00"
now_epoch=$(date +%s)
window_epoch=$(date -d "${WINDOW}" +%s)
if (( now_epoch < window_epoch )); then
  echo "Refusing to check before ${WINDOW} CST; Track A dataset-negative posthoc is a long task." >&2
  exit 3
fi

bash /data/cyx/1030/scLatent/ops/check_latentfm_tracka_scf_dataset_negative_after_0256_20260623.sh
