#!/usr/bin/env bash
set -euo pipefail

WINDOW="2026-06-23 04:24:00"
now_epoch=$(date +%s)
window_epoch=$(date -d "${WINDOW}" +%s)
if (( now_epoch < window_epoch )); then
  echo "Refusing to check before ${WINDOW} CST; support-FiLM retry1 posthoc is a long task." >&2
  exit 3
fi

bash /data/cyx/1030/scLatent/ops/check_latentfm_trackc_support_film_retry1_after_0354_20260623.sh
