#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent

now_epoch=$(date +%s)
boundary_epoch=$(date -d '2026-06-22 21:40:00' +%s)
if (( now_epoch < boundary_epoch )); then
  echo "Refusing to check all memory mc256 decisions before 2026-06-22 21:40:00 CST" >&2
  exit 3
fi

echo "# Track C train-only memory mc256 combined status"
echo "checked_at=$(date '+%F %T %Z')"
echo
echo "## Base block"
bash "${ROOT}/ops/check_latentfm_trackc_trainonly_memory_mc256_after_2058_20260622.sh"
echo
echo "## Extension block"
bash "${ROOT}/ops/check_latentfm_trackc_trainonly_memory_mc256_ext_after_2110_20260622.sh"
echo
echo "## Combined summary"
bash "${ROOT}/ops/summarize_latentfm_trackc_trainonly_memory_mc256_all_decisions_20260622.sh"
