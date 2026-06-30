#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent

now_epoch=$(date +%s)
boundary_epoch=$(date -d '2026-06-22 23:55:00' +%s)
if (( now_epoch < boundary_epoch )); then
  echo "Refusing to re-check Track C support-context smokes before 2026-06-22 23:55:00 CST" >&2
  echo "The 23:25 check found posthoc still pending; AGENTS.md requires a >=30 minute check interval." >&2
  exit 3
fi

bash "${ROOT}/ops/check_latentfm_trackc_support_context_after_2255_20260622.sh"
