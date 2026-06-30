#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

NOT_BEFORE="2026-06-24 04:26:00"

"${PYTHON}" - "${NOT_BEFORE}" <<'PY'
from __future__ import annotations

import sys
from datetime import datetime

not_before = datetime.strptime(sys.argv[1], "%Y-%m-%d %H:%M:%S")
now = datetime.now()
if now < not_before:
    print(
        f"Refusing to check general exposure long job before {not_before:%F %T} CST; "
        f"now={now:%F %T} CST",
        file=sys.stderr,
    )
    raise SystemExit(3)
PY

RUN=${ROOT}/runs/latentfm_xverse_scaling_count_smokes_20260624/xverse_scaling_general_exposure_cap_v2_3k_seed42
TRAIN_EXIT=${RUN}/xverse_scaling_general_exposure_cap_v2_3k_seed42.EXIT_CODE
POSTHOC_EXIT=${RUN}/POSTHOC_EXIT_CODE

echo "[$(date '+%F %T %Z')] general exposure long-job marker check"

if [[ -e "${TRAIN_EXIT}" ]]; then
  echo "general_train_exit=$(cat "${TRAIN_EXIT}")"
else
  echo "general_train_exit=still_running"
fi

if [[ -e "${POSTHOC_EXIT}" ]]; then
  echo "general_posthoc_exit=$(cat "${POSTHOC_EXIT}")"
else
  echo "general_posthoc_exit=posthoc_not_complete"
fi

if [[ -e "${POSTHOC_EXIT}" && "$(cat "${POSTHOC_EXIT}")" == "0" ]]; then
  echo "Summarizing general exposure-cap v2 internal decision"
  "${PYTHON}" "${ROOT}/ops/summarize_latentfm_xverse_scaling_count_smokes_20260624.py"
else
  echo "General exposure-cap v2 summary skipped because posthoc is not complete with exit 0"
fi
