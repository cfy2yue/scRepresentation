#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PY=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PY}" ]]; then
  PY=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi
RUN_NAME=scldm_tracka_gene_shrink_k4_dataset_negative_adapter_2k_seed42
RUN_DIR=${ROOT}/runs/latentfm_tracka_scldm_guarded_fallback_20260623/${RUN_NAME}
GATE_JSON=${ROOT}/reports/latentfm_tracka_scldm_guarded_fallback_adapter_${RUN_NAME}_gate_20260623.json
DECISION_MD=${ROOT}/reports/LATENTFM_TRACKA_SCLDM_GUARDED_FALLBACK_ADAPTER_${RUN_NAME}_DECISION_20260623.md
WINDOW="2026-06-23 04:14:00"

now_epoch=$(date +%s)
window_epoch=$(date -d "${WINDOW}" +%s)
if (( now_epoch < window_epoch )); then
  echo "Refusing to check before ${WINDOW} CST; Track A SCLDM guarded fallback smoke is a long GPU task." >&2
  exit 3
fi

echo "[$(date '+%F %T %Z')] Track A SCLDM guarded fallback check"
echo "run_dir=${RUN_DIR}"
cat "${RUN_DIR}/EXIT_CODE" 2>/dev/null || echo "train still running or marker absent"
cat "${RUN_DIR}/POSTHOC_EXIT_CODE" 2>/dev/null || echo "posthoc still running or marker absent"
if [[ -f "${DECISION_MD}" ]]; then
  sed -n '1,140p' "${DECISION_MD}"
elif [[ -f "${GATE_JSON}" ]]; then
  "${PY}" - <<PY
import json
from pathlib import Path
p = Path("${GATE_JSON}")
payload = json.loads(p.read_text(encoding="utf-8"))
print(json.dumps(payload.get("gate") or payload, indent=2)[:4000])
PY
else
  echo "decision report not available yet"
fi
