#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
LAUNCHER=${ROOT}/ops/launch_latentfm_true_cell_count_single_smoke_20260624.sh

MATERIALIZER_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_capped_h5_materializer_gate_20260624.json
SCHEMA_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_capped_h5_schema_gate_20260624.json
DRYLOAD_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_dryload_gate_20260624.json
DESIGN_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_design_controls_gate_20260624.json

RUN_ROOT=${ROOT}/runs/latentfm_true_cell_count_nested_smokes_20260624
OUT_ROOT=${ROOT}/CoupledFM/output/latentfm_runs/true_cell_count_nested_smokes_20260624
LOG_ROOT=${ROOT}/logs/latentfm_true_cell_count_nested_smokes_20260624

TOTAL_STEPS=${LATENTFM_TRUE_CELL_COUNT_TOTAL_STEPS:-3000}

if [[ "${LATENTFM_TRUE_CELL_COUNT_NESTED_MATRIX_ACK:-}" != "launch_nested_true_cell_count_matrix" ]]; then
  cat >&2 <<'EOF'
Refusing to launch nested true-cell-count matrix.

Set:
  LATENTFM_TRUE_CELL_COUNT_NESTED_MATRIX_ACK=launch_nested_true_cell_count_matrix

Boundary:
  - requires nested materializer/schema/dry-load/design gates to pass
  - launches leakage-safe train-only/internal true cell-count smokes only
  - no canonical multi or Track C query
  - promotion still requires nested decision summary, controls, and frozen no-harm only after route freeze
EOF
  exit 4
fi

PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

"${PYTHON}" - "${MATERIALIZER_JSON}" "${SCHEMA_JSON}" "${DRYLOAD_JSON}" "${DESIGN_JSON}" <<'PY'
import json, sys
from pathlib import Path

mat, schema, dry, design = [Path(x) for x in sys.argv[1:]]
payloads = [json.loads(p.read_text()) for p in [mat, schema, dry, design]]
expected = [
    ("materialized", True, payloads[0].get("materialized")),
    ("schema_status", "capped_h5_schema_gate_pass_no_gpu", payloads[1].get("status")),
    ("dryload_status", "true_cell_count_dryload_pass_no_gpu", payloads[2].get("status")),
    ("design_status", "true_cell_count_design_controls_pass_preliminary_only_no_gpu", payloads[3].get("status")),
]
bad = [f"{name}: expected {want!r}, got {got!r}" for name, want, got in expected if got != want]
if bad:
    raise SystemExit("; ".join(bad))
PY

mkdir -p "${RUN_ROOT}/logs" "${OUT_ROOT}" "${LOG_ROOT}"

run_ids=(
  gene_only_fixed256_budget64_128_256_budget64_seed42
  gene_only_fixed256_budget64_128_256_budget64_seed43
  gene_only_fixed256_budget64_128_256_budget64_seed44
  gene_only_fixed256_budget64_128_256_budget128_seed42
  gene_only_fixed256_budget64_128_256_budget128_seed43
  gene_only_fixed256_budget64_128_256_budget128_seed44
  gene_only_fixed256_budget64_128_256_budget256_seed42
  gene_only_fixed256_budget64_128_256_budget256_seed43
  gene_only_fixed256_budget64_128_256_budget256_seed44
)

for run_id in "${run_ids[@]}"; do
  run_name="xverse_truecell_nested_${run_id}_${TOTAL_STEPS}"
  echo "[$(date '+%F %T %Z')] launching ${run_name}"
  LATENTFM_TRUE_CELL_COUNT_SMOKE_ACK=bounded_capped_data_smoke \
  LATENTFM_TRUE_CELL_COUNT_RUN_ID="${run_id}" \
  LATENTFM_TRUE_CELL_COUNT_RUN_NAME="${run_name}" \
  LATENTFM_TRUE_CELL_COUNT_MATERIALIZER_JSON="${MATERIALIZER_JSON}" \
  LATENTFM_TRUE_CELL_COUNT_SCHEMA_JSON="${SCHEMA_JSON}" \
  LATENTFM_TRUE_CELL_COUNT_DRYLOAD_JSON="${DRYLOAD_JSON}" \
  LATENTFM_TRUE_CELL_COUNT_DESIGN_JSON="${DESIGN_JSON}" \
  LATENTFM_TRUE_CELL_COUNT_RUN_ROOT="${RUN_ROOT}" \
  LATENTFM_TRUE_CELL_COUNT_OUT_ROOT="${OUT_ROOT}" \
  LATENTFM_TRUE_CELL_COUNT_LOG_ROOT="${LOG_ROOT}" \
  LATENTFM_TRUE_CELL_COUNT_TOTAL_STEPS="${TOTAL_STEPS}" \
  bash "${LAUNCHER}"
done

echo "[$(date '+%F %T %Z')] nested true-cell-count matrix launch loop complete"
