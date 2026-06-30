#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

MATERIALIZER_JSON=${ROOT}/reports/latentfm_true_cell_count_allmodality_doseaware_materializer_gate_20260625.json

"${PYTHON}" - "${MATERIALIZER_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "allmodality_doseaware_materialized_no_gpu":
    raise SystemExit(f"materializer not complete/pass: {payload.get('status')}")
if not payload.get("materialized_rows"):
    raise SystemExit("materializer has no materialized_rows")
PY

"${PYTHON}" "${ROOT}/ops/backfill_latentfm_true_cell_count_allmodality_doseaware_condition_metadata_20260625.py" --write
"${PYTHON}" "${ROOT}/ops/audit_latentfm_true_cell_count_allmodality_doseaware_schema_gate_20260625.py"
"${PYTHON}" "${ROOT}/ops/audit_latentfm_true_cell_count_allmodality_doseaware_dryload_gate_20260625.py" --max-batches "${LATENTFM_ALLMODALITY_DRYLOAD_MAX_BATCHES:-2}"
"${PYTHON}" "${ROOT}/ops/audit_latentfm_true_cell_count_allmodality_doseaware_chemical_conditioning_gate_20260625.py"
"${PYTHON}" "${ROOT}/ops/audit_latentfm_true_cell_count_allmodality_doseaware_design_controls_20260625.py"
"${PYTHON}" "${ROOT}/ops/build_latentfm_true_cell_count_allmodality_doseaware_loader_splits_20260625.py"

"${PYTHON}" - <<'PY'
import json
from pathlib import Path

checks = {
    "metadata": (
        Path("/data/cyx/1030/scLatent/reports/latentfm_true_cell_count_allmodality_doseaware_condition_metadata_backfill_20260625.json"),
        "allmodality_doseaware_condition_metadata_written_no_gpu",
    ),
    "schema": (
        Path("/data/cyx/1030/scLatent/reports/latentfm_true_cell_count_allmodality_doseaware_schema_gate_20260625.json"),
        "allmodality_doseaware_schema_pass_no_gpu",
    ),
    "dryload": (
        Path("/data/cyx/1030/scLatent/reports/latentfm_true_cell_count_allmodality_doseaware_dryload_gate_20260625.json"),
        "allmodality_doseaware_dryload_pass_no_gpu",
    ),
    "chemical_conditioning": (
        Path("/data/cyx/1030/scLatent/reports/latentfm_true_cell_count_allmodality_doseaware_chemical_conditioning_gate_20260625.json"),
        "allmodality_doseaware_chemical_conditioning_pass_no_gpu",
    ),
    "loader_splits": (
        Path("/data/cyx/1030/scLatent/reports/latentfm_true_cell_count_allmodality_doseaware_loader_splits_20260625.json"),
        "allmodality_doseaware_loader_splits_ready_no_gpu",
    ),
}
rows = {}
for name, (path, expected) in checks.items():
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows[name] = payload.get("status")
    if rows[name] != expected:
        raise SystemExit(f"{name} gate failed: {rows[name]} != {expected}")
design = json.loads(Path("/data/cyx/1030/scLatent/reports/latentfm_true_cell_count_allmodality_doseaware_design_controls_20260625.json").read_text(encoding="utf-8"))
if not design.get("smoke_ready_after_schema_dryload"):
    raise SystemExit(f"design gate does not authorize smoke-readiness: {design.get('status')}")
print(json.dumps({"status": "allmodality_doseaware_post_materialization_gates_pass", "gates": rows, "design_status": design.get("status")}, indent=2))
PY
