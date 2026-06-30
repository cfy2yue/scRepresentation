#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

NESTED_MATERIALIZER_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_capped_h5_materializer_gate_20260624.json
NESTED_SAMPLE_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_sample_provenance_gate_20260624.json
NESTED_SAMPLE_MD=${ROOT}/reports/LATENTFM_TRUE_CELL_COUNT_NESTED_SAMPLE_PROVENANCE_GATE_20260624.md
NESTED_SCHEMA_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_capped_h5_schema_gate_20260624.json
NESTED_SCHEMA_MD=${ROOT}/reports/LATENTFM_TRUE_CELL_COUNT_NESTED_CAPPED_H5_SCHEMA_GATE_20260624.md
NESTED_DRYLOAD_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_dryload_gate_20260624.json
NESTED_DRYLOAD_MD=${ROOT}/reports/LATENTFM_TRUE_CELL_COUNT_NESTED_DRYLOAD_GATE_20260624.md
NESTED_DESIGN_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_design_controls_gate_20260624.json
NESTED_DESIGN_MD=${ROOT}/reports/LATENTFM_TRUE_CELL_COUNT_NESTED_DESIGN_CONTROLS_GATE_20260624.md

echo "[$(date '+%F %T %Z')] nested true cell-count post-materialization gates"

echo "[nested gate] sample provenance"
"${PYTHON}" - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "${ROOT}/ops")
import backfill_latentfm_true_cell_count_sample_provenance_20260624 as m
import materialize_latentfm_true_cell_count_nested_capped_h5_20260624 as nested
m.MATERIALIZER_JSON = Path("${NESTED_MATERIALIZER_JSON}")
m.OUT_JSON = Path("${NESTED_SAMPLE_JSON}")
m.OUT_MD = Path("${NESTED_SAMPLE_MD}")
m.sample_indices = nested.nested_sample_indices
sys.argv = ["nested_sample_provenance", "--write"]
raise SystemExit(m.main())
PY

echo "[nested gate] capped-H5 schema/provenance"
"${PYTHON}" - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "${ROOT}/ops")
import audit_latentfm_true_cell_count_capped_h5_schema_gate_20260624 as m
m.MATERIALIZER_JSON = Path("${NESTED_MATERIALIZER_JSON}")
m.OUT_JSON = Path("${NESTED_SCHEMA_JSON}")
m.OUT_MD = Path("${NESTED_SCHEMA_MD}")
sys.argv = ["nested_schema_gate"]
raise SystemExit(m.main())
PY

echo "[nested gate] dry-load"
"${PYTHON}" - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "${ROOT}/ops")
import audit_latentfm_true_cell_count_dryload_gate_20260624 as m
m.MATERIALIZER_JSON = Path("${NESTED_MATERIALIZER_JSON}")
m.SCHEMA_JSON = Path("${NESTED_SCHEMA_JSON}")
m.OUT_JSON = Path("${NESTED_DRYLOAD_JSON}")
m.OUT_MD = Path("${NESTED_DRYLOAD_MD}")
sys.argv = ["nested_dryload_gate"]
raise SystemExit(m.main())
PY

echo "[nested gate] design controls"
"${PYTHON}" - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "${ROOT}/ops")
import audit_latentfm_true_cell_count_design_controls_20260624 as m
m.MATERIALIZER_JSON = Path("${NESTED_MATERIALIZER_JSON}")
m.SCHEMA_JSON = Path("${NESTED_SCHEMA_JSON}")
m.OUT_JSON = Path("${NESTED_DESIGN_JSON}")
m.OUT_MD = Path("${NESTED_DESIGN_MD}")
sys.argv = ["nested_design_gate"]
raise SystemExit(m.main())
PY

echo "[$(date '+%F %T %Z')] nested true cell-count post-materialization gates complete"
