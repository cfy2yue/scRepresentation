#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
VENV="${ZSCAPE_BIO_VENV:-${ROOT}/.venvs/zscape_bio_20260628}"
SETUP="${ROOT}/ops/setup_zscape_bio_venv_20260628.sh"
SCRIPT="${ROOT}/ops/audit_zscape_gprofiler_enrichment_preflight_20260628.py"
EXPR_DIR="${ZSCAPE_EXPR_DIR:-${ROOT}/reports/zscape_expression_latent_biology_preflight_20260628}"
FLOW_ROWS="${ZSCAPE_FLOW_ROWS:-${ROOT}/reports/zscape_flow_constraint_feasibility_20260628/zscape_flow_constraint_feasibility_rows.csv}"
BACKGROUND="${ZSCAPE_BACKGROUND_GENES:-${ROOT}/runs/zscape_raw_counts_cell_manifest_extraction_20260628/zscape_raw_counts_cell_manifest_extraction_20260628_074523/outputs/zscape_manifest_selected_gene_names.txt}"
RUN_NAME="zscape_formal_gprofiler_enrichment_20260628_$(date +%H%M%S)"
OUT_DIR="${ZSCAPE_FORMAL_ENRICH_OUT:-${ROOT}/reports/zscape_formal_gprofiler_enrichment_20260628/${RUN_NAME}}"

if [[ "${ZSCAPE_FORMAL_ENRICHMENT_ACK:-}" != "frozen_env_qc_log1p_recorded" ]]; then
  echo "Set ZSCAPE_FORMAL_ENRICHMENT_ACK=frozen_env_qc_log1p_recorded after confirming the expression inputs and QC/log1p policy." >&2
  exit 2
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  "${SETUP}"
fi

mkdir -p "${OUT_DIR}"

"${VENV}/bin/python" "${SCRIPT}" \
  --expr-dir "${EXPR_DIR}" \
  --flow-rows "${FLOW_ROWS}" \
  --background "${BACKGROUND}" \
  --out-dir "${OUT_DIR}" \
  --organism drerio \
  --sources GO:BP GO:MF GO:CC REAC WP KEGG \
  --top-n "${ZSCAPE_ENRICH_TOP_N:-50}" \
  --min-query-genes "${ZSCAPE_ENRICH_MIN_QUERY_GENES:-10}" \
  --user-threshold "${ZSCAPE_ENRICH_THRESHOLD:-0.05}" \
  --threshold-method g_SCS \
  --custom-background \
  --primary-only

"${VENV}/bin/python" -m pip freeze | sort > "${OUT_DIR}/pip_freeze.txt"

cat > "${OUT_DIR}/FORMAL_RUN_CONTEXT.md" <<EOF
# ZSCAPE Formal g:Profiler Context

Timestamp: \`$(date '+%F %T %Z')\`

## Inputs

- expression dir: \`${EXPR_DIR}\`
- flow rows: \`${FLOW_ROWS}\`
- background genes: \`${BACKGROUND}\`
- venv: \`${VENV}\`
- pip freeze: \`${OUT_DIR}/pip_freeze.txt\`

## Boundary

- CPU/network-only biological interpretation.
- No LatentFM training, inference, checkpoint selection, canonical multi
  selection, or Track C query use.
- The expression-space input must have already recorded whether QC filtering
  was applied and that log1p was applied exactly once.
EOF

echo "${OUT_DIR}"
