#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PY=/data/cyx/software/miniconda3/envs/scdfm/bin/python
RUN_FAMILY=zscape_prospective_partial_crossfit_specificity_20260630
RUN_NAME=zscape_prospective_partial_crossfit_specificity_20260630_1148
RUN_DIR="$ROOT/runs/$RUN_FAMILY/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
OUT_DIR="$ROOT/reports/zscape_prospective_partial_crossfit_specificity_20260630"
SESSION="$RUN_NAME"

COUNTS="$ROOT/runs/zscape_prospective_expansion_extract_atlas_20260630/zscape_prospective_expansion_extract_atlas_20260630_0435/outputs/zscape_manifest_selected_counts_csc.npz"
CELL_INDEX="$ROOT/runs/zscape_prospective_expansion_extract_atlas_20260630/zscape_prospective_expansion_extract_atlas_20260630_0435/outputs/zscape_manifest_selected_expression_cell_index.csv"
MATCHED_MANIFEST="$ROOT/runs/zscape_prospective_expansion_extract_atlas_20260630/zscape_prospective_expansion_extract_atlas_20260630_0435/outputs/zscape_expression_selected_cell_ids_matched.csv"
GENE_NAMES="$ROOT/runs/zscape_prospective_expansion_extract_atlas_20260630/zscape_prospective_expansion_extract_atlas_20260630_0435/outputs/zscape_manifest_selected_gene_names.txt"
GENE_METADATA="$ROOT/dataset/external/zscape_20260628/GSE202639_zperturb_full_gene_metadata.csv.gz"
ROW_IDS="mesodermal_progenitor_cells_contains_psm__tbx16_tbx16l__18p0h,mesodermal_progenitor_cells_contains_psm__tbx16_msgn1__18p0h,notochord__tfap2a__72p0h"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
if [ -e "$RUN_DIR" ] || [ -e "$OUT_DIR" ]; then
  echo "Refusing to overwrite existing run or output directory." >&2
  echo "$RUN_DIR" >&2
  echo "$OUT_DIR" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$OUT_DIR"

COMMAND="cd $ROOT && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 $PY $ROOT/ops/audit_zscape_crossfit_residual_specificity_repair_gate_20260628.py --counts-npz $COUNTS --cell-index $CELL_INDEX --matched-manifest $MATCHED_MANIFEST --gene-names $GENE_NAMES --gene-metadata $GENE_METADATA --primary-row-ids $ROW_IDS --max-splits 70 --bootstrap-repeats 120 --random-repeats 40 --module-size 24 --heldout-embryos 4 --positive-fraction 0.75 --specificity-margin 0.02 --out-dir $OUT_DIR"

cat > "$RUN_DIR/RUN_STATUS.md" <<EOF
# Run Status: $RUN_NAME

## Command

\`\`\`bash
$COMMAND
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux session: \`$SESSION\`

## Log path

\`$LOG_DIR/run.log\`

## Expected outputs

* \`$OUT_DIR/LATENTFM_ZSCAPE_CROSSFIT_RESIDUAL_SPECIFICITY_REPAIR_GATE_20260628.md\`
* \`$OUT_DIR/zscape_crossfit_specificity_gate_20260628.json\`
* \`$OUT_DIR/zscape_crossfit_specificity_query_rows.csv\`
* \`$OUT_DIR/zscape_crossfit_specificity_row_summary.csv\`

## Hypothesis

The 3 prospective strict-control pass rows may represent focused biological
specificity signals only if crossfit residual modules rediscovered without
heldout perturb embryos remain positive against wrong-target/time/lineage and
matched-random controls.

## Resource plan

CPU-only, 8 thread caps. No GPU, no training, no inference, no checkpoint
selection, no canonical multi selection, and no Track C query access.

## Gate / promotion signal

Focused biological specificity support only if the CPU report shows stable
heldout module specificity for the partial pass rows. This still does not
authorize LatentFM/RawFM loss, sampling, model positives, constraints, or GPU
training; it can only justify another CPU train-set translation/no-harm gate.

## Fail-close rule

If the focused specificity panel fails, close prospective second-lineage rescue
and retain these rows as ZSCAPE descriptor/failure-analysis evidence only.

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 $LOG_DIR/run.log
cat $RUN_DIR/EXIT_CODE 2>/dev/null || echo "still running"
free -h
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Launched only after prospective strict-control decision
\`zscape_prospective_strict_control_partial_signal_fail_design_gate_no_gpu\`.
The predeclared broad design gate failed because only 3 rows passed; this
focused panel is failure-analysis/design-review only.
EOF

echo "$SESSION" > "$RUN_DIR/SESSION_NAME"
date '+%F %T %Z' > "$RUN_DIR/STARTED"

tmux new -d -s "$SESSION" "bash -lc '$COMMAND > $LOG_DIR/run.log 2>&1; status=\$?; echo \$status > $RUN_DIR/EXIT_CODE; date +\"%F %T %Z\" > $RUN_DIR/FINISHED; if [ \$status -eq 0 ]; then printf \"\\n## Final status\\n\\nFinished.\\n\" >> $RUN_DIR/RUN_STATUS.md; else printf \"\\n## Final status\\n\\nFailed with exit code %s.\\n\" \"\$status\" >> $RUN_DIR/RUN_STATUS.md; fi'"

echo "$RUN_DIR"
