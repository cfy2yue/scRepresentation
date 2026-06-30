#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
LAUNCHER=${ROOT}/ops/launch_latentfm_true_cell_count_single_smoke_20260624.sh

MATERIALIZER_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_capped_h5_materializer_gate_20260624.json
SCHEMA_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_capped_h5_schema_gate_20260624.json
DRYLOAD_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_dryload_gate_20260624.json
DESIGN_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_design_controls_gate_20260624.json

RUN_ROOT=${ROOT}/runs/latentfm_true_cell_count_budget128_tail_stability_6k_20260625
OUT_ROOT=${ROOT}/CoupledFM/output/latentfm_runs/true_cell_count_budget128_tail_stability_6k_20260625
LOG_ROOT=${ROOT}/logs/latentfm_true_cell_count_budget128_tail_stability_6k_20260625

if [[ "${LATENTFM_TRUE_CELL_COUNT_BUDGET128_TAIL_6K_ACK:-}" != "launch_budget128_tail_stability_6k" ]]; then
  cat >&2 <<'EOF'
Refusing to launch budget128 tail-stability 6k smokes.

Set:
  LATENTFM_TRUE_CELL_COUNT_BUDGET128_TAIL_6K_ACK=launch_budget128_tail_stability_6k

Boundary:
  - reuses nested-v2 budget128 capped-H5 artifacts that passed materializer/schema/dry-load/design gates
  - train-only/internal posthoc only
  - no canonical multi, Track C query, or checkpoint promotion
  - hypothesis: best 3k budget128 mean may be limited by short training / seed-noisy Schmidt tail
  - stop rule: if 6k keeps cross-background negative dataset tails or non-positive CI, close strict cell-count law claim
EOF
  exit 4
fi

mkdir -p "${RUN_ROOT}/logs" "${OUT_ROOT}" "${LOG_ROOT}"

run_ids=(
  gene_only_fixed256_budget64_128_256_budget128_seed42
  gene_only_fixed256_budget64_128_256_budget128_seed43
  gene_only_fixed256_budget64_128_256_budget128_seed44
)

for run_id in "${run_ids[@]}"; do
  seed="${run_id##*_seed}"
  run_name="xverse_truecell_nested_budget128_tailstable_seed${seed}_6000"
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
  LATENTFM_TRUE_CELL_COUNT_TOTAL_STEPS=6000 \
  LATENTFM_TRUE_CELL_COUNT_HYPOTHESIS="Budget128 was the peak 3k nested true-cell-count budget and all seeds passed individual internal gates, but the matrix failed strict law criteria due to a small Schmidt negative dataset tail. This 6k bounded smoke tests whether that tail is short-training/seed-noise rather than an intrinsic tail-safety failure. This is not a promotion claim." \
  LATENTFM_TRUE_CELL_COUNT_STOP_RULE="Summarize train-only/internal outputs only. If cross-background negative dataset tails persist, CI is non-positive, or seed stability weakens, close strict cell-count scaling-law promotion and keep only mechanism insight; canonical multi and Track C query remain unused." \
  bash "${LAUNCHER}"
done

echo "[$(date '+%F %T %Z')] budget128 tail-stability launch loop complete"
