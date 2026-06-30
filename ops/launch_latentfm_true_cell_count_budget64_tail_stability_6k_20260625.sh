#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
LAUNCHER=${ROOT}/ops/launch_latentfm_true_cell_count_single_smoke_20260624.sh

MATERIALIZER_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_capped_h5_materializer_gate_20260624.json
SCHEMA_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_capped_h5_schema_gate_20260624.json
DRYLOAD_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_dryload_gate_20260624.json
DESIGN_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_design_controls_gate_20260624.json

RUN_ROOT=${ROOT}/runs/latentfm_true_cell_count_budget64_tail_stability_6k_20260625
OUT_ROOT=${ROOT}/CoupledFM/output/latentfm_runs/true_cell_count_budget64_tail_stability_6k_20260625
LOG_ROOT=${ROOT}/logs/latentfm_true_cell_count_budget64_tail_stability_6k_20260625

if [[ "${LATENTFM_TRUE_CELL_COUNT_BUDGET64_TAIL_6K_ACK:-}" != "launch_budget64_tail_stability_6k" ]]; then
  cat >&2 <<'EOF'
Refusing to launch budget64 tail-stability 6k smokes.

Set:
  LATENTFM_TRUE_CELL_COUNT_BUDGET64_TAIL_6K_ACK=launch_budget64_tail_stability_6k

Boundary:
  - reuses nested-v2 budget64 capped-H5 artifacts that passed materializer/schema/dry-load/design gates
  - train-only/internal posthoc only
  - no canonical multi, Track C query, or checkpoint promotion
  - hypothesis: after budget128 6k tail-stability passed, budget64 6k tests whether the cell-count curve is a genuine exposure/cell-cap effect rather than a 3k short-training artifact
  - stop rule: if budget64 6k has negative dataset tails, non-positive CI, or matches/exceeds budget128 only through unsafe tails, keep the scaling claim mechanism-only and do not run canonical no-harm from this branch
EOF
  exit 4
fi

mkdir -p "${RUN_ROOT}/logs" "${OUT_ROOT}" "${LOG_ROOT}"

run_ids=(
  gene_only_fixed256_budget64_128_256_budget64_seed42
  gene_only_fixed256_budget64_128_256_budget64_seed43
  gene_only_fixed256_budget64_128_256_budget64_seed44
)

for run_id in "${run_ids[@]}"; do
  seed="${run_id##*_seed}"
  run_name="xverse_truecell_nested_budget64_tailstable_seed${seed}_6000"
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
  LATENTFM_TRUE_CELL_COUNT_HYPOTHESIS="Budget128 6k is the current strongest true-cell-count/cell-cap mechanism candidate. This budget64 6k bounded smoke tests the low-budget side of the same nested fixed-condition curve at the same training length, separating true cell-count exposure effects from 3k short-training artifacts. This is not a promotion claim." \
  LATENTFM_TRUE_CELL_COUNT_STOP_RULE="Summarize train-only/internal outputs only. If budget64 6k has cross-background negative dataset tails, non-positive CI, or only wins through unsafe dataset dominance, keep the branch mechanism-only; canonical multi and Track C query remain unused, and no canonical no-harm is authorized from budget64." \
  bash "${LAUNCHER}"
done

echo "[$(date '+%F %T %Z')] budget64 tail-stability launch loop complete"
