#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
LAUNCHER=${ROOT}/ops/launch_latentfm_true_cell_count_single_smoke_20260624.sh

MATERIALIZER_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_capped_h5_materializer_gate_20260624.json
SCHEMA_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_capped_h5_schema_gate_20260624.json
DRYLOAD_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_dryload_gate_20260624.json
DESIGN_JSON=${ROOT}/reports/latentfm_true_cell_count_nested_design_controls_gate_20260624.json

RUN_ROOT=${ROOT}/runs/latentfm_true_cell_count_budget128_anchor_replay005_6k_20260625
OUT_ROOT=${ROOT}/CoupledFM/output/latentfm_runs/true_cell_count_budget128_anchor_replay005_6k_20260625
LOG_ROOT=${ROOT}/logs/latentfm_true_cell_count_budget128_anchor_replay005_6k_20260625

if [[ "${LATENTFM_TRUE_CELL_COUNT_BUDGET128_AR005_6K_ACK:-}" != "launch_budget128_anchor_replay005_6k" ]]; then
  cat >&2 <<'EOF'
Refusing to launch budget128 anchor-replay repair smokes.

Set:
  LATENTFM_TRUE_CELL_COUNT_BUDGET128_AR005_6K_ACK=launch_budget128_anchor_replay005_6k

Boundary:
  - reuses nested-v2 budget128 capped-H5 artifacts that passed materializer/schema/dry-load/design gates
  - train-only/internal posthoc only
  - no canonical multi, Track C query, or checkpoint promotion
  - hypothesis: light train-only anchor replay (0.05) reduces canonical Pearson drift while retaining budget128 cell-cap internal gain
  - stop rule: if internal cross/family gain or tail safety weakens, close this repair before canonical no-harm
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
  run_name="xverse_truecell_nested_budget128_ar005_seed${seed}_6000"
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
  LATENTFM_TRUE_CELL_COUNT_ANCHOR_REPLAY_LOSS_WEIGHT=0.05 \
  LATENTFM_TRUE_CELL_COUNT_HYPOTHESIS="Budget128 6k cell-cap route passed train-only/internal tail gates but failed frozen canonical Pearson no-harm while improving MMD. This repair adds light train-only anchor replay (0.05) to reduce direction drift without using canonical metrics for training or selection." \
  LATENTFM_TRUE_CELL_COUNT_STOP_RULE="Summarize train-only/internal outputs only first. If cross/family gains weaken below gate, MMD worsens, or negative dataset tails return, close this repair before canonical no-harm; canonical multi and Track C query remain unused." \
  bash "${LAUNCHER}"
done

echo "[$(date '+%F %T %Z')] budget128 anchor-replay repair launch loop complete"
