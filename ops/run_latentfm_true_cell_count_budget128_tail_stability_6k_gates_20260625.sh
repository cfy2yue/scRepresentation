#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_ROOT=${ROOT}/runs/latentfm_true_cell_count_budget128_tail_stability_6k_20260625

if [[ "${LATENTFM_TRUE_CELL_COUNT_BUDGET128_TAIL_6K_GATE_ACK:-}" != "summarize_budget128_tail_stability_6k" ]]; then
  cat >&2 <<'EOF'
Refusing to summarize budget128 tail-stability 6k runs.

Set:
  LATENTFM_TRUE_CELL_COUNT_BUDGET128_TAIL_6K_GATE_ACK=summarize_budget128_tail_stability_6k

Boundary:
  - reads train-only/internal posthoc JSON from the 6k budget128 tail-stability runs only
  - does not read canonical multi or Track C query
  - no training, inference, or GPU use
EOF
  exit 4
fi

for run_dir in "${RUN_ROOT}"/xverse_truecell_nested_budget128_tailstable_*_6000; do
  [[ -d "${run_dir}" ]] || continue
  if [[ "$(cat "${run_dir}/EXIT_CODE" 2>/dev/null || echo missing)" != "0" ]]; then
    echo "Run is not train-complete: ${run_dir}" >&2
    exit 3
  fi
  if [[ "$(cat "${run_dir}/POSTHOC_EXIT_CODE" 2>/dev/null || echo missing)" != "0" ]]; then
    echo "Run is not posthoc-complete: ${run_dir}" >&2
    exit 3
  fi
done

LATENTFM_TRUE_CELL_COUNT_NESTED_RUN_ROOT="${RUN_ROOT}" \
LATENTFM_TRUE_CELL_COUNT_NESTED_EXPECTED_RUNS=3 \
LATENTFM_TRUE_CELL_COUNT_NESTED_OUT_JSON=${ROOT}/reports/latentfm_true_cell_count_budget128_tail_stability_6k_decision_20260625.json \
LATENTFM_TRUE_CELL_COUNT_NESTED_OUT_MD=${ROOT}/reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_TAIL_STABILITY_6K_DECISION_20260625.md \
"${PYTHON}" "${ROOT}/ops/summarize_latentfm_true_cell_count_nested_matrix_20260624.py"

LATENTFM_TRUE_CELL_COUNT_NESTED_RUN_ROOT="${RUN_ROOT}" \
LATENTFM_TRUE_CELL_COUNT_NESTED_CONTROLS_EXPECTED_RUNS=3 \
LATENTFM_TRUE_CELL_COUNT_NESTED_CONTROLS_OUT_JSON=${ROOT}/reports/latentfm_true_cell_count_budget128_tail_stability_6k_controls_20260625.json \
LATENTFM_TRUE_CELL_COUNT_NESTED_CONTROLS_OUT_MD=${ROOT}/reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_TAIL_STABILITY_6K_CONTROLS_20260625.md \
"${PYTHON}" "${ROOT}/ops/audit_latentfm_true_cell_count_nested_controls_20260624.py"
