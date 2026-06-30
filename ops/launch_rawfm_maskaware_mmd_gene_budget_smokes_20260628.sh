#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
PY="/data/cyx/software/miniconda3/envs/scdfm/bin/python"
SPLIT="$ROOT/dataset/biFlow_data/split_seed42.json"
BIFLOW="$ROOT/dataset/biFlow_data"
OUT_ROOT="$ROOT/CoupledFM/output/rawfm_maskaware_mmd_gene_budget_smoke_20260628"
RUN_ROOT="$ROOT/runs"

COMMON_ARGS=(
  --biflow-dir "$BIFLOW"
  --latent-backbone stack
  --split-file "$SPLIT"
  --datasets Wessels
  --mode ot
  --ot-feature raw
  --epochs 2
  --batch-size 4
  --micro-batch 2
  --grad-accum-steps 1
  --lr 5e-5
  --val-every-steps 0
  --max-train-steps-per-epoch 10
  --selection-protocol fixed_steps_no_selection
  --fixed-step-no-selection
  --test-every-epoch 0
  --early-stop-patience 0
  --eval-ode-steps 5
  --val-ode-steps 5
  --gene-mask-prob 0
  --gene-mask-all-prob 0
  --use-mmd
  --mmd-gamma-max 0.001
  --mmd-every 1
  --mmd-epoch-start 0
  --mmd-warmup-start-frac 0.0
  --mmd-warmup-end-frac 0.5
  --mmd-micro-chunk 2
)

launch_one() {
  local run_label="$1"
  local output_slug="$2"
  local manifest="$3"
  local gpu="$4"
  local role="$5"
  local run_name="rawfm_wessels_mmd_${run_label}_k256_smoke_20260628_2127"
  local run_dir="$RUN_ROOT/$run_name"
  local output_dir="$OUT_ROOT/wessels_${output_slug}_k256_seed42"
  local log="$run_dir/logs/run.log"
  local cmd="cd $ROOT/CoupledFM && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 $PY model/tools/launch_stack_train.py ${COMMON_ARGS[*]} --output-dir $output_dir --gene-budget-manifest $manifest --gene-budget-label ${output_slug}_k256_seed42"

  if [ -e "$output_dir" ]; then
    echo "Refusing to overwrite existing output_dir: $output_dir" >&2
    exit 1
  fi
  mkdir -p "$run_dir/logs"
  date '+%F %T %z' > "$run_dir/STARTED"
  echo "$run_name" > "$run_dir/SESSION_NAME"
  cat > "$run_dir/RUN_STATUS.md" <<STATUS
# Run Status: $run_name

## Hypothesis

Mask-aware budgeted MMD may reduce the endpoint-distribution harm seen in residual or hybrid RawFM gene budgets while preserving the observed corr_pert signal. This run is the $role member of the packet.

## Command

\`\`\`bash
$cmd
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux session: \`$run_name\`

## Log path

\`$log\`

## Expected outputs

* \`$output_dir/ot/last.pt\`
* \`$output_dir/ot/train_log.jsonl\`
* \`$output_dir/ot/split_provenance.json\`
* \`$output_dir/ot/run_meta.json\`

## Resource Plan

* GPU: physical GPU $gpu from three-sample audit at 2026-06-28 21:27 CST.
* CPU: 4 BLAS/OpenMP threads.
* RAM: Wessels k=256 budget smokes stayed low-memory in prior runs.
* Portfolio: two physical GPUs total, two training jobs per GPU, 16 project CPU threads.
* Strict empty GPU left unused: GPU7.

## Gate

Promotion signal:
* exit code 0;
* no \`best.pt\` under fixed-step/no-selection;
* train log records \`train_mmd_visible_genes=256\` when MMD runs;
* candidate beats its matched control on corr_pert by at least \`+0.01\`;
* candidate MMD is no worse than matched control by more than \`0.005\`;
* MMD improves versus the corresponding no-MMD candidate.

Failure close rule:
* close this exact MMD stabilizer if corr_pert advantage collapses below \`+0.005\`, MMD remains materially worse than matched control, \`best.pt\` is written, or train MMD does not use the masked 256-gene subspace.

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 $log
cat $run_dir/EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Bounded fixed-step/no-selection smoke. Canonical split is explicit read-only \`split_seed42.json\`; canonical multi and Track C query are not used for training or checkpoint selection. Random gene masking is explicitly disabled so MMD projection corresponds to the deterministic gene-budget keep set.
STATUS

  tmux new -d -s "$run_name" "bash -lc '$cmd > $log 2>&1; status=\$?; echo \$status > $run_dir/EXIT_CODE; date '+%F %T %z' > $run_dir/FINISHED'"
}

STRUCT="$ROOT/reports/rawfm_structural_gene_budget_manifest_20260628"
HYBRID="$ROOT/reports/rawfm_hybrid_gene_budget_manifest_20260628"

launch_one "residual_topk" "response_abundance_residual_topk_mmd" \
  "$STRUCT/response_abundance_residual_topk_k256_seed42.json" 2 \
  "full residualized-response candidate"
launch_one "residual_confound_control" "residual_confound_matched_random_mmd" \
  "$STRUCT/residual_confound_matched_random_k256_seed42.json" 2 \
  "matched residual-confound control"
launch_one "hybrid_residual128_abundance128" "residual128_abundance128_hybrid_mmd" \
  "$HYBRID/residual128_abundance128_hybrid_k256_seed42.json" 3 \
  "hybrid residual+abundance candidate"
launch_one "hybrid_confound128_abundance128_control" "confound128_abundance128_hybrid_control_mmd" \
  "$HYBRID/confound128_abundance128_hybrid_control_k256_seed42.json" 3 \
  "matched hybrid control"
