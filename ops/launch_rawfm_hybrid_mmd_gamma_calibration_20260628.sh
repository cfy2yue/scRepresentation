#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
PY="/data/cyx/software/miniconda3/envs/scdfm/bin/python"
SPLIT="$ROOT/dataset/biFlow_data/split_seed42.json"
BIFLOW="$ROOT/dataset/biFlow_data"
MANIFEST_DIR="$ROOT/reports/rawfm_hybrid_gene_budget_manifest_20260628"
OUT_ROOT="$ROOT/CoupledFM/output/rawfm_hybrid_mmd_gamma_calibration_20260628"
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
  --mmd-every 1
  --mmd-epoch-start 0
  --mmd-warmup-start-frac 0.0
  --mmd-warmup-end-frac 0.25
  --mmd-micro-chunk 2
)

launch_one() {
  local label="$1"
  local role="$2"
  local gamma_label="$3"
  local gamma_value="$4"
  local gpu="$5"
  local run_name="rawfm_wessels_hybrid_mmd_${gamma_label}_${label}_k256_smoke_20260628_2144"
  local run_dir="$RUN_ROOT/$run_name"
  local manifest="$MANIFEST_DIR/${label}_k256_seed42.json"
  local output_dir="$OUT_ROOT/wessels_${gamma_label}_${label}_k256_seed42"
  local log="$run_dir/logs/run.log"
  local cmd="cd $ROOT/CoupledFM && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 $PY model/tools/launch_stack_train.py ${COMMON_ARGS[*]} --mmd-gamma-max $gamma_value --output-dir $output_dir --gene-budget-manifest $manifest --gene-budget-label ${gamma_label}_${label}_k256_seed42"

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

The prior mask-aware MMD packet with gamma \`0.001\` verified the 256-gene masked MMD path but was too weak to move final metrics. This calibration tests whether a stronger MMD gamma can reduce hybrid candidate MMD without collapsing the residual corr_pert signal. This run is the $role for gamma \`$gamma_value\`.

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
* \`$output_dir/ot/run_meta.json\`
* \`$output_dir/ot/split_provenance.json\`

## Resource Plan

* GPU: physical GPU $gpu from three-sample audit at 2026-06-28 21:44 CST.
* CPU: 4 BLAS/OpenMP threads.
* Portfolio: two physical GPUs total, two training jobs per GPU, 16 project CPU threads.
* Random gene masking disabled so MMD projection should stay at 256 visible genes.

## Gate

Promotion signal:
* exit code 0;
* no \`best.pt\`;
* \`mmd_genes=256\` appears in the log;
* at the same gamma, candidate beats matched control on corr_pert by at least \`+0.01\`;
* candidate MMD is not worse than matched control by more than \`0.005\`;
* candidate MMD improves over no-MMD hybrid by at least \`0.01\`.

Failure close rule:
* close this gamma if MMD remains unchanged, MMD still trails matched control materially, corr_pert advantage collapses below \`+0.005\`, or training becomes unstable.

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

Bounded fixed-step/no-selection calibration. Canonical split is explicit read-only \`split_seed42.json\`; canonical multi and Track C query are not used for training or checkpoint selection.
STATUS

  tmux new -d -s "$run_name" "bash -lc '$cmd > $log 2>&1; status=\$?; echo \$status > $run_dir/EXIT_CODE; date '+%F %T %z' > $run_dir/FINISHED'"
}

launch_one "residual128_abundance128_hybrid" "hybrid candidate" "g002" "0.02" 2
launch_one "confound128_abundance128_hybrid_control" "hybrid matched control" "g002" "0.02" 2
launch_one "residual128_abundance128_hybrid" "hybrid candidate" "g010" "0.10" 3
launch_one "confound128_abundance128_hybrid_control" "hybrid matched control" "g010" "0.10" 3
