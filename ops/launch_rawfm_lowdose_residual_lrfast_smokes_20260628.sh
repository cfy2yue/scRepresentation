#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
PY="/data/cyx/software/miniconda3/envs/scdfm/bin/python"
SPLIT="$ROOT/dataset/biFlow_data/split_seed42.json"
BIFLOW="$ROOT/dataset/biFlow_data"
MANIFEST_DIR="$ROOT/reports/rawfm_lowdose_residual_manifest_20260628"
OUT_ROOT="$ROOT/CoupledFM/output/rawfm_lowdose_residual_lrfast_smoke_20260628"
RUN_ROOT="$ROOT/runs"
REPORT_DIR="$ROOT/reports/rawfm_lowdose_residual_lrfast_smoke_comparison_20260628"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
GPU_A="${GPU_A:?set GPU_A to the first audited GPU id}"
GPU_B="${GPU_B:?set GPU_B to the second audited GPU id}"
MANIFEST="$REPORT_DIR/launch_manifest_${STAMP}.tsv"

mkdir -p "$REPORT_DIR"
printf "label\trole\trun_dir\toutput\n" > "$MANIFEST"

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
  --warmup-steps 1
  --min-lr-ratio 0.5
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
  --no-mmd
)

launch_one() {
  local label="$1"
  local role="$2"
  local gpu="$3"
  local run_name="rawfm_wessels_lowdose_lrfast_${label}_smoke_${STAMP}"
  local run_dir="$RUN_ROOT/$run_name"
  local manifest="$MANIFEST_DIR/${label}_k256_seed42.json"
  local output_dir="$OUT_ROOT/wessels_lrfast_${label}_seed42_${STAMP}"
  local log="$run_dir/logs/run.log"
  local cmd="cd $ROOT/CoupledFM && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 $PY model/tools/launch_stack_train.py ${COMMON_ARGS[*]} --output-dir $output_dir --gene-budget-manifest $manifest --gene-budget-label lrfast_${label}_seed42"

  if [ ! -f "$manifest" ]; then
    echo "Missing manifest: $manifest" >&2
    exit 1
  fi
  if [ -e "$output_dir" ]; then
    echo "Refusing to overwrite existing output_dir: $output_dir" >&2
    exit 1
  fi
  mkdir -p "$run_dir/logs"
  date '+%F %T %z' > "$run_dir/STARTED"
  echo "$run_name" > "$run_dir/SESSION_NAME"
  printf "%s\t%s\t%s\t%s\n" "$label" "$role" "$run_dir" "$output_dir/ot" >> "$MANIFEST"
  cat > "$run_dir/RUN_STATUS.md" <<STATUS
# Run Status: $run_name

## Hypothesis

The residual-response gene signal appears real but full residual/hybrid budgets caused MMD harm. This low-dose add-back smoke tests whether $label preserves candidate-control corr_pert signal while sharing abundance/random ballast and using a short-warmup schedule that produces actual optimizer movement in a 20-step smoke.

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

* GPU: physical GPU $gpu from immediate pre-launch audit.
* CPU: 4 BLAS/OpenMP threads.
* Portfolio: two physical GPUs total, two training jobs per GPU, 16 project CPU threads.

## Gate

Promotion signal:
* exit code 0;
* no \`best.pt\`;
* candidate beats its matched control by \`corr_pert >= +0.01\`;
* candidate MMD is not worse than matched control by more than \`0.005\` and is \`<= 0.030\`;
* candidate direct Pearson is not worse than control by more than \`0.02\`.

Failure close rule:
* close the tested residual dose if MMD harm remains, corr_pert advantage is below \`+0.01\`, direct Pearson collapses, or the same-ballast control wins.

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

launch_one "residual32_abundance96_random128" "candidate" "$GPU_A"
launch_one "confound32_abundance96_random128_control" "control" "$GPU_A"
launch_one "residual64_abundance96_random96" "candidate" "$GPU_B"
launch_one "confound64_abundance96_random96_control" "control" "$GPU_B"

echo "Wrote launch manifest: $MANIFEST"
