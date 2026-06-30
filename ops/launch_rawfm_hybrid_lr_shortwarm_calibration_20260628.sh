#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
PY="/data/cyx/software/miniconda3/envs/scdfm/bin/python"
SPLIT="$ROOT/dataset/biFlow_data/split_seed42.json"
BIFLOW="$ROOT/dataset/biFlow_data"
MANIFEST_DIR="$ROOT/reports/rawfm_hybrid_gene_budget_manifest_20260628"
OUT_ROOT="$ROOT/CoupledFM/output/rawfm_hybrid_lr_shortwarm_calibration_20260628"
RUN_ROOT="$ROOT/runs"
REPORT_DIR="$ROOT/reports/rawfm_hybrid_lr_shortwarm_calibration_20260628"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
GPU_A="${GPU_A:?set GPU_A to the first audited GPU id}"
GPU_B="${GPU_B:?set GPU_B to the second audited GPU id}"
MANIFEST="$REPORT_DIR/launch_manifest_${STAMP}.tsv"

mkdir -p "$REPORT_DIR"
printf "label\tgamma\trole\trun_dir\toutput\n" > "$MANIFEST"

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
)

launch_one() {
  local label="$1"
  local role="$2"
  local branch="$3"
  local gpu="$4"
  local gamma_value="$5"
  local run_name="rawfm_wessels_hybrid_lrfast_${branch}_${label}_k256_smoke_${STAMP}"
  local run_dir="$RUN_ROOT/$run_name"
  local manifest="$MANIFEST_DIR/${label}_k256_seed42.json"
  local output_dir="$OUT_ROOT/wessels_lrfast_${branch}_${label}_k256_seed42_${STAMP}"
  local log="$run_dir/logs/run.log"
  local mmd_args=()
  local gate_note
  if [ "$gamma_value" = "none" ]; then
    mmd_args=(--no-mmd)
    gate_note="This is the short-warmup no-MMD reference for isolating LR schedule effects."
  else
    mmd_args=(--use-mmd --mmd-every 1 --mmd-epoch-start 0 --mmd-warmup-start-frac 0.0 --mmd-warmup-end-frac 0.25 --mmd-micro-chunk 2 --mmd-gamma-max "$gamma_value")
    gate_note="This is the short-warmup MMD test for gamma ${gamma_value}."
  fi
  local cmd="cd $ROOT/CoupledFM && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 $PY model/tools/launch_stack_train.py ${COMMON_ARGS[*]} ${mmd_args[*]} --output-dir $output_dir --gene-budget-manifest $manifest --gene-budget-label lrfast_${branch}_${label}_k256_seed42"

  if [ -e "$output_dir" ]; then
    echo "Refusing to overwrite existing output_dir: $output_dir" >&2
    exit 1
  fi
  mkdir -p "$run_dir/logs"
  date '+%F %T %z' > "$run_dir/STARTED"
  echo "$run_name" > "$run_dir/SESSION_NAME"
  printf "%s\t%s\t%s\t%s\t%s\n" "lrfast_${branch}_${role}" "$gamma_value" "$role" "$run_dir" "$output_dir/ot" >> "$MANIFEST"
  cat > "$run_dir/RUN_STATUS.md" <<STATUS
# Run Status: $run_name

## Hypothesis

The earlier 20-step RawFM hybrid MMD smokes used the default \`warmup_steps=1000\`, so actual optimizer LR was tiny throughout the run. This smoke tests whether short-warmup training (\`warmup_steps=1\`, \`min_lr_ratio=0.5\`) reveals a real signal from the same hybrid gene-budget branch. $gate_note

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
* Random gene masking disabled so MMD projection should stay at 256 visible genes for MMD runs.

## Gate

Promotion signal for the MMD short-warmup branch:
* exit code 0;
* no \`best.pt\`;
* MMD runs log \`mmd_genes=256\`;
* MMD candidate beats matched MMD control on corr_pert by at least \`+0.01\`;
* MMD candidate MMD is not worse than matched MMD control by more than \`0.005\`;
* MMD candidate improves MMD over the short-warmup no-MMD candidate by at least \`0.01\`;
* no-MMD short-warmup reference does not show obvious instability relative to default-warmup no-MMD reference.

Failure close rule:
* close this LR-schedule/MMD mutation if short-warmup destabilizes direct/pp metrics, if MMD still gives no distributional improvement, or if candidate advantage is no better than matched control.

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

launch_one "residual128_abundance128_hybrid" "candidate" "nommd" "$GPU_A" "none"
launch_one "confound128_abundance128_hybrid_control" "control" "nommd" "$GPU_A" "none"
launch_one "residual128_abundance128_hybrid" "candidate" "g010" "$GPU_B" "0.10"
launch_one "confound128_abundance128_hybrid_control" "control" "g010" "$GPU_B" "0.10"

echo "Wrote launch manifest: $MANIFEST"
