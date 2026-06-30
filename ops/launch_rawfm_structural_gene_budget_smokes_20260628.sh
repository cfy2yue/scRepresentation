#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
PY="/data/cyx/software/miniconda3/envs/scdfm/bin/python"
SPLIT="$ROOT/dataset/biFlow_data/split_seed42.json"
BIFLOW="$ROOT/dataset/biFlow_data"
MANIFEST_DIR="$ROOT/reports/rawfm_structural_gene_budget_manifest_20260628"
OUT_ROOT="$ROOT/CoupledFM/output/rawfm_structural_gene_budget_smoke_20260628"
RUN_ROOT="$ROOT/runs"

launch_one() {
  local label="$1"
  local gpu="$2"
  local run_name="rawfm_wessels_structural_${label}_k256_smoke_20260628_1956"
  local run_dir="$RUN_ROOT/$run_name"
  local manifest="$MANIFEST_DIR/${label}_k256_seed42.json"
  local output_dir="$OUT_ROOT/wessels_${label}_k256_seed42"
  local log="$run_dir/logs/run.log"
  local cmd="cd $ROOT/CoupledFM && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 $PY model/tools/launch_stack_train.py --biflow-dir $BIFLOW --latent-backbone stack --split-file $SPLIT --datasets Wessels --output-dir $output_dir --mode ot --ot-feature raw --epochs 2 --batch-size 4 --micro-batch 2 --grad-accum-steps 1 --lr 5e-5 --val-every-steps 0 --max-train-steps-per-epoch 10 --selection-protocol fixed_steps_no_selection --fixed-step-no-selection --test-every-epoch 0 --early-stop-patience 0 --eval-ode-steps 5 --val-ode-steps 5 --gene-budget-manifest $manifest --gene-budget-label ${label}_k256_seed42 --no-mmd"

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

Structural RawFM gene budgets that separate train-only response information from abundance/variance/detection confounds may beat the prior naive response-topk and matched controls under the same fixed-step/no-selection Wessels smoke.

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

* GPU: physical GPU $gpu from three-sample audit at 2026-06-28 19:56 CST.
* CPU: 4 BLAS/OpenMP threads; four-run packet total 16 project threads.
* RAM: Wessels loader dry-run stayed below 1.5 GiB RSS per process.
* MMD disabled because budgeted training MMD is not yet mask-aware.

## Gate

Promotion signal:
* exit code 0;
* \`last.pt\`, \`train_log.jsonl\`, \`split_provenance.json\`, and \`run_meta.json\` exist;
* no \`best.pt\` is created under fixed-step/no-selection;
* final-only Wessels metrics beat the matched control(s) before any expansion.

Failure close rule:
* close this structural budget if it loses to residual matched controls, writes \`best.pt\`, uses unbudgeted inputs, or worsens MMD relative to matched controls.

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

Bounded smoke only. Canonical split is explicit read-only \`split_seed42.json\`; canonical multi and Track C query are not used for training or checkpoint selection.
STATUS

  tmux new -d -s "$run_name" "bash -lc '$cmd > $log 2>&1; status=\$?; echo \$status > $run_dir/EXIT_CODE; date '+%F %T %z' > $run_dir/FINISHED; if [ \$status -eq 0 ]; then printf \"\\n## Final status\\n\\nFinished.\\n\" >> $run_dir/RUN_STATUS.md; else printf \"\\n## Final status\\n\\nFailed with exit code %s.\\n\" \"\$status\" >> $run_dir/RUN_STATUS.md; fi'"
}

launch_one "response_abundance_residual_topk" 2
launch_one "residual_confound_matched_random" 2
launch_one "condition_diversity_topk" 3
launch_one "residual_abundance_matched_random" 3
