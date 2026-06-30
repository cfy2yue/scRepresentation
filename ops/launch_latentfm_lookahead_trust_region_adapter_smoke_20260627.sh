#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
PYTHON="/data/cyx/software/miniconda3/envs/scdfm/bin/python"
GPU_ID="${1:?usage: $0 <physical_gpu_id> [run_name]}"
RUN_NAME="${2:-xverse_lookahead_trust_region_adapter_seed42_40accepted_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$ROOT/runs/latentfm_lookahead_trust_region_adapter_smoke_20260627"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
SAFE_SPLIT="$ROOT/dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
DATA_DIR="$ROOT/dataset/latentfm_full/xverse"
BIFLOW_DIR="$ROOT/dataset/biFlow_data"

mkdir -p "$LOG_DIR"

TRAIN_CMD="cd '$ROOT' && \
CUDA_VISIBLE_DEVICES='$GPU_ID' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH='$ROOT/CoupledFM' \
'$PYTHON' ops/train_latentfm_lookahead_trust_region_adapter_smoke_20260627.py \
  --save-dir '$RUN_DIR' \
  --device cuda:0 \
  --seed 42 \
  --batch-size 16 \
  --max-attempts 80 \
  --max-accepted 40 \
  --step-grid 1,3,10,30,100,300 \
  --anchor-threshold 1e-6 \
  --min-task-delta 1e-10 \
  --min-footprint 1e-7"

EVAL_CMD="cd '$ROOT/CoupledFM' && \
CUDA_VISIBLE_DEVICES='$GPU_ID' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH='$ROOT/CoupledFM' \
'$PYTHON' -m model.latent.eval_split_groups \
  --checkpoint '$RUN_DIR/latest.pt' \
  --split-file '$SAFE_SPLIT' \
  --data-dir '$DATA_DIR' \
  --biflow-dir '$BIFLOW_DIR' \
  --groups internal_val_cross_background_seen_gene_proxy internal_val_family_gene_proxy \
  --out '$RUN_DIR/internal_eval_split_groups.json' \
  --device cuda:0 \
  --ode-steps 20 \
  --eval-seed 42 \
  --save-condition-means"

SUM_CMD="cd '$ROOT' && \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH='$ROOT/CoupledFM' \
'$PYTHON' ops/summarize_latentfm_lookahead_trust_region_adapter_smoke_20260627.py \
  --run-dir '$RUN_DIR' \
  --eval-json '$RUN_DIR/internal_eval_split_groups.json'"

cat > "$RUN_DIR/RUN_STATUS.md" <<EOF
# Run Status: $RUN_NAME

## Hypothesis

A zero-initialized condition-delta adapter trained with lookahead/trust-region
projected updates can make small train-task progress while preserving
anchor-replay/no-harm behavior.

## Command

\`\`\`bash
$TRAIN_CMD
$EVAL_CMD
$SUM_CMD
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%Y-%m-%d %H:%M:%S %Z')

## PID / tmux / scheduler ID

tmux session: \`$RUN_NAME\`

## Log path

\`$LOG_DIR/run.log\`

## Expected outputs

* \`$RUN_DIR/latest.pt\`
* \`$RUN_DIR/train_metrics.csv\`
* \`$RUN_DIR/summary.json\`
* \`$RUN_DIR/internal_eval_split_groups.json\`
* \`$RUN_DIR/posthoc/LATENTFM_LOOKAHEAD_TRUST_REGION_INTERNAL_EVAL_DECISION.md\`
* \`$RUN_DIR/EXIT_CODE\`
* \`$RUN_DIR/FINISHED\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 $LOG_DIR/run.log
cat $RUN_DIR/EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Safe split: \`$SAFE_SPLIT\`

No canonical multi is used for selection. Track C query is not read. Internal
eval is diagnostic for the frozen latest checkpoint produced by this smoke.
Canonical no-harm, if run, must happen only after this route/checkpoint is
frozen.
EOF

echo "$RUN_NAME" > "$RUN_DIR/SESSION_NAME"
date > "$RUN_DIR/STARTED"

tmux new -d -s "$RUN_NAME" "bash -lc \"$TRAIN_CMD && $EVAL_CMD && $SUM_CMD\" > '$LOG_DIR/run.log' 2>&1; code=\$?; echo \$code > '$RUN_DIR/EXIT_CODE'; date > '$RUN_DIR/FINISHED'; exit \$code"

echo "$RUN_NAME"
echo "$RUN_DIR"
