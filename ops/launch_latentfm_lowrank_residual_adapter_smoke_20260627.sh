#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
PYTHON="/data/cyx/software/miniconda3/envs/scdfm/bin/python"
GPU_ID="${1:?usage: $0 <physical_gpu_id> [run_name]}"

SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-60}"
MAX_ACCEPTED="${MAX_ACCEPTED:-20}"
LOWRANK_RANK="${LOWRANK_RANK:-32}"
STEP_GRID="${STEP_GRID:-0.03,0.1,0.3,1,3,10,30}"
ANCHOR_THRESHOLD="${ANCHOR_THRESHOLD:-1e-6}"
MIN_TASK_DELTA="${MIN_TASK_DELTA:-1e-10}"
MIN_FOOTPRINT="${MIN_FOOTPRINT:-5e-6}"

RUN_NAME="${2:-xverse_lowrank_residual_adapter_seed${SEED}_${MAX_ACCEPTED}accepted_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$ROOT/runs/latentfm_lowrank_residual_adapter_smoke_20260627"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
SAFE_SPLIT="$ROOT/dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
DATA_DIR="$ROOT/dataset/latentfm_full/xverse"
BIFLOW_DIR="$ROOT/dataset/biFlow_data"

if [[ -e "$RUN_DIR" ]]; then
  echo "run directory already exists: $RUN_DIR" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

TRAIN_CMD="cd '$ROOT' && \
CUDA_VISIBLE_DEVICES='$GPU_ID' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH='$ROOT/CoupledFM' \
'$PYTHON' ops/train_latentfm_lookahead_trust_region_adapter_smoke_20260627.py \
  --save-dir '$RUN_DIR' \
  --device cuda:0 \
  --seed '$SEED' \
  --batch-size '$BATCH_SIZE' \
  --max-attempts '$MAX_ATTEMPTS' \
  --max-accepted '$MAX_ACCEPTED' \
  --adapter-kind lowrank_residual \
  --lowrank-rank '$LOWRANK_RANK' \
  --step-grid '$STEP_GRID' \
  --anchor-threshold '$ANCHOR_THRESHOLD' \
  --min-task-delta '$MIN_TASK_DELTA' \
  --min-footprint '$MIN_FOOTPRINT'"

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

A default-off low-rank condition residual adapter trained with the same
lookahead/trust-region line search can produce substantially larger controlled
footprint than the closed condition-delta adapter while preserving anchor
replay/no-harm. Parameterized early-stop runs test whether the 20-accepted
negative internal signal was caused by over-updating rather than by the adapter
family itself.

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

This run follows CPU gate:
\`$ROOT/reports/LATENTFM_LOWRANK_RESIDUAL_MODEL_PATCH_UNIT_GATE_20260627.md\`.

No canonical multi is used for selection. Track C query is not read. Canonical
single/family no-harm may be launched only if this frozen checkpoint passes
the internal posthoc gate.
EOF

echo "$RUN_NAME" > "$RUN_DIR/SESSION_NAME"
date > "$RUN_DIR/STARTED"

tmux new -d -s "$RUN_NAME" "bash -lc \"$TRAIN_CMD && $EVAL_CMD && $SUM_CMD\" > '$LOG_DIR/run.log' 2>&1; code=\$?; echo \$code > '$RUN_DIR/EXIT_CODE'; date > '$RUN_DIR/FINISHED'; exit \$code"

echo "$RUN_NAME"
echo "$RUN_DIR"
