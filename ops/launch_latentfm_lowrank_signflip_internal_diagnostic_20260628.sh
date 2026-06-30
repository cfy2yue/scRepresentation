#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
PYTHON="/data/cyx/software/miniconda3/envs/scdfm/bin/python"
GPU_ID="${1:?usage: $0 <physical_gpu_id> [run_name]}"
RUN_NAME="${2:-xverse_lowrank_signflip_from5step_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$ROOT/runs/latentfm_lowrank_signflip_diagnostic_20260628"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
LOG_DIR="$RUN_DIR/logs"
SAFE_SPLIT="$ROOT/dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
DATA_DIR="$ROOT/dataset/latentfm_full/xverse"
BIFLOW_DIR="$ROOT/dataset/biFlow_data"
SOURCE_RUN="${SOURCE_RUN:-$ROOT/runs/latentfm_lowrank_residual_adapter_smoke_20260627/xverse_lowrank_residual_adapter_seed42_5accepted_20260628_0000}"
ALPHAS="${ALPHAS:--1,-0.5,-0.25,0.25}"

if [[ -e "$RUN_DIR" ]]; then
  echo "run directory already exists: $RUN_DIR" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

cat > "$RUN_DIR/RUN_STATUS.md" <<EOF
# Run Status: $RUN_NAME

## Hypothesis

The low-rank residual v1 checkpoints may be directionally anti-aligned with the
safe internal pearson_pert proxy. Scaling only the low-rank up layer tests
whether reversing or shrinking the learned residual direction repairs internal
proxy deltas without retraining.

## Command

\`\`\`bash
SOURCE_RUN='$SOURCE_RUN' ALPHAS='$ALPHAS' $0 $GPU_ID $RUN_NAME
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

* \`$RUN_DIR/alpha_*/latest.pt\`
* \`$RUN_DIR/alpha_*/internal_eval_split_groups.json\`
* \`$RUN_DIR/alpha_*/posthoc/LATENTFM_LOOKAHEAD_TRUST_REGION_INTERNAL_EVAL_DECISION.md\`
* \`$RUN_DIR/signflip_summary.tsv\`
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

Source run: \`$SOURCE_RUN\`

Alphas: \`$ALPHAS\`

Boundary: safe internal proxy diagnostic only. No training, no canonical multi,
no Track C query. A passing alpha would authorize only frozen canonical
single/family no-harm, not a final model claim.
EOF

echo "$RUN_NAME" > "$RUN_DIR/SESSION_NAME"
date > "$RUN_DIR/STARTED"

tmux new -d -s "$RUN_NAME" "bash -lc '
set -euo pipefail
cd \"$ROOT\"
echo -e \"alpha\tstatus\treport\" > \"$RUN_DIR/signflip_summary.tsv\"
IFS=\",\" read -ra ALPHA_ARR <<< \"$ALPHAS\"
for alpha in \"\${ALPHA_ARR[@]}\"; do
  clean=\$(echo \"\$alpha\" | sed \"s/-/m/g; s/\\./p/g\")
  subdir=\"$RUN_DIR/alpha_\$clean\"
  mkdir -p \"\$subdir\"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \"$PYTHON\" ops/create_latentfm_lowrank_scaled_checkpoint_20260628.py \
    --source-checkpoint \"$SOURCE_RUN/latest.pt\" \
    --source-summary \"$SOURCE_RUN/summary.json\" \
    --out-dir \"\$subdir\" \
    --alpha \"\$alpha\"
  cd \"$ROOT/CoupledFM\"
  CUDA_VISIBLE_DEVICES=\"$GPU_ID\" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=\"$ROOT/CoupledFM\" \
    \"$PYTHON\" -m model.latent.eval_split_groups \
      --checkpoint \"\$subdir/latest.pt\" \
      --split-file \"$SAFE_SPLIT\" \
      --data-dir \"$DATA_DIR\" \
      --biflow-dir \"$BIFLOW_DIR\" \
      --groups internal_val_cross_background_seen_gene_proxy internal_val_family_gene_proxy \
      --out \"\$subdir/internal_eval_split_groups.json\" \
      --device cuda:0 \
      --ode-steps 20 \
      --eval-seed 42 \
      --save-condition-means
  cd \"$ROOT\"
  set +e
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=\"$ROOT/CoupledFM\" \
    \"$PYTHON\" ops/summarize_latentfm_lookahead_trust_region_adapter_smoke_20260627.py \
      --run-dir \"\$subdir\" \
      --eval-json \"\$subdir/internal_eval_split_groups.json\"
  rc=\$?
  set -e
  status=\$(\"$PYTHON\" - <<PY
import json, pathlib
p = pathlib.Path(\"\$subdir/posthoc/internal_eval_vs_anchor_summary.json\")
print(json.loads(p.read_text())[\"status\"] if p.exists() else \"missing_summary\")
PY
)
  echo -e \"\$alpha\t\$status\t\$subdir/posthoc/LATENTFM_LOOKAHEAD_TRUST_REGION_INTERNAL_EVAL_DECISION.md\" >> \"$RUN_DIR/signflip_summary.tsv\"
done
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=\"$ROOT/CoupledFM\" \
  \"$PYTHON\" \"$ROOT/ops/summarize_latentfm_lowrank_signflip_diagnostic_20260628.py\" \
    --run-dir \"$RUN_DIR\"
' > '$LOG_DIR/run.log' 2>&1; code=\$?; echo \$code > '$RUN_DIR/EXIT_CODE'; date > '$RUN_DIR/FINISHED'; exit \$code"

echo "$RUN_NAME"
echo "$RUN_DIR"
