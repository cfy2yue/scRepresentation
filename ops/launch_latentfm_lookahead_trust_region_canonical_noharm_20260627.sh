#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
PYTHON="/data/cyx/software/miniconda3/envs/scdfm/bin/python"
GPU_ID="${1:?usage: $0 <physical_gpu_id> <run_dir>}"
RUN_DIR="${2:?usage: $0 <physical_gpu_id> <run_dir>}"
RUN_DIR="$(readlink -f "$RUN_DIR")"
RUN_NAME="$(basename "$RUN_DIR")_canonical_noharm_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$RUN_DIR/posthoc_eval_canonical"
LOG_DIR="$OUT_DIR/logs"
ANCHOR_CKPT="$ROOT/CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
CAND_CKPT="$RUN_DIR/latest.pt"
INTERNAL_JSON="$RUN_DIR/posthoc/internal_eval_vs_anchor_summary.json"
PROVENANCE_JSON="$RUN_DIR/posthoc/lookahead_trust_region_smoke_provenance_audit.json"
CANON_SPLIT="$ROOT/dataset/biFlow_data/split_seed42.json"
DATA_DIR="$ROOT/dataset/latentfm_full/xverse"
BIFLOW_DIR="$ROOT/dataset/biFlow_data"

if [[ ! -f "$CAND_CKPT" ]]; then
  echo "candidate checkpoint not found: $CAND_CKPT" >&2
  exit 2
fi

if [[ ! -f "$INTERNAL_JSON" ]]; then
  echo "internal gate JSON not found: $INTERNAL_JSON" >&2
  exit 2
fi

if [[ ! -f "$PROVENANCE_JSON" ]]; then
  echo "provenance audit JSON not found: $PROVENANCE_JSON" >&2
  exit 2
fi

"$PYTHON" - "$INTERNAL_JSON" "$PROVENANCE_JSON" <<'PY'
import json
import sys
from pathlib import Path

internal_path = Path(sys.argv[1])
provenance_path = Path(sys.argv[2])
internal = json.loads(internal_path.read_text(encoding="utf-8"))
provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
required_internal = "lookahead_trust_region_internal_eval_pass_needs_canonical_noharm"
required_provenance = "lookahead_trust_region_smoke_provenance_pass"
if internal.get("status") != required_internal:
    raise SystemExit(
        f"internal gate status {internal.get('status')!r} != {required_internal!r}; "
        "canonical no-harm launch refused"
    )
if provenance.get("status") != required_provenance:
    raise SystemExit(
        f"provenance status {provenance.get('status')!r} != {required_provenance!r}; "
        "canonical no-harm launch refused"
    )
PY

mkdir -p "$LOG_DIR"

ANCHOR_SPLIT="$OUT_DIR/split_group_eval_anchor_ode20_canonical.json"
CAND_SPLIT="$OUT_DIR/split_group_eval_candidate_ode20_canonical.json"
ANCHOR_FAMILY="$OUT_DIR/condition_family_eval_anchor_ode20_canonical.json"
CAND_FAMILY="$OUT_DIR/condition_family_eval_candidate_ode20_canonical.json"
GATE_JSON="$OUT_DIR/single_background_candidate_gate.json"
GATE_MD="$OUT_DIR/LATENTFM_LOOKAHEAD_TRUST_REGION_CANONICAL_NOHARM_DECISION.md"

CMD="cd '$ROOT/CoupledFM' && \
CUDA_VISIBLE_DEVICES='$GPU_ID' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH='$ROOT/CoupledFM' \
'$PYTHON' -m model.latent.eval_split_groups \
  --checkpoint '$ANCHOR_CKPT' \
  --split-file '$CANON_SPLIT' \
  --data-dir '$DATA_DIR' \
  --biflow-dir '$BIFLOW_DIR' \
  --groups test_single \
  --out '$ANCHOR_SPLIT' \
  --device cuda:0 \
  --ode-steps 20 \
  --eval-seed 42 \
  --save-condition-means && \
CUDA_VISIBLE_DEVICES='$GPU_ID' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH='$ROOT/CoupledFM' \
'$PYTHON' -m model.latent.eval_condition_families \
  --checkpoint '$ANCHOR_CKPT' \
  --split-file '$CANON_SPLIT' \
  --data-dir '$DATA_DIR' \
  --biflow-dir '$BIFLOW_DIR' \
  --groups family_gene \
  --out '$ANCHOR_FAMILY' \
  --device cuda:0 \
  --ode-steps 20 \
  --eval-seed 42 \
  --save-condition-means && \
CUDA_VISIBLE_DEVICES='$GPU_ID' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH='$ROOT/CoupledFM' \
'$PYTHON' -m model.latent.eval_split_groups \
  --checkpoint '$CAND_CKPT' \
  --split-file '$CANON_SPLIT' \
  --data-dir '$DATA_DIR' \
  --biflow-dir '$BIFLOW_DIR' \
  --groups test_single \
  --out '$CAND_SPLIT' \
  --device cuda:0 \
  --ode-steps 20 \
  --eval-seed 42 \
  --save-condition-means && \
CUDA_VISIBLE_DEVICES='$GPU_ID' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH='$ROOT/CoupledFM' \
'$PYTHON' -m model.latent.eval_condition_families \
  --checkpoint '$CAND_CKPT' \
  --split-file '$CANON_SPLIT' \
  --data-dir '$DATA_DIR' \
  --biflow-dir '$BIFLOW_DIR' \
  --groups family_gene \
  --out '$CAND_FAMILY' \
  --device cuda:0 \
  --ode-steps 20 \
  --eval-seed 42 \
  --save-condition-means && \
cd '$ROOT' && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH='$ROOT/CoupledFM' \
'$PYTHON' ops/evaluate_latentfm_single_background_candidate_gate_20260623.py \
  --anchor-split-json '$ANCHOR_SPLIT' \
  --candidate-split-json '$CAND_SPLIT' \
  --anchor-family-json '$ANCHOR_FAMILY' \
  --candidate-family-json '$CAND_FAMILY' \
  --split-file '$CANON_SPLIT' \
  --data-dir '$DATA_DIR' \
  --n-boot 2000 \
  --seed 42 \
  --out-json '$GATE_JSON' && \
'$PYTHON' ops/render_latentfm_single_background_candidate_gate_md_20260627.py \
  --gate-json '$GATE_JSON' \
  --out-md '$GATE_MD' \
  --title 'LatentFM Lookahead Trust-Region Canonical No-Harm Decision'"

cat > "$OUT_DIR/RUN_STATUS.md" <<EOF
# Run Status: $RUN_NAME

## Command

\`\`\`bash
$CMD
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

* \`$ANCHOR_SPLIT\`
* \`$CAND_SPLIT\`
* \`$ANCHOR_FAMILY\`
* \`$CAND_FAMILY\`
* \`$GATE_JSON\`
* \`$GATE_MD\`
* \`$OUT_DIR/EXIT_CODE\`
* \`$OUT_DIR/FINISHED\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 $LOG_DIR/run.log
cat $OUT_DIR/EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Run only after the internal lookahead posthoc gate passes. Canonical multi is
not evaluated or selected. Track C query is not read.
EOF

echo "$RUN_NAME" > "$OUT_DIR/SESSION_NAME"
date > "$OUT_DIR/STARTED"

tmux new -d -s "$RUN_NAME" "bash -lc \"$CMD\" > '$LOG_DIR/run.log' 2>&1; code=\$?; echo \$code > '$OUT_DIR/EXIT_CODE'; date > '$OUT_DIR/FINISHED'; exit \$code"

echo "$RUN_NAME"
echo "$OUT_DIR"
