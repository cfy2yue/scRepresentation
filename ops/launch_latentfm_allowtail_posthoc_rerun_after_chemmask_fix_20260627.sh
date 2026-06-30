#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_BLOCK="$ROOT/runs/latentfm_tracka_xverse_allowlisted_tail_hybrid_20260627"
LOG_BLOCK="$ROOT/logs/latentfm_tracka_xverse_allowlisted_tail_hybrid_20260627"
REPORT_DIR="$ROOT/reports/tracka_exact_tail_candidate_gate_20260627"
PYTHON_BIN=/data/cyx/software/miniconda3/envs/scdfm/bin/python
GPU_ID="${LATENTFM_ALLOWTAIL_POSTHOC_GPU:-2}"
THREADS="${LATENTFM_ALLOWTAIL_POSTHOC_THREADS:-4}"

mkdir -p "$REPORT_DIR"

launch_seed() {
  local seed="$1"
  local run_name="xverse_allowtail_hybrid_pertresid_prior_w003_p002_replay1_2k_seed${seed}"
  local session="lfm_allowtail_posthoc_rerun_seed${seed}_20260627"
  local run_dir="$RUN_BLOCK/$run_name"
  local log_dir="$LOG_BLOCK/$run_name"
  local posthoc_dir="$run_dir/posthoc_canonical_tracka"
  local log_path="$log_dir/posthoc_rerun_after_chemmask_fix.log"
  local ckpt="$ROOT/CoupledFM/output/latentfm_runs/latentfm_tracka_xverse_allowlisted_tail_hybrid_20260627/$run_name/best.pt"
  local anchor_json="$posthoc_dir/condition_family_eval_anchor_ode20_canonical.json"
  local candidate_json="$posthoc_dir/condition_family_eval_candidate_ode20_canonical.json"
  local gate_prefix="$REPORT_DIR/$run_name"

  mkdir -p "$log_dir" "$posthoc_dir"
  if [[ ! -s "$ckpt" ]]; then
    echo "Missing candidate checkpoint: $ckpt" >&2
    return 2
  fi
  if [[ ! -s "$anchor_json" ]]; then
    echo "Missing anchor posthoc JSON: $anchor_json" >&2
    return 2
  fi

  cat >> "$run_dir/RUN_STATUS.md" <<STATUS

## Posthoc rerun after chem_mask fix

Start time: $(date '+%F %T %Z')

tmux: ${session}

Log path: \`${log_path}\`

Reason: original posthoc failed because the stricter fail-closed allowlist gate
required explicit chem_mask metadata; \`_pert_to_device\` now supplies an
explicit false no-chemical mask for no-chemical conditions. This rerun reuses
the existing trained checkpoint and completed anchor JSON, then recomputes
candidate canonical Track A posthoc and the predeclared exact-tail gate.
Canonical multi and Track C query are not used.
STATUS

  rm -f "$run_dir/POSTHOC_RERUN_EXIT_CODE" "$run_dir/POSTHOC_RERUN_FINISHED"
  tmux new -d -s "$session" \
    "bash -lc 'set -euo pipefail; \
      source $ROOT/init-scdfm.sh >/dev/null; \
      cd $ROOT/CoupledFM; \
      export CUDA_VISIBLE_DEVICES=$GPU_ID; \
      export OMP_NUM_THREADS=$THREADS; \
      export MKL_NUM_THREADS=$THREADS; \
      export OPENBLAS_NUM_THREADS=$THREADS; \
      export NUMEXPR_NUM_THREADS=$THREADS; \
      export BLIS_NUM_THREADS=$THREADS; \
      export PYTHONPATH=$ROOT/CoupledFM:\${PYTHONPATH:-}; \
      export PERT_EMBED_SOURCE=scgpt_embed_gene; \
      common=(--data-dir $ROOT/dataset/latentfm_full/xverse --biflow-dir $ROOT/dataset/biFlow_data --split-file $ROOT/dataset/biFlow_data/split_seed42.json --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024); \
      set +e; \
      ( \
        set -euo pipefail; \
        echo \"[posthoc-rerun] seed=$seed gpu=$GPU_ID start=\$(date)\"; \
        $PYTHON_BIN -m model.latent.eval_condition_families \
          --checkpoint $ckpt \
          --groups test_all family_gene family_drug structure_single test_single \
          --out $candidate_json \"\${common[@]}\"; \
        $PYTHON_BIN $ROOT/ops/evaluate_latentfm_tracka_exact_tail_candidate_gate_20260627.py \
          --anchor-json $anchor_json \
          --candidate-json $candidate_json \
          --out-prefix $gate_prefix \
          --title \"xverse allowlisted-tail hybrid exact-tail gate rerun after chem_mask fix\" \
          --n-boot 5000 \
          --seed 42; \
        echo \"[posthoc-rerun] seed=$seed finished=\$(date)\"; \
      ) > $log_path 2>&1; \
      code=\$?; set -e; echo \$code > $run_dir/POSTHOC_RERUN_EXIT_CODE; date > $run_dir/POSTHOC_RERUN_FINISHED; exit \$code'"
  echo "$session" > "$run_dir/POSTHOC_RERUN_SESSION_NAME"
}

launch_seed 42
launch_seed 43

tmux ls | grep 'lfm_allowtail_posthoc_rerun' || true
