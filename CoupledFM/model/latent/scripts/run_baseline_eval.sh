#!/usr/bin/env bash
# Evaluate baseline checkpoints: mse_only, ode_mmd_v2_a, ode_mmd_v2_b, ode_mmd_v2_c
# Uses default split (full dataset test set). Output: $RUNS/*/eval_results.json
#
# Env vars:
#   PYTHON=python               — python executable
#   RUNS=<abs path>             — run dir (default: <latent>/runs/baseline)
#   EVAL_SCRIPT=<abs path>      — path to eval_checkpoint.py; legacy repo had it
#                                 under experiment/scripts/. Set this explicitly
#                                 when running evaluation.
#   GPU=0                       — CUDA device id
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LATENT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON=${PYTHON:-python}
RUNS=${RUNS:-"$LATENT_DIR/runs/baseline"}
PLOT_SCRIPT="$SCRIPT_DIR/plot_baseline_bar.py"
GPU=${GPU:-0}

# EVAL_SCRIPT must be provided via env (legacy expriement/scripts/eval_checkpoint.py
# was not ported into this repo yet).
if [ -z "${EVAL_SCRIPT:-}" ]; then
    echo "[warn] EVAL_SCRIPT not set; skipping per-run evaluation."
    echo "       Set EVAL_SCRIPT=/path/to/eval_checkpoint.py to enable."
else
    for name in mse_only ode_mmd_v2_a ode_mmd_v2_b ode_mmd_v2_c; do
        ckpt="$RUNS/$name/best.pt"
        if [ ! -f "$ckpt" ]; then
            echo "[SKIP] $name — best.pt not found"
            continue
        fi
        echo "=== Evaluating $name ==="
        "$PYTHON" "$EVAL_SCRIPT" \
            --checkpoint "$ckpt" \
            --model-type control_mlp \
            --gpu "$GPU" \
            --save-dir "$RUNS/$name" \
            --tag "$name"
        echo ""
    done
    echo "=== Baseline evaluation done ==="
    echo "Results: $RUNS/*/eval_results.json"
fi

if [ -f "$PLOT_SCRIPT" ]; then
    echo "=== Plotting baseline bar chart ==="
    "$PYTHON" "$PLOT_SCRIPT" --runs-dir "$RUNS" --out-dir "$RUNS/plots" --format png
    echo "Figure: $RUNS/plots/baseline_metrics_bar.png"
fi
