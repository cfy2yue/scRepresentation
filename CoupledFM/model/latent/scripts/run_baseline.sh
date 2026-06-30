#!/usr/bin/env bash
# Baseline training: mse_only, ode_mmd_v2_a, ode_mmd_v2_b
# Full dataset, default split. Output: $RUNS/{mse_only,ode_mmd_v2_a,ode_mmd_v2_b}
#
# Env vars:
#   PYTHON=python                  — python executable
#   RUNS=<abs path>                — run output base dir (default: ./runs/baseline)
#   GPU_MSE=0  GPU_A=2  GPU_B=3    — per-variant CUDA device id
set -eo pipefail

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
RUNS=${RUNS:-"$(pwd)/runs/baseline"}
GPU_MSE=${GPU_MSE:-0}
GPU_A=${GPU_A:-2}
GPU_B=${GPU_B:-3}

COMMON="--model-type control_mlp --gpu 0 --batch-size 256 --min-cells 32 --scale-noise 0.02
  --lr 1e-4 --total-steps 10000000 --warmup-steps 1000 --lr-decay-steps 150000
  --n-ot-workers 6 --patience 10 --print-every 200 --eval-every 0"

mkdir -p "$RUNS/mse_only" "$RUNS/ode_mmd_v2_a" "$RUNS/ode_mmd_v2_b"

# mse_only: no MMD
echo "[mse_only] MSE only, no MMD"
CUDA_VISIBLE_DEVICES=$GPU_MSE nohup "$PYTHON" train.py $COMMON \
    --no-use-mmd --save-dir "$RUNS/mse_only" \
    > "$RUNS/mse_only/nohup.out" 2>&1 &
echo "  PID=$!"

# ode_mmd_v2_a: γ=0.001, warmup 60k→200k, 10 ODE steps
echo "[ode_mmd_v2_a] γ=0.001, warmup 60k→200k, 10 ODE steps"
CUDA_VISIBLE_DEVICES=$GPU_A nohup "$PYTHON" train.py $COMMON \
    --use-mmd --gamma 0.001 --gamma-warmup-start 60000 --gamma-warmup-end 200000 \
    --mmd-ode-steps 10 --mmd-every 30 --ds-loss-alpha 0.0 \
    --save-dir "$RUNS/ode_mmd_v2_a" \
    > "$RUNS/ode_mmd_v2_a/nohup.out" 2>&1 &
echo "  PID=$!"

# ode_mmd_v2_b: γ=0.002, warmup 60k→150k, 5 ODE steps
echo "[ode_mmd_v2_b] γ=0.002, warmup 60k→150k, 5 ODE steps"
CUDA_VISIBLE_DEVICES=$GPU_B nohup "$PYTHON" train.py $COMMON \
    --use-mmd --gamma 0.002 --gamma-warmup-start 60000 --gamma-warmup-end 150000 \
    --mmd-ode-steps 5 --mmd-every 30 --ds-loss-alpha 0.0 \
    --save-dir "$RUNS/ode_mmd_v2_b" \
    > "$RUNS/ode_mmd_v2_b/nohup.out" 2>&1 &
echo "  PID=$!"

echo "=== Baseline training submitted ==="
