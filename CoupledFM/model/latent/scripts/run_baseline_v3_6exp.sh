#!/usr/bin/env bash
# Baseline v3: 6 组新实验，基于当前趋势设计
# 趋势: γ=0.002(v2_b) 最优 dp≈0.99; γ≥0.02(v2_c) 导致 dp 崩塌至 0.81
# 设计: 在低 γ 区间精细搜索 + warmup/mmd_every/ds_loss 变体
#
# GPU 1,3,4 串行两轮:
#   第一轮: v2_f, v2_g, v2_h 并行 (GPU 1,3,4)
#   第二轮: v2_i, v2_j, v2_k 并行 (GPU 1,3,4) — 第一轮跑完后执行
#
# 用法: 先执行本脚本第一轮; 等 3 个任务都结束后再执行第二轮
#
# Env vars:
#   PYTHON=python        — python executable
#   RUNS=<abs path>      — run output base dir (default: <latent>/runs/baseline)
#   GPU_1=1  GPU_2=3  GPU_3=4   — 3 GPUs per round
set -eo pipefail

cd "$(dirname "$0")/.."
LATENT_DIR="$(pwd)"
PYTHON=${PYTHON:-python}
RUNS=${RUNS:-"$LATENT_DIR/runs/baseline"}
GPU_1=${GPU_1:-1}
GPU_2=${GPU_2:-3}
GPU_3=${GPU_3:-4}

COMMON="--model-type control_mlp --gpu 0 --batch-size 256 --min-cells 32 --scale-noise 0.02
  --lr 1e-4 --total-steps 10000000 --warmup-steps 1000 --lr-decay-steps 150000
  --n-ot-workers 6 --patience 10 --print-every 200
  --use-mmd --mmd-ode-steps 5 --ds-loss-alpha 0.0"

# ============================================================================
# 第一轮: 3 个实验 (GPU 1, 3, 4 并行)
# ============================================================================
run_round1() {
  mkdir -p "$RUNS/ode_mmd_v2_f" "$RUNS/ode_mmd_v2_g" "$RUNS/ode_mmd_v2_h"

  # v2_f: γ=0.003, warmup 60k→150k (v2_b 基础上略增 γ)
  echo "[ode_mmd_v2_f] γ=0.003, warmup 60k→150k, ode=5, mmd_every=30"
  CUDA_VISIBLE_DEVICES=$GPU_1 nohup "$PYTHON" train.py $COMMON \
    --gamma 0.003 --gamma-warmup-start 60000 --gamma-warmup-end 150000 --mmd-every 30 \
    --save-dir "$RUNS/ode_mmd_v2_f" \
    > "$RUNS/ode_mmd_v2_f/nohup.out" 2>&1 &
  echo "  GPU $GPU_1 PID=$!"

  # v2_g: γ=0.004, warmup 50k→180k (稍早 warmup，略高 γ)
  echo "[ode_mmd_v2_g] γ=0.004, warmup 50k→180k, ode=5, mmd_every=30"
  CUDA_VISIBLE_DEVICES=$GPU_2 nohup "$PYTHON" train.py $COMMON \
    --gamma 0.004 --gamma-warmup-start 50000 --gamma-warmup-end 180000 --mmd-every 30 \
    --save-dir "$RUNS/ode_mmd_v2_g" \
    > "$RUNS/ode_mmd_v2_g/nohup.out" 2>&1 &
  echo "  GPU $GPU_2 PID=$!"

  # v2_h: γ=0.005, warmup 60k→200k (γ 上限探索，仍保持低区)
  echo "[ode_mmd_v2_h] γ=0.005, warmup 60k→200k, ode=5, mmd_every=30"
  CUDA_VISIBLE_DEVICES=$GPU_3 nohup "$PYTHON" train.py $COMMON \
    --gamma 0.005 --gamma-warmup-start 60000 --gamma-warmup-end 200000 --mmd-every 30 \
    --save-dir "$RUNS/ode_mmd_v2_h" \
    > "$RUNS/ode_mmd_v2_h/nohup.out" 2>&1 &
  echo "  GPU $GPU_3 PID=$!"

  echo ""
  echo "=== 第一轮已提交 (v2_f, v2_g, v2_h) ==="
  echo "等 3 个任务都结束后，执行: $0 round2"
}

# ============================================================================
# 第二轮: 3 个实验 (GPU 1, 3, 4 并行)
# ============================================================================
run_round2() {
  mkdir -p "$RUNS/ode_mmd_v2_i" "$RUNS/ode_mmd_v2_j" "$RUNS/ode_mmd_v2_k"

  # v2_i: γ=0.002, 更早 warmup 40k→120k, mmd_every=20 (更频繁 MMD)
  echo "[ode_mmd_v2_i] γ=0.002, warmup 40k→120k, ode=5, mmd_every=20"
  CUDA_VISIBLE_DEVICES=$GPU_1 nohup "$PYTHON" train.py $COMMON \
    --gamma 0.002 --gamma-warmup-start 40000 --gamma-warmup-end 120000 --mmd-every 20 \
    --save-dir "$RUNS/ode_mmd_v2_i" \
    > "$RUNS/ode_mmd_v2_i/nohup.out" 2>&1 &
  echo "  GPU $GPU_1 PID=$!"

  # v2_j: γ=0.002, ds_loss_alpha=0.5 (per-dataset 逆频率加权)
  echo "[ode_mmd_v2_j] γ=0.002, warmup 60k→150k, ds_loss_alpha=0.5"
  CUDA_VISIBLE_DEVICES=$GPU_2 nohup "$PYTHON" train.py $COMMON \
    --gamma 0.002 --gamma-warmup-start 60000 --gamma-warmup-end 150000 --mmd-every 30 \
    --ds-loss-alpha 0.5 \
    --save-dir "$RUNS/ode_mmd_v2_j" \
    > "$RUNS/ode_mmd_v2_j/nohup.out" 2>&1 &
  echo "  GPU $GPU_2 PID=$!"

  # v2_k: γ=0.003, ode_steps=8 (更多 ODE 步数)
  echo "[ode_mmd_v2_k] γ=0.003, warmup 60k→150k, ode=8, mmd_every=30"
  CUDA_VISIBLE_DEVICES=$GPU_3 nohup "$PYTHON" train.py $COMMON \
    --gamma 0.003 --gamma-warmup-start 60000 --gamma-warmup-end 150000 --mmd-every 30 \
    --mmd-ode-steps 8 \
    --save-dir "$RUNS/ode_mmd_v2_k" \
    > "$RUNS/ode_mmd_v2_k/nohup.out" 2>&1 &
  echo "  GPU $GPU_3 PID=$!"

  echo ""
  echo "=== 第二轮已提交 (v2_i, v2_j, v2_k) ==="
}

# ============================================================================
# Main
# ============================================================================
case "${1:-round1}" in
  round1) run_round1 ;;
  round2) run_round2 ;;
  *)
    echo "用法: $0 [round1|round2]"
    echo "  round1: 提交 v2_f, v2_g, v2_h (GPU 1,3,4)"
    echo "  round2: 提交 v2_i, v2_j, v2_k (GPU 1,3,4)"
    echo ""
    echo "建议流程:"
    echo "  1. $0 round1"
    echo "  2. 等待 3 个任务完成 (nvidia-smi / tail -f runs/baseline/ode_mmd_v2_*/nohup.out)"
    echo "  3. $0 round2"
    exit 1
    ;;
esac
