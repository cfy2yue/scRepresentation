#!/usr/bin/env bash
# One-shot checks on an H20 / Hopper-class node (adjust NCCL vars per your cluster admin).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${RAW_PYTHON_BIN:-python}"
echo "[bootstrap_h20] PYTHON=$PYTHON ROOT=$ROOT"

$PYTHON - <<'PY'
import os, sys
import torch
print("[bootstrap_h20] torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA required"
cap = torch.cuda.get_device_capability(0)
sm = cap[0] * 10 + cap[1]
print(f"[bootstrap_h20] device0 capability sm_{sm}")
# H20 reports as 9.0 (Hopper family)
x = torch.randn(2, 2, device="cuda", dtype=torch.bfloat16)
y = (x @ x).sum()
assert torch.isfinite(y), y
print("[bootstrap_h20] bfloat16 matmul OK")
print("bootstrap_h20: PASS")
PY
