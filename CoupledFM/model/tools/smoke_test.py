"""Smoke test: 不依赖 biFlow 数据，验证 import + 一次 OT Sinkhorn + 轻量前向。

用法：bash scripts/smoke_test.sh
退出 0 且末行打印 "SMOKE OK" 即通过。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.environ.setdefault("RAW_INDEPENDENT_ROOT", str(_REPO))

import torch

from model.config import Config  # noqa: F401
from model.data.vocab import GeneVocab  # noqa: F401
from model.models.velocity_field import RawExprVelocityField
from model.utils.data.ot_pairer import compute_ot_cost, sinkhorn_log_torch
from model.utils.data.split import canonical_split_path  # noqa: F401
from model.utils.train.ema import ModelEMA  # noqa: F401
from model.utils.train.schedulers import get_ode_prob_curriculum  # noqa: F401

print("[smoke] imports OK")

device = "cuda" if torch.cuda.is_available() else "cpu"
x0 = torch.randn(8, 4, device=device)
x1 = torch.randn(12, 4, device=device)
cost = compute_ot_cost(x0, x1, "l2")
pi = sinkhorn_log_torch(cost, reg=0.05, n_iter=20)
assert pi.shape == (8, 12)
print(f"[smoke] OT-on-DE OK plan_sum={float(pi.sum().cpu()):.4f}")

# Tiny velocity field (no pert encoder, no checkpoint)
B, G = 2, 48
vf = RawExprVelocityField(
    d_model=64,
    n_layer=2,
    n_head=4,
    d_ff=128,
    d_latent=32,
    coupling_mode="ot",
    use_pert_condition=False,
    grad_ckpt=False,
    attn_backend="sdpa",
).to(device)

x_t = torch.randn(B, G, device=device)
x_c = torch.randn(B, G, device=device)
t = torch.rand(B, device=device)
gid = torch.randint(0, 40000, (G,), device=device)  # (G,) – shared across batch, per dataset.gene_ids_valid
out = vf(x_t, x_c, t, gid, aux_emb=torch.randn(B, 32, device=device))
assert out.shape[0] == B
loss = out.float().mean()
loss.backward()
print("[smoke] RawExprVelocityField forward+backward OK")

# Optional resource smoke: only run when pretrain/cache files are present
# so this stays a portable CPU smoke (no fail on bare checkout).
try:
    from model import paths
    from model.condition_emb.genepert.gene_cache import GeneEmbeddingCache

    cache_dir = paths.cellnavi_cache_dir()
    if (cache_dir / "gene_embeddings.npy").is_file():
        c = GeneEmbeddingCache(str(cache_dir))
        c.validate_index_bounds()
        print(f"[smoke] GeneEmbeddingCache dim={c.embed_dim} rows={c.num_embeddings}")
    else:
        print(f"[smoke] skip GeneEmbeddingCache (no {cache_dir}/gene_embeddings.npy)")

    ckpt = paths.cellnavi_pretrain_ckpt_path()
    if ckpt.is_file():
        sd = torch.load(str(ckpt), map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        nkeys = len(sd) if hasattr(sd, "__len__") else -1
        print(f"[smoke] pretrain ckpt loaded keys={nkeys}")
    else:
        print(f"[smoke] skip pretrain ckpt (no {ckpt})")
except Exception as e:
    print(f"[smoke] optional resource check skipped: {type(e).__name__}: {e}")

print("SMOKE OK")
