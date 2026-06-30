# utils — shared library

Shared primitives used by `latent/`, raw-expression CoupledFM entrypoints, raw
pretraining, and the data pipelines. The subpackages expose stable APIs; avoid
cross-imports between task-specific training modules and instead add common
code here.

## Subpackages

| Path | Contents |
|------|----------|
| [`utils/models/`](models/) | `MultiHeadAttention` (sdpa / flash / linear / sparse), `FeedForward`, `VelocityFieldBase` |
| [`utils/data/`](data/) | `GeneVocab`, `_LazyH5`, `_DatasetHandle`, `LatentOTPairer` (GPU Sinkhorn), split builders |
| [`utils/train/`](train/) | `ModelEMA`, LR schedulers (`cosine_with_min_lr`, teacher-forcing curriculum), AMP autocast helper |
| [`utils/io/`](io/) | HDF5 helpers with `HDF5_USE_FILE_LOCKING=FALSE` semantics, NPZ utilities |
| [`utils/conditioning/`](conditioning/) | `IdentityConditionEncoder` (`cond_vec` placeholder) |

## Attention backends (`utils/models/attention.py`)

Canonical source for `MultiHeadAttention`; model-level attention imports
delegate here. Four backends:

| Backend | Supports `attn_bias` | Supports `edge_index` | Typical throughput |
|---------|---------------------|----------------------|-------------------|
| `sdpa` (default) | ✅ dense additive | — | 90–95 % of flash |
| `flash` | fallback to SDPA when bias given | — | 100 % reference |
| `linear` | — | — | O(N), no bias |
| `sparse` (CellNavi-style) | — | ✅ required | O(B·E·H) |

`sparse` is the batched port of CellNavi's `SparseScaledDotProductAttention`
(`scatter_softmax` implemented in log-sum-exp form for bf16/fp16 safety).
Verified numerically equivalent to `sdpa` on fully-connected `edge_index`
(max relative diff ≈ 3e-7); see `output/attn_bias_design.md` §9 for test
details.

```python
from utils.models.attention import MultiHeadAttention
# Dense SDPA + additive bias
mha = MultiHeadAttention(d_model=256, d_key=256, n_head=16, dropout=0.1, attn_backend="sdpa")
out = mha(x, attn_bias=bias)
# Sparse scatter
mha = MultiHeadAttention(..., attn_backend="sparse")
out = mha(x, edge_index=edge_index, num_nodes=N)
```

## Train utilities (`utils/train/`)

- [`ema.py`](train/ema.py) — `ModelEMA(decay=0.999, update_after=..., update_every=...)`
  with `apply_to(model)` context manager for evaluation and `state_dict` /
  `load_state_dict` for checkpoint round-trip.
- [`schedulers.py`](train/schedulers.py) —
  - `get_cosine_with_min_lr_schedule_with_warmup(optimizer, warmup, total, min_lr_ratio=0.1)`
    — HuggingFace-style cosine decay with a floor so LR never drops below
    `base_lr * min_lr_ratio`.
  - `get_linear_warmup_then_const(optimizer, warmup)` — linear warmup + flat
    thereafter (for stage-1 fine-tuning with a frozen backbone).
  - `get_ode_prob_curriculum(step, warmup, anneal, max_prob)` — scalar
    schedule for teacher-forcing curriculum; returns the probability of
    using ODE-evolved `z_t` instead of interpolated `z_t` at a given step.
- AMP autocast wrapper used by both `latent/train.py` and `model/train.py`.

## Data utilities (`utils/data/`)

- `GeneVocab` — gene-name → CellNavi-token mapping (`gene_name.txt`) and
  NicheNet-node index (`node2idx.json`). Read-only; shared across DDP ranks.
- `_LazyH5` / `_DatasetHandle` — lazy H5 wrappers; open an AnnData file once
  per rank via `HDF5_USE_FILE_LOCKING=FALSE` and serve rows by fancy indexing.
  Audit in [`docs/lazy_loading_audit.md`](../docs/lazy_loading_audit.md).
- `LatentOTPairer` — entropic OT pairing with a GPU Sinkhorn path
  (log-stabilised) and a CPU POT fallback. See
  [`output/minibatch_ot_cpu_optimization.md`](../output/minibatch_ot_cpu_optimization.md)
  for the CPU bottleneck analysis and the GPU migration record.
- `split` — canonical train/test split builder keyed by `(biflow_dir, seed)`,
  unified across latent / raw / CoupledFM experiments to prevent leakage.

## Conventions

- `utils/` **never** imports from task-specific training entrypoints. The
  reverse is the only allowed direction.
- All public APIs are `from __future__ import annotations`-compatible (Python ≥ 3.10).
- fp16/bf16-safety: any softmax or reduction over long sequences promotes to
  fp32 internally and casts back. See `_scatter_softmax` and
  `_linear_attention`.
