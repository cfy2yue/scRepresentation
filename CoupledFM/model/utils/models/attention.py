"""
Multi-head attention with four backends.

Backends
--------
* ``sdpa``   — ``F.scaled_dot_product_attention``. Supports optional additive
               ``attn_bias`` (broadcast to ``(B, H, N, N)``). PyTorch will route
               to FlashAttention-2 when bias is ``None`` and dtype is fp16/bf16.
* ``flash``  — ``flash_attn.flash_attn_func``. No bias/mask support; automatically
               falls back to SDPA when ``attn_bias`` is provided.
* ``linear`` — ELU+1 kernel linear attention, O(N) memory. Cannot carry bias.
* ``sparse`` — CellNavi-style scatter attention over ``edge_index`` edges.
               Complexity O(B·E·H) rather than O(B·N²·H). Shares the same
               ``fc_query/fc_key/fc_value/fc_out`` parameter names as the
               dense variants so CellNavi pretrained weights load unchanged.

The ``attn_bias`` (dense) and ``edge_index`` (sparse) inputs are the two
different ways to inject graph priors:
  - ``attn_bias``:  additive float bias on logits (dense; ``-inf`` to mask out).
  - ``edge_index``: explicit (src, dst) edge list — only edges in the list
                    contribute, nothing else exists (hard mask).

``attn_bias`` is ignored when ``attn_backend='sparse'`` (the ``edge_index``
already defines the allowed pairs); a warning is emitted once per call where
this applies so misconfiguration is visible.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


# ---------------------------------------------------------------------------
# Numerics helpers
# ---------------------------------------------------------------------------

def _scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Numerically stable softmax over variable-size groups defined by ``index``.

    Args:
        src:  ``(E, H)`` — raw logits per edge per head.
        index: ``(E,)`` — destination node id in ``[0, num_nodes)``.
        num_nodes: total node count (= ``B * N`` after batch flattening).

    Returns:
        ``(E, H)`` softmax weights, normalised per destination node.

    Under AMP (bf16/fp16), intermediate math is promoted to fp32 to avoid
    overflow in the ``exp`` / row-sum step; output is cast back to the input
    dtype before returning.
    """
    compute_dtype = src.dtype
    if compute_dtype in (torch.float16, torch.bfloat16):
        src_fp = src.float()
    else:
        src_fp = src

    idx_exp = index.unsqueeze(-1).expand_as(src_fp)
    # amax per destination; include_self=True keeps the -inf initial value so
    # that groups with no members remain -inf (they will never be indexed).
    src_max = torch.full(
        (num_nodes, src_fp.shape[-1]),
        float("-inf"),
        device=src_fp.device,
        dtype=src_fp.dtype,
    )
    src_max.scatter_reduce_(0, idx_exp, src_fp, reduce="amax", include_self=True)

    out = (src_fp - src_max[index]).exp()
    sum_out = torch.zeros(
        (num_nodes, src_fp.shape[-1]),
        device=src_fp.device,
        dtype=src_fp.dtype,
    )
    sum_out.scatter_add_(0, idx_exp, out)
    out = out / (sum_out[index] + 1e-12)
    return out.to(compute_dtype)


def _linear_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """ELU+1 linear attention. ``q, k, v: (B, H, N, d)``.

    Accumulate einsums in float32 when inputs are half-precision — fp16/bf16
    einsums on long sequences (gwps) can overflow to inf/nan.
    """
    dtype = q.dtype
    if dtype in (torch.float16, torch.bfloat16):
        q = q.float()
        k = k.float()
        v = v.float()
    q = F.elu(q) + 1.0
    k = F.elu(k) + 1.0
    kv = torch.einsum("bhnd,bhnv->bhdv", k, v)
    k_sum = k.sum(dim=2)
    num = torch.einsum("bhnd,bhdv->bhnv", q, kv)
    den = torch.einsum("bhnd,bhd->bhn", q, k_sum).clamp(min=1e-6)
    out = num / den.unsqueeze(-1)
    return out.to(dtype)


def _sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    scale: float,
) -> torch.Tensor:
    """Batched CellNavi-style sparse scatter attention.

    Args:
        q, k, v:    ``(B, N, H, d_k)``.
        edge_index: ``(2, E)`` — ``[src, dst]`` pairs, messages flow ``src → dst``.
                    Shared across the batch (every item has the same N nodes /
                    same sparse pattern).
        num_nodes:  ``N`` (sequence length).
        scale:      softmax temperature (typically ``sqrt(d_k)``).

    Returns:
        ``(B, N, H, d_k)`` — per-node aggregated messages.
    """
    B, N, H, Dk = q.shape
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}"
        )
    E = edge_index.shape[1]

    # Offset edge_index per batch item: flatten (B, N) → B*N with base N*i.
    base = (
        torch.arange(B, device=q.device, dtype=edge_index.dtype) * N
    ).view(B, 1, 1)
    ei = edge_index.unsqueeze(0) + base          # (B, 2, E)
    ei = ei.permute(1, 0, 2).reshape(2, B * E)   # (2, B*E)
    src = ei[0]
    dst = ei[1]

    qf = q.reshape(B * N, H, Dk)
    kf = k.reshape(B * N, H, Dk)
    vf = v.reshape(B * N, H, Dk)

    logits = (qf[dst] * kf[src]).sum(dim=-1) / scale          # (B*E, H)
    attn_w = _scatter_softmax(logits, dst, B * N)             # (B*E, H)
    msgs = attn_w.unsqueeze(-1) * vf[src]                     # (B*E, H, Dk)

    # ``scatter_add_`` requires identical dtypes; under autocast the mul above
    # can promote through bf16 → fp32, so allocate ``out`` with the msgs dtype.
    out = torch.zeros(
        (B * N, H, Dk), device=qf.device, dtype=msgs.dtype,
    )
    dst_idx = dst.view(-1, 1, 1).expand_as(msgs)
    out.scatter_add_(0, dst_idx, msgs)
    return out.view(B, N, H, Dk).to(qf.dtype)


# ---------------------------------------------------------------------------
# Public modules
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Global / sparse MHA with pretrained-compatible ``fc_{query,key,value,out}``.

    ``forward`` takes an optional ``attn_bias`` (dense, for SDPA backend) OR
    ``edge_index`` (sparse, for ``sparse`` backend). They are mutually
    exclusive by backend choice.
    """

    _VALID_BACKENDS = frozenset({"sdpa", "flash", "linear", "sparse"})

    def __init__(
        self,
        d_model: int,
        d_key: int,
        n_head: int,
        dropout: float,
        attn_backend: str = "sdpa",
    ):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        b = attn_backend.lower()
        if b not in self._VALID_BACKENDS:
            raise ValueError(
                f"attn_backend must be one of {sorted(self._VALID_BACKENDS)}, got {attn_backend!r}"
            )
        if b == "flash" and flash_attn_func is None:
            raise ImportError(
                "attn_backend='flash' requires flash-attn: pip install flash-attn"
            )

        self.n_head = n_head
        self.d_k = d_key // n_head
        self.attn_backend = b

        self.fc_query = nn.Linear(d_model, d_key, bias=False)
        self.fc_key = nn.Linear(d_model, d_key, bias=False)
        self.fc_value = nn.Linear(d_model, d_key, bias=False)
        self.fc_out = nn.Linear(d_key, d_model, bias=False)

        self.attn_drop_p = dropout
        self.out_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_src: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         ``(B, N, d_model)`` — queries.
            kv_src:    ``(B, N, d_model)`` or ``None`` for self-attention.
            attn_bias: optional additive bias broadcastable to ``(B, H, N, N)``.
                       Used by the SDPA backend; ``flash``/``linear`` fall back
                       to SDPA when this is non-None.
            edge_index: ``(2, E)`` integer tensor. **Required** when
                        ``attn_backend='sparse'``. Shared across batch.
            num_nodes: int, defaults to ``x.shape[1]``. Must equal ``N`` for
                       the sparse backend.
        """
        B, N, _ = x.shape
        kv = kv_src if kv_src is not None else x

        q = self.fc_query(x).view(B, N, self.n_head, self.d_k)
        k = self.fc_key(kv).view(B, N, self.n_head, self.d_k)
        v = self.fc_value(kv).view(B, N, self.n_head, self.d_k)
        return self.attend_with_kv(
            q, k, v,
            attn_bias=attn_bias,
            edge_index=edge_index,
            num_nodes=num_nodes,
        )

    def attend_with_kv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """Attention + output projection from pre-projected Q/K/V.

        Args:
            q, k, v: ``(B, N, n_head, d_k)`` — must match this module's head layout.
        """
        Bq, Nq, H, Dk = q.shape
        Bk, Nk, Hk, Dk2 = k.shape
        if v.shape != k.shape:
            raise ValueError(
                "attend_with_kv: K/V must match, got "
                f"{tuple(k.shape)} vs {tuple(v.shape)}"
            )
        if Bq != Bk or H != Hk or Dk != Dk2:
            raise ValueError(
                "attend_with_kv: Q/K batch or head layout mismatch: "
                f"{tuple(q.shape)} vs {tuple(k.shape)}"
            )
        if H != self.n_head or Dk != self.d_k:
            raise ValueError(
                f"attend_with_kv: tensor has (n_head={H}, d_k={Dk}) but module "
                f"expects (n_head={self.n_head}, d_k={self.d_k})"
            )
        dp = self.attn_drop_p if self.training else 0.0
        B, N = Bq, Nq

        # ---- Sparse backend (CellNavi-style) -----------------------------
        if self.attn_backend == "sparse":
            if Nq != Nk:
                raise ValueError(
                    "attend_with_kv: sparse backend requires Q.len == KV.len, "
                    f"got Nq={Nq} Nk={Nk}"
                )
            if edge_index is None:
                raise ValueError(
                    "attn_backend='sparse' requires edge_index of shape (2, E)."
                )
            if attn_bias is not None:
                warnings.warn(
                    "attn_bias is ignored when attn_backend='sparse' "
                    "(edge_index already defines the allowed pairs).",
                    stacklevel=2,
                )
            scale = float(self.d_k) ** 0.5
            nn_ = int(num_nodes) if num_nodes is not None else N
            if nn_ != N:
                raise ValueError(
                    f"num_nodes={nn_} != sequence length N={N}"
                )
            out = _sparse_attention(q, k, v, edge_index, nn_, scale)
            out = out.reshape(B, N, -1)
            return self.out_drop(self.fc_out(out))

        # ---- Dense backends ----------------------------------------------
        use_flash = (
            attn_bias is None
            and self.attn_backend == "flash"
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
            and Nq == Nk
        )
        use_linear = (
            self.attn_backend == "linear" and attn_bias is None and Nq == Nk
        )

        if use_flash:
            out = flash_attn_func(
                q, k, v,
                dropout_p=dp,
                softmax_scale=None,
                causal=False,
            )
        elif use_linear:
            qh = q.transpose(1, 2)
            kh = k.transpose(1, 2)
            vh = v.transpose(1, 2)
            out = _linear_attention(qh, kh, vh).transpose(1, 2)
        else:
            qh = q.transpose(1, 2)
            kh = k.transpose(1, 2)
            vh = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                qh, kh, vh,
                attn_mask=attn_bias,
                dropout_p=dp,
                is_causal=False,
            )
            out = out.transpose(1, 2)

        out = out.reshape(B, N, -1)
        return self.out_drop(self.fc_out(out))


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ff(x)


# ---------------------------------------------------------------------------
# Re-exports for legacy imports
# ---------------------------------------------------------------------------

__all__ = ["MultiHeadAttention", "FeedForward"]
