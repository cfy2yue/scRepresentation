"""Pretrain-only direction / gene-set adapter (discarded after pretraining)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from model.raw_pretrain.data_source import PADDING_GENE_TOKEN


class PretrainDirectionAdapter(nn.Module):
    """Pool perturbed-gene tokens into a ``cond_vec`` for ``RawExprVelocityField``.

    Shares ``embed_gene`` weights with the velocity field (caller passes the module).
    """

    def __init__(
        self,
        gene_emb: nn.Module,
        d_model: int,
        *,
        d_cond: int = 128,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gene_emb = gene_emb
        self.d_model = int(d_model)
        self.d_cond = int(d_cond)
        self.up_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.down_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.up_token, std=0.02)
        nn.init.normal_(self.down_token, std=0.02)
        self.mag_proj = nn.Linear(1, d_model)
        nn.init.zeros_(self.mag_proj.weight)
        nn.init.zeros_(self.mag_proj.bias)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.query, std=0.02)
        self.out = nn.Linear(d_model, d_cond)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        gene_ids: Tensor,
        signs: Tensor,
        mags: Tensor,
        pert_mask: Tensor,
    ) -> Tensor:
        """
        gene_ids: (B, K) cellnavi indices; -1 / invalid → padding token.
        signs:    (B, K) +1 / -1 / 0 (pad)
        mags:     (B, K)
        pert_mask:(B, K) 1 = valid pert slot
        """
        b, k = gene_ids.shape
        gid = gene_ids.long()
        gid = torch.where(gid < 0, torch.full_like(gid, PADDING_GENE_TOKEN), gid)
        gid = torch.where(gid > 40001, torch.full_like(gid, PADDING_GENE_TOKEN), gid)
        h = self.gene_emb(gid).to(dtype=self.query.dtype)
        s = signs.unsqueeze(-1)
        dir_tok = torch.where(s > 0, self.up_token, self.down_token)
        h = h + dir_tok + self.mag_proj(mags.unsqueeze(-1))
        key_padding = pert_mask <= 0
        q = self.query.expand(b, -1, -1).to(dtype=h.dtype)
        attn_out, _ = self.attn(q, h, h, key_padding_mask=key_padding)
        return self.out(attn_out.squeeze(1))


__all__ = ["PretrainDirectionAdapter"]
