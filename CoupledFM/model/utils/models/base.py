"""Abstract velocity field interface (control → GT flow; no IR)."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class VelocityFieldBase(nn.Module, ABC):
    """Raw-expression velocity field v(x_t, x_ctrl, t, ...).

    Subclasses (rawexprFM / CoupledFM) implement ``forward``.

    Args (all forward passes):
        x_t: current state (B, G)
        x_ctrl: paired control expression (B, G)
        t: flow time (B,)
        gene_ids: vocabulary token ids (G,) or (B, G)
        cond_vec: optional condition vector (B, D_cond); placeholder zeros until designed
        aux_emb: optional latent guidance (B, D_latent); CoupledFM only
        attn_bias: optional (B, H, N, N) or broadcastable additive bias for graph / NicheNet
    """

    @abstractmethod
    def forward(
        self,
        x_t: torch.Tensor,
        x_ctrl: torch.Tensor,
        t: torch.Tensor,
        gene_ids: torch.Tensor,
        cond_vec: Optional[torch.Tensor] = None,
        aux_emb: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return velocity (B, G)."""
        raise NotImplementedError
