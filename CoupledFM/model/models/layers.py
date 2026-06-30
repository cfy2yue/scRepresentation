"""
Building-block layers for the raw-expression flow matching model.

- TimestepEmbedder: sinusoidal + MLP  (from DiT / latent FM)
- GeneadaLN:        node-level adaptive LayerNorm  (from scDFM), with optional
                    latent conditioning for gene-specific latent modulation
- ContinuousValueEncoder: scalar expression -> d_model  (replaces CellNavi embed_exp)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep -> MLP  (DiT style)."""

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_dim = freq_dim

    @staticmethod
    def sinusoidal(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.sinusoidal(t, self.freq_dim))


class GeneadaLN(nn.Module):
    """Node-level adaptive LayerNorm with optional latent conditioning.

    Each gene position gets its own shift / scale / gate derived from
    the gene identity embedding.  When ``use_latent=True``, a projected
    latent vector is concatenated with the gene embedding so that each
    gene receives modulation that depends on *both* its identity and
    the current global latent state — enabling gene-specific responses
    to latent guidance.

    ``gene_emb`` is constant across layers; ``value_emb`` evolves.
    """

    def __init__(self, d_model: int, use_latent: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.use_latent = use_latent
        cond_dim = d_model * 2 if use_latent else d_model
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * d_model, bias=True),
        )

    def forward(
        self,
        gene_emb: Tensor,
        value_emb: Tensor,
        z_proj: Optional[Tensor] = None,
    ) -> Tensor:
        """
        gene_emb:  (1, G, d) or (B, G, d) — gene identity (broadcasts)
        value_emb: (B, G, d)              — current representation
        z_proj:    (B, d) or None          — projected latent for gene-specific modulation
        """
        if self.use_latent:
            B, G, d = value_emb.shape
            gene_exp = gene_emb.expand(B, G, d)
            if z_proj is not None:
                z_exp = z_proj.unsqueeze(1).expand(B, G, d)
            else:
                z_exp = gene_exp.new_zeros(B, G, d)
            cond = torch.cat([gene_exp, z_exp], dim=-1)  # (B, G, 2d)
        else:
            cond = gene_emb
        shift, scale, gate = self.adaLN_modulation(cond).chunk(3, dim=-1)
        return value_emb + gate * (self.norm(value_emb) * (1.0 + scale) + shift)


class ContinuousValueEncoder(nn.Module):
    """Maps a scalar expression value to *d_model* dimensions.

    Replaces CellNavi's ``embed_exp`` + ``embed_rawcount`` (which required
    raw UMI counts).  Works directly on log1p-normalised expression.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        lin1 = nn.Linear(2, d_model)
        nn.init.zeros_(lin1.weight[:, 1:2])
        self.enc = nn.Sequential(
            lin1,
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (N,), (N, 1) expression, or (N, 2) [value, visibility] -> (N, d_model)"""
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.size(-1) == 1:
            x = torch.cat([x, torch.ones_like(x)], dim=-1)
        return self.enc(x)


class FourierValueEncoder(nn.Module):
    """Random Fourier features on ``[value, visibility]`` → ``d_model``."""

    def __init__(self, d_model: int, n_freqs: int = 32, dropout: float = 0.1):
        super().__init__()
        self.n_freqs = int(n_freqs)
        lin = nn.Linear(2, self.n_freqs, bias=False)
        nn.init.normal_(lin.weight, std=1.0)
        self.proj = lin
        self.enc = nn.Sequential(
            nn.Linear(2 * self.n_freqs, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.size(-1) == 1:
            x = torch.cat([x, torch.ones_like(x)], dim=-1)
        z = self.proj(x)
        f = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
        return self.enc(f)
