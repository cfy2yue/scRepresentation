"""
Model B: Latent DiffPerceiver velocity field.

Adapts scDFM's DiffPerceiverBlock to operate in latent embedding space.

Architecture (following scDFM):
  x_t  (B, D)  -> Tokenizer_xt  -> (B, K, d)
  x_ir (B, D)  -> Tokenizer_ir  -> (B, K, d)
  Fusion: cat(tok_xt, tok_ir) -> MLP -> (B, K, d)  <- y (contains both x_t and IR)
  t    (B,)    -> TimestepEmb   -> (B, d)

  N x DiffPerceiverBlock:
      DiffSelfAttn(y, y, t_emb)  + adaLN
      DiffCrossAttn(y, ir, t_emb) + adaLN
  Detokenizer: (B, K, d) -> flatten -> Linear -> (B, D)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ======================= Building blocks =======================


def _lambda_init(depth: int) -> float:
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


class MultiheadDiffAttn(nn.Module):
    """Differential attention (no rotary), matching scDFM's implementation.

    cross=True:  Q1/K1 from noisy_y, Q2/K2 from x  (used for self-attn in DiffPerceiverBlock)
    cross=False: Q1/K1 from noisy_y, Q2/K2 from x  (used for cross-attn in DiffPerceiverBlock)
    V always from noisy_y.
    """

    def __init__(self, embed_dim: int, num_heads: int, depth: int, cross: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.cross = cross

        self.q_proj_1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj_1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj_2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj_2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = _lambda_init(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(0, 0.1))

        self.subln = RMSNorm(self.head_dim)

    def forward(self, noisy_y: Tensor, x: Tensor) -> Tensor:
        B, T, _ = noisy_y.size()
        S = x.size(1)

        if self.cross:
            q1 = self.q_proj_1(noisy_y)
            k1 = self.k_proj_1(x)
            q2 = self.q_proj_2(noisy_y)
            k2 = self.k_proj_2(x)
        else:
            q1 = self.q_proj_1(noisy_y)
            k1 = self.k_proj_1(noisy_y)
            q2 = self.q_proj_2(x)
            k2 = self.k_proj_2(x)

        v = self.v_proj(noisy_y)

        def _reshape(t, seq_len):
            return t.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.cross:
            q1 = _reshape(q1, T) * self.scaling
            k1 = _reshape(k1, S)
            q2 = _reshape(q2, T) * self.scaling
            k2 = _reshape(k2, S)
        else:
            q1 = _reshape(q1, T) * self.scaling
            k1 = _reshape(k1, T)
            q2 = _reshape(q2, S) * self.scaling
            k2 = _reshape(k2, S)

        v = _reshape(v, T)

        a1 = torch.softmax(q1 @ k1.transpose(-1, -2), dim=-1, dtype=torch.float32).type_as(q1)
        a2 = torch.softmax(q2 @ k2.transpose(-1, -2), dim=-1, dtype=torch.float32).type_as(q1)

        lam1 = torch.exp((self.lambda_q1 * self.lambda_k1).sum().float()).type_as(q1)
        lam2 = torch.exp((self.lambda_q2 * self.lambda_k2).sum().float()).type_as(q1)
        lam = lam1 - lam2 + self.lambda_init

        attn = (a1 - lam * a2) @ v
        attn = self.subln(attn)
        attn = attn * (1.0 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(B, T, self.embed_dim)
        return self.out_proj(attn)


def _modulate(x, shift, scale):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiffTransformerBlock(nn.Module):
    """DiffAttn + adaLN + MLP (matching scDFM's DifferentialTransformerBlock)."""

    def __init__(self, d_model: int, n_heads: int, depth: int, mlp_ratio: float = 4.0, cross: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = MultiheadDiffAttn(d_model, n_heads, depth, cross=cross)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, d_model),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )

    def forward(self, y: Tensor, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        y = y + gate_msa.unsqueeze(1) * self.attn(
            _modulate(self.norm1(y), shift_msa, scale_msa), x)
        y = y + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(self.norm2(y), shift_mlp, scale_mlp))
        return y


class DiffPerceiverBlock(nn.Module):
    """DiffSelfAttn + DiffCrossAttn (matching scDFM's DiffPerceiverBlock)."""

    def __init__(self, d_model: int, n_heads: int, depth: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.diff_self_attn = DiffTransformerBlock(d_model, n_heads, depth, mlp_ratio, cross=True)
        self.diff_cross_attn = DiffTransformerBlock(d_model, n_heads, depth, mlp_ratio, cross=False)

    def forward(self, y: Tensor, x: Tensor, c: Tensor) -> Tensor:
        y = self.diff_self_attn(y, y, c)
        y = self.diff_cross_attn(y, x, c)
        return y


# ======================= Timestep embedder =======================

class TimestepEmbedder(nn.Module):
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


# ======================= Main model =======================

class LatentDiffPerceiver(nn.Module):
    """Latent DiffPerceiver velocity field v_theta(x_t, t, x_ir) -> v.

    Key design following scDFM:
      1) Tokenize x_t and x_ir into K virtual tokens each
      2) Fuse: cat(tok_xt, tok_ir) -> fusion MLP -> y  (y already contains IR info)
      3) DiffPerceiverBlocks: self-attn(y,y) + cross-attn(y, ir_tokens)
      4) Detokenize back to emb_dim
    """

    def __init__(
        self,
        emb_dim: int = 2058,
        n_tokens: int = 8,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model

        self.tok_xt = nn.Linear(emb_dim, n_tokens * d_model)
        self.tok_ir = nn.Linear(emb_dim, n_tokens * d_model)

        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.t_embed = TimestepEmbedder(d_model)

        self.blocks = nn.ModuleList(
            [DiffPerceiverBlock(d_model, n_heads, i, mlp_ratio) for i in range(n_layers)]
        )

        self.out_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.detok = nn.Linear(n_tokens * d_model, emb_dim)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.zeros_(self.detok.weight)
        nn.init.zeros_(self.detok.bias)

    def forward(self, x_t: Tensor, t: Tensor, x_src: Tensor) -> Tensor:
        B = x_t.size(0)
        yt = self.tok_xt(x_t).view(B, self.n_tokens, self.d_model)
        h_src = self.tok_ir(x_src).view(B, self.n_tokens, self.d_model)

        y = self.fusion_layer(torch.cat([yt, h_src], dim=-1))

        c = self.t_embed(t)

        for block in self.blocks:
            y = block(y, h_src, c)

        y = self.out_norm(y)
        y = y.reshape(B, self.n_tokens * self.d_model)
        return self.detok(y)
