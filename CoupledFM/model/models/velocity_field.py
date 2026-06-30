"""
Raw-expression flow matching velocity field.

Architecture per FlowTransformerLayer:
  1. Latent-conditioned GeneadaLN — gene-specific modulation from
     (gene_identity, z_proj), enabling per-gene latent sensitivity
  2. adaLN-Zero  — sample-level conditioning from fused (t, z_t)
  3. DiffAttn    — SelfAttn(h_t) - lambda * CrossAttn(h_t, h_ctrl)
  4. adaLN-Zero  — sample-level conditioning
  5. FFN         — pretrained feed-forward

Latent guidance:
  - z_t (2058-dim) is injected via three pathways:
    (a) CLS gated injection: bottleneck MLP → gated content into h_t's CLS
    (b) adaLN conditioning: t projected UP to 2058-dim, fused with z_t in
        latent space, then projected down to d_model (residual zero-init)
    (c) GeneadaLN: z_t projected → d_model, concatenated with gene_emb
        so each gene gets modulation dependent on BOTH identity and latent

All attention Q/K/V/out and FFN weights are loaded from CellNavi pretrained
checkpoint.  adaLN / GeneadaLN / lambda / output-proj are trained from scratch.
"""

import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as ckpt_fn

from .attention import MultiHeadAttention, FeedForward
from .layers import (
    TimestepEmbedder,
    GeneadaLN,
    ContinuousValueEncoder,
    FourierValueEncoder,
)
from model.condition_emb.genepert.perturbation import unpack_perturbation_tuple
from model.condition_emb.genepert.perturbation_encoder import (
    PerturbationConditionEncoder,
    UnifiedConditionEncoder,
)
from model.condition_emb.genepert.gene_cache import GeneEmbeddingCache


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale) + shift


class LatentResampler(nn.Module):
    """Cross-attention from learnable queries to a single latent KV token."""

    def __init__(
        self,
        d_latent: int,
        d_model: int,
        n_tokens: int = 8,
        n_head: int = 4,
        dropout: float = 0.1,
        attn_backend: str = "sdpa",
    ):
        super().__init__()
        del attn_backend  # Q_len != KV_len; only SDPA supports this cross-attn shape
        self.kv_proj = nn.Linear(d_latent, d_model)
        nn.init.zeros_(self.kv_proj.weight)
        nn.init.zeros_(self.kv_proj.bias)
        self.query = nn.Parameter(torch.zeros(n_tokens, d_model))
        nn.init.normal_(self.query, std=0.02)
        self.attn = MultiHeadAttention(
            d_model, d_model, n_head, dropout, attn_backend="sdpa",
        )

    def forward(self, aux_emb: Tensor) -> Tensor:
        B = aux_emb.shape[0]
        n_tok = self.query.shape[0]
        q_in = self.query.unsqueeze(0).expand(B, -1, -1)
        kv = self.kv_proj(aux_emb).unsqueeze(1)
        q = self.attn.fc_query(q_in).view(B, n_tok, self.attn.n_head, self.attn.d_k)
        k = self.attn.fc_key(kv).view(B, 1, self.attn.n_head, self.attn.d_k)
        v = self.attn.fc_value(kv).view(B, 1, self.attn.n_head, self.attn.d_k)
        out = self.attn.attend_with_kv(q, k, v)
        return out.mean(dim=1)


class FlowTransformerLayer(nn.Module):
    """Single layer: GeneadaLN -> DiffAttn -> adaLN-Zero -> FFN.

    All tensors are batched: (B, N, d).
    """

    def __init__(
        self,
        d_model: int = 256,
        d_key: int = 256,
        n_head: int = 16,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        layer_idx: int = 0,
        use_latent: bool = False,
        attn_backend: str = "sdpa",
        cross_attn_independent_kv: bool = False,
    ):
        super().__init__()

        self.attn = MultiHeadAttention(
            d_model, d_key, n_head, dropout, attn_backend=attn_backend,
        )
        self.cross_attn_independent_kv = cross_attn_independent_kv
        self.attn_cross_kv = None
        if cross_attn_independent_kv:
            self.attn_cross_kv = nn.ModuleDict({
                "fc_key": nn.Linear(d_model, d_key, bias=False),
                "fc_value": nn.Linear(d_model, d_key, bias=False),
            })
            with torch.no_grad():
                self.attn_cross_kv["fc_key"].weight.copy_(self.attn.fc_key.weight)
                self.attn_cross_kv["fc_value"].weight.copy_(self.attn.fc_value.weight)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)

        self.gene_adaln = GeneadaLN(d_model, use_latent=use_latent)

        self.norm_attn = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm_ffn = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        self.adaln_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )

        init_val = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self.diff_lambda = nn.Parameter(torch.tensor(init_val))

        self.use_diff = True

    def forward(
        self,
        h_t: Tensor,
        h_ctrl: Tensor,
        c_vec: Tensor,
        gene_emb: Tensor,
        cls_mask: Tensor,
        z_proj: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            h_t:        (B, N, d)  evolving stream
            h_ctrl:     (B, N, d)  fixed control reference
            c_vec:      (B, d)  conditioning (time+latent fused); broadcast to N
            gene_emb:   (1, N, d)  gene identity embedding (broadcasts over B)
            cls_mask:   (N,) bool — True for gene nodes, False for CLS
            z_proj:     (B, d) or None — projected latent for GeneadaLN
            attn_bias:  optional (B, H, N, N) additive bias (SDPA backend).
            edge_index: optional (2, E) sparse graph (sparse backend). Shared
                        across batch; same prior is applied to self- and
                        cross-attention legs of the diff attention.
        """
        # ---- GeneadaLN (gene nodes only, latent-conditioned) -------------
        h_genes = self.gene_adaln(
            gene_emb[:, cls_mask], h_t[:, cls_mask], z_proj=z_proj,
        )
        B, N, D = h_t.shape
        gene_slots = torch.zeros(B, N, D, device=h_t.device, dtype=h_t.dtype)
        gene_slots[:, cls_mask] = h_genes
        mask3 = cls_mask.view(1, N, 1).to(device=h_t.device)
        h_t = torch.where(mask3, gene_slots, h_t)

        # ---- adaLN-Zero modulation from c_vec (single Linear on (B,d)) ----
        _six = self.adaln_mod(c_vec).chunk(6, dim=-1)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = [
            t.unsqueeze(1) for t in _six
        ]

        # ---- DiffAttn: SelfAttn - lambda * CrossAttn --------------------
        h_normed = _modulate(self.norm_attn(h_t), shift_a, scale_a)

        self_out = self.attn(
            h_normed, attn_bias=attn_bias, edge_index=edge_index,
        )

        if self.use_diff:
            h_ctrl_normed = self.norm_attn(h_ctrl)
            if self.cross_attn_independent_kv and self.attn_cross_kv is not None:
                q = self.attn.fc_query(h_normed).view(
                    B, N, self.attn.n_head, self.attn.d_k,
                )
                k = self.attn_cross_kv["fc_key"](h_ctrl_normed).view(
                    B, N, self.attn.n_head, self.attn.d_k,
                )
                v = self.attn_cross_kv["fc_value"](h_ctrl_normed).view(
                    B, N, self.attn.n_head, self.attn.d_k,
                )
                cross_out = self.attn.attend_with_kv(
                    q, k, v,
                    attn_bias=attn_bias,
                    edge_index=edge_index,
                )
            else:
                cross_out = self.attn(
                    h_normed, kv_src=h_ctrl_normed,
                    attn_bias=attn_bias, edge_index=edge_index,
                )
            attn_out = self_out - self.diff_lambda * cross_out
        else:
            attn_out = self_out

        h_t = h_t + gate_a * attn_out

        # ---- FFN with adaLN-Zero ----------------------------------------
        h_normed = _modulate(self.norm_ffn(h_t), shift_f, scale_f)
        h_t = h_t + gate_f * self.ffn(h_normed)

        return h_t


class RawExprVelocityField(nn.Module):
    """v_theta(x_t, t, x_ctrl) -> per-gene velocity (global attention).

    Loads CellNavi pretrained ``embed_gene`` + 6-layer encoder (attention + FFN).
    Adds ContinuousValueEncoder, TimestepEmbedder, GeneadaLN, adaLN-Zero,
    diff-lambda, and output projection.
    """

    CLS_TOKEN_ID = 40001

    def __init__(
        self,
        d_model: int = 256,
        n_layer: int = 6,
        n_head: int = 16,
        d_ff: int = 1024,
        dropout: float = 0.1,
        attn_mode: str = "diff",
        d_latent: int = 2058,
        grad_ckpt: bool = True,
        attn_backend: str = "sdpa",
        coupling_mode: str = "ot",
        use_pert_token: bool = False,
        num_pert_ids: int = 10000,
        graph_bias_mode: str = "none",
        use_latent_resampler: bool = False,
        latent_resampler_n_tokens: int = 8,
        latent_resampler_n_head: int = 4,
        cross_attn_independent_kv: bool = False,
        value_encoder: str = "linear",
        fourier_n_freqs: int = 32,
        use_residual_flow: bool = False,
        use_pert_condition: bool = False,
        pert_embed_mode: str = "random_learned",
        pert_cond_dim: int = 512,
        pert_type_emb_dim: int = 32,
        pert_encoder_num_embeddings: int = 8192,
        pert_gene_emb_dim: int = 256,
        pert_encoder_dropout: float = 0.0,
        max_combo_id_exclusive: int = 4096,
        gene_embedding_cache: Optional[GeneEmbeddingCache] = None,
        legacy_cond_vec_dim: int = 0,
        pert_chem_emb_dim: int = 512,
        pert_chem_projector_hidden: int = 0,
        pert_gene_projector_hidden: int = 0,
        pert_type_scale_init: Optional[Tuple[float, ...]] = None,
        pool_aggregations: Tuple[str, ...] = ("mean",),
        pool_scale_init: Tuple[float, ...] = (1.0,),
        pool_fusion_mode: str = "sum",
        type_adapter_mode: str = "scalar",
        condition_embedding_source: Optional[str] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.attn_mode = attn_mode
        self.grad_ckpt = grad_ckpt
        self.attn_backend = attn_backend
        self.coupling_mode = coupling_mode
        self.use_residual_flow = bool(use_residual_flow)
        self.use_pert_condition = bool(use_pert_condition)
        self.graph_bias_mode = (graph_bias_mode or "none").lower()
        self.graph_bias_alpha = nn.Parameter(torch.tensor(0.0))

        ldc = int(legacy_cond_vec_dim)
        self.legacy_cond_vec_dim = ldc
        self.cond_vec_proj: Optional[nn.Linear] = None
        if ldc > 0:
            self.cond_vec_proj = nn.Linear(ldc, d_model)

        self.embed_gene = nn.Embedding(40002, d_model, padding_idx=40000)

        ve = (value_encoder or "linear").lower()
        if ve == "fourier":
            self.value_encoder = FourierValueEncoder(
                d_model, n_freqs=fourier_n_freqs, dropout=dropout,
            )
        else:
            self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        self.t_embed = TimestepEmbedder(d_model)

        self.pert_token = None
        if use_pert_token:
            self.pert_token = nn.Embedding(num_pert_ids, d_model)
            nn.init.normal_(self.pert_token.weight, std=0.02)

        # ── latent injection modules ──────────────────────────────────
        d_mid = max(d_model * 4, 1024)
        self.use_latent_resampler = bool(use_latent_resampler)
        self.pert_encoder: Optional[Union[PerturbationConditionEncoder, UnifiedConditionEncoder]] = None
        self.pert_to_c = nn.Identity()
        if self.use_pert_condition:
            mode = (pert_embed_mode or "random_learned").lower().strip()
            cache = gene_embedding_cache
            if mode.startswith("pretrained") and cache is None:
                raise ValueError(
                    "RawExprVelocityField(use_pert_condition=True, pretrained*): "
                    "gene_embedding_cache must be set (matches training cache / lookup)"
                )
            pcd = int(pert_chem_emb_dim)
            pchem_h = int(pert_chem_projector_hidden)
            gph = int(pert_gene_projector_hidden)
            pti = tuple(pert_type_scale_init) if pert_type_scale_init is not None else (
                0.0, -1.0, -1.0, -1.0, 1.0, 1.0
            )
            if mode == "combo_id_baseline":
                self.pert_encoder = PerturbationConditionEncoder(
                    mode,
                    out_dim=int(pert_cond_dim),
                    cache=cache,
                    num_embeddings_random=int(pert_encoder_num_embeddings),
                    embed_dim_random=int(pert_gene_emb_dim),
                    max_combo_id_exclusive=int(max_combo_id_exclusive),
                    type_embed_dim=int(pert_type_emb_dim),
                    dropout_p=float(pert_encoder_dropout),
                    chem_emb_dim=(None if pcd <= 0 else pcd),
                    chem_projector_hidden=(None if pchem_h <= 0 else pchem_h),
                )
            else:
                self.pert_encoder = UnifiedConditionEncoder(
                    mode,
                    out_dim=int(pert_cond_dim),
                    cache=cache,
                    num_embeddings_random=int(pert_encoder_num_embeddings),
                    embed_dim_random=int(pert_gene_emb_dim),
                    gene_projector_hidden=gph,
                    chem_emb_dim=(None if pcd <= 0 else pcd),
                    chem_projector_hidden=(None if pchem_h <= 0 else pchem_h),
                    dropout_p=float(pert_encoder_dropout),
                    pert_type_scale_init=pti,
                    pool_aggregations=tuple(pool_aggregations),
                    pool_scale_init=tuple(pool_scale_init),
                    pool_fusion_mode=str(pool_fusion_mode).lower().strip(),
                    type_adapter_mode=str(type_adapter_mode).lower().strip(),
                    condition_embedding_source=condition_embedding_source,
                )
            pc = int(pert_cond_dim)
            self.pert_to_c = nn.Linear(pc, d_model) if pc != d_model else nn.Identity()

        if self.use_latent_resampler:
            self.latent_cls_mlp = None
            self.latent_resampler = LatentResampler(
                d_latent=d_latent,
                d_model=d_model,
                n_tokens=latent_resampler_n_tokens,
                n_head=latent_resampler_n_head,
                dropout=dropout,
                attn_backend=attn_backend,
            )
        else:
            self.latent_resampler = None
            self.latent_cls_mlp = nn.Sequential(
                nn.Linear(d_latent, d_mid),
                nn.SiLU(),
                nn.Linear(d_mid, d_model),
            )
        self.cls_gate = nn.Sequential(
            nn.Linear(d_latent, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # adaLN conditioning: fuse t and z_t in the LATENT dimension (2058),
        # then project down.  Residual: c_vec = t_emb + fuse(t_up + z_t),
        # with fuse zero-initialised so the model starts with c_vec ≈ t_emb.
        self.t_up = nn.Sequential(
            nn.Linear(d_model, d_latent),
            nn.SiLU(),
        )
        self.cond_fuse = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.SiLU(),
            nn.Linear(d_latent, d_model),
        )

        # Gene-level latent projection for Latent-conditioned GeneadaLN.
        # z_gene_proj: (B, d_latent) → (B, d_model).
        # Zero-init last layer so GeneadaLN starts with gene_emb-only behaviour.
        self.z_gene_proj = nn.Sequential(
            nn.Linear(d_latent, d_mid),
            nn.SiLU(),
            nn.Linear(d_mid, d_model),
        )

        # ── transformer layers ────────────────────────────────────────
        use_latent_gene = coupling_mode == "coupled"
        self.layers = nn.ModuleList([
            FlowTransformerLayer(
                d_model=d_model,
                d_key=d_model,
                n_head=n_head,
                dim_feedforward=d_ff,
                dropout=dropout,
                layer_idx=i,
                use_latent=use_latent_gene,
                attn_backend=attn_backend,
                cross_attn_independent_kv=cross_attn_independent_kv,
            )
            for i in range(n_layer)
        ])

        if attn_mode == "self_only":
            for layer in self.layers:
                layer.use_diff = False

        self.out_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.out_proj = nn.Linear(d_model, 1)

        self._init_new_weights()

    def _init_new_weights(self):
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        for layer in self.layers:
            nn.init.zeros_(layer.adaln_mod[-1].weight)
            nn.init.zeros_(layer.adaln_mod[-1].bias)
        nn.init.zeros_(self.cls_gate[-2].bias)
        # Residual zero-init: latent contribution starts at 0
        nn.init.zeros_(self.cond_fuse[-1].weight)
        nn.init.zeros_(self.cond_fuse[-1].bias)
        nn.init.zeros_(self.z_gene_proj[-1].weight)
        nn.init.zeros_(self.z_gene_proj[-1].bias)
        if (
            self.use_pert_condition
            and self.pert_encoder is not None
            and isinstance(self.pert_to_c, nn.Linear)
        ):
            nn.init.zeros_(self.pert_to_c.weight)
            nn.init.zeros_(self.pert_to_c.bias)
        if self.cond_vec_proj is not None:
            nn.init.zeros_(self.cond_vec_proj.weight)
            nn.init.zeros_(self.cond_vec_proj.bias)
        if self.latent_cls_mlp is not None:
            nn.init.zeros_(self.latent_cls_mlp[-1].weight)
            nn.init.zeros_(self.latent_cls_mlp[-1].bias)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        x_t: Tensor,
        x_ctrl: Tensor,
        t: Tensor,
        gene_ids: Tensor,
        aux_emb: Optional[Tensor] = None,
        gene_mask: Optional[Tensor] = None,
        pert_idx: Optional[Tensor] = None,
        cond_vec: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        perturbation_batch: Optional[Tuple[Tensor, ...]] = None,
        *,
        pert_gene_ids: Optional[Tensor] = None,
        pert_mask: Optional[Tensor] = None,
        pert_type_id: Optional[Tensor] = None,
        nperts: Optional[Tensor] = None,
        combo_id: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            cond_vec:   optional (B, D) legacy continuous condition when
                        ``legacy_cond_vec_dim=D`` matches shape; otherwise warning.
            perturbation_batch / pert_* : optional unified perturbation tensors.
        Returns:
            v: (B, G)  predicted per-gene velocity
        """
        if perturbation_batch is not None:
            pert_gene_ids, pert_mask, pert_type_id, nperts, combo_id, chem_emb, chem_mask = (
                unpack_perturbation_tuple(perturbation_batch)
            )
        else:
            chem_emb, chem_mask = None, None

        cond_vec_addon: Optional[Tensor] = None
        if cond_vec is not None:
            if self.cond_vec_proj is not None:
                dexp = self.legacy_cond_vec_dim
                if cond_vec.shape[-1] != dexp:
                    raise ValueError(
                        f"cond_vec last dim {cond_vec.shape[-1]} "
                        f"!= legacy_cond_vec_dim {dexp}"
                    )
                cond_vec_addon = self.cond_vec_proj(
                    cond_vec.to(device=x_t.device, dtype=torch.float32)
                )
            else:
                warnings.warn(
                    "RawExprVelocityField: cond_vec unused (legacy_cond_vec_dim=0; "
                    "set legacy_cond_vec_dim / cond_vec_proj to project it).",
                    UserWarning,
                    stacklevel=2,
                )

        if self.use_pert_condition:
            if self.pert_encoder is None:
                raise RuntimeError("use_pert_condition but pert_encoder missing")
            pe = self.pert_encoder(
                pert_gene_ids=pert_gene_ids,
                pert_mask=pert_mask,
                pert_type_id=pert_type_id,
                nperts=nperts,
                combo_id=combo_id,
                chem_emb=chem_emb,
                chem_mask=chem_mask,
            )
        else:
            if any(
                x is not None
                for x in (
                    perturbation_batch,
                    pert_gene_ids,
                    pert_mask,
                    pert_type_id,
                    nperts,
                    combo_id,
                )
            ):
                raise ValueError(
                    "RawExprVelocityField(use_pert_condition=False): do not pass perturbation tensors."
                )
            pe = None
        B, G = x_t.shape
        device = x_t.device
        N = G + 1  # CLS + G genes

        # ---- gene embedding (pretrained, shared) -------------------------
        gene_emb = self.embed_gene(gene_ids)  # (G, d)

        # ---- CLS token (gated latent injection for h_t only) ---------------
        cls_id = torch.tensor([self.CLS_TOKEN_ID], dtype=torch.long, device=device)
        cls_emb = self.embed_gene(cls_id).squeeze(0)  # (d,)
        cls_ctrl_batch = cls_emb.unsqueeze(0).expand(B, -1)  # (B, d) plain

        use_latent_paths = (
            aux_emb is not None and self.coupling_mode == "coupled"
        )
        if use_latent_paths:
            gate = self.cls_gate(aux_emb)  # (B, 1)
            if self.latent_resampler is not None:
                cls_t_batch = (
                    cls_emb.unsqueeze(0) + gate * self.latent_resampler(aux_emb)
                )
            else:
                cls_t_batch = (
                    cls_emb.unsqueeze(0) + gate * self.latent_cls_mlp(aux_emb)
                )
        else:
            cls_t_batch = cls_ctrl_batch  # (B, d)

        # ---- encode x_t and x_ctrl: (B, G) -> (B, G, d) -----------------
        gene_emb_exp = gene_emb.unsqueeze(0).expand(B, -1, -1)  # (B, G, d)

        if gene_mask is None:
            vis_t = torch.ones_like(x_t)
        else:
            vis_t = 1.0 - gene_mask
        vals_t = x_t * vis_t
        h_t_in = torch.stack([vals_t, vis_t], dim=-1).reshape(-1, 2)
        h_t_genes = (
            self.value_encoder(h_t_in).view(B, G, self.d_model)
            + gene_emb_exp
        )
        vis_c = torch.ones_like(x_ctrl)
        h_ctrl_in = torch.stack([x_ctrl, vis_c], dim=-1).reshape(-1, 2)
        h_ctrl_genes = (
            self.value_encoder(h_ctrl_in).view(B, G, self.d_model)
            + gene_emb_exp
        )

        # ---- assemble [CLS, gene_0, ..., gene_{G-1}] → (B, N, d) --------
        h_t = torch.cat([cls_t_batch.unsqueeze(1), h_t_genes], dim=1)
        h_ctrl = torch.cat([cls_ctrl_batch.unsqueeze(1), h_ctrl_genes], dim=1)

        # ---- gene_emb with CLS placeholder → (1, N, d) ------------------
        gene_emb_with_cls = torch.cat(
            [torch.zeros(1, self.d_model, device=device), gene_emb], dim=0
        ).unsqueeze(0)  # (1, N, d)

        # ---- cls_mask: True for gene nodes, False for CLS → (N,) --------
        cls_mask = torch.ones(N, dtype=torch.bool, device=device)
        cls_mask[0] = False

        # ---- conditioning: fuse t + latent in high-dim space → (B, N, d) --
        t_emb = self.t_embed(t)  # (B, d)
        z_proj = None
        if use_latent_paths:
            t_high = self.t_up(t_emb)                     # (B, d_latent)
            latent_delta = self.cond_fuse(t_high + aux_emb)  # (B, d_model)
            c_vec = t_emb + latent_delta                   # residual start ≈ t_emb
            z_proj = self.z_gene_proj(aux_emb)             # (B, d_model) for GeneadaLN
        else:
            c_vec = t_emb
        if cond_vec_addon is not None:
            c_vec = c_vec + cond_vec_addon.to(dtype=c_vec.dtype)
        if self.pert_token is not None and pert_idx is not None:
            c_vec = c_vec + self.pert_token(pert_idx)
        if pe is not None:
            c_vec = c_vec + self.pert_to_c(pe.to(dtype=c_vec.dtype))

        attn_bias_eff = attn_bias
        if (
            self.graph_bias_mode == "sdpa_bias"
            and edge_index is not None
            and edge_index.numel() > 0
        ):
            ab = torch.zeros(1, 1, N, N, device=device, dtype=x_t.dtype)
            ab[0, 0, edge_index[0].long(), edge_index[1].long()] = (
                self.graph_bias_alpha.to(dtype=ab.dtype)
            )
            if attn_bias_eff is not None:
                attn_bias_eff = attn_bias_eff + ab
            else:
                attn_bias_eff = ab

        # ---- transformer layers -----------------------------------------
        for layer in self.layers:
            if self.grad_ckpt and self.training:
                h_t = ckpt_fn(
                    layer, h_t, h_ctrl, c_vec, gene_emb_with_cls,
                    cls_mask, z_proj, attn_bias_eff, edge_index,
                    use_reentrant=False,
                )
            else:
                h_t = layer(
                    h_t, h_ctrl, c_vec, gene_emb_with_cls, cls_mask,
                    z_proj=z_proj,
                    attn_bias=attn_bias_eff,
                    edge_index=edge_index,
                )

        # ---- output: per-gene velocity (exclude CLS) --------------------
        h_genes = h_t[:, 1:, :]  # (B, G, d)
        v = self.out_proj(self.out_norm(h_genes)).squeeze(-1)  # (B, G)
        return v

    # -----------------------------------------------------------------
    # Pretrained weight loading
    # -----------------------------------------------------------------

    def load_pretrained_weights(self, ckpt_path: str, verbose: bool = True):
        """Load CellNavi encoder weights into this model.

        Maps CellNavi's ``SparseCellNaviEncoder`` state dict:
          - embed_gene         -> self.embed_gene
          - encoder.layers[i].attn.*  -> self.layers[i].attn.*
          - encoder.layers[i].ffn.*   -> self.layers[i].ffn.*

        Skips: embed_exp, embed_rawcount, embed_ratio, norm1/norm2, decoder, fc.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        src_state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        key_map = {}

        for k in src_state:
            clean = k.replace("module.", "", 1) if k.startswith("module.") else k

            if clean == "embed_gene.0.weight":
                key_map[clean] = "embed_gene.weight"

            if clean.startswith("encoder.layers."):
                rest = clean[len("encoder."):]
                parts = rest.split(".", 2)
                if len(parts) < 3:
                    continue
                suffix = parts[2]
                layer_idx = parts[1]
                if suffix.startswith("attn."):
                    key_map[clean] = f"layers.{layer_idx}.attn.{suffix[5:]}"
                elif suffix.startswith("ffn."):
                    key_map[clean] = f"layers.{layer_idx}.ffn.{suffix[4:]}"

        loaded, skipped = [], []
        new_state = {}
        for src_key, dst_key in key_map.items():
            raw_key = src_key
            if raw_key not in src_state:
                raw_key = "module." + raw_key
            if raw_key not in src_state:
                skipped.append(src_key)
                continue
            if dst_key in self.state_dict():
                src_tensor = src_state[raw_key]
                dst_tensor = self.state_dict()[dst_key]
                if src_tensor.shape == dst_tensor.shape:
                    new_state[dst_key] = src_tensor
                    loaded.append(f"{src_key} -> {dst_key}")
                else:
                    skipped.append(
                        f"{src_key} shape mismatch "
                        f"{src_tensor.shape} vs {dst_tensor.shape}"
                    )
            else:
                skipped.append(f"{dst_key} not in model")

        info = self.load_state_dict(new_state, strict=False)

        if verbose:
            print(
                f"[load_pretrained] Loaded {len(loaded)} params, "
                f"skipped {len(skipped)}, "
                f"missing {len(info.missing_keys)}"
            )
            if skipped:
                print(
                    f"  Skipped: {skipped[:10]}"
                    f"{'...' if len(skipped) > 10 else ''}"
                )

        return {
            "loaded": loaded,
            "skipped": skipped,
            "missing": info.missing_keys,
            "unexpected": info.unexpected_keys,
        }

    def load_raw_pretrain_backbone(
        self,
        ckpt_path: str,
        strict: bool = False,
        verbose: bool = True,
    ):
        """Load a flat checkpoint from ``raw_pretrain`` (``torch.save(uv.state_dict())``).

        This is **not** CellNavi format; use :meth:`load_pretrained_weights` for
        ``pretrain_weights.pth``.
        """
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "model" in blob and isinstance(blob["model"], dict):
            # Likely main-training checkpoint fragment — caller should use full resume path.
            state = blob["model"]
        elif isinstance(blob, dict):
            state = blob
        else:
            raise ValueError(f"Unsupported checkpoint type at {ckpt_path!r}")
        info = self.load_state_dict(state, strict=strict)
        if verbose:
            print(
                f"[load_raw_pretrain_backbone] strict={strict} missing="
                f"{len(info.missing_keys)} unexpected={len(info.unexpected_keys)}",
                flush=True,
            )
            if info.missing_keys[:5]:
                print(f"  missing (head): {info.missing_keys[:5]}", flush=True)
            if info.unexpected_keys[:5]:
                print(f"  unexpected (head): {info.unexpected_keys[:5]}", flush=True)
        return {
            "kind": "raw_backbone",
            "missing": list(info.missing_keys),
            "unexpected": list(info.unexpected_keys),
        }


def load_velocity_pretrained_bundle(
    model: "RawExprVelocityField",
    path: str,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Pick the right loader for ``train.pretrained_ckpt``.

    Priority:
      1. Full training checkpoint (``optimizer`` + ``model`` keys) → ``ckpt['model']``.
      2. CellNavi ``pretrain_weights.pth`` (``encoder.layers.*`` / ``embed_gene.0.weight``)
         → :meth:`RawExprVelocityField.load_pretrained_weights`.
      3. Otherwise flat ``state_dict`` (e.g. ``raw_pretrain`` ``backbone.pt``) →
         :meth:`RawExprVelocityField.load_raw_pretrain_backbone`.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Expected dict checkpoint, got {type(ckpt)} from {path!r}")

    if "optimizer" in ckpt and "model" in ckpt and isinstance(ckpt["model"], dict):
        info = model.load_state_dict(ckpt["model"], strict=False)
        if verbose:
            print(
                f"[load_velocity_pretrained_bundle] loaded train-format ckpt['model'] "
                f"missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}",
                flush=True,
            )
        return {
            "kind": "train_pack",
            "missing": list(info.missing_keys),
            "unexpected": list(info.unexpected_keys),
        }

    # CellNavi vs raw flat
    src = ckpt.get("state_dict", ckpt)
    if not isinstance(src, dict):
        src = ckpt
    keys: List[str] = list(src.keys())
    is_cellnavi = any(
        k.startswith("encoder.layers.") or k.startswith("module.encoder.layers.")
        for k in keys
    ) or any(k.endswith("embed_gene.0.weight") for k in keys)
    if is_cellnavi:
        out = model.load_pretrained_weights(path, verbose=verbose)
        out["kind"] = "cellnavi"
        return out

    out = model.load_raw_pretrain_backbone(path, strict=False, verbose=verbose)
    return out
