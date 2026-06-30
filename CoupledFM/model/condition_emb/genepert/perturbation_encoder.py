"""Perturbation condition encoder mapping gene ids / type ids to ``cond_model`` (B, out_dim)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from . import perturbation as P
from .gene_cache import GeneEmbeddingCache, GeneEmbeddingTable


_ENCODER_MODES = frozenset(
    {
        "random_learned",
        "pretrained_frozen",
        "pretrained_tunable",
        "pretrained_with_type_gate",
        "pretrained_with_chem",
        "combo_id_baseline",
    }
)


class PerturbationConditionEncoder(nn.Module):
    """Encode multi-gene perturbations with masked mean + optional perturbation-type signal.

    * Index ``0`` on ``pert_gene_ids`` marks an empty slot; the corresponding
      embedding row should be zeros (CFG / unconditional drop-friendly).
    * ``nperts == 0`` rows get zero contribution after pooling.
    Modes:

    - ``random_learned``: fresh ``nn.Embedding`` + MLP projection.
    - ``pretrained_frozen`` / ``pretrained_tunable``: weights from ``GeneEmbeddingCache``.
    - ``pretrained_with_type_gate``: gated fusion with type embeddings.
    - ``combo_id_baseline``: ``nn.Embedding`` over discrete ``combo_ids`` only (minimal).
    """

    def __init__(
        self,
        mode: str,
        out_dim: int,
        *,
        cache: Optional[GeneEmbeddingCache] = None,
        num_embeddings_random: int = 8192,
        embed_dim_random: int = 256,
        max_combo_id_exclusive: int = 4096,
        type_embed_dim: int = 32,
        num_types: Optional[int] = None,
        dropout_p: float = 0.0,
        chem_emb_dim: Optional[int] = None,
        chem_projector_hidden: Optional[int] = None,
        chem_hidden: Optional[int] = None,
    ):
        super().__init__()
        mode = mode.lower().strip()
        if mode not in _ENCODER_MODES:
            raise ValueError(f"unknown mode={mode}; expected one of {_ENCODER_MODES}")
        if mode == "pretrained_with_chem":
            mode = "pretrained_tunable"
        self.mode = mode
        self.out_dim = int(out_dim)
        nt = int(num_types) if num_types is not None else P.num_perturbation_types()

        if mode.startswith("pretrained"):
            if cache is None:
                raise ValueError("pretrained* modes require GeneEmbeddingCache")
            self.embed_dim = cache.embed_dim
            freeze = mode == "pretrained_frozen"
            self.gene_table = GeneEmbeddingTable.from_cache(cache, freeze=freeze)
            self.pad_idx_gene = int(cache.pad_index)
        elif mode == "combo_id_baseline":
            self.embed_dim = 0  # unused
            self.pad_idx_gene = 0
            self._combo_emb = nn.Embedding(int(max_combo_id_exclusive), self.out_dim, padding_idx=0)
            nn.init.normal_(self._combo_emb.weight, std=0.02)
            with torch.no_grad():
                self._combo_emb.weight[0].zero_()
            self.gene_table = None  # type: ignore[assignment]
        else:
            self.embed_dim = int(embed_dim_random)
            ed = nn.Embedding(int(num_embeddings_random), self.embed_dim, padding_idx=0)
            nn.init.normal_(ed.weight, std=0.02)
            with torch.no_grad():
                ed.weight[0].zero_()
            self.gene_table = GeneEmbeddingTable(
                num_embeddings_random,
                self.embed_dim,
                weights=ed.weight.data.clone(),
                padding_idx=0,
                freeze=False,
            )
            self.pad_idx_gene = 0

        self.type_emb = nn.Embedding(nt, type_embed_dim, padding_idx=P.PERT_TYPE_NULL)
        with torch.no_grad():
            self.type_emb.weight[P.PERT_TYPE_NULL].zero_()
        nn.init.normal_(self.type_emb.weight[P.PERT_TYPE_NULL + 1 :], std=0.02)

        if mode == "pretrained_with_type_gate":
            self._gate_lin = nn.Linear(type_embed_dim, self.embed_dim)
            nn.init.zeros_(self._gate_lin.weight)
            nn.init.constant_(self._gate_lin.bias, 2.0)
            fused_in = self.embed_dim
        else:
            fused_in = self.embed_dim + type_embed_dim

        mid = max(fused_in, self.out_dim)
        layers = [
            nn.Linear(fused_in, mid),
            nn.GELU(),
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(mid, self.out_dim))
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        ced_in = chem_emb_dim
        if ced_in is None or int(ced_in) <= 0:
            ced = 0
        else:
            ced = int(ced_in)
        self.chem_emb_dim = ced
        self._chem_mlp: Optional[nn.Module] = None
        if ced > 0 and mode != "combo_id_baseline":
            cph = chem_projector_hidden
            if cph is None or int(cph) <= 0:
                cph = chem_hidden
            cph_i = 0 if cph is None else int(cph)
            hidden = max(ced, self.embed_dim) if cph_i <= 0 else cph_i
            self._chem_mlp = nn.Sequential(
                nn.Linear(ced, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.embed_dim),
            )
            nn.init.normal_(self._chem_mlp[0].weight, std=0.02)
            nn.init.normal_(self._chem_mlp[-1].weight, std=0.02)
            nn.init.zeros_(self._chem_mlp[-1].bias)

        if mode == "combo_id_baseline":
            self.forward = self._forward_combo  # type: ignore[method-assign]

    @staticmethod
    def _masked_mean_slots(emb: torch.Tensor, pert_mask: torch.Tensor) -> torch.Tensor:
        """Masked mean over gene slots (B, K, D) -> (B, D)."""
        m = pert_mask.to(dtype=emb.dtype).unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1e-6)
        return (emb * m).sum(dim=1) / denom

    def _masked_mean_genes(self, pert_gene_ids: torch.Tensor, pert_mask: torch.Tensor) -> torch.Tensor:
        assert self.gene_table is not None
        emb = self.gene_table(pert_gene_ids)
        return self._masked_mean_slots(emb, pert_mask)

    def _forward_combo(
        self,
        pert_gene_ids: torch.Tensor,
        pert_mask: torch.Tensor,
        pert_type_id: torch.Tensor,
        nperts: torch.Tensor,
        combo_id: Optional[torch.Tensor] = None,
        chem_emb: Optional[torch.Tensor] = None,
        chem_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del chem_emb, chem_mask
        batch = int(nperts.shape[0])
        del pert_gene_ids, pert_mask, pert_type_id
        device = self._combo_emb.weight.device
        if combo_id is None:
            cid = torch.zeros(batch, dtype=torch.long, device=device)
        else:
            cid = combo_id.long()
        out = self._combo_emb(cid.clamp(min=0, max=self._combo_emb.num_embeddings - 1)).to(dtype=torch.float32)
        dead = cid <= 0
        out = out * (~dead).to(out.dtype).unsqueeze(-1)
        return out

    def forward(
        self,
        pert_gene_ids: Optional[torch.Tensor] = None,
        pert_mask: Optional[torch.Tensor] = None,
        pert_type_id: Optional[torch.Tensor] = None,
        nperts: Optional[torch.Tensor] = None,
        combo_id: Optional[torch.Tensor] = None,
        pert_gene_emb: Optional[torch.Tensor] = None,
        chem_emb: Optional[torch.Tensor] = None,
        chem_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``cond_model`` (B, out_dim).

        Provide either ``pert_gene_ids`` + ``pert_mask`` **or** ``pert_gene_emb`` + ``pert_mask``
        (precomputed rows ``(B, K, embed_dim)``). ``combo_id`` baseline ignores gene tensors except ``nperts``
        batch sizing when ``combo_id`` is absent.

        ``pert_type_id`` and ``nperts`` default to zeros when omitted (unconditioned / control-like).

        When ``chem_emb`` / ``chem_mask`` are provided and the encoder was built with
        ``chem_emb_dim > 0``, the gene-slot embedding is blended with a learned projection
        of ``chem_emb`` (mask in ``[0, 1]`` per row).
        """
        if chem_emb is not None and self._chem_mlp is None:
            raise ValueError("chem_emb given but encoder has chem_emb_dim=0 (no chem MLP)")

        if pert_type_id is None or nperts is None:
            if pert_gene_emb is not None:
                batch = int(pert_gene_emb.shape[0])
                device = pert_gene_emb.device
            elif pert_gene_ids is not None:
                batch = int(pert_gene_ids.shape[0])
                device = pert_gene_ids.device
            else:
                raise ValueError("need pert_gene_emb or pert_gene_ids for batch size")
            if pert_type_id is None:
                pert_type_id = torch.zeros(batch, dtype=torch.long, device=device)
            if nperts is None:
                nperts = torch.zeros(batch, dtype=torch.long, device=device)

        assert pert_mask is not None
        if pert_gene_emb is not None:
            if pert_gene_emb.shape[-1] != self.embed_dim:
                raise ValueError(
                    f"pert_gene_emb last dim {pert_gene_emb.shape[-1]} != encoder embed_dim {self.embed_dim}"
                )
            pooled = self._masked_mean_slots(pert_gene_emb.to(dtype=torch.float32), pert_mask)
        else:
            if pert_gene_ids is None:
                raise ValueError("need pert_gene_ids or pert_gene_emb")
            pooled = self._masked_mean_genes(pert_gene_ids, pert_mask)
        alive = (nperts > 0).to(dtype=pooled.dtype).unsqueeze(-1)
        pooled = pooled * alive

        batch_sz = int(pooled.shape[0])
        chem_alive = torch.zeros(batch_sz, dtype=torch.bool, device=pooled.device)
        m_row: Optional[torch.Tensor] = None
        if self._chem_mlp is not None and chem_emb is not None:
            cem = chem_emb.to(dtype=torch.float32)
            if cem.dim() == 2:
                if cem.shape[-1] != self.chem_emb_dim:
                    raise ValueError(
                        f"chem_emb last dim {cem.shape[-1]} != chem_emb_dim {self.chem_emb_dim}"
                    )
                if chem_mask is None:
                    m_row = torch.ones(batch_sz, device=pooled.device, dtype=pooled.dtype)
                else:
                    m_row = chem_mask.to(device=pooled.device, dtype=pooled.dtype).reshape(
                        batch_sz
                    ).clamp(0.0, 1.0)
                m_exp = m_row.unsqueeze(-1)
                chem_slot = self._chem_mlp(cem)
                pooled = pooled * (1.0 - m_exp) + chem_slot * m_exp
                chem_alive = m_row > 0
            elif cem.dim() == 3:
                b3, kkc, dd = cem.shape
                if b3 != batch_sz:
                    raise ValueError(f"chem_emb batch {b3} != {batch_sz}")
                if dd != self.chem_emb_dim:
                    raise ValueError(
                        f"chem_emb last dim {dd} != chem_emb_dim {self.chem_emb_dim}"
                    )
                chem_flat = cem.reshape(-1, dd)
                chem_proj = self._chem_mlp(chem_flat).reshape(b3, kkc, self.embed_dim)
                if chem_mask is None:
                    mk = torch.ones(b3, kkc, device=pooled.device, dtype=pooled.dtype)
                else:
                    mk = chem_mask.to(device=pooled.device, dtype=pooled.dtype).reshape(
                        b3, kkc
                    ).clamp(0.0, 1.0)
                denom = mk.sum(dim=1).clamp(min=1e-6).unsqueeze(-1)
                chem_slot_vec = (chem_proj * mk.unsqueeze(-1)).sum(dim=1) / denom
                m_row = (mk.sum(dim=1) > 0).to(dtype=pooled.dtype)
                m_exp = m_row.unsqueeze(-1)
                pooled = pooled * (1.0 - m_exp) + chem_slot_vec * m_exp
                chem_alive = m_row > 0
            else:
                raise ValueError(
                    f"chem_emb must be 2D or 3D float tensor, got shape {tuple(cem.shape)}"
                )

        eff_tid = pert_type_id.long()
        if m_row is not None:
            drug_fb = (m_row > 0) & (eff_tid == P.PERT_TYPE_NULL)
            eff_tid = torch.where(
                drug_fb,
                torch.full_like(eff_tid, P.PERT_TYPE_DRUG),
                eff_tid,
            )

        tt = eff_tid.clamp(min=0, max=self.type_emb.num_embeddings - 1)
        te = self.type_emb(tt)
        inactive = eff_tid == P.PERT_TYPE_NULL
        te = te * (~inactive).to(te.dtype).unsqueeze(-1)

        if self.mode == "pretrained_with_type_gate":
            gate = torch.sigmoid(self._gate_lin(te))
            pooled = pooled * gate
            h = pooled
        else:
            h = torch.cat([pooled, te], dim=-1)
        out = self.mlp(h)
        gene_alive = nperts > 0
        row_alive = gene_alive | chem_alive
        zero_row = ~row_alive
        if zero_row.any():
            zs = torch.zeros_like(out[:1]).expand_as(out)
            zs_mask = zero_row.unsqueeze(-1).to(out.dtype)
            out = out * (1.0 - zs_mask) + zs * zs_mask
        return out


_DEFAULT_UNIFIED_TYPE_SCALE = (0.0, -1.0, -1.0, -1.0, 1.0, 1.0)


class UnifiedConditionEncoder(nn.Module):
    """Gene + chem **content** pooled in ``out_dim``, with optional concat fusion and per-type scaling."""

    def __init__(
        self,
        mode: str,
        out_dim: int,
        *,
        cache: Optional[GeneEmbeddingCache] = None,
        num_embeddings_random: int = 8192,
        embed_dim_random: int = 256,
        gene_projector_hidden: int = 0,
        chem_emb_dim: Optional[int] = None,
        chem_projector_hidden: Optional[int] = None,
        dropout_p: float = 0.0,
        pert_type_scale_init: Optional[Tuple[float, ...]] = None,
        pool_aggregations: Tuple[str, ...] = ("mean",),
        pool_scale_init: Tuple[float, ...] = (1.0,),
        pool_fusion_mode: str = "sum",
        type_adapter_mode: str = "scalar",
        condition_embedding_source: Optional[str] = None,
        pairwise_mode: str = "off",
    ):
        super().__init__()
        mode = mode.lower().strip()
        if mode not in _ENCODER_MODES or mode == "combo_id_baseline":
            raise ValueError(
                "UnifiedConditionEncoder requires a gene-table mode "
                f"(not {mode!r}); use PerturbationConditionEncoder for combo_id_baseline."
            )
        if mode == "pretrained_with_chem":
            mode = "pretrained_tunable"
        if mode == "pretrained_with_type_gate":
            raise ValueError(
                "UnifiedConditionEncoder no longer downgrades pretrained_with_type_gate. "
                "Use pretrained_tunable (or pretrained_frozen) with "
                "type_adapter_mode='vector_scale_gate' for per-dimension gated scaling, "
                "or use PerturbationConditionEncoder(mode='pretrained_with_type_gate')."
            )
        if mode.startswith("pretrained") and cache is None:
            raise ValueError("pretrained* UnifiedConditionEncoder requires GeneEmbeddingCache")

        self.mode = mode
        self.out_dim = int(out_dim)
        self.condition_embedding_source = condition_embedding_source

        pfm = str(pool_fusion_mode).lower().strip()
        if pfm not in ("sum", "concat_linear"):
            raise ValueError("pool_fusion_mode must be 'sum' or 'concat_linear'")
        self.pool_fusion_mode = pfm

        tam = str(type_adapter_mode).lower().strip()
        if tam not in ("scalar", "vector_scale", "vector_scale_gate"):
            raise ValueError(
                "type_adapter_mode must be 'scalar', 'vector_scale', or 'vector_scale_gate'"
            )
        self.type_adapter_mode = tam

        pwm = str(pairwise_mode).lower().strip()
        if pwm not in ("off", "hadamard_mean"):
            raise ValueError("pairwise_mode must be 'off' or 'hadamard_mean'")
        self.pairwise_mode = pwm

        if mode.startswith("pretrained"):
            assert cache is not None
            freeze = mode == "pretrained_frozen"
            self.gene_table = GeneEmbeddingTable.from_cache(cache, freeze=freeze)
            self.pad_idx_gene = int(cache.pad_index)
            gene_in = cache.embed_dim
        else:
            self.embed_dim_rand = int(embed_dim_random)
            ed = nn.Embedding(int(num_embeddings_random), self.embed_dim_rand, padding_idx=0)
            nn.init.normal_(ed.weight, std=0.02)
            with torch.no_grad():
                ed.weight[0].zero_()
            self.gene_table = GeneEmbeddingTable(
                int(num_embeddings_random),
                self.embed_dim_rand,
                weights=ed.weight.data.clone(),
                padding_idx=0,
                freeze=False,
            )
            self.pad_idx_gene = 0
            gene_in = self.embed_dim_rand

        self._build_gene_proj(gene_in, int(gene_projector_hidden), float(dropout_p))

        ced = int(chem_emb_dim) if chem_emb_dim is not None and int(chem_emb_dim) > 0 else 0
        self.chem_emb_dim = ced
        self._chem_to_out: Optional[nn.Module] = None
        if ced > 0:
            hidden = int(chem_projector_hidden) if chem_projector_hidden and int(chem_projector_hidden) > 0 else max(
                ced, self.out_dim
            )
            self._chem_to_out = nn.Sequential(
                nn.Linear(ced, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.out_dim),
            )
            for m in self._chem_to_out.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)

        init_t = tuple(float(x) for x in (pert_type_scale_init or _DEFAULT_UNIFIED_TYPE_SCALE))
        if len(init_t) != P.num_perturbation_types():
            raise ValueError(
                f"pert_type_scale_init must have length {P.num_perturbation_types()}, got {len(init_t)}"
            )
        nt = P.num_perturbation_types()

        if tam == "scalar":
            self.type_scale = nn.Parameter(torch.tensor(init_t, dtype=torch.float32))
            self.register_parameter("type_vector_scale", None)
            self.register_parameter("type_gate_delta", None)
        else:
            row = torch.tensor(init_t, dtype=torch.float32).unsqueeze(1)
            mat = row.expand(nt, self.out_dim).clone()
            self.type_vector_scale = nn.Parameter(mat)
            self.register_parameter("type_scale", None)
            if tam == "vector_scale_gate":
                self.type_gate_delta = nn.Parameter(torch.zeros(nt, self.out_dim))
            else:
                self.register_parameter("type_gate_delta", None)

        ops = tuple(str(o).lower().strip() for o in pool_aggregations)
        if not ops:
            raise ValueError("pool_aggregations must be non-empty")
        _ALLOWED_POOL_OPS = ("mean", "sum", "max", "min")
        for o in ops:
            if o not in _ALLOWED_POOL_OPS:
                raise ValueError(
                    f"unknown pool op {o!r}; allowed={_ALLOWED_POOL_OPS}"
                )
        init_p = tuple(float(x) for x in pool_scale_init)
        if len(init_p) != len(ops):
            raise ValueError(
                f"pool_scale_init length {len(init_p)} must match "
                f"pool_aggregations length {len(ops)}"
            )
        self.pool_aggregations: Tuple[str, ...] = ops
        self.pool_scale = nn.Parameter(torch.tensor(init_p, dtype=torch.float32))

        if self.pool_fusion_mode == "concat_linear":
            n_br = len(ops)
            self.pool_fuse = nn.Linear(n_br * self.out_dim, self.out_dim)
            nn.init.normal_(self.pool_fuse.weight, std=0.02)
            nn.init.zeros_(self.pool_fuse.bias)
        else:
            self.register_parameter("pool_fuse", None)

        if self.pairwise_mode == "hadamard_mean":
            self.pair_to_out = nn.Linear(self.out_dim, self.out_dim)
            nn.init.zeros_(self.pair_to_out.weight)
            nn.init.zeros_(self.pair_to_out.bias)
        else:
            self.pair_to_out = None

        self.layer_norm = nn.LayerNorm(self.out_dim, elementwise_affine=True, eps=1e-6)

    def _build_gene_proj(self, gene_in: int, gh: int, dropout_p: float) -> None:
        if gh <= 0:
            self.gene_to_out = nn.Linear(gene_in, self.out_dim)
            nn.init.normal_(self.gene_to_out.weight, std=0.02)
            nn.init.zeros_(self.gene_to_out.bias)
        else:
            mid = max(gene_in, self.out_dim, gh)
            layers: list[nn.Module] = [
                nn.Linear(gene_in, mid),
                nn.GELU(),
            ]
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            layers.append(nn.Linear(mid, self.out_dim))
            self.gene_to_out = nn.Sequential(*layers)
            for m in self.gene_to_out.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)

    def _type_multiplier(self, eff_tid: torch.Tensor, *, comp_dtype: torch.dtype) -> torch.Tensor:
        """Return (B, out_dim) or (B, 1) scale tensor (broadcastable on last dim)."""
        if self.type_adapter_mode == "scalar":
            assert self.type_scale is not None
            tt = eff_tid.clamp(min=0, max=self.type_scale.numel() - 1)
            return self.type_scale[tt].unsqueeze(-1).to(dtype=comp_dtype)
        assert self.type_vector_scale is not None
        tt = eff_tid.clamp(min=0, max=self.type_vector_scale.shape[0] - 1)
        vs = self.type_vector_scale[tt].to(dtype=comp_dtype)
        if self.type_adapter_mode == "vector_scale_gate" and self.type_gate_delta is not None:
            gd = self.type_gate_delta[tt].to(dtype=comp_dtype)
            vs = vs * (1.0 + 0.01 * gd)
        return vs

    @staticmethod
    def _masked_mean_slots(emb: torch.Tensor, pert_mask: torch.Tensor) -> torch.Tensor:
        m = pert_mask.to(dtype=emb.dtype).unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1e-6)
        return (emb * m).sum(dim=1) / denom

    @staticmethod
    def _masked_sum_slots(emb: torch.Tensor, pert_mask: torch.Tensor) -> torch.Tensor:
        m = pert_mask.to(dtype=emb.dtype).unsqueeze(-1)
        return (emb * m).sum(dim=1)

    @staticmethod
    def _masked_max_slots(emb: torch.Tensor, pert_mask: torch.Tensor) -> torch.Tensor:
        m = pert_mask.to(dtype=torch.bool).unsqueeze(-1)
        has_any = m.any(dim=1)
        neg_inf = torch.finfo(emb.dtype).min
        masked = torch.where(m, emb, torch.full_like(emb, neg_inf))
        val = masked.amax(dim=1)
        return torch.where(has_any, val, torch.zeros_like(val))

    @staticmethod
    def _masked_min_slots(emb: torch.Tensor, pert_mask: torch.Tensor) -> torch.Tensor:
        m = pert_mask.to(dtype=torch.bool).unsqueeze(-1)
        has_any = m.any(dim=1)
        pos_inf = torch.finfo(emb.dtype).max
        masked = torch.where(m, emb, torch.full_like(emb, pos_inf))
        val = masked.amin(dim=1)
        return torch.where(has_any, val, torch.zeros_like(val))

    def _pool_one(self, op: str, proj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if op == "mean":
            return self._masked_mean_slots(proj, mask)
        if op == "sum":
            return self._masked_sum_slots(proj, mask)
        if op == "max":
            return self._masked_max_slots(proj, mask)
        if op == "min":
            return self._masked_min_slots(proj, mask)
        raise ValueError(f"unknown pool op {op!r}")

    def _pairwise_content(self, proj: torch.Tensor, pert_mask: torch.Tensor) -> torch.Tensor:
        """Mean pairwise Hadamard interaction over active gene slots."""
        if self.pair_to_out is None:
            return torch.zeros(proj.shape[0], self.out_dim, device=proj.device, dtype=proj.dtype)
        m = pert_mask.to(dtype=proj.dtype).unsqueeze(-1)
        z = proj * m
        pair_raw = 0.5 * ((z.sum(dim=1) ** 2) - (z ** 2).sum(dim=1))
        n = m.sum(dim=1)
        n_pairs = n * (n - 1.0) * 0.5
        pair = pair_raw / n_pairs.clamp(min=1.0)
        pair = pair * (n_pairs > 0).to(dtype=proj.dtype)
        return self.pair_to_out(pair)

    def _gene_content(self, pert_gene_ids: torch.Tensor, pert_mask: torch.Tensor) -> torch.Tensor:
        emb = self.gene_table(pert_gene_ids).to(dtype=torch.float32)
        if isinstance(self.gene_to_out, nn.Linear):
            proj = self.gene_to_out(emb)
        else:
            proj = self.gene_to_out(emb)
        return self._masked_mean_slots(proj, pert_mask)

    def _gene_content_list(
        self, pert_gene_ids: torch.Tensor, pert_mask: torch.Tensor
    ) -> list:
        emb = self.gene_table(pert_gene_ids).to(dtype=torch.float32)
        proj = self.gene_to_out(emb)
        return [self._pool_one(op, proj, pert_mask) for op in self.pool_aggregations]

    def _chem_content(
        self,
        chem_emb: torch.Tensor,
        chem_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ((B, out_dim) aggregated chem, (B,) alive float)."""
        if self._chem_to_out is None:
            z = torch.zeros(chem_emb.shape[0], self.out_dim, device=chem_emb.device, dtype=torch.float32)
            return z, torch.zeros(chem_emb.shape[0], device=chem_emb.device, dtype=torch.float32)

        x = chem_emb.to(dtype=torch.float32)
        if x.dim() == 2:
            b = x.shape[0]
            mk = (
                torch.ones(b, device=x.device, dtype=x.dtype)
                if chem_mask is None
                else chem_mask.to(device=x.device, dtype=x.dtype).reshape(b).clamp(0.0, 1.0)
            )
            vec = self._chem_to_out(x)
            return vec * mk.unsqueeze(-1), mk
        if x.dim() == 3:
            b, kk, dd = x.shape
            if dd != self.chem_emb_dim:
                raise ValueError(f"chem last dim {dd} != chem_emb_dim {self.chem_emb_dim}")
            flat = x.reshape(-1, dd)
            proj = self._chem_to_out(flat).reshape(b, kk, self.out_dim)
            if chem_mask is None:
                mk = torch.ones(b, kk, device=x.device, dtype=x.dtype)
            else:
                mk = chem_mask.to(device=x.device, dtype=x.dtype).reshape(b, kk).clamp(0.0, 1.0)
            denom = mk.sum(dim=1).clamp(min=1e-6).unsqueeze(-1)
            vec = (proj * mk.unsqueeze(-1)).sum(dim=1) / denom
            alive = (mk.sum(dim=1) > 0).to(dtype=x.dtype)
            return vec, alive
        raise ValueError(f"chem_emb shape {tuple(x.shape)}; expected (B,D) or (B,K,D)")

    def _chem_content_list(
        self,
        chem_emb: torch.Tensor,
        chem_mask: Optional[torch.Tensor],
    ) -> Tuple[list, torch.Tensor]:
        if self._chem_to_out is None:
            b = chem_emb.shape[0]
            zeros = torch.zeros(b, self.out_dim, device=chem_emb.device, dtype=torch.float32)
            return [zeros for _ in self.pool_aggregations], torch.zeros(
                b, device=chem_emb.device, dtype=torch.float32
            )
        x = chem_emb.to(dtype=torch.float32)
        if x.dim() == 2:
            b = x.shape[0]
            mk = (
                torch.ones(b, device=x.device, dtype=x.dtype)
                if chem_mask is None
                else chem_mask.to(device=x.device, dtype=x.dtype).reshape(b).clamp(0.0, 1.0)
            )
            vec = self._chem_to_out(x) * mk.unsqueeze(-1)
            return [vec for _ in self.pool_aggregations], mk
        if x.dim() == 3:
            b, kk, dd = x.shape
            if dd != self.chem_emb_dim:
                raise ValueError(f"chem last dim {dd} != chem_emb_dim {self.chem_emb_dim}")
            flat = x.reshape(-1, dd)
            proj = self._chem_to_out(flat).reshape(b, kk, self.out_dim)
            if chem_mask is None:
                mk = torch.ones(b, kk, device=x.device, dtype=x.dtype)
            else:
                mk = chem_mask.to(device=x.device, dtype=x.dtype).reshape(b, kk).clamp(0.0, 1.0)
            alive = (mk.sum(dim=1) > 0).to(dtype=x.dtype)
            return [self._pool_one(op, proj, mk) for op in self.pool_aggregations], alive
        raise ValueError(f"chem_emb shape {tuple(x.shape)}; expected (B,D) or (B,K,D)")

    def forward(
        self,
        pert_gene_ids: Optional[torch.Tensor] = None,
        pert_mask: Optional[torch.Tensor] = None,
        pert_type_id: Optional[torch.Tensor] = None,
        nperts: Optional[torch.Tensor] = None,
        combo_id: Optional[torch.Tensor] = None,
        pert_gene_emb: Optional[torch.Tensor] = None,
        chem_emb: Optional[torch.Tensor] = None,
        chem_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del combo_id  # not used in unified path

        if pert_type_id is None or nperts is None:
            if pert_gene_emb is not None:
                batch = int(pert_gene_emb.shape[0])
                device = pert_gene_emb.device
            elif pert_gene_ids is not None:
                batch = int(pert_gene_ids.shape[0])
                device = pert_gene_ids.device
            elif chem_emb is not None:
                batch = int(chem_emb.shape[0])
                device = chem_emb.device
            else:
                raise ValueError("need perturbation tensors for batch size")
            if pert_type_id is None:
                pert_type_id = torch.zeros(batch, dtype=torch.long, device=device)
            if nperts is None:
                nperts = torch.zeros(batch, dtype=torch.long, device=device)

        assert pert_mask is not None

        if pert_gene_emb is not None:
            ge = pert_gene_emb.to(dtype=torch.float32)
            proj = self.gene_to_out(ge)
            gene_list = [self._pool_one(op, proj, pert_mask) for op in self.pool_aggregations]
        else:
            if pert_gene_ids is None:
                raise ValueError("need pert_gene_ids or pert_gene_emb")
            emb = self.gene_table(pert_gene_ids).to(dtype=torch.float32)
            proj = self.gene_to_out(emb)
            gene_list = [self._pool_one(op, proj, pert_mask) for op in self.pool_aggregations]

        alive_g = (nperts > 0).to(dtype=gene_list[0].dtype).unsqueeze(-1)
        gene_list = [g * alive_g for g in gene_list]
        pair_out = self._pairwise_content(proj, pert_mask) * alive_g

        chem_alive_f = torch.zeros(
            gene_list[0].shape[0], device=gene_list[0].device, dtype=gene_list[0].dtype
        )
        chem_list = [torch.zeros_like(g) for g in gene_list]
        if chem_emb is not None:
            if self._chem_to_out is None:
                raise ValueError(
                    "chem_emb provided but chem_emb_dim=0 in UnifiedConditionEncoder"
                )
            chem_list, chem_alive_f = self._chem_content_list(chem_emb, chem_mask)

        ps = self.pool_scale.to(dtype=gene_list[0].dtype)
        comps = [
            ps[i] * (gene_list[i] + chem_list[i])
            for i in range(len(self.pool_aggregations))
        ]
        eff_tid = pert_type_id.long()
        if chem_alive_f is not None and chem_alive_f.numel() > 0:
            drug_fb = (chem_alive_f > 0) & (eff_tid == P.PERT_TYPE_NULL)
            eff_tid = torch.where(
                drug_fb,
                torch.full_like(eff_tid, P.PERT_TYPE_DRUG),
                eff_tid,
            )

        mult = self._type_multiplier(eff_tid, comp_dtype=comps[0].dtype)
        gene_alive = nperts > 0
        gene_unknown_type = gene_alive & (chem_alive_f <= 0) & (eff_tid == P.PERT_TYPE_NULL)
        if gene_unknown_type.any():
            # Missing perturbation_type metadata should not erase known gene content.
            # True no-perturb rows are still zeroed by ``zero_row`` below.
            mult = torch.where(
                gene_unknown_type.unsqueeze(-1),
                torch.ones_like(mult),
                mult,
            )
        if self.pool_fusion_mode == "sum":
            content = sum(comps)
            out = self.layer_norm(content * mult + pair_out * mult)
        else:
            assert self.pool_fuse is not None
            n_br = len(comps)
            # mult may be (B,1) for scalar or (B,out_dim) for vector modes.
            # Expand to (B, out_dim) then repeat to (B, n_br*out_dim) to match stacked.
            mult_full = mult.expand(-1, self.out_dim)        # (B, out_dim)
            mult_rep = mult_full.repeat(1, n_br)             # (B, n_br*out_dim)
            stacked = torch.cat(comps, dim=-1)               # (B, n_br*out_dim)
            fused = self.pool_fuse(stacked * mult_rep)
            out = self.layer_norm(fused + pair_out * mult_full)
        chem_alive_b = chem_alive_f > 0
        zero_row = ~(gene_alive | chem_alive_b)
        if zero_row.any():
            zs = torch.zeros_like(out[:1]).expand_as(out)
            zs_mask = zero_row.unsqueeze(-1).to(out.dtype)
            out = out * (1.0 - zs_mask) + zs * zs_mask
        return out
