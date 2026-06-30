"""
MLP velocity field models with adaLN-Zero conditioning.

MLPVelocityField:      v(x_t, t) — baseline without control input
ControlMLPVelocityField: v(x_t, t, x_0) — with control (IR) input

ControlMLP architecture:
  SharedEncoder(x_t) → h_t,  SharedEncoder(x_0) → h_0   (same weights)
  Fusion: cat(h_t, h_0) → Linear → h
  Conditioning: c = TimestepEmb(t) + CtrlProj(h_0)
  N x adaLN-Zero ResBlock(h, c)
  OutputProj: h → v
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.utils.conditioning.perturbation_encoder import (
    PerturbationConditionEncoder,
    UnifiedConditionEncoder,
)
from model.condition_emb.genepert import PERT_TYPE_DRUG
from model.utils.embeddings.gene_cache import GeneEmbeddingCache
from model.utils.models.layers import TimestepEmbedder


class ResidualBlock(nn.Module):
    """adaLN-Zero conditioned residual MLP block."""

    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 3 * d_model),
        )

    def forward(self, h: Tensor, c: Tensor) -> Tensor:
        shift, scale, gate = self.ada(c).chunk(3, dim=-1)
        y = self.norm(h)
        y = y * (1.0 + scale) + shift
        y = self.mlp(y)
        return h + gate * y


class MLPVelocityField(nn.Module):
    """Baseline MLP velocity field v_theta(x_t, t) -> v.  No control input."""

    def __init__(
        self,
        emb_dim: int = 2058,
        d_model: int = 1024,
        n_layers: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(emb_dim, d_model)
        self.t_embed = TimestepEmbedder(d_model)
        self.blocks = nn.ModuleList(
            [ResidualBlock(d_model, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.output_proj = nn.Linear(d_model, emb_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        for block in self.blocks:
            nn.init.zeros_(block.ada[-1].weight)
            nn.init.zeros_(block.ada[-1].bias)

    def forward(self, x_t: Tensor, t: Tensor, x_0: Tensor = None) -> Tensor:
        h = self.input_proj(x_t)
        t_emb = self.t_embed(t)
        for block in self.blocks:
            h = block(h, t_emb)
        h = self.out_norm(h)
        return self.output_proj(h)


class ControlMLPVelocityField(nn.Module):
    """Control-conditioned MLP velocity field v_theta(x_t, t, x_0) -> v.

    Control (x_0 / IR) enters via two paths:
      1. Data path:  SharedEncoder encodes both x_t and x_0 into the same
         representation space; their encodings are fused via concatenation.
      2. Conditioning path:  h_0 is projected to a conditioning vector that
         is added to the timestep embedding and fed into adaLN at every layer.

    Optional perturbation conditioner (experimental): sums a
    :class:`~utils.conditioning.perturbation_encoder.PerturbationConditionEncoder`
    output (projected to ``d_model``) into the adaLN conditioning vector ``c``.
    """

    def __init__(
        self,
        emb_dim: int = 2058,
        d_model: int = 512,
        n_layers: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        *,
        use_pert_condition: bool = False,
        pert_embed_mode: str = "random_learned",
        pert_cond_dim: int = 512,
        pert_type_emb_dim: int = 32,
        pert_encoder_num_embeddings: int = 8192,
        pert_gene_emb_dim: int = 256,
        pert_encoder_dropout: float = 0.0,
        max_combo_id_exclusive: int = 4096,
        gene_embedding_cache: Optional[GeneEmbeddingCache] = None,
        pert_chem_emb_dim: Optional[int] = None,
        pert_chem_projector_hidden: Optional[int] = None,
        pert_gene_projector_hidden: int = 0,
        pert_type_scale_init: Optional[Tuple[float, ...]] = None,
        pool_aggregations: Tuple[str, ...] = ("mean",),
        pool_scale_init: Tuple[float, ...] = (1.0,),
        pool_fusion_mode: str = "sum",
        type_adapter_mode: str = "scalar",
        pairwise_mode: str = "off",
        condition_embedding_source: Optional[str] = None,
        pert_to_c_init_mode: str = "zero",
        use_pert_in_fusion: bool = False,
        use_condition_delta_head: bool = False,
        condition_delta_head_hidden: int = 1024,
        condition_delta_head_use_in_model: bool = False,
        condition_lowrank_residual_use_in_model: bool = False,
        condition_lowrank_residual_rank: int = 32,
        condition_delta_in_model_filter: str = "all",
        trackc_support_context_use_in_model: bool = False,
        trackc_support_residual_use_in_model: bool = False,
        trackc_support_film_use_in_model: bool = False,
        trackc_support_context_dim: int = 0,
        trackc_support_set_task_use_in_model: bool = False,
        trackc_support_set_task_dim: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_pert_condition = bool(use_pert_condition)
        self.pert_to_c_init_mode = str(pert_to_c_init_mode).lower().strip()
        self.use_pert_in_fusion = bool(use_pert_in_fusion)
        self.use_condition_delta_head = bool(use_condition_delta_head)
        self.condition_delta_head_use_in_model = bool(condition_delta_head_use_in_model)
        self.condition_lowrank_residual_use_in_model = bool(condition_lowrank_residual_use_in_model)
        self.trackc_support_context_use_in_model = bool(trackc_support_context_use_in_model)
        self.trackc_support_residual_use_in_model = bool(trackc_support_residual_use_in_model)
        self.trackc_support_film_use_in_model = bool(trackc_support_film_use_in_model)
        self.trackc_support_context_dim = int(trackc_support_context_dim)
        self.trackc_support_set_task_use_in_model = bool(trackc_support_set_task_use_in_model)
        self.trackc_support_set_task_dim = int(trackc_support_set_task_dim)
        if (
            self.trackc_support_context_use_in_model
            or self.trackc_support_residual_use_in_model
            or self.trackc_support_film_use_in_model
        ) and self.trackc_support_context_dim <= 0:
            raise ValueError(
                "trackc_support_context_dim must be positive when support context is enabled"
            )
        if self.trackc_support_set_task_use_in_model and self.trackc_support_set_task_dim <= 0:
            raise ValueError(
                "trackc_support_set_task_dim must be positive when support-set task adapter is enabled"
            )
        filt = str(condition_delta_in_model_filter or "all").strip().lower()
        if filt not in {
            "all",
            "gene_single",
            "gene_multi",
            "prior_covered_gene_multi",
            "allowlisted_gene_single",
        }:
            raise ValueError(
                "condition_delta_in_model_filter must be one of: "
                "all, gene_single, gene_multi, prior_covered_gene_multi, "
                "allowlisted_gene_single"
            )
        self.condition_delta_in_model_filter = filt

        self.shared_enc = nn.Sequential(
            nn.Linear(emb_dim, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.fusion = nn.Linear(2 * d_model, d_model)

        self.t_embed = TimestepEmbedder(d_model)
        self.ctrl_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList(
            [ResidualBlock(d_model, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.output_proj = nn.Linear(d_model, emb_dim)
        if self.trackc_support_context_use_in_model:
            self.support_context_to_c = nn.Linear(self.trackc_support_context_dim, d_model, bias=False)
        else:
            self.support_context_to_c = None
        if self.trackc_support_residual_use_in_model or self.trackc_support_film_use_in_model:
            self.support_context_to_v = nn.Linear(self.trackc_support_context_dim, emb_dim, bias=False)
        else:
            self.support_context_to_v = None
        if self.trackc_support_film_use_in_model:
            self.support_context_to_v_scale = nn.Linear(self.trackc_support_context_dim, emb_dim, bias=False)
        else:
            self.support_context_to_v_scale = None
        if self.trackc_support_set_task_use_in_model:
            self.support_set_task_to_c = nn.Linear(self.trackc_support_set_task_dim, d_model, bias=False)
        else:
            self.support_set_task_to_c = None

        self.pert_encoder: Optional[Union[PerturbationConditionEncoder, UnifiedConditionEncoder]]
        self.pert_to_c: nn.Module
        if self.use_pert_condition:
            cache = gene_embedding_cache
            mode = pert_embed_mode.lower().strip()
            if mode.startswith("pretrained"):
                if cache is None:
                    raise ValueError("pretrained perturbation modes require gene_embedding_cache")
            pcd = pert_chem_emb_dim
            if pcd is None:
                pcd = 0
            pcd = int(pcd)
            pch = pert_chem_projector_hidden
            if pch is None:
                pch = 0
            pch = int(pch)
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
                    chem_projector_hidden=(None if pch <= 0 else pch),
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
                    chem_projector_hidden=(None if pch <= 0 else pch),
                    dropout_p=float(pert_encoder_dropout),
                    pert_type_scale_init=pti,
                    pool_aggregations=tuple(pool_aggregations),
                    pool_scale_init=tuple(pool_scale_init),
                    pool_fusion_mode=str(pool_fusion_mode).lower().strip(),
                    type_adapter_mode=str(type_adapter_mode).lower().strip(),
                    pairwise_mode=str(pairwise_mode).lower().strip(),
                    condition_embedding_source=condition_embedding_source,
                )
            pc = int(pert_cond_dim)
            self.pert_to_c = nn.Linear(pc, d_model) if pc != d_model else nn.Identity()
            gene_table = getattr(self.pert_encoder, "gene_table", None)
            embed = getattr(gene_table, "embed", None)
            n_gene_embeddings = int(getattr(embed, "num_embeddings", 0) or 0)
        else:
            self.pert_encoder = None
            self.pert_to_c = nn.Identity()
            n_gene_embeddings = 0
        self.register_buffer(
            "condition_delta_prior_gene_allowlist",
            torch.zeros(n_gene_embeddings, dtype=torch.bool),
            persistent=True,
        )
        if self.use_condition_delta_head:
            if not self.use_pert_condition:
                raise ValueError("use_condition_delta_head requires use_pert_condition=True")
            hidden = max(int(condition_delta_head_hidden), d_model, 8)
            self.condition_delta_head = nn.Sequential(
                nn.LayerNorm(d_model, elementwise_affine=True, eps=1e-6),
                nn.Linear(d_model, hidden),
                nn.SiLU(),
                nn.Linear(hidden, emb_dim),
            )
            if self.condition_delta_head_use_in_model:
                self.condition_delta_to_c = nn.Linear(emb_dim, d_model)
            else:
                self.condition_delta_to_c = None
        else:
            self.condition_delta_head = None
            self.condition_delta_to_c = None
        if self.condition_lowrank_residual_use_in_model:
            if not self.use_pert_condition:
                raise ValueError("condition_lowrank_residual_use_in_model requires use_pert_condition=True")
            rank = max(int(condition_lowrank_residual_rank), 1)
            self.condition_lowrank_residual_down = nn.Linear(d_model, rank, bias=False)
            self.condition_lowrank_residual_up = nn.Linear(rank, emb_dim, bias=True)
        else:
            self.condition_lowrank_residual_down = None
            self.condition_lowrank_residual_up = None

        self._init_weights()

    def set_condition_delta_prior_gene_ids(self, gene_ids) -> None:
        """Set deployable train-prior gene-id allowlist for gated delta injection."""
        allow = torch.zeros_like(self.condition_delta_prior_gene_allowlist, dtype=torch.bool)
        if allow.numel() > 0:
            ids = torch.as_tensor(list(gene_ids), dtype=torch.long, device=allow.device)
            ids = ids[(ids >= 0) & (ids < allow.numel())]
            if ids.numel() > 0:
                allow[ids] = True
        self.condition_delta_prior_gene_allowlist = allow

    def _condition_delta_in_model_gate(
        self,
        *,
        pert_gene_ids: Optional[Tensor],
        pert_mask: Optional[Tensor],
        pert_type_id: Optional[Tensor],
        nperts: Optional[Tensor],
        chem_mask: Optional[Tensor],
    ) -> Optional[Tensor]:
        mode = getattr(self, "condition_delta_in_model_filter", "all")
        if mode == "all":
            return None
        if pert_gene_ids is None or pert_mask is None:
            raise RuntimeError(
                "condition_delta_in_model_filter requires pert_gene_ids and pert_mask; "
                f"got mode={mode!r}"
            )
        active = pert_mask.to(dtype=torch.bool)
        active_count = active.sum(dim=1)
        if nperts is None:
            multi = active_count >= 2
            single = active_count == 1
        else:
            nperts_flat = nperts.reshape(-1).to(device=active.device)
            multi = nperts_flat >= 2
            single = nperts_flat == 1
        if pert_type_id is None or chem_mask is None:
            raise RuntimeError(
                "condition_delta_in_model_filter requires pert_type_id and chem_mask "
                f"to avoid fail-open injection; got mode={mode!r}"
            )
        not_drug = pert_type_id.reshape(-1).to(device=active.device) != PERT_TYPE_DRUG
        no_chem = ~(chem_mask.to(device=active.device) > 0).any(dim=1)
        if mode == "gene_single":
            return single & not_drug & no_chem
        if mode == "gene_multi":
            return multi & not_drug & no_chem
        if mode == "allowlisted_gene_single":
            gate = single & not_drug & no_chem
        else:
            gate = multi & not_drug & no_chem
        allow = self.condition_delta_prior_gene_allowlist.to(device=pert_gene_ids.device)
        if allow.numel() <= 0:
            return gate & torch.zeros_like(gate, dtype=torch.bool)
        ids = pert_gene_ids.to(device=allow.device, dtype=torch.long)
        in_bounds = (ids >= 0) & (ids < allow.numel())
        clamped = ids.clamp(min=0, max=max(allow.numel() - 1, 0))
        covered = torch.zeros_like(active, dtype=torch.bool)
        covered[in_bounds] = allow[clamped[in_bounds]]
        all_active_covered = (covered | ~active).all(dim=1)
        return gate & all_active_covered

    def _init_weights(self):
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        for block in self.blocks:
            nn.init.zeros_(block.ada[-1].weight)
            nn.init.zeros_(block.ada[-1].bias)
        if self.use_pert_condition and isinstance(self.pert_to_c, nn.Linear):
            mode = getattr(self, "pert_to_c_init_mode", "zero")
            if mode == "xavier_small":
                nn.init.xavier_uniform_(self.pert_to_c.weight, gain=0.1)
                nn.init.zeros_(self.pert_to_c.bias)
            elif mode == "zero":
                nn.init.zeros_(self.pert_to_c.weight)
                nn.init.zeros_(self.pert_to_c.bias)
            else:
                raise ValueError(f"Unknown pert_to_c_init_mode={mode!r}")
        if self.condition_delta_head is not None:
            for m in self.condition_delta_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)
        if self.condition_delta_to_c is not None:
            nn.init.zeros_(self.condition_delta_to_c.weight)
            nn.init.zeros_(self.condition_delta_to_c.bias)
        if self.condition_lowrank_residual_down is not None:
            nn.init.normal_(self.condition_lowrank_residual_down.weight, std=0.02)
        if self.condition_lowrank_residual_up is not None:
            nn.init.zeros_(self.condition_lowrank_residual_up.weight)
            nn.init.zeros_(self.condition_lowrank_residual_up.bias)
        if self.support_context_to_c is not None:
            nn.init.zeros_(self.support_context_to_c.weight)
        if self.support_context_to_v is not None:
            nn.init.zeros_(self.support_context_to_v.weight)
        if self.support_context_to_v_scale is not None:
            nn.init.zeros_(self.support_context_to_v_scale.weight)

        if self.support_set_task_to_c is not None:
            nn.init.zeros_(self.support_set_task_to_c.weight)

    def _support_context_present_mask(
        self,
        support_context: Optional[Tensor],
        batch_size: int,
        support_context_present: Optional[Tensor] = None,
    ) -> Tensor:
        context = self._validate_support_context(support_context, batch_size)
        if support_context_present is None:
            present = context.abs().sum(dim=1, keepdim=True).gt(0)
        else:
            present = support_context_present
            if present.ndim == 1:
                present = present.unsqueeze(-1)
            if present.ndim != 2 or present.shape[0] != int(batch_size) or present.shape[1] != 1:
                raise ValueError("support_context_present must be shaped (B,) or (B, 1)")
            if not torch.isfinite(present.to(dtype=torch.float32)).all():
                raise ValueError("support_context_present contains non-finite values")
            present_f = present.to(dtype=torch.float32)
            if not torch.logical_or(present_f == 0.0, present_f == 1.0).all():
                raise ValueError("support_context_present must contain only 0/1 values")
            present = present_f.bool()
        return present.to(device=context.device, dtype=context.dtype)

    def _validate_support_context(self, support_context: Optional[Tensor], batch_size: int) -> Tensor:
        if support_context is None:
            raise RuntimeError(
                "Track C support context/residual paths requires support_context for every forward pass"
            )
        if support_context.ndim != 2:
            raise ValueError("support_context must be shaped (B, trackc_support_context_dim)")
        if support_context.shape[0] != int(batch_size):
            raise ValueError(
                "support_context batch size must match x_t batch size: "
                f"{support_context.shape[0]} != {int(batch_size)}"
            )
        if support_context.shape[1] != self.trackc_support_context_dim:
            raise ValueError(
                "support_context feature dimension must equal trackc_support_context_dim: "
                f"{support_context.shape[1]} != {self.trackc_support_context_dim}"
            )
        if not torch.isfinite(support_context).all():
            raise ValueError("support_context contains non-finite values")
        return support_context

    def _support_context_projection(
        self,
        support_context: Optional[Tensor],
        batch_size: int,
        support_context_present: Optional[Tensor] = None,
    ) -> Tensor:
        if not self.trackc_support_context_use_in_model or self.support_context_to_c is None:
            raise RuntimeError("support context projection requested but support context is disabled")
        context = self._validate_support_context(support_context, batch_size)
        present = self._support_context_present_mask(context, batch_size, support_context_present)
        return self.support_context_to_c(context) * present.to(dtype=context.dtype)

    def _support_residual_projection(
        self,
        support_context: Optional[Tensor],
        batch_size: int,
        support_context_present: Optional[Tensor] = None,
    ) -> Tensor:
        if (
            not (self.trackc_support_residual_use_in_model or self.trackc_support_film_use_in_model)
            or self.support_context_to_v is None
        ):
            raise RuntimeError("support residual projection requested but support residual is disabled")
        context = self._validate_support_context(support_context, batch_size)
        present = self._support_context_present_mask(context, batch_size, support_context_present)
        return self.support_context_to_v(context) * present.to(dtype=context.dtype)

    def _support_film_scale_projection(
        self,
        support_context: Optional[Tensor],
        batch_size: int,
        support_context_present: Optional[Tensor] = None,
    ) -> Tensor:
        if not self.trackc_support_film_use_in_model or self.support_context_to_v_scale is None:
            raise RuntimeError("support FiLM projection requested but support FiLM is disabled")
        context = self._validate_support_context(support_context, batch_size)
        present = self._support_context_present_mask(context, batch_size, support_context_present)
        return self.support_context_to_v_scale(context) * present.to(dtype=context.dtype)

    def _support_set_task_present_mask(
        self,
        support_set_task: Optional[Tensor],
        batch_size: int,
        support_set_task_present: Optional[Tensor] = None,
    ) -> Tensor:
        task = self._validate_support_set_task(support_set_task, batch_size)
        if support_set_task_present is None:
            present = task.abs().sum(dim=1, keepdim=True).gt(0)
        else:
            present = support_set_task_present
            if present.ndim == 1:
                present = present.unsqueeze(-1)
            if present.ndim != 2 or present.shape[0] != int(batch_size) or present.shape[1] != 1:
                raise ValueError("support_set_task_present must be shaped (B,) or (B, 1)")
            if not torch.isfinite(present.to(dtype=torch.float32)).all():
                raise ValueError("support_set_task_present contains non-finite values")
            present_f = present.to(dtype=torch.float32)
            if not torch.logical_or(present_f == 0.0, present_f == 1.0).all():
                raise ValueError("support_set_task_present must contain only 0/1 values")
            present = present_f.bool()
        return present.to(device=task.device, dtype=task.dtype)

    def _validate_support_set_task(self, support_set_task: Optional[Tensor], batch_size: int) -> Tensor:
        if support_set_task is None:
            raise RuntimeError(
                "Track C support-set task adapter requires support_set_task for every forward pass"
            )
        if support_set_task.ndim != 2:
            raise ValueError("support_set_task must be shaped (B, trackc_support_set_task_dim)")
        if support_set_task.shape[0] != int(batch_size):
            raise ValueError(
                "support_set_task batch size must match x_t batch size: "
                f"{support_set_task.shape[0]} != {int(batch_size)}"
            )
        if support_set_task.shape[1] != self.trackc_support_set_task_dim:
            raise ValueError(
                "support_set_task feature dimension must equal trackc_support_set_task_dim: "
                f"{support_set_task.shape[1]} != {self.trackc_support_set_task_dim}"
            )
        if not torch.isfinite(support_set_task).all():
            raise ValueError("support_set_task contains non-finite values")
        return support_set_task

    def _support_set_task_projection(
        self,
        support_set_task: Optional[Tensor],
        batch_size: int,
        support_set_task_present: Optional[Tensor] = None,
    ) -> Tensor:
        if not self.trackc_support_set_task_use_in_model or self.support_set_task_to_c is None:
            raise RuntimeError("support-set task projection requested but support-set task adapter is disabled")
        task = self._validate_support_set_task(support_set_task, batch_size)
        present = self._support_set_task_present_mask(task, batch_size, support_set_task_present)
        return self.support_set_task_to_c(task) * present.to(dtype=task.dtype)

    def _pert_projection(
        self,
        *,
        pert_gene_ids: Optional[Tensor] = None,
        pert_mask: Optional[Tensor] = None,
        pert_type_id: Optional[Tensor] = None,
        nperts: Optional[Tensor] = None,
        combo_id: Optional[Tensor] = None,
        chem_emb: Optional[Tensor] = None,
        chem_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if not self.use_pert_condition or self.pert_encoder is None:
            raise RuntimeError("perturbation projection requested but use_pert_condition=False")
        pe = self.pert_encoder(
            pert_gene_ids=pert_gene_ids,
            pert_mask=pert_mask,
            pert_type_id=pert_type_id,
            nperts=nperts,
            combo_id=combo_id,
            chem_emb=chem_emb,
            chem_mask=chem_mask,
        )
        return self.pert_to_c(pe)

    def predict_condition_delta(
        self,
        *,
        pert_gene_ids: Optional[Tensor] = None,
        pert_mask: Optional[Tensor] = None,
        pert_type_id: Optional[Tensor] = None,
        nperts: Optional[Tensor] = None,
        combo_id: Optional[Tensor] = None,
        chem_emb: Optional[Tensor] = None,
        chem_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.condition_delta_head is None:
            raise RuntimeError("condition_delta_head is disabled")
        p_d = self._pert_projection(
            pert_gene_ids=pert_gene_ids,
            pert_mask=pert_mask,
            pert_type_id=pert_type_id,
            nperts=nperts,
            combo_id=combo_id,
            chem_emb=chem_emb,
            chem_mask=chem_mask,
        )
        return self.condition_delta_head(p_d)

    def predict_additive_condition_delta(
        self,
        *,
        pert_gene_ids: Tensor,
        pert_mask: Tensor,
        pert_type_id: Optional[Tensor] = None,
        nperts: Optional[Tensor] = None,
        combo_id: Optional[Tensor] = None,
        chem_emb: Optional[Tensor] = None,
        chem_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict a response delta by summing single-gene condition atoms."""
        del nperts, combo_id, chem_emb, chem_mask
        if self.condition_delta_head is None:
            raise RuntimeError("condition_delta_head is disabled")
        if pert_gene_ids is None or pert_mask is None:
            raise RuntimeError("additive condition delta requires gene ids and masks")
        if pert_gene_ids.ndim != 2 or pert_mask.ndim != 2:
            raise ValueError("pert_gene_ids and pert_mask must be shaped (B, K)")
        if pert_gene_ids.shape != pert_mask.shape:
            raise ValueError("pert_gene_ids and pert_mask shapes must match")

        bsz, k_slots = pert_gene_ids.shape
        device = pert_gene_ids.device
        flat_gene = pert_gene_ids.reshape(-1)
        flat_mask = pert_mask.reshape(-1).to(dtype=torch.bool)
        single_gene_ids = torch.zeros(
            bsz * k_slots,
            k_slots,
            dtype=pert_gene_ids.dtype,
            device=device,
        )
        single_mask = torch.zeros(
            bsz * k_slots,
            k_slots,
            dtype=torch.bool,
            device=device,
        )
        single_gene_ids[:, 0] = flat_gene
        single_mask[:, 0] = flat_mask
        if pert_type_id is None:
            single_type = torch.zeros(bsz * k_slots, dtype=torch.long, device=device)
        else:
            single_type = pert_type_id.reshape(bsz, 1).expand(bsz, k_slots).reshape(-1)
        single_nperts = flat_mask.to(dtype=torch.long)

        p_d = self._pert_projection(
            pert_gene_ids=single_gene_ids,
            pert_mask=single_mask,
            pert_type_id=single_type,
            nperts=single_nperts,
            combo_id=None,
            chem_emb=None,
            chem_mask=None,
        )
        atoms = self.condition_delta_head(p_d).reshape(bsz, k_slots, -1)
        atom_mask = pert_mask.to(dtype=atoms.dtype).unsqueeze(-1)
        return (atoms * atom_mask).sum(dim=1)

    def predict_interaction_condition_delta(
        self,
        *,
        pert_gene_ids: Tensor,
        pert_mask: Tensor,
        pert_type_id: Optional[Tensor] = None,
        nperts: Optional[Tensor] = None,
        combo_id: Optional[Tensor] = None,
        chem_emb: Optional[Tensor] = None,
        chem_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return the diagnostic combo-minus-additive condition-delta residual."""
        combo_delta = self.predict_condition_delta(
            pert_gene_ids=pert_gene_ids,
            pert_mask=pert_mask,
            pert_type_id=pert_type_id,
            nperts=nperts,
            combo_id=combo_id,
            chem_emb=chem_emb,
            chem_mask=chem_mask,
        )
        additive_delta = self.predict_additive_condition_delta(
            pert_gene_ids=pert_gene_ids,
            pert_mask=pert_mask,
            pert_type_id=pert_type_id,
            nperts=nperts,
            combo_id=combo_id,
            chem_emb=chem_emb,
            chem_mask=chem_mask,
        )
        return combo_delta - additive_delta

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        x_0: Tensor,
        *,
        pert_gene_ids: Optional[Tensor] = None,
        pert_mask: Optional[Tensor] = None,
        pert_type_id: Optional[Tensor] = None,
        nperts: Optional[Tensor] = None,
        combo_id: Optional[Tensor] = None,
        chem_emb: Optional[Tensor] = None,
        chem_mask: Optional[Tensor] = None,
        support_context: Optional[Tensor] = None,
        support_context_present: Optional[Tensor] = None,
        support_set_task: Optional[Tensor] = None,
        support_set_task_present: Optional[Tensor] = None,
    ) -> Tensor:
        if not self.use_pert_condition:
            if any(
                x is not None
                for x in (pert_gene_ids, pert_mask, pert_type_id, nperts, combo_id, chem_emb, chem_mask)
            ):
                raise ValueError(
                    "ControlMLPVelocityField(use_pert_condition=False): do not pass perturbation tensors."
                )

        h_t = self.shared_enc(x_t)
        h_0 = self.shared_enc(x_0)

        h = self.fusion(torch.cat([h_t, h_0], dim=-1))

        c = self.t_embed(t) + self.ctrl_proj(h_0)
        if self.trackc_support_context_use_in_model:
            c = c + self._support_context_projection(
                support_context,
                x_t.shape[0],
                support_context_present=support_context_present,
            ).to(dtype=c.dtype)
        elif (
            (support_context is not None or support_context_present is not None)
            and not self.trackc_support_residual_use_in_model
            and not self.trackc_support_film_use_in_model
        ):
            raise ValueError(
                "support_context was passed but trackc_support_context_use_in_model=False"
            )
        if self.trackc_support_set_task_use_in_model:
            c = c + self._support_set_task_projection(
                support_set_task,
                x_t.shape[0],
                support_set_task_present=support_set_task_present,
            ).to(dtype=c.dtype)
        elif support_set_task is not None or support_set_task_present is not None:
            raise ValueError(
                "support_set_task was passed but trackc_support_set_task_use_in_model=False"
            )
        p_d = None
        if self.use_pert_condition and self.pert_encoder is not None:
            p_d = self._pert_projection(
                pert_gene_ids=pert_gene_ids,
                pert_mask=pert_mask,
                pert_type_id=pert_type_id,
                nperts=nperts,
                combo_id=combo_id,
                chem_emb=chem_emb,
                chem_mask=chem_mask,
            ).to(dtype=c.dtype)
            c = c + p_d
            if self.use_pert_in_fusion:
                h = h + p_d.to(dtype=h.dtype)
            if self.condition_delta_head_use_in_model:
                if self.condition_delta_head is None or self.condition_delta_to_c is None:
                    raise RuntimeError("condition_delta_head_use_in_model requires condition_delta_head")
                delta_c = self.condition_delta_to_c(self.condition_delta_head(p_d))
                gate = self._condition_delta_in_model_gate(
                    pert_gene_ids=pert_gene_ids,
                    pert_mask=pert_mask,
                    pert_type_id=pert_type_id,
                    nperts=nperts,
                    chem_mask=chem_mask,
                )
                if gate is not None:
                    delta_c = delta_c * gate.to(dtype=delta_c.dtype).unsqueeze(-1)
                c = c + delta_c.to(dtype=c.dtype)

        for block in self.blocks:
            h = block(h, c)

        h = self.out_norm(h)
        out = self.output_proj(h)
        if self.condition_lowrank_residual_use_in_model:
            if p_d is None or self.condition_lowrank_residual_down is None or self.condition_lowrank_residual_up is None:
                raise RuntimeError("condition_lowrank_residual_use_in_model requires perturbation conditioning")
            lowrank_residual = self.condition_lowrank_residual_up(
                F.silu(self.condition_lowrank_residual_down(p_d))
            )
            gate = self._condition_delta_in_model_gate(
                pert_gene_ids=pert_gene_ids,
                pert_mask=pert_mask,
                pert_type_id=pert_type_id,
                nperts=nperts,
                chem_mask=chem_mask,
            )
            if gate is not None:
                lowrank_residual = lowrank_residual * gate.to(dtype=lowrank_residual.dtype).unsqueeze(-1)
            out = out + lowrank_residual.to(dtype=out.dtype)
        if self.trackc_support_film_use_in_model:
            support_shift = self._support_residual_projection(
                support_context,
                x_t.shape[0],
                support_context_present=support_context_present,
            ).to(dtype=out.dtype)
            support_scale = self._support_film_scale_projection(
                support_context,
                x_t.shape[0],
                support_context_present=support_context_present,
            ).to(dtype=out.dtype)
            out = out + support_shift + support_scale * out.abs()
        elif self.trackc_support_residual_use_in_model:
            out = out + self._support_residual_projection(
                support_context,
                x_t.shape[0],
                support_context_present=support_context_present,
            ).to(dtype=out.dtype)
        return out
