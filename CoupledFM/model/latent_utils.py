"""
Frozen latent FM model wrapper for inference-time CLS guidance.

Loads the best latent ControlMLP checkpoint and provides:
  - encode_z_t(z_src, t): linear interpolation for training
  - ode_step(z, z_src, t, dt): one Euler step for inference
  - ode_integrate(z_src, n_steps): full ODE t=0→1

Multi-GPU (DDP): each rank constructs ``FrozenLatentFM`` on its own ``cuda:{local_rank}`` and
runs ``ode_at_t`` / ``predict_velocity`` on **that rank's batch only** — no cross-rank latent
all-gather；与主模型 DDP 一致的数据并行（每卡一份冻结 latent 权重，显存线性于 rank 数）。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def _src_dir() -> Path:
    return Path(__file__).resolve().parent


def _resolve_latent_code_root() -> Path:
    """Root of **latent FM** codebase (contains ``models/mlp.py``).

    Set ``LATENT_FM_CODE_ROOT`` to use a sibling checkout.
    Default: bundled ``model/latent`` under the CoupledFM repo.
    """
    raw_s = os.environ.get("LATENT_FM_CODE_ROOT", "").strip()
    if raw_s:
        return Path(raw_s).expanduser().resolve()
    return (_src_dir() / "latent").expanduser().resolve()


def _ensure_repo_paths() -> None:
    """Ensure repo root is on ``sys.path`` so ``import model.*`` resolves when loading MLP via importlib."""
    repo_root = _src_dir().parent
    p = str(repo_root)
    if p not in sys.path:
        sys.path.insert(0, p)


def _dynamic_load_control_mlp_class():
    latent_dir = _resolve_latent_code_root()
    mlp_py = latent_dir / "models" / "mlp.py"
    if not mlp_py.is_file():
        raise FileNotFoundError(
            f"FrozenLatentFM: missing latent MLP at {mlp_py}. "
            f"Clone the latent FM codebase or set LATENT_FM_CODE_ROOT."
        )
    spec = importlib.util.spec_from_file_location(
        "latent_models_mlp", str(mlp_py),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ControlMLPVelocityField


def _predict_velocity_inner(
    model: nn.Module,
    z_t: Tensor,
    t: Tensor,
    z_src: Tensor,
    perturbation_batch: Optional[Tuple[Tensor, ...]],
    *,
    require_pert_when_enabled: bool,
    max_pert_genes: int,
) -> Tensor:
    if getattr(model, "use_pert_condition", False):
        use_pb = perturbation_batch
        if use_pb is None and not require_pert_when_enabled:
            from model.latent.perturb_helpers import null_perturbation_tensors

            gid, mk, tid, npt, cid = null_perturbation_tensors(
                z_t.shape[0], max_pert_genes, device=z_t.device
            )
            use_pb = (gid, mk, tid, npt, cid)
        if use_pb is None:
            raise ValueError(
                "use_pert_condition checkpoint requires perturbation_batch=... "
                "or pass require_pert_when_enabled=False for null perturbation tensors."
            )
        gid, mk, tid, npt, cid = use_pb
        return model(
            z_t,
            t,
            z_src,
            pert_gene_ids=gid,
            pert_mask=mk,
            pert_type_id=tid,
            nperts=npt,
            combo_id=cid,
        )
    if perturbation_batch is not None:
        raise ValueError("Legacy checkpoint: do not pass perturbation_batch.")
    return model(z_t, t, z_src)


class FrozenLatentFM(nn.Module):
    """Frozen latent flow matching model for CLS guidance.

    DDP 训练时：每个进程各自 ``to(device)`` 一份；前向只处理本 rank 的 ``(B, D)`` 张量，无需跨卡通信。
    """

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
    ):
        super().__init__()
        _ensure_repo_paths()

        ControlMLPVelocityField = _dynamic_load_control_mlp_class()

        ckpt_path_p = Path(ckpt_path)
        ckpt = torch.load(str(ckpt_path_p), map_location="cpu", weights_only=False)

        config = ckpt.get("config", None)
        if config is None:
            config_path = ckpt_path_p.parent / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
        cfg: Dict[str, Any] = dict(config or {})

        emb_dim = int(cfg.get("emb_dim", 2058))
        d_model = int(cfg.get("mlp_d_model", 512))
        n_layers = int(cfg.get("mlp_n_layers", 8))
        mlp_ratio = float(cfg.get("mlp_ratio", 4.0))
        self.max_pert_genes = int(cfg.get("max_pert_genes", 16))

        kwargs: Dict[str, Any] = dict(
            emb_dim=emb_dim,
            d_model=d_model,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            dropout=0.0,
        )

        use_pert = bool(cfg.get("use_pert_condition", False))
        pretrained_cache = None
        if use_pert:
            from model.condition_emb.genepert.gene_cache import GeneEmbeddingCache

            mode_s = str(cfg.get("pert_embed_mode", "random_learned")).lower().strip()
            if mode_s.startswith("pretrained"):
                cached = str(cfg.get("pert_gene_emb_cache_dir", "") or "").strip()
                if not cached:
                    raise ValueError(
                        "pretrained perturbation_encoder requires pert_gene_emb_cache_dir in saved config"
                    )
                pretrained_cache = GeneEmbeddingCache(Path(cached).expanduser())
            kwargs.update(
                use_pert_condition=True,
                pert_embed_mode=str(cfg.get("pert_embed_mode", "random_learned")),
                pert_cond_dim=int(cfg.get("pert_cond_dim", d_model)),
                pert_type_emb_dim=int(cfg.get("pert_type_emb_dim", 32)),
                pert_encoder_num_embeddings=int(cfg.get("pert_encoder_num_embeddings", 8192)),
                pert_gene_emb_dim=int(cfg.get("pert_gene_emb_dim", 256)),
                pert_encoder_dropout=float(cfg.get("pert_encoder_dropout", 0.0)),
                max_combo_id_exclusive=int(cfg.get("max_combo_id_exclusive", 4096)),
                gene_embedding_cache=pretrained_cache,
            )

        self.model = ControlMLPVelocityField(**kwargs)
        missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
        n_total = len(self.model.state_dict())
        if len(missing) > max(8, n_total // 2):
            raise ValueError(
                f"[FrozenLatentFM] incompatible checkpoint: {len(missing)}/{n_total} keys missing "
                f"(first missing: {missing[:5]}). Re-export ckpt or fix ControlMLPVelocityField layout."
            )
        if missing or unexpected:
            warnings.warn(
                f"[FrozenLatentFM] load_state_dict strict=False; missing={len(missing)} unexpected={len(unexpected)}",
                stacklevel=1,
            )
        self.model.eval()
        self.model.to(device)

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.emb_dim = emb_dim
        self._device = device
        self._config = cfg

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[FrozenLatentFM] loaded {ckpt_path}, {n_params:,} params (frozen)")

    @torch.no_grad()
    def make_null_perturbation_tensors(
        self, batch_size: int, *, device: Optional[torch.device] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Control / unconditioned rows for :meth:`predict_velocity` when ``use_pert_condition`` is on."""
        from model.latent.perturb_helpers import null_perturbation_tensors

        dev = device or torch.device(self._device)
        return null_perturbation_tensors(
            int(batch_size), self.max_pert_genes, device=dev
        )

    @torch.no_grad()
    def predict_velocity(
        self,
        z_t: Tensor,
        t: Tensor,
        z_src: Tensor,
        perturbation_batch: Optional[Tuple[Tensor, ...]] = None,
        *,
        require_pert_when_enabled: bool = False,
    ) -> Tensor:
        """v_latent(z_t, t, z_src) → velocity in latent space.

        For pert-conditioned checkpoints, pass ``perturbation_batch`` (tuple on ``z_t.device``), or
        ``require_pert_when_enabled=True`` raises if you omit tensors; default ``False`` uses null perturbation rows.
        """
        return _predict_velocity_inner(
            self.model,
            z_t,
            t,
            z_src,
            perturbation_batch,
            require_pert_when_enabled=require_pert_when_enabled,
            max_pert_genes=self.max_pert_genes,
        )

    @torch.no_grad()
    def ode_step(
        self,
        z: Tensor,
        z_src: Tensor,
        t_val: float,
        dt: float,
        perturbation_batch: Optional[Tuple[Tensor, ...]] = None,
        *,
        require_pert_when_enabled: bool = False,
    ) -> Tensor:
        """One Euler step: z_{t+dt} = z_t + dt * v(z_t, t, z_src)."""
        B = z.shape[0]
        t = torch.full((B,), t_val, device=z.device, dtype=torch.float32)
        v = self.predict_velocity(
            z,
            t,
            z_src,
            perturbation_batch,
            require_pert_when_enabled=require_pert_when_enabled,
        )
        return z + dt * v

    @torch.no_grad()
    def ode_integrate(
        self,
        z_src: Tensor,
        n_steps: int = 100,
        perturbation_batch: Optional[Tuple[Tensor, ...]] = None,
        *,
        require_pert_when_enabled: bool = False,
    ) -> Tensor:
        """Euler integration from t=0 (z_src) to t=1. Returns predicted z_1."""
        dt = 1.0 / n_steps
        z = z_src.clone()
        for i in range(n_steps):
            z = self.ode_step(
                z,
                z_src,
                i * dt,
                dt,
                perturbation_batch,
                require_pert_when_enabled=require_pert_when_enabled,
            )
        return z

    @torch.no_grad()
    def ode_integrate_trajectory(
        self,
        z_src: Tensor,
        n_steps: int = 100,
        perturbation_batch: Optional[Tuple[Tensor, ...]] = None,
        *,
        require_pert_when_enabled: bool = False,
    ) -> Tensor:
        """Returns full trajectory (n_steps+1, B, D) including z_0=z_src."""
        dt = 1.0 / n_steps
        z = z_src.clone()
        traj = [z.clone()]
        for i in range(n_steps):
            z = self.ode_step(
                z,
                z_src,
                i * dt,
                dt,
                perturbation_batch,
                require_pert_when_enabled=require_pert_when_enabled,
            )
            traj.append(z.clone())
        return torch.stack(traj, dim=0)

    @torch.no_grad()
    def ode_at_t(
        self,
        z_src: Tensor,
        t_values: Tensor,
        n_steps: int = 20,
        perturbation_batch: Optional[Tuple[Tensor, ...]] = None,
        *,
        require_pert_when_enabled: bool = False,
    ) -> Tensor:
        """Euler-integrate from z_src to each sample's target time t.

        For a batch where each sample has a different t in [0, 1],
        runs shared Euler steps and picks z(t) via linear interpolation
        between the two nearest step boundaries.

        Args:
            z_src:    (B, D) initial latent embeddings (flow source)
            t_values: (B,)   target times in [0, 1]
            n_steps:  number of Euler steps for the full [0, 1] interval

        Returns:
            z_t: (B, D) latent state at each sample's target time
        """
        dt = 1.0 / n_steps
        z = z_src.clone()
        result = z_src.clone()

        for i in range(n_steps):
            t_step = i * dt
            t_next = (i + 1) * dt
            t_tensor = torch.full_like(t_values, t_step, dtype=torch.float32)
            v = self.predict_velocity(
                z,
                t_tensor,
                z_src,
                perturbation_batch,
                require_pert_when_enabled=require_pert_when_enabled,
            )
            z_next = z + dt * v

            mask = (t_values >= t_step) & (t_values < t_next)
            if mask.any():
                alpha = ((t_values[mask] - t_step) / dt).unsqueeze(-1)
                result[mask] = (1.0 - alpha) * z[mask] + alpha * z_next[mask]

            z = z_next

        tail = t_values >= 1.0 - 1e-6
        if tail.any():
            result[tail] = z[tail]

        return result
