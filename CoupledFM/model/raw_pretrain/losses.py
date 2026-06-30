"""Flow-matching losses aligned with ``model/train.py`` (masked MSE on velocity)."""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

LossType = Literal["mse", "smooth_l1"]


def compute_loss_weight(
    t: torch.Tensor,
    *,
    mode: str = "none",
    snr_gamma: float = 5.0,
) -> torch.Tensor:
    from model.utils.train.loss_weights import compute_loss_weight as _cw

    return _cw(t, mode=mode, snr_gamma=snr_gamma)


def velocity_loss(
    v_pred: torch.Tensor,
    dx_target: torch.Tensor,
    gene_mask: torch.Tensor,
    *,
    loss_type: LossType = "mse",
    smooth_beta: float = 1.0,
    time_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    gene_mask: same convention as CoupledFM training — 0 = visible, 1 = masked out.
    """
    vis = (1.0 - gene_mask).to(v_pred.dtype)
    d = vis.sum(dim=-1).clamp(min=1e-6)
    w_time = time_w if time_w is not None else torch.ones(
        v_pred.shape[0], device=v_pred.device, dtype=v_pred.dtype,
    )
    if loss_type == "mse":
        per_elem = (v_pred - dx_target).pow(2)
    elif loss_type == "smooth_l1":
        per_elem = F.smooth_l1_loss(
            v_pred, dx_target, beta=smooth_beta, reduction="none",
        )
    else:
        raise ValueError(loss_type)
    per_sample = (per_elem * vis).sum(dim=-1) / d
    per_sample = torch.where(vis.sum(dim=-1) > 0, per_sample, torch.zeros_like(per_sample))
    return (per_sample * w_time).mean()


def endpoint_loss(
    x_t: torch.Tensor,
    v_pred: torch.Tensor,
    t: torch.Tensor,
    x_gt: torch.Tensor,
    gene_mask: torch.Tensor,
    *,
    loss_type: LossType = "mse",
    smooth_beta: float = 1.0,
    time_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """One-step Euler endpoint: x1_hat = x_t + (1-t) v."""
    t_col = t.view(-1, 1)
    x1_hat = x_t + (1.0 - t_col) * v_pred
    return velocity_loss(
        x1_hat, x_gt, gene_mask, loss_type=loss_type, smooth_beta=smooth_beta, time_w=time_w,
    )


__all__ = [
    "velocity_loss",
    "endpoint_loss",
    "compute_loss_weight",
]
