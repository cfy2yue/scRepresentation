"""Per-sample loss weights over flow time t (e.g. Min-SNR style)."""

from __future__ import annotations

import torch


def compute_loss_weight(
    t: torch.Tensor,
    mode: str = "min_snr",
    snr_gamma: float = 5.0,
) -> torch.Tensor:
    """Return weight per batch element, shape (B,). ``t`` in [0, 1]."""
    mode = (mode or "none").lower()
    if mode in ("none", "uniform", ""):
        return torch.ones(t.shape[0], device=t.device, dtype=t.dtype)
    if mode == "min_snr":
        # SNR ~ t^2 / (1-t)^2 for linear interpolation FM; clip for stability
        eps = 1e-5
        a = t.clamp(eps, 1.0 - eps)
        snr = (a * a) / ((1.0 - a) * (1.0 - a) + eps)
        w = snr / (snr + float(snr_gamma))
        return w.clamp(min=1e-3)
    raise ValueError(f"Unknown loss_weighting mode: {mode!r}")


__all__ = ["compute_loss_weight"]
