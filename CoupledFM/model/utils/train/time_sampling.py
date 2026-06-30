"""Shared flow time t ~ [0,1] sampling for raw / coupled datasets and latent FM."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def sample_t(
    batch_rng: np.random.RandomState,
    B: int,
    mode: str = "logit_normal",
) -> np.ndarray:
    """Sample t in [0, 1] for a batch (numpy, CPU dataset iterator)."""
    mode = (mode or "uniform").lower()
    if mode == "uniform":
        return batch_rng.rand(B).astype(np.float32)
    if mode in ("logit_normal", "logit-normal"):
        x = batch_rng.standard_normal(B).astype(np.float32)
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
    if mode == "lognormal":
        # SD3-style: sample logit-normal via log-normal then squash
        z = batch_rng.standard_normal(B).astype(np.float32)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    raise ValueError(f"Unknown time_sampling mode: {mode!r}")


def sample_t_torch(
    B: int,
    device: torch.device,
    mode: str = "logit_normal",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample t in [0, 1], shape (B,), on ``device``."""
    mode = (mode or "uniform").lower()
    if mode == "uniform":
        return torch.rand(B, device=device, generator=generator)
    if mode in ("logit_normal", "logit-normal"):
        x = torch.randn(B, device=device, generator=generator)
        return torch.sigmoid(x)
    if mode == "lognormal":
        z = torch.randn(B, device=device, generator=generator)
        return torch.sigmoid(z)
    raise ValueError(f"Unknown time_sampling mode: {mode!r}")


__all__ = ["sample_t", "sample_t_torch"]
