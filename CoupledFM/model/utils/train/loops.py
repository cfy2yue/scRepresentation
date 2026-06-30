"""Minimal training loop helpers (modules wire their own train())."""

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn


def train_step_fm(
    model: nn.Module,
    batch: tuple,
    loss_fn: Callable[..., torch.Tensor],
    scaler: Optional[Any] = None,
) -> torch.Tensor:
    """Generic single-step loss from batch; subclasses unpack batch themselves."""
    raise NotImplementedError("Use module-specific train.py")


def detach_loss(loss: torch.Tensor) -> float:
    return float(loss.detach().cpu().item())
