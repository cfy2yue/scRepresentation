"""LoRA adapters for selective linear fine-tuning."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps ``nn.Linear`` with low-rank delta; base weights frozen."""

    def __init__(self, linear: nn.Linear, rank: int):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False
        self.rank = rank
        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.empty(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.lora_A.t() @ self.lora_B.t())


def apply_lora_to_model(
    root: nn.Module,
    rank: int,
    target_substrs: Sequence[str],
) -> int:
    """In-place replace matching ``nn.Linear`` modules with :class:`LoRALinear`.

    Returns the number of modules replaced.
    """
    n_replaced = 0

    def _visit(module: nn.Module, prefix: str) -> None:
        nonlocal n_replaced
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LoRALinear):
                _visit(child, full)
                continue
            if isinstance(child, nn.Linear) and any(s in full for s in target_substrs):
                setattr(module, name, LoRALinear(child, rank))
                n_replaced += 1
            else:
                _visit(child, full)

    _visit(root, "")
    return n_replaced


__all__ = ["LoRALinear", "apply_lora_to_model"]
