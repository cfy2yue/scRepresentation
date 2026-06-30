"""Condition vector encoders (placeholder until user design)."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from model.utils.data.dataset_base import COND_VEC_DIM


class ConditionEncoder(nn.Module, ABC):
    """Maps raw cond_vec or metadata to model conditioning."""

    @abstractmethod
    def forward(self, cond_vec: torch.Tensor) -> torch.Tensor:
        """Return tensor broadcastable to model (B, D)."""
        raise NotImplementedError


class IdentityConditionEncoder(ConditionEncoder):
    """Returns zeros (B, out_dim); replace with MLP / lookup later."""

    def __init__(self, out_dim: int = COND_VEC_DIM):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, cond_vec: torch.Tensor) -> torch.Tensor:
        b = cond_vec.shape[0]
        return cond_vec.new_zeros(b, self.out_dim)
