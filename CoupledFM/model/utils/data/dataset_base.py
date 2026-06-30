"""Base dataset contract: control → GT flow (no IR)."""

from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, Tuple

import torch
from torch.utils.data import IterableDataset

# Canonical obsm keys (new project)
OBS_KEYS = {
    "emb": "emb",
    "cond_vec": "cond_vec",
}

# Placeholder condition vector dimension until user design
COND_VEC_DIM = 128


class BaseFMDataset(IterableDataset, ABC):
    """Iterable dataset yielding 9-tuples for FM training.

    Yield per batch (conceptual; concrete subclasses implement ``__iter__``):

    0. x_t: (B, G_vocab) float32 — interpolated state
    1. x_ctrl: (B, G_vocab) float32 — paired control
    2. t: (B,) float32 — flow time
    3. gene_ids_valid: (G_vocab,) int64
    4. dx_t: (B, G_vocab) float32 — target velocity / residual (e.g. x_gt - x_ctrl)
    5. ds_name: str
    6. cond: str — perturbation condition name
    7. cond_vec: (B, D_cond) float32 — placeholder zeros (COND_VEC_DIM)
    8. latent_data: Optional[Tensor] — z_ctrl / z_t for CoupledFM; None for baseline
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[Any, ...]]:
        raise NotImplementedError

    @staticmethod
    def placeholder_cond_vec(batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, COND_VEC_DIM, device=device, dtype=torch.float32)
