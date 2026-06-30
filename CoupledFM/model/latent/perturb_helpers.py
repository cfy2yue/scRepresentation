"""Perturbation metadata helpers for latent FM."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from model.utils.conditioning.perturbation import ConditionMetadata


def condition_metadata_from_cond_string(cond: str) -> ConditionMetadata:
    """Parse HDF5 ``conditions`` string into :class:`ConditionMetadata` (no h5ad)."""
    return ConditionMetadata.from_obs_fields(cond, perturbation_field=None)


def null_perturbation_tensors(
    batch_size: int,
    max_genes: int,
    *,
    device: torch.device,
    dtype_ids: Optional[torch.dtype] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """All-control / unconditioned rows (encoder output zeros; chem slots ``None``)."""
    del dtype_ids  # API placeholder
    b = int(batch_size)
    k = max(1, int(max_genes))
    z = torch.zeros(b, k, dtype=torch.long, device=device)
    m = torch.zeros(b, k, dtype=torch.bool, device=device)
    tid = torch.zeros(b, dtype=torch.long, device=device)
    npt = torch.zeros(b, dtype=torch.long, device=device)
    combo = torch.zeros(b, dtype=torch.long, device=device)
    return z, m, tid, npt, combo, None, None
