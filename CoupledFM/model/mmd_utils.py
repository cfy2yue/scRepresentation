"""
MMD² utilities for training (aligned with FM/latent/fm_ot.py).
"""

from typing import List, Sequence

import torch
from torch import Tensor


def _pairwise_sq_dists(X: Tensor, Y: Tensor) -> Tensor:
    return torch.cdist(X, Y, p=2) ** 2


@torch.no_grad()
def median_sigmas(
    X: Tensor, scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    return_D2: bool = False,
) -> List[float]:
    """Adaptive bandwidth selection via median heuristic."""
    D2 = _pairwise_sq_dists(X, X)
    mask = ~torch.eye(D2.size(0), dtype=torch.bool, device=D2.device)
    tri = D2[mask]
    m = torch.median(tri).clamp_min(1e-12)
    sigs = [float(torch.sqrt(s * m).item()) for s in scales]
    if return_D2:
        return sigs, D2
    return sigs


def mmd2_unbiased(
    X: Tensor, Y: Tensor, sigmas: List[float],
    Dyy: Tensor = None,
) -> Tensor:
    """Multi-kernel unbiased MMD^2."""
    m, n = X.size(0), Y.size(0)
    Dxx = _pairwise_sq_dists(X, X)
    if Dyy is None:
        Dyy = _pairwise_sq_dists(Y, Y)
    Dxy = _pairwise_sq_dists(X, Y)

    vals = []
    for sigma in sigmas:
        beta = 1.0 / (2.0 * sigma ** 2 + 1e-12)
        Kxx = torch.exp(-beta * Dxx)
        Kyy = torch.exp(-beta * Dyy)
        Kxy = torch.exp(-beta * Dxy)

        txx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1) + 1e-12)
        tyy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1) + 1e-12)
        txy = Kxy.mean()
        vals.append(txx + tyy - 2.0 * txy)

    return torch.stack(vals).mean()
