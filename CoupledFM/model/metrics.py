"""
Perturbation prediction evaluation metrics.

Implements:
  - PearsonDelta: Pearson correlation on delta profiles (pred - ref) vs (gt - ref)
    with control-center or perturbation-center reference.
    Reference: Systema (Vinas et al., Nature Biotechnology 2025)
  - MMD: Multi-kernel unbiased MMD^2 with median-heuristic bandwidth.
    Reference: scDFM
"""

from typing import List

import numpy as np
import torch
from scipy.stats import pearsonr


def pearson_delta(pred_mean: np.ndarray, gt_mean: np.ndarray,
                  reference: np.ndarray) -> float:
    """Pearson correlation on delta profiles relative to a reference center.

    Args:
        pred_mean: predicted pseudobulk mean expression (G,)
        gt_mean: ground-truth pseudobulk mean expression (G,)
        reference: reference center expression (G,), e.g. ctrl_mean or ir_mean

    Returns:
        Pearson r between (gt_mean - reference) and (pred_mean - reference).
        Returns ``nan`` if either delta vector has zero variance (undefined correlation),
        matching ``utils.train.metrics.pearson_np`` semantics.
    """
    delta_gt = gt_mean - reference
    delta_pred = pred_mean - reference
    if np.std(delta_gt) < 1e-12 or np.std(delta_pred) < 1e-12:
        return float("nan")
    r, _ = pearsonr(delta_gt, delta_pred)
    return float(r) if np.isfinite(r) else float("nan")


# ── MMD ──────────────────────────────────────────────────────────

def _pairwise_sq_dists(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(X, Y, p=2) ** 2


@torch.no_grad()
def _median_sigmas(X: torch.Tensor,
                   scales=(0.5, 1.0, 2.0, 4.0)) -> List[float]:
    D2 = _pairwise_sq_dists(X, X)
    mask = ~torch.eye(D2.size(0), dtype=torch.bool, device=D2.device)
    m = torch.median(D2[mask]).clamp_min(1e-12)
    s2 = torch.tensor(scales, device=X.device) * m
    return [float(s.sqrt().item()) for s in s2]


@torch.no_grad()
def mmd2_multi_sigma(X: torch.Tensor, Y: torch.Tensor,
                     scales=(0.5, 1.0, 2.0, 4.0)) -> float:
    """Multi-kernel unbiased MMD^2 with median-heuristic bandwidth.

    Args:
        X: generated samples (m, d)
        Y: reference samples (n, d)
        scales: bandwidth multipliers for median heuristic

    Returns:
        Scalar MMD^2 value (lower is better, 0 = identical distributions).
    """
    sigmas = _median_sigmas(Y, scales)
    m, n = X.size(0), Y.size(0)
    if m < 2 or n < 2:
        return float("nan")

    Dxx = _pairwise_sq_dists(X, X)
    Dyy = _pairwise_sq_dists(Y, Y)
    Dxy = _pairwise_sq_dists(X, Y)

    vals = []
    for sigma in sigmas:
        beta = 1.0 / (2.0 * sigma ** 2 + 1e-12)
        Kxx = torch.exp(-beta * Dxx)
        Kyy = torch.exp(-beta * Dyy)
        Kxy = torch.exp(-beta * Dxy)

        term_xx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1) + 1e-12)
        term_yy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1) + 1e-12)
        term_xy = Kxy.mean()
        vals.append(term_xx + term_yy - 2.0 * term_xy)

    return float(torch.stack(vals).mean().item())
