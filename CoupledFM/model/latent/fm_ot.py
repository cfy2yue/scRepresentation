"""
Optimal Transport pairing and MMD utilities for Flow Matching.

OTPlanSampler is adapted from scDFM (Meta Flow Matching, CC-by-NC).
MMD functions are adapted from scDFM's run.py.
"""

import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

try:
    import ot as pot
except ImportError:
    pot = None


# ======================= OT Plan Sampler =======================

class OTPlanSampler:
    """Mini-batch OT pairing (squared Euclidean cost).

    Backends:
      - ``"exact"`` / ``"sinkhorn"``：CPU POT（原行为，会通过 ``OTPrefetchIter`` 的多线程路径）。
      - ``"torch_sinkhorn"``：GPU 纯 torch Sinkhorn（推荐，``OTPrefetchIter`` 走单线程 GPU 路径）。

    ``torch_sinkhorn`` 模式下 ``ot_fn`` 留空；真正的配对在 ``utils.data.ot_pairer.sinkhorn_pair``
    中完成，``OTPrefetchIter`` 读取本对象的 ``method / reg / n_iter`` 作为配置。
    """

    def __init__(
        self,
        method: str = "torch_sinkhorn",
        reg: float = 0.05,
        num_threads: int = 4,
        n_iter: int = 50,
    ):
        self.method = method
        self.reg = reg
        self.num_threads = num_threads
        self.n_iter = n_iter

        if method in ("exact", "sinkhorn"):
            if pot is None:
                raise ImportError(
                    "POT (Python Optimal Transport) is required for CPU backend: pip install pot"
                )
            if method == "exact":
                self.ot_fn = partial(pot.emd, numThreads=num_threads)
            else:
                self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "torch_sinkhorn":
            self.ot_fn = None  # GPU path, handled by OTPrefetchIter._iter_gpu
        else:
            raise ValueError(f"Unknown OT method: {method}")

    @staticmethod
    def _sq_euclidean_np(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        """||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b, via BLAS matmul."""
        x0 = np.ascontiguousarray(x0, dtype=np.float64)
        x1 = np.ascontiguousarray(x1, dtype=np.float64)
        a_sq = (x0 * x0).sum(axis=1, keepdims=True)   # (n, 1)
        b_sq = (x1 * x1).sum(axis=1, keepdims=True)    # (m, 1)
        M = a_sq + b_sq.T - 2.0 * (x0 @ x1.T)         # BLAS dgemm
        np.maximum(M, 0.0, out=M)
        return M

    @staticmethod
    def _assignment_from_plan_np(pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, m = pi.shape
        k_target = min(n, m)
        order = np.argsort(pi.ravel())[::-1]
        used_r = np.zeros(n, dtype=bool)
        used_c = np.zeros(m, dtype=bool)
        rows: list[int] = []
        cols: list[int] = []
        for li in order:
            if len(rows) >= k_target:
                break
            i, j = divmod(int(li), m)
            if used_r[i] or used_c[j]:
                continue
            used_r[i] = True
            used_c[j] = True
            rows.append(i)
            cols.append(j)
        if len(rows) < k_target:
            fr = np.flatnonzero(~used_r)
            fc = np.flatnonzero(~used_c)
            for t in range(min(len(fr), len(fc), k_target - len(rows))):
                rows.append(int(fr[t]))
                cols.append(int(fc[t]))
        return np.asarray(rows[:k_target], dtype=np.int64), np.asarray(cols[:k_target], dtype=np.int64)

    @staticmethod
    def _resample_indices_np(i: np.ndarray, j: np.ndarray, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        k = int(i.shape[0])
        if k <= 0:
            raise RuntimeError("OT assignment produced zero pairs.")
        if k < n_samples:
            idx = np.random.choice(k, size=n_samples, replace=True)
            return i[idx], j[idx]
        idx = np.random.permutation(k)[:n_samples]
        return i[idx], j[idx]

    def sample_plan_np(self, x0: np.ndarray, x1: np.ndarray, *, use_assignment: bool = False) -> tuple:
        """OT pairing entirely on numpy arrays (no torch, no GPU)."""
        a = np.ones(x0.shape[0], dtype=np.float64) / x0.shape[0]
        b = np.ones(x1.shape[0], dtype=np.float64) / x1.shape[0]
        M = self._sq_euclidean_np(x0, x1)
        pi = self.ot_fn(a, b, M)
        if not np.all(np.isfinite(pi)) or np.abs(pi.sum()) < 1e-8:
            pi = np.ones_like(pi) / pi.size
        if use_assignment:
            i, j = self._assignment_from_plan_np(pi)
            i, j = self._resample_indices_np(i, j, x0.shape[0])
            return x0[i], x1[j]
        p = pi.ravel()
        p = p / p.sum()
        choices = np.random.choice(len(p), p=p, size=x0.shape[0], replace=True)
        i, j = np.divmod(choices, pi.shape[1])
        return x0[i], x1[j]

    def sample_plan(self, x0: Tensor, x1: Tensor) -> tuple:
        """Return OT-paired (x0_reordered, x1_reordered). Tensor interface."""
        i, j = self._ot_indices(x0, x1)
        return x0[i], x1[j]

    def _ot_indices(self, x0: Tensor, x1: Tensor):
        a = np.ones(x0.shape[0], dtype=np.float64) / x0.shape[0]
        b = np.ones(x1.shape[0], dtype=np.float64) / x1.shape[0]
        M = torch.cdist(x0, x1).pow(2).detach().cpu().numpy()
        pi = self.ot_fn(a, b, M.astype(np.float64))
        if not np.all(np.isfinite(pi)) or np.abs(pi.sum()) < 1e-8:
            pi = np.ones_like(pi) / pi.size
        p = pi.ravel()
        p = p / p.sum()
        choices = np.random.choice(len(p), p=p, size=x0.shape[0], replace=True)
        return np.divmod(choices, pi.shape[1])


# ======================= CondOT Path =======================

@dataclass
class PathSample:
    x_0: Tensor
    x_1: Tensor
    t: Tensor
    x_t: Tensor
    dx_t: Tensor


class CondOTPath:
    """Linear interpolation path: x_t = (1-t)*x0 + t*x1, dx_t = x1 - x0."""

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> PathSample:
        t_ = t.unsqueeze(-1)  # (B, 1)
        x_t = (1.0 - t_) * x_0 + t_ * x_1
        dx_t = x_1 - x_0
        return PathSample(x_0=x_0, x_1=x_1, t=t, x_t=x_t, dx_t=dx_t)


# ======================= MMD =======================

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


def mmd2_biased(
    X: Tensor, Y: Tensor, sigmas: List[float],
    Dyy: Tensor = None,
) -> Tensor:
    """Multi-kernel biased MMD^2 (includes diagonal terms). Usually non-negative for RBF kernels."""
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
        vals.append(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())

    return torch.stack(vals).mean()
