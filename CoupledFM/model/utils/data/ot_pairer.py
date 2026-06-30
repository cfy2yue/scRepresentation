"""Optimal Transport pairing between control and GT latent embeddings.

**双后端**：
  - CPU（POT，``pot.emd`` / ``pot.sinkhorn``）：保留兼容，适合小 batch / 调试；
  - GPU（纯 PyTorch Sinkhorn，log-space stable）：默认推荐，把原本 CPU bound 的 OT
    放到 GPU，使训练循环不再被 ``pot.emd`` 拖住。
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import ot as pot
except ImportError:  # POT 可选
    pot = None

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # SciPy is optional unless ``ot_pair_mode=hungarian`` is used.
    linear_sum_assignment = None


# ===========================================================================
# Cost + assignment
# ===========================================================================

def compute_ot_cost(x0: torch.Tensor, x1: torch.Tensor, cost_fn: str = "l2") -> torch.Tensor:
    """Pairwise cost matrix (N, M)."""
    fn = (cost_fn or "l2").lower()
    if fn == "l2":
        return torch.cdist(x0, x1, p=2).pow(2)
    if fn == "cosine":
        u = F.normalize(x0, p=2, dim=1, eps=1e-12)
        v = F.normalize(x1, p=2, dim=1, eps=1e-12)
        sim = u @ v.t()
        return (1.0 - sim).clamp(min=0.0)
    if fn == "zscore_l2":
        u = (x0 - x0.mean(0, keepdim=True)) / (x0.std(0, keepdim=True).clamp_min(1e-6))
        v = (x1 - x1.mean(0, keepdim=True)) / (x1.std(0, keepdim=True).clamp_min(1e-6))
        return torch.cdist(u, v, p=2).pow(2)
    if fn == "rank_l2":
        r0 = x0.argsort(dim=0).argsort(dim=0).to(dtype=x0.dtype)
        r1 = x1.argsort(dim=0).argsort(dim=0).to(dtype=x1.dtype)
        return torch.cdist(r0, r1, p=2).pow(2)
    raise ValueError(f"Unknown OT cost_fn: {cost_fn!r}")


@torch.no_grad()
def assign_from_plan_greedy(pi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedy one-to-one matching from transport plan (N, M), max weight first."""
    n, m = pi.shape
    k_target = min(n, m)
    flat = pi.reshape(-1)
    order = torch.argsort(flat, descending=True)
    used_r = torch.zeros(n, device=pi.device, dtype=torch.bool)
    used_c = torch.zeros(m, device=pi.device, dtype=torch.bool)
    rows: list = []
    cols: list = []
    for s in range(order.numel()):
        if len(rows) >= k_target:
            break
        li = order[s]
        i = (li // m).long()
        j = (li % m).long()
        if used_r[i] or used_c[j]:
            continue
        used_r[i] = True
        used_c[j] = True
        rows.append(i)
        cols.append(j)
    # Fallback: fill missing with first free slots
    if len(rows) < k_target:
        fr = (~used_r).nonzero(as_tuple=True)[0]
        fc = (~used_c).nonzero(as_tuple=True)[0]
        for t in range(min(len(fr), len(fc), k_target - len(rows))):
            rows.append(fr[t])
            cols.append(fc[t])
    return torch.stack(rows[:k_target]), torch.stack(cols[:k_target])


def _resample_assignment_pairs(
    i: torch.Tensor,
    j: torch.Tensor,
    n_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand or subsample greedy one-to-one pairs to length ``n_samples``."""
    k = int(i.shape[0])
    if k == 0:
        raise RuntimeError("OT assignment produced zero pairs (empty cost / plan).")
    if k < n_samples:
        idx = torch.randint(
            0, k, (n_samples,), device=i.device, dtype=torch.long,
        )
        return i[idx], j[idx]
    perm = torch.randperm(k, device=i.device)[:n_samples]
    return i[perm], j[perm]


@torch.no_grad()
def assign_from_cost_hungarian(
    cost: torch.Tensor,
    n_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """True min-cost one-to-one assignment from a pairwise cost matrix."""
    if linear_sum_assignment is None:
        raise ImportError("scipy is required for ot_pair_mode='hungarian'")
    if cost.ndim != 2:
        raise ValueError(f"cost must be 2D, got shape={tuple(cost.shape)}")
    device = cost.device
    med = cost.median().clamp_min(1e-12)
    cost_np = (cost / med).detach().cpu().numpy()
    rows, cols = linear_sum_assignment(cost_np)
    i = torch.as_tensor(rows, device=device, dtype=torch.long)
    j = torch.as_tensor(cols, device=device, dtype=torch.long)
    return _resample_assignment_pairs(i, j, int(n_samples))


# ===========================================================================
# GPU Sinkhorn (pure torch, log-space)
# ===========================================================================

@torch.no_grad()
def sinkhorn_log_torch(
    cost: torch.Tensor,
    reg: float = 0.05,
    n_iter: int = 50,
    tol: float = 1e-5,
) -> torch.Tensor:
    """Log-stabilized Sinkhorn on a ``(n, m)`` cost matrix on GPU."""
    n, m = cost.shape
    device = cost.device
    dtype = cost.dtype

    log_a = torch.full((n,), -np.log(n), device=device, dtype=dtype)
    log_b = torch.full((m,), -np.log(m), device=device, dtype=dtype)
    K = -cost / reg

    log_u = torch.zeros(n, device=device, dtype=dtype)
    log_v = torch.zeros(m, device=device, dtype=dtype)
    check_convergence = bool(tol and tol > 0 and cost.device.type != "cuda")

    for _ in range(n_iter):
        log_u_new = log_a - torch.logsumexp(K + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(K + log_u_new.unsqueeze(1), dim=0)
        if check_convergence:
            diff = (log_u_new - log_u).abs().max()
            log_u = log_u_new
            if diff.item() < tol:
                break
            continue
        log_u = log_u_new

    pi = torch.exp(K + log_u.unsqueeze(1) + log_v.unsqueeze(0))
    if not torch.isfinite(pi).all() or pi.sum() < 1e-8:
        pi = torch.full_like(pi, 1.0 / (n * m))
    return pi


@torch.no_grad()
def sinkhorn_pair(
    x0: torch.Tensor,
    x1: torch.Tensor,
    n_samples: int,
    reg: float = 0.05,
    n_iter: int = 50,
    normalize_cost: bool = True,
    generator: Optional[torch.Generator] = None,
    cost_fn: str = "l2",
    use_assignment: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GPU Sinkhorn + either greedy one-to-one matching or multinomial sampling."""
    assert x0.device == x1.device, "x0 and x1 must be on the same device"

    cost = compute_ot_cost(x0, x1, cost_fn)
    if normalize_cost:
        med = cost.median().clamp_min(1e-12)
        cost = cost / med

    pi = sinkhorn_log_torch(cost, reg=reg, n_iter=n_iter)

    if use_assignment:
        i, j = assign_from_plan_greedy(pi)
        i, j = _resample_assignment_pairs(i, j, n_samples)
        return i, j

    p = pi.reshape(-1)
    p_sum = p.sum()
    if p_sum <= 0 or not torch.isfinite(p_sum):
        p = torch.full_like(p, 1.0 / p.numel())
    else:
        p = p / p_sum

    m = x1.shape[0]
    choices = torch.multinomial(
        p, num_samples=n_samples, replacement=True, generator=generator,
    )
    i = torch.div(choices, m, rounding_mode="floor")
    j = choices % m
    return i, j


@torch.no_grad()
def hungarian_pair(
    x0: torch.Tensor,
    x1: torch.Tensor,
    n_samples: int,
    cost_fn: str = "l2",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cost-only Hungarian pairing for explicit marginal-preserving ablations."""
    assert x0.device == x1.device, "x0 and x1 must be on the same device"
    cost = compute_ot_cost(x0, x1, cost_fn)
    return assign_from_cost_hungarian(cost, n_samples=n_samples)


# ===========================================================================
# Unified pairer
# ===========================================================================

class LatentOTPairer:
    """统一 OT pairer：CPU (POT) or GPU (torch sinkhorn)。"""

    def __init__(
        self,
        method: str = "torch_sinkhorn",
        num_threads: int = 4,
        reg: float = 0.05,
        n_iter: int = 50,
        device: Optional[torch.device] = None,
        cost_fn: str = "l2",
    ):
        self.method = method
        self.reg = reg
        self.n_iter = n_iter
        self.device = device
        self.cost_fn = cost_fn
        self._torch_generator: Optional[torch.Generator] = None

        if method in ("exact", "sinkhorn"):
            if pot is None:
                raise ImportError("POT required for CPU backend: pip install pot")
            if method == "exact":
                self.ot_fn = partial(pot.emd, numThreads=num_threads)
            else:
                self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "torch_sinkhorn":
            self.ot_fn = None
        else:
            raise ValueError(f"Unknown OT method: {method}")

    @staticmethod
    def _sq_euclidean(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        x0 = np.ascontiguousarray(x0, dtype=np.float32)
        x1 = np.ascontiguousarray(x1, dtype=np.float32)
        a_sq = (x0 * x0).sum(axis=1, keepdims=True)
        b_sq = (x1 * x1).sum(axis=1, keepdims=True)
        m = a_sq + b_sq.T - 2.0 * (x0 @ x1.T)
        np.maximum(m, 0.0, out=m)
        return m.astype(np.float64, copy=False)

    def pair(
        self,
        emb_ctrl: np.ndarray,
        emb_gt: np.ndarray,
        n_samples: int,
        use_assignment: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.method == "torch_sinkhorn":
            dev = self.device or (torch.device("cuda") if torch.cuda.is_available()
                                  else torch.device("cpu"))
            t0 = torch.as_tensor(emb_ctrl, dtype=torch.float32, device=dev)
            t1 = torch.as_tensor(emb_gt, dtype=torch.float32, device=dev)
            i, j = sinkhorn_pair(
                t0, t1, n_samples=n_samples,
                reg=self.reg, n_iter=self.n_iter,
                generator=self._torch_generator,
                cost_fn=self.cost_fn,
                use_assignment=use_assignment,
            )
            return i.cpu().numpy(), j.cpu().numpy()

        t0 = torch.from_numpy(np.ascontiguousarray(emb_ctrl, dtype=np.float32))
        t1 = torch.from_numpy(np.ascontiguousarray(emb_gt, dtype=np.float32))
        m = compute_ot_cost(t0, t1, self.cost_fn).double().numpy()

        a = np.ones(emb_ctrl.shape[0], dtype=np.float64) / emb_ctrl.shape[0]
        b = np.ones(emb_gt.shape[0], dtype=np.float64) / emb_gt.shape[0]
        pi = self.ot_fn(a, b, m)

        if not np.all(np.isfinite(pi)) or np.abs(pi.sum()) < 1e-8:
            pi = np.ones_like(pi) / pi.size

        if use_assignment:
            pi_t = torch.from_numpy(np.asarray(pi, dtype=np.float64)).to(
                dtype=torch.float32,
            )
            i, j = assign_from_plan_greedy(pi_t)
            i, j = _resample_assignment_pairs(i, j, n_samples)
            return i.cpu().numpy(), j.cpu().numpy()

        p = pi.ravel()
        p = p / p.sum()
        choices = np.random.choice(len(p), p=p, size=n_samples, replace=True)
        ctrl_idx, gt_idx = np.divmod(choices, pi.shape[1])
        return ctrl_idx, gt_idx

    def pair_torch(
        self,
        emb_ctrl: torch.Tensor,
        emb_gt: torch.Tensor,
        n_samples: int,
        cost_fn: Optional[str] = None,
        use_assignment: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cf = cost_fn if cost_fn is not None else self.cost_fn
        return sinkhorn_pair(
            emb_ctrl, emb_gt, n_samples=n_samples,
            reg=self.reg, n_iter=self.n_iter,
            generator=self._torch_generator,
            cost_fn=cf,
            use_assignment=use_assignment,
        )

    def set_generator(self, generator: torch.Generator) -> None:
        self._torch_generator = generator


__all__ = [
    "LatentOTPairer",
    "assign_from_plan_greedy",
    "compute_ot_cost",
    "hungarian_pair",
    "sinkhorn_log_torch",
    "sinkhorn_pair",
]
