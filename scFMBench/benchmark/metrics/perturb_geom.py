"""
Perturbation geometry in latent space (centroid shifts; optional cell-level OT).

Uses only numpy / pandas; optional ``POT`` (``pip install POT``) for exact EMD helpers,
or ``geomloss`` for energy distance between subsampled clouds.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import ot as _pot
except ImportError:
    _pot = None


def mean_latent_by_perturbation(
    latent: np.ndarray,
    obs: pd.DataFrame,
    pert_col: str,
    *,
    is_control_col: str = "is_control",
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Return (centroid_matrix, pert_names, is_control_flags aligned with pert_names).

    If ``is_control_col`` is present and has any True rows, row ``__control__`` is the
    pooled centroid of all control cells; otherwise centroids are computed for every
    distinct ``pert_col`` value (no control row).
    """
    if len(obs) != latent.shape[0]:
        raise ValueError(f"obs rows {len(obs)} != latent rows {latent.shape[0]}")
    has_ctrl = is_control_col in obs.columns
    ctrl = obs[is_control_col].astype(bool).to_numpy() if has_ctrl else np.zeros(len(obs), dtype=bool)
    perts = obs[pert_col].astype(str).to_numpy()

    names: List[str] = []
    rows: List[np.ndarray] = []
    flags: List[bool] = []

    if has_ctrl and ctrl.any():
        names.append("__control__")
        rows.append(latent[ctrl].mean(axis=0))
        flags.append(True)

    for p in sorted(np.unique(perts)):
        if has_ctrl:
            m = (perts == p) & (~ctrl)
        else:
            m = perts == p
        if not m.any():
            continue
        names.append(str(p))
        rows.append(latent[m].mean(axis=0))
        flags.append(False)

    C = np.stack(rows, axis=0)
    return C, names, np.array(flags, dtype=bool)


def centroid_shift_metrics(
    latent: np.ndarray,
    obs: pd.DataFrame,
    pert_col: str,
    *,
    is_control_col: str = "is_control",
) -> Dict[str, Any]:
    """L2 distance from global control centroid to each non-control perturbation centroid."""
    C, names, is_ctrl = mean_latent_by_perturbation(
        latent, obs, pert_col, is_control_col=is_control_col
    )
    ctrl_idx = np.where(is_ctrl)[0]
    if ctrl_idx.size != 1:
        return {
            "error": "need exactly one pooled control centroid (__control__); check is_control_col",
            "n_centroids": float(C.shape[0]),
        }
    mu0 = C[ctrl_idx[0]]
    dists = {}
    for i, name in enumerate(names):
        if is_ctrl[i]:
            continue
        dists[name] = float(np.linalg.norm(C[i] - mu0))
    arr = np.array(list(dists.values()), dtype=np.float64) if dists else np.array([])
    out: Dict[str, Any] = {
        "per_pert_l2": dists,
        "mean_l2_to_control": float(arr.mean()) if arr.size else float("nan"),
        "median_l2_to_control": float(np.median(arr)) if arr.size else float("nan"),
        "n_perts": float(len(dists)),
    }
    return out


def subsampled_emd2(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_n: int = 2048,
    seed: int = 0,
) -> Optional[float]:
    """
    POT ``emd2`` objective with Euclidean ground cost, subsampled for speed.

    Note: despite POT's function name, this is W1/EMD when the cost matrix is
    pairwise Euclidean distance. Use a squared cost matrix if W2^2 is desired.
    Returns None if POT unavailable.
    """
    if _pot is None:
        return None
    rng = np.random.default_rng(seed)
    if x.shape[0] > max_n:
        x = x[rng.choice(x.shape[0], size=max_n, replace=False)]
    if y.shape[0] > max_n:
        y = y[rng.choice(y.shape[0], size=max_n, replace=False)]
    M = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)
    a = np.ones(x.shape[0]) / x.shape[0]
    b = np.ones(y.shape[0]) / y.shape[0]
    return float(_pot.emd2(a, b, M))


def ot_delta_control_vs_pert(
    latent: np.ndarray,
    obs: pd.DataFrame,
    pert_col: str,
    pert_name: str,
    *,
    is_control_col: str = "is_control",
    max_n: int = 2048,
    seed: int = 0,
) -> Dict[str, Any]:
    """EMD between control cells and cells of one perturbation (optional POT)."""
    ctrl = obs[is_control_col].astype(bool).to_numpy()
    perts = obs[pert_col].astype(str).to_numpy()
    X0 = latent[ctrl]
    X1 = latent[(perts == pert_name) & (~ctrl)]
    emd = subsampled_emd2(X0, X1, max_n=max_n, seed=seed)
    return {"pert": pert_name, "emd_subsampled": emd, "n_ctrl": int(X0.shape[0]), "n_pert": int(X1.shape[0])}


def summarize_ot_deltas(
    latent: np.ndarray,
    obs: pd.DataFrame,
    pert_col: str,
    *,
    is_control_col: str = "is_control",
    max_perts: int = 50,
    max_n: int = 1024,
    seed: int = 0,
) -> Dict[str, Any]:
    """Mean/median EMD over first ``max_perts`` non-control perturbations (alphabetical)."""
    ctrl = obs[is_control_col].astype(bool).to_numpy()
    perts = obs[pert_col].astype(str).to_numpy()
    names = sorted(
        str(p)
        for p in np.unique(perts)
        if p and ((perts == p) & (~ctrl)).any()
    )
    vals: List[float] = []
    for p in names[:max_perts]:
        r = ot_delta_control_vs_pert(
            latent,
            obs,
            pert_col,
            p,
            is_control_col=is_control_col,
            max_n=max_n,
            seed=seed,
        )
        v = r.get("emd_subsampled")
        if v is not None:
            vals.append(v)
    return {
        "emd_mean": float(np.mean(vals)) if vals else None,
        "emd_median": float(np.median(vals)) if vals else None,
        "n_computed": len(vals),
        "pot_available": _pot is not None,
    }
