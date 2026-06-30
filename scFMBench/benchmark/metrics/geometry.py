"""
Tier-2 latent geometry metrics (G1–G6) plus an LDM-readiness composite proxy.

All metrics operate on a latent matrix Z of shape (n, d), optionally with labels / batches in obs.
Designed to be cheap on large n via subsampling where noted.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

try:
    from .obs_io import read_obs_table
except ImportError:  # noqa: PERF203
    from obs_io import read_obs_table


def _subsample_idx(n: int, max_n: int, seed: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_n, replace=False)


def _cov_effective_rank(C: np.ndarray) -> float:
    """Shannon effective rank of covariance eigenvalues."""
    w = np.linalg.eigvalsh(C)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= 0:
        return 0.0
    p = w / s
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def g1_spectrum_metrics(Z: np.ndarray, *, max_cells_pca: int = 8000, seed: int = 0) -> Dict[str, float]:
    """G1: PCA variance decay, participation ratio, effective rank of covariance."""
    n = Z.shape[0]
    idx = _subsample_idx(n, max_cells_pca, seed)
    X = Z[idx].astype(np.float64)
    X -= X.mean(axis=0)
    d = X.shape[1]
    n_comp = min(d, X.shape[0] - 1, 200)
    if n_comp < 2:
        return {"G1_pca_90_ratio": float("nan"), "G1_participation_ratio": float("nan"), "G1_effective_rank_cov": float("nan")}
    pca = PCA(n_components=n_comp).fit(X)
    ev = pca.explained_variance_ratio_
    c90 = int(np.searchsorted(np.cumsum(ev), 0.9) + 1)
    pr = (ev.sum() ** 2) / (ev @ ev) if (ev @ ev) > 0 else float("nan")
    C = np.cov(X, rowvar=False, ddof=1)
    er = _cov_effective_rank(C)
    return {
        "G1_pca_90_ratio": float(np.cumsum(ev)[min(c90 - 1, len(ev) - 1)]),
        "G1_pca_k90": float(c90),
        "G1_participation_ratio": float(pr),
        "G1_effective_rank_cov": float(er),
    }


def g2_local_label_consistency(
    Z: np.ndarray,
    labels: np.ndarray,
    *,
    n_neighbors: int = 15,
    max_cells: int = 10000,
    seed: int = 0,
) -> Dict[str, float]:
    """G2: mean fraction of kNN sharing the same discrete label."""
    n = Z.shape[0]
    k = min(n_neighbors, n - 1)
    if k < 1:
        return {"G2_knn_label_consistency": float("nan")}
    idx = _subsample_idx(n, max_cells, seed)
    X = Z[idx].astype(np.float64)
    lab = np.asarray(labels)[idx]
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    neigh = nn.kneighbors(X, return_distance=False)[:, 1:]
    same = (lab[neigh] == lab[:, None]).mean(axis=1).mean()
    return {"G2_knn_label_consistency": float(same)}


def g3_isotropy(Z: np.ndarray, *, max_cells: int = 8000, seed: int = 0) -> Dict[str, float]:
    """G3: anisotropy via λ_max / tr(C); lower means more isotropic."""
    idx = _subsample_idx(Z.shape[0], max_cells, seed)
    X = Z[idx].astype(np.float64)
    X -= X.mean(axis=0)
    C = np.cov(X, rowvar=False, ddof=1)
    w = np.linalg.eigvalsh(C)
    w = np.clip(w, 0.0, None)
    tr = w.sum()
    lam_max = float(w.max()) if w.size else 0.0
    ratio = float(lam_max / tr) if tr > 1e-12 else float("nan")
    cond = float(lam_max / (w.min() + 1e-12)) if w.size else float("nan")
    return {"G3_lambda_max_over_trace": ratio, "G3_condition_eig": cond}


def g4_label_silhouette(
    Z: np.ndarray,
    labels: np.ndarray,
    *,
    max_cells: int = 5000,
    seed: int = 0,
) -> Dict[str, float]:
    """G4: silhouette on subsample (Euclidean); requires ≥2 classes."""
    n = Z.shape[0]
    idx = _subsample_idx(n, max_cells, seed)
    X = Z[idx].astype(np.float64)
    lab = np.asarray(labels)[idx]
    uniq = np.unique(lab)
    if len(uniq) < 2:
        return {"G4_silhouette_euclidean": float("nan")}
    s = silhouette_score(X, lab, metric="euclidean")
    return {"G4_silhouette_euclidean": float(s)}


def g5_noise_stability(
    Z: np.ndarray,
    *,
    sigma: float = 0.01,
    sub_n: int = 512,
    seed: int = 0,
) -> Dict[str, float]:
    """G5: Spearman correlation of pairwise distances before vs after Gaussian noise (subsample)."""
    from scipy.stats import spearmanr

    n = Z.shape[0]
    m = min(sub_n, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)
    X = Z[idx].astype(np.float64)
    D0 = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    tri = np.triu_indices(m, k=1)
    v0 = D0[tri]
    noise = rng.standard_normal(X.shape) * sigma * (X.std(axis=0, keepdims=True) + 1e-8)
    Xn = X + noise
    D1 = np.linalg.norm(Xn[:, None, :] - Xn[None, :, :], axis=2)
    v1 = D1[tri]
    r, _ = spearmanr(v0, v1)
    return {"G5_dist_spearman_under_noise": float(r)}


def g6_laplacian_dirichlet_energy(
    Z: np.ndarray,
    *,
    n_neighbors: int = 15,
    max_cells: int = 6000,
    seed: int = 0,
) -> Dict[str, float]:
    """G6: average squared edge difference 0.5 Σ w_ij ||z_i-z_j||² / n on kNN graph (Gaussian weights)."""
    n = Z.shape[0]
    k = min(n_neighbors, n - 1)
    if k < 1:
        return {"G6_laplacian_energy": float("nan")}
    idx = _subsample_idx(n, max_cells, seed)
    X = Z[idx].astype(np.float64)
    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    dist, ind = nn_model.kneighbors(X)
    dist, ind = dist[:, 1:], ind[:, 1:]
    sigma = np.median(dist[:, -1]) + 1e-8
    w = np.exp(-(dist**2) / (2 * sigma**2))
    m = X.shape[0]
    energy = 0.0
    wsum = 0.0
    for i in range(m):
        for t in range(k):
            j = ind[i, t]
            ww = w[i, t]
            energy += 0.5 * ww * float(np.sum((X[i] - X[j]) ** 2))
            wsum += ww
    if wsum <= 0:
        return {"G6_laplacian_energy": float("nan")}
    return {"G6_laplacian_energy": float(energy / m)}


def ldm_readiness_proxy(geom: Dict[str, float]) -> Dict[str, float]:
    """
    Heuristic composite in [0, 1]: favors stable distances, moderate spectrum, not extreme anisotropy.
    """
    g5 = geom.get("G5_dist_spearman_under_noise", 0.0)
    if np.isnan(g5):
        g5 = 0.0
    g5_n = max(0.0, min(1.0, (g5 + 1) / 2))

    pr = geom.get("G1_participation_ratio", float("nan"))
    if np.isnan(pr) or pr <= 0:
        g1n = 0.0
    else:
        g1n = float(np.tanh(pr / 10.0))

    ani = geom.get("G3_lambda_max_over_trace", float("nan"))
    if np.isnan(ani):
        g3n = 0.0
    else:
        g3n = float(1.0 / (1.0 + np.exp(25.0 * (ani - 0.35))))

    score = (g5_n + g1n + g3n) / 3.0
    return {
        "LDM_proxy_score": float(score),
        "LDM_proxy_g5_stability": float(g5_n),
        "LDM_proxy_g1_participation": float(g1n),
        "LDM_proxy_g3_isotropy": float(g3n),
    }


def run_geometry_metrics(
    Z: np.ndarray,
    obs: Optional[pd.DataFrame] = None,
    *,
    label_col: Optional[str] = None,
    batch_col: Optional[str] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run G1–G6 (+ optional label-based metrics) and LDM proxy."""
    Z = np.asarray(Z, dtype=np.float64)
    out: Dict[str, Any] = {}
    out.update(g1_spectrum_metrics(Z, seed=seed))
    out.update(g3_isotropy(Z, seed=seed))
    out.update(g5_noise_stability(Z, seed=seed))
    out.update(g6_laplacian_dirichlet_energy(Z, seed=seed))

    if obs is not None and label_col is not None and label_col in obs.columns:
        labels = obs[label_col].astype(str).to_numpy()
        out.update(g2_local_label_consistency(Z, labels, seed=seed))
        out.update(g4_label_silhouette(Z, labels, seed=seed))

    out.update(ldm_readiness_proxy(out))

    if batch_col is not None and obs is not None and batch_col in obs.columns:
        batches = obs[batch_col].astype(str).to_numpy()
        out["batch_n_unique"] = float(len(np.unique(batches)))
    return out


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latent", type=Path, required=True)
    ap.add_argument("--obs", type=Path, required=True, help="obs table (.parquet or .csv.gz; needs target columns)")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--label-col", type=str, default=None)
    ap.add_argument("--batch-col", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Z = np.load(args.latent)
    obs = read_obs_table(args.obs)
    metrics = run_geometry_metrics(
        Z, obs, label_col=args.label_col, batch_col=args.batch_col, seed=args.seed
    )
    metrics["n_cells"] = int(Z.shape[0])
    metrics["latent_dim"] = int(Z.shape[1])
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
