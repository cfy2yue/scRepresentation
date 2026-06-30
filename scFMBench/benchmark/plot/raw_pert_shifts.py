"""Compute centroid L2 distances between control and each perturbation in the
**raw (log1p-normalized) gene-expression** space.

These act as the ground-truth ranking against which each model's latent
scale-normalized centroid shifts are compared (Spearman correlation, top-K).

Sources:
- chempert: ``data/staging/chempert/<dataset_id>.h5ad`` (sciplex3_*)
- genepert: ``data/staging/genepert/<dataset_id>.h5ad``

Both staging files store log1p-normalized expression in ``X`` (sparse CSR) and
have a control marker column: ``is_control`` (bool) for genepert/chempert
staging, falling back to ``perturbation == "control"`` if absent.

Per-dataset results are cached as JSON in
``output/figures/_cache_raw_pert_shifts/<dataset_id>.json`` to avoid repeated
heavy I/O on rebuilds.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

LOG = logging.getLogger(__name__)
FM_ROOT = Path(__file__).resolve().parents[2] / "fm"
if str(FM_ROOT) not in sys.path:
    sys.path.insert(0, str(FM_ROOT))
import paths


def _staging_path(scfm_root: Path, dataset_id: str) -> Optional[Path]:
    if dataset_id.startswith("sciplex3_"):
        p = paths.staging_root() / "chempert" / f"{dataset_id}.h5ad"
    else:
        p = paths.staging_root() / "genepert" / f"{dataset_id}.h5ad"
    return p if p.is_file() else None


def _cache_path(out_dir: Path, dataset_id: str) -> Path:
    return out_dir / "_cache_raw_pert_shifts" / f"{dataset_id}.json"


def compute_raw_pert_diff_matrix(
    scfm_root: Path,
    dataset_id: str,
    perts: list[str],
    *,
    pert_col: str = "perturbation",
    control_label: str = "control",
) -> Optional[np.ndarray]:
    """Return ``(K, n_genes)`` matrix of raw centroid difference vectors
    ``mu_pert - mu_control`` in the staging log1p-normalized expression space,
    one row per ``perts[k]``. Rows for missing perturbations are filled with
    NaN; callers should mask those.

    Not cached (cheap given upstream caches), but the per-pert centroid
    expression mean is computed once per call.
    """
    staging = _staging_path(scfm_root, dataset_id)
    if staging is None:
        return None

    import anndata as ad
    import scipy.sparse as sp

    a = ad.read_h5ad(staging, backed="r")
    if pert_col not in a.obs.columns:
        return None

    pert_arr = a.obs[pert_col].astype(str).to_numpy()
    if "is_control" in a.obs.columns:
        is_ctrl = a.obs["is_control"].to_numpy().astype(bool)
    elif "control" in a.obs.columns and a.obs["control"].dtype.kind in "biu":
        is_ctrl = a.obs["control"].to_numpy().astype(bool)
    else:
        is_ctrl = pert_arr == control_label

    ctrl_idx = np.sort(np.where(is_ctrl)[0])
    if ctrl_idx.size < 2:
        return None
    Xc = a.X[ctrl_idx, :]
    mu_ctrl = (np.asarray(Xc.mean(axis=0)).ravel()
               if sp.issparse(Xc) else np.asarray(Xc).mean(axis=0))

    K = len(perts)
    n_genes = a.shape[1]
    out = np.full((K, n_genes), np.nan, dtype=np.float32)
    for i, p in enumerate(perts):
        mask = (pert_arr == p) & (~is_ctrl)
        idx = np.sort(np.where(mask)[0])
        if idx.size < 2:
            continue
        Xp = a.X[idx, :]
        mu_p = (np.asarray(Xp.mean(axis=0)).ravel()
                if sp.issparse(Xp) else np.asarray(Xp).mean(axis=0))
        out[i] = (mu_p - mu_ctrl).astype(np.float32)
    return out


def compute_raw_pert_shifts(
    scfm_root: Path,
    out_dir: Path,
    dataset_id: str,
    *,
    control_label: str = "control",
    pert_col: str = "perturbation",
    use_cache: bool = True,
) -> Dict[str, float]:
    """Return ``{perturbation_name: ||mu_pert - mu_control||_2}`` in raw expression space.

    Centroids are means over all cells in each group; the distance is the
    Euclidean norm of the centroid difference vector. Uses the dataset's own
    log1p-normalized ``X`` so the magnitude lives on a comparable scale across
    perturbations within the same dataset.

    Cached to ``out_dir/_cache_raw_pert_shifts/<dataset_id>.json`` after the
    first computation. Returns an empty dict if the staging file or required
    columns are missing.
    """
    cache = _cache_path(out_dir, dataset_id)
    if use_cache and cache.is_file():
        try:
            return {str(k): float(v) for k, v in json.loads(cache.read_text()).items()}
        except Exception as exc:
            LOG.warning("Ignoring unreadable raw-shift cache %s: %s", cache, exc)

    staging = _staging_path(scfm_root, dataset_id)
    if staging is None:
        LOG.warning("No staging .h5ad for %s; raw GT shifts unavailable.", dataset_id)
        return {}

    import anndata as ad
    import scipy.sparse as sp

    a = ad.read_h5ad(staging, backed="r")
    if pert_col not in a.obs.columns:
        LOG.warning("Staging %s missing column %s", staging, pert_col)
        return {}

    pert = a.obs[pert_col].astype(str).to_numpy()

    if "is_control" in a.obs.columns:
        is_ctrl = a.obs["is_control"].to_numpy().astype(bool)
    elif "control" in a.obs.columns and a.obs["control"].dtype.kind in "biu":
        is_ctrl = a.obs["control"].to_numpy().astype(bool)
    else:
        is_ctrl = pert == control_label

    if is_ctrl.sum() < 2:
        LOG.warning("Too few controls in %s (%d)", dataset_id, int(is_ctrl.sum()))
        return {}

    ctrl_idx = np.where(is_ctrl)[0]
    ctrl_idx_sorted = np.sort(ctrl_idx)
    X_ctrl = a.X[ctrl_idx_sorted, :]
    if sp.issparse(X_ctrl):
        mu_ctrl = np.asarray(X_ctrl.mean(axis=0)).ravel()
    else:
        mu_ctrl = np.asarray(X_ctrl).mean(axis=0)

    shifts: Dict[str, float] = {}
    perts = sorted(set(pert[~is_ctrl]) - {control_label})
    for p in perts:
        mask = (pert == p) & (~is_ctrl)
        idx = np.where(mask)[0]
        if idx.size < 2:
            continue
        idx_sorted = np.sort(idx)
        Xp = a.X[idx_sorted, :]
        if sp.issparse(Xp):
            mu_p = np.asarray(Xp.mean(axis=0)).ravel()
        else:
            mu_p = np.asarray(Xp).mean(axis=0)
        shifts[p] = float(np.linalg.norm(mu_p - mu_ctrl))

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(shifts, indent=2, sort_keys=True))
    return shifts
