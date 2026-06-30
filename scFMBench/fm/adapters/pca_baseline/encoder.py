"""
Dataset-fitted **PCA** baseline: fit PCA on **all cells** in one input AnnData,
then transform that same full dataset to get non-parametric cell representations.

The implementation lives under
``third_party/dataset_fitted_baseline/PCA/pca.py`` and is intentionally minimal.
This adapter only handles latent_bench metadata and path loading.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import paths




def _load_pca_impl():
    mod_path = (
        paths.third_party_root()
        / "dataset_fitted_baseline"
        / "PCA"
        / "pca.py"
    )
    if not mod_path.is_file():
        raise FileNotFoundError(f"PCA implementation not found: {mod_path}")
    spec = importlib.util.spec_from_file_location("latent_bench_dataset_fitted_pca", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def encode(
    adata: ad.AnnData,
    *,
    n_components: int = 50,
    random_state: int = 0,
    svd_solver: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Fit PCA on the **entire input dataset** in ``adata.X`` and return latents
    for every row.

    Args:
        adata: One dataset h5ad already loaded as AnnData. The expected benchmark
            path is a single file such as ``data/raw/DE5000_bench/<dataset>.h5ad``.
        n_components: Requested PCA dimension (clipped to
            ``min(n_obs, n_vars, n_components)``).
        random_state: Passed to PCA when applicable.
        svd_solver: sklearn PCA ``svd_solver``.

    Returns:
        ``(latent, meta)`` with ``latent`` shape ``(n_obs, n_components_actual)``.
    """
    n_obs, n_vars = adata.shape
    n_comp_req = int(n_components)
    if n_comp_req < 1:
        raise ValueError("n_components must be >= 1")
    if adata.X is None:
        raise ValueError(
            "PCA baseline requires adata.X to contain expression values. "
            "If this is a metadata-only benchmark h5ad, first align expression "
            "from a raw h5ad via adapters.dataset_fitted_io.attach_expression_from_h5ad "
            "or the runner's --pca-expression-from-h5ad option."
        )

    n_comp = int(min(n_comp_req, n_obs, n_vars))
    if n_comp < 1:
        raise ValueError(f"Cannot run PCA on shape {adata.shape}")

    pca_mod = _load_pca_impl()
    latent, pca = pca_mod.fit_transform_expression(
        adata.X,
        n_components=n_comp,
        random_state=random_state,
        svd_solver=svd_solver,
    )

    evr = pca.explained_variance_ratio_
    meta: dict[str, Any] = {
        "encoder_role": "ExpressionOnlyEncoder",
        "fit_scope": "dataset",
        "fit_method": "pca",
        "fit_target": "all_input_cells",
        "force_pert_effective": False,
        "pert_source": None,
        "n_components_requested": n_comp_req,
        "n_components_actual": n_comp,
        "n_obs": int(n_obs),
        "n_vars": int(n_vars),
        "explained_variance_ratio": evr.astype(np.float64).tolist(),
        "explained_variance_ratio_sum": float(np.sum(evr)),
        "svd_solver": svd_solver,
        "implementation": "third_party/dataset_fitted_baseline/PCA/pca.py",
    }
    return latent, meta
