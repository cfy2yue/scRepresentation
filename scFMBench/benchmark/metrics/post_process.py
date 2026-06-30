"""
Latent post-processing: center / center-scale / TVN on control cells (Tx-Evaluation style).

Embeddings are ``(n_cells, d)`` aligned row-wise with ``metadata``. No AnnData dependency.
"""

from __future__ import annotations

from typing import Any, Hashable, Optional

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _control_mask(metadata: pd.DataFrame, pert_col: str, control_key: Any) -> np.ndarray:
    return (metadata[pert_col] == control_key).to_numpy()


def center_embeddings(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    control_key: Hashable,
    batch_col: Optional[str] = None,
) -> np.ndarray:
    """Subtract mean control embedding per batch (or global control mean)."""
    out = embeddings.astype(np.float64, copy=True)
    if batch_col is not None:
        for batch in metadata[batch_col].unique():
            batch_ind = (metadata[batch_col] == batch).to_numpy()
            ctrl = batch_ind & _control_mask(metadata, pert_col, control_key)
            if not ctrl.any():
                continue
            mu = embeddings[ctrl].mean(axis=0)
            out[batch_ind] -= mu
    else:
        ctrl = _control_mask(metadata, pert_col, control_key)
        if ctrl.any():
            out -= embeddings[ctrl].mean(axis=0)
    return out


def centerscale_on_controls(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    control_key: Hashable,
    batch_col: Optional[str] = None,
) -> np.ndarray:
    """Per-batch (or global) StandardScaler fit on control cells only, transform all."""
    out = embeddings.astype(np.float64, copy=True)
    if batch_col is not None:
        for batch in metadata[batch_col].unique():
            batch_ind = (metadata[batch_col] == batch).to_numpy()
            ctrl = batch_ind & _control_mask(metadata, pert_col, control_key)
            if not ctrl.any():
                continue
            out[batch_ind] = StandardScaler().fit(embeddings[ctrl]).transform(embeddings[batch_ind])
        return out
    ctrl = _control_mask(metadata, pert_col, control_key)
    if not ctrl.any():
        return out
    return StandardScaler().fit(embeddings[ctrl]).transform(out)


def tvn_on_controls(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    control_key: Hashable,
    batch_col: Optional[str] = None,
) -> np.ndarray:
    """TVN: global centerscale -> PCA fit on controls -> global centerscale -> optional per-batch whitening."""
    Z = embeddings.astype(np.float64, copy=True)
    Z = centerscale_on_controls(Z, metadata, pert_col, control_key)
    ctrl = _control_mask(metadata, pert_col, control_key)
    if not ctrl.any():
        return Z
    Z = PCA().fit(Z[ctrl]).transform(Z)
    Z = centerscale_on_controls(Z, metadata, pert_col, control_key)
    if batch_col is not None:
        for batch in metadata[batch_col].unique():
            batch_ind = (metadata[batch_col] == batch).to_numpy()
            ctrl_b = batch_ind & ctrl
            if not ctrl_b.any():
                continue
            cov = np.cov(Z[ctrl_b], rowvar=False, ddof=1) + 0.5 * np.eye(Z.shape[1])
            W = linalg.fractional_matrix_power(cov, -0.5)
            Z[batch_ind] = Z[batch_ind] @ W
    return Z
