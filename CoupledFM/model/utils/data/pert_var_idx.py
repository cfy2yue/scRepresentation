"""
Per-cell perturbation gene indices for AnnData.

Builds or reads ``obsm['pert_var_idx']``: (n_obs, max_pert) int32, -1 padded.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

import numpy as np


def parse_perturbation_string(cond: str) -> List[str]:
    """Split ``A+B``, ``A,B``, ``A B`` style condition into gene symbols."""
    s = str(cond).strip()
    if not s or s.lower() in ("control", "nan", "none", ""):
        return []
    out: List[str] = []
    for g in re.split(r"[\s+,_]+", s):
        g = g.strip()
        if g and not g.lower().startswith("chr"):
            out.append(g)
    return out


def perturbation_strings_to_matrix(
    perturbations: Sequence[str],
    var_names: Sequence[str],
) -> Tuple[np.ndarray, int]:
    """
    Map each cell's perturbation string to column indices in ``var_names``.

    Returns:
        matrix: (n_obs, max_pert) int32, -1 padding
        max_pert: width used
    """
    var_map = {str(v).upper(): i for i, v in enumerate(var_names)}
    rows: List[List[int]] = []
    max_p = 0
    for ps in perturbations:
        genes = parse_perturbation_string(ps)
        idxs = [var_map[g.upper()] for g in genes if g.upper() in var_map]
        rows.append(idxs)
        max_p = max(max_p, len(idxs))
    if max_p == 0:
        return np.full((len(rows), 1), -1, dtype=np.int32), 0
    mat = np.full((len(rows), max_p), -1, dtype=np.int32)
    for i, idxs in enumerate(rows):
        if idxs:
            mat[i, : len(idxs)] = idxs
    return mat, max_p


def ensure_pert_var_idx(
    adata,
    obs_key: str = "perturbation",
    inplace: bool = True,
) -> np.ndarray:
    """
    Return ``obsm['pert_var_idx']``, building it from ``obs[obs_key]`` if missing.

    If already present, returns a copy/view without overwriting unless empty.
    """
    import anndata as ad

    if not isinstance(adata, ad.AnnData):
        raise TypeError("expected AnnData")

    if "pert_var_idx" in adata.obsm and adata.obsm["pert_var_idx"].size > 0:
        return np.asarray(adata.obsm["pert_var_idx"], dtype=np.int32)

    if obs_key not in adata.obs.columns:
        z = np.full((adata.n_obs, 1), -1, dtype=np.int32)
        if inplace:
            adata.obsm["pert_var_idx"] = z
        return z

    pert_strs = adata.obs[obs_key].astype(str).tolist()
    mat, _ = perturbation_strings_to_matrix(pert_strs, adata.var_names)
    if inplace:
        adata.obsm["pert_var_idx"] = mat
    return mat


def pert_indices_for_cell(
    pert_var_idx: np.ndarray,
    cell_idx: int,
) -> List[int]:
    """Row ``cell_idx`` as a list of valid gene column indices (no -1)."""
    if pert_var_idx is None or pert_var_idx.size == 0:
        return []
    row = pert_var_idx[cell_idx]
    return [int(x) for x in np.asarray(row).ravel() if x >= 0]
