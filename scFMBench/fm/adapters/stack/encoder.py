"""
arc-stack (StateICL / Stack-Large) cell-embedding adapter.

Benchmark protocol: encoding uses only ``adata.X`` (after optional log1p undo). No
perturbation metadata is passed as a separate conditional input.

Stack applies a full-gene linear ``gene_reduction`` and tabular attention—no gene
subset sampling or truncation—so there is no protected-gene coverage path.
``obsm['pert_var_idx']``, if present, is reflected only in optional manifest
``pert_kept_histogram`` / ``pert_source`` for cross-adapter consistency; ``force_pert``
does not alter the forward pass.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import anndata as ad
import numpy as np
import paths

from .._common import histogram_pert_kept




def _ensure_stack_path() -> Path:
    src = paths.third_party_root() / "stack" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def encode(
    adata: ad.AnnData,
    *,
    checkpoint: Optional[str] = None,
    genelist: Optional[str] = None,
    force_pert: bool = True,
    input_is_log1p: bool = True,
    batch_size: int = 8,
    num_workers: int = 0,
    gene_name_col: Optional[str] = None,
    filter_organism: bool = False,
    device: Optional[str] = None,
    show_progress: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Returns:
        embeddings (n_cells, n_hidden * token_dim) float32
        meta dict with ``pert_kept_histogram`` when ``obsm['pert_var_idx']`` is available.

    Notes:
        * ``force_pert`` does not change embeddings: Stack consumes the full gene vector
          from ``X`` only. The flag is recorded in meta; ``force_pert_effective`` is always
          False (no sampling/truncation coverage). When ``obsm['pert_var_idx']`` exists,
          we still emit ``pert_kept_histogram`` for manifest alignment—it is not a model input.
        * Stack applies ``log1p`` internally. For the embedding path specifically, the
          ``torch.log1p(features)`` call lives inline in
          ``stack.models.core.inference.InferenceMixin.get_latent_representation``
          (``inference.py:~477``) — it does **not** go through
          ``StateICLModelBase.forward`` (the ``forward`` log1p at ``base.py:~153`` is only
          hit on the training / prediction paths). The benchmark convention is that
          ``adata.X`` has already been log1p(normalize_total)-transformed, so we undo it
          with ``expm1`` before handing to Stack (``input_is_log1p=True`` by default). Set
          ``input_is_log1p=False`` if you are feeding genuine raw counts.
        * ``adata`` is never mutated — a shallow copy with ``X = expm1(clip(X, 0, None))``
          is created when ``input_is_log1p`` is True. The transient does **not** carry
          ``adata.raw``; this is intentional because Stack's ``TestSamplerDataset`` prefers
          ``adata.raw.X`` when present (see ``_load_adata_metadata``). If ``adata.raw`` were
          copied across, Stack would bypass our expm1 and re-log1p the log-space data.
    """
    _ensure_stack_path()

    import torch

    ckpt = checkpoint or os.environ.get("LATENT_BENCH_STACK_CKPT", str(paths.pretrained_root() / "stack" / "bc_large.ckpt"))
    gl_path = genelist or os.environ.get(
        "LATENT_BENCH_STACK_GENELIST",
        str(paths.pretrained_root() / "stack" / "basecount_1000per_15000max.pkl"),
    )
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"Stack checkpoint not found: {ckpt}")
    if not Path(gl_path).is_file():
        raise FileNotFoundError(f"Stack genelist pickle not found: {gl_path}")

    from stack.model_loading import load_model_from_checkpoint

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model_from_checkpoint(ckpt, device=dev)

    meta: dict = {}
    pert_present = "pert_var_idx" in adata.obsm
    if pert_present:
        m = np.asarray(adata.obsm["pert_var_idx"], dtype=np.int32)
        counts = [int(np.sum(r >= 0)) for r in m]
        meta["pert_kept_histogram"] = histogram_pert_kept(counts)
    meta["force_pert"] = bool(force_pert)
    meta["force_pert_effective"] = False  # no subset packing; nothing to "protect"
    meta["pert_source"] = "obsm_pert_var_idx" if pert_present else None
    meta["input_is_log1p"] = bool(input_is_log1p)

    # Undo benchmark-level log1p before feeding Stack (which re-applies log1p internally).
    # We build a transient AnnData so the caller's object is never mutated. log1p data is
    # non-negative in theory, but we clip to [0, +inf) defensively to avoid numerical noise
    # pushing values below zero (expm1(<0) is finite but would silently encode synthetic
    # "negative counts" that Stack's log1p would then turn into NaN).
    adata_in = adata
    if input_is_log1p:
        import scipy.sparse as sp

        X = adata.X
        if sp.issparse(X):
            X2 = X.copy()
            data = np.asarray(X2.data, dtype=np.float32)
            np.clip(data, 0.0, None, out=data)
            X2.data = np.expm1(data)
        else:
            X_arr = np.asarray(X, dtype=np.float32)
            # np.asarray may return the caller's buffer when dtype already matches; copy
            # before in-place clip so we never mutate the caller's array.
            X_arr = np.array(X_arr, copy=True)
            np.clip(X_arr, 0.0, None, out=X_arr)
            X2 = np.expm1(X_arr)
        adata_in = ad.AnnData(X=X2, obs=adata.obs, var=adata.var, obsm=dict(adata.obsm),
                              varm=dict(adata.varm), uns=dict(adata.uns))

    # Stack's TestSamplerDataset accepts an in-memory AnnData object (see
    # ``stack.data.training.datasets.TestSamplerDataset._load_adata_metadata``); passing it
    # directly avoids an extra disk round-trip. ``filter_organism=False`` lets the smoke
    # subset run even without an ``organism`` column.
    cell_embeddings, _dataset_embeddings = model.get_latent_representation(
        adata_path=adata_in,
        genelist_path=str(gl_path),
        gene_name_col=gene_name_col,
        batch_size=batch_size,
        show_progress=show_progress,
        num_workers=num_workers,
        filter_organism=filter_organism,
    )

    emb = np.asarray(cell_embeddings, dtype=np.float32)
    meta["encoder_role"] = "ExpressionOnlyEncoder"
    meta["pert_var_idx_present"] = bool(pert_present)
    meta["hidden_dim"] = int(emb.shape[1])
    return emb, meta
