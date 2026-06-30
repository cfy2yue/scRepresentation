"""
Load and merge biFlow-style control / GT h5ad for **dataset-fitted** baselines.

Layout (same as ``coupled.data.dataset`` / ``utils.data.split``):

  ``{biflow_dir}/control_center/{dataset_stem}.h5ad``
  ``{biflow_dir}/gt/{dataset_stem}.h5ad``

Merged AnnData is **control + perturbed GT** in one object (``ad.concat`` on obs).
Gene dimension uses ``join="inner"`` by default so var alignment is safe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import anndata as ad
import numpy as np
from scipy.sparse import issparse, csr_matrix


def _record_alignment_stats(result: ad.AnnData, adata_before_align: ad.AnnData) -> None:
    """Observability: bench↔reference intersection may drop obs/var without logging."""
    result.uns["dataset_fitted_align"] = {
        "n_obs_input": int(adata_before_align.n_obs),
        "n_obs_output": int(result.n_obs),
        "n_obs_dropped": int(adata_before_align.n_obs) - int(result.n_obs),
        "n_var_input": int(adata_before_align.n_vars),
        "n_var_output": int(result.n_vars),
        "n_var_dropped": int(adata_before_align.n_vars) - int(result.n_vars),
    }


def load_biflow_merged_anndata(
    biflow_dir: str | Path,
    dataset_stem: str,
    *,
    join: Literal["inner", "outer"] = "inner",
    label_sources: bool = True,
) -> ad.AnnData:
    """
    Read control_center and gt pair, concatenate on cells.

    Args:
        biflow_dir: Root with ``control_center/`` and ``gt/`` subdirs.
        dataset_stem: File stem (e.g. ``Adamson`` for ``Adamson.h5ad``).
        join: ``inner`` keeps intersection of genes (recommended).
        label_sources: If True, set ``obs['dataset_fitted_split']`` to
            ``control`` or ``perturbed_gt`` and ``obs['dataset_fitted_stem']``.
    """
    root = Path(biflow_dir)
    ctrl_p = root / "control_center" / f"{dataset_stem}.h5ad"
    gt_p = root / "gt" / f"{dataset_stem}.h5ad"
    if not ctrl_p.is_file():
        raise FileNotFoundError(f"control_center h5ad not found: {ctrl_p}")
    if not gt_p.is_file():
        raise FileNotFoundError(f"gt h5ad not found: {gt_p}")

    ctrl = ad.read_h5ad(ctrl_p)
    gt = ad.read_h5ad(gt_p)
    if label_sources:
        ctrl.obs["dataset_fitted_split"] = "control"
        gt.obs["dataset_fitted_split"] = "perturbed_gt"
        ctrl.obs["dataset_fitted_stem"] = dataset_stem
        gt.obs["dataset_fitted_stem"] = dataset_stem

    # Avoid duplicate obs index collisions between splits.
    merged = ad.concat(
        [ctrl, gt],
        join=join,
        merge="same",
        label=None,
        keys=None,
        index_unique="-",
    )
    merged.uns["dataset_fitted_biflow_dir"] = str(root.resolve())
    merged.uns["dataset_fitted_stem"] = dataset_stem
    return merged


def attach_counts_from_h5ad(
    adata: ad.AnnData,
    counts_h5ad: str | Path,
    *,
    source_layer: Optional[str] = None,
    target_layer: str = "counts",
) -> ad.AnnData:
    """
    Align ``adata`` to a reference h5ad that holds **raw counts** (same cells / genes).

    Keeps cell order as in ``adata`` (restricted to ``obs`` intersection). Restricts
    both objects to the **intersection of var_names** so layer assignment is shape-safe.

    Args:
        adata: Benchmark or merged object (e.g. log1p in ``X``).
        counts_h5ad: Path to h5ad whose ``X`` or ``layers[source_layer]`` are counts.
        source_layer: If set, read counts from this layer; else use ``reference.X``.
        target_layer: Write counts into ``adata.layers[target_layer]``.
    """
    adata_before = adata
    ref = ad.read_h5ad(str(counts_h5ad))
    obs_order = [o for o in adata.obs_names if o in ref.obs_names]
    if not obs_order:
        raise ValueError(
            f"No overlapping obs names between adata and reference {counts_h5ad}"
        )

    left = adata[obs_order].copy()
    right = ref[obs_order].copy()
    common_var = left.var_names.intersection(right.var_names)
    if len(common_var) == 0:
        raise ValueError("No overlapping var names between adata and counts reference")
    left = left[:, common_var].copy()
    right = right[:, common_var].copy()

    if source_layer:
        if source_layer not in right.layers:
            raise KeyError(f"source_layer {source_layer!r} not in reference.layers")
        cnt = right.layers[source_layer]
    else:
        cnt = right.X

    if issparse(cnt):
        left.layers[target_layer] = csr_matrix(cnt, dtype=np.float32)
    else:
        left.layers[target_layer] = np.asarray(cnt, dtype=np.float32)

    left.uns["dataset_fitted_counts_ref"] = str(Path(counts_h5ad).resolve())
    left.uns["dataset_fitted_counts_source_layer"] = source_layer
    _record_alignment_stats(left, adata_before)
    return left


def attach_expression_from_h5ad(
    adata: ad.AnnData,
    expression_h5ad: str | Path,
    *,
    source_layer: Optional[str] = None,
) -> ad.AnnData:
    """
    Align ``adata`` to a reference h5ad that holds expression matrix data and
    replace ``adata.X`` with that aligned matrix.

    This is useful when a benchmark h5ad carries dataset-level metadata but has
    ``X = null`` and the actual expression lives in a paired raw h5ad.
    """
    adata_before = adata
    ref = ad.read_h5ad(str(expression_h5ad))
    obs_order = [o for o in adata.obs_names if o in ref.obs_names]
    if not obs_order:
        raise ValueError(
            f"No overlapping obs names between adata and reference {expression_h5ad}"
        )

    left = adata[obs_order].copy()
    right = ref[obs_order].copy()
    common_var = left.var_names.intersection(right.var_names)
    if len(common_var) == 0:
        raise ValueError("No overlapping var names between adata and expression reference")
    left = left[:, common_var].copy()
    right = right[:, common_var].copy()

    if source_layer:
        if source_layer not in right.layers:
            raise KeyError(f"source_layer {source_layer!r} not in reference.layers")
        expr = right.layers[source_layer]
    else:
        expr = right.X

    if issparse(expr):
        left.X = csr_matrix(expr, dtype=np.float32)
    else:
        left.X = np.asarray(expr, dtype=np.float32)

    left.uns["dataset_fitted_expression_ref"] = str(Path(expression_h5ad).resolve())
    left.uns["dataset_fitted_expression_source_layer"] = source_layer
    _record_alignment_stats(left, adata_before)
    return left
