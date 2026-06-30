"""
Dataset-fitted **scVI** baseline: train one SCVI model on all cells in a single
AnnData (typically control + GT merged), return **latent representation** only.

**Default (``input_is_log1p=False``): counts are required** — the adapter does
**not** silently train on log1p ``adata.X``. Provide raw counts in
``adata.layers[counts_layer]`` (default ``\"counts\"``) or pass
``counts_layer=None`` and ensure ``adata.X`` holds integer (non-log) counts
(smoke/tests only; discouraged for real benchmarks).

**Log1p expression (``input_is_log1p=True``):** opt in to scvi-tools
``gene_likelihood='normal'`` (Gaussian decoder head) on ``adata.X`` or a named
layer — **no pseudo-counts**, no ``expm1``. See ``docs/encoder_impl/scvi_baseline.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import anndata as ad
import numpy as np
from scipy.sparse import issparse


def _validate_count_matrix(arr: np.ndarray) -> None:
    """Reject obvious log1p / non-count matrices for scVI (same spirit as the X path)."""
    if not np.isfinite(arr).all():
        raise ValueError("counts matrix has non-finite values")
    if (arr < 0).any():
        raise ValueError("counts matrix has negative entries; expected raw counts")
    flat = arr.ravel()
    nz = flat[flat > 1e-8]
    if nz.size == 0:
        return
    frac_nz_int = float(np.mean(np.isclose(nz, np.round(nz), rtol=0.0, atol=1e-4)))
    nz_max = float(np.max(nz))
    # Typical raw UMI: positive entries are mostly integers; log1p norm is fractional with modest max.
    if nz_max <= 20 and frac_nz_int < 0.88:
        raise ValueError(
            "counts matrix does not look like raw counts (positive entries are not mostly integers). "
            "Do not put log1p-normalized expression in layers['counts']."
        )
    frac_all_int = float(np.mean(np.isclose(flat, np.round(flat), rtol=0.0, atol=1e-4)))
    if frac_all_int < 0.9 and nz_max > 25:
        raise ValueError(
            "counts matrix does not look like raw counts (not integer-like). "
            "Do not put log1p-normalized expression in layers['counts']."
        )


def _ensure_counts_matrix(adata: ad.AnnData, counts_layer: Optional[str]) -> tuple[ad.AnnData, str]:
    """
    Return a **copy** of adata with counts available for SCVI.setup_anndata.

    If ``counts_layer`` is a non-empty string and present in ``adata.layers``,
    SCVI will use ``layer=counts_layer``.

    If ``counts_layer is None``, require ``adata.X`` to look like counts
    (finite, non-negative, mostly integer-ish) and use X.
    """
    work = adata.copy()
    if counts_layer:
        if counts_layer not in work.layers:
            raise ValueError(
                f"scVI baseline requires raw counts in adata.layers[{counts_layer!r}]. "
                f"Available layers: {list(work.layers.keys())!r}. "
                "Do not train scVI on log1p-normalized X without supplying counts."
            )
        cnt = work.layers[counts_layer]
        if issparse(cnt):
            arr = np.asarray(cnt.toarray())
        else:
            arr = np.asarray(cnt)
        _validate_count_matrix(arr)
        return work, counts_layer

    # counts_layer is None → use X with sanity checks (testing / advanced only).
    x = work.X
    if issparse(x):
        arr = np.asarray(x.toarray())
    else:
        arr = np.asarray(x)
    _validate_count_matrix(arr)
    return work, "X"  # sentinel for setup: no layer kwarg


def _ensure_log1p_matrix(adata: ad.AnnData, counts_layer: Optional[str]) -> tuple[ad.AnnData, str]:
    """
    Return a **copy** of adata with log1p expression in ``X`` or ``layers[counts_layer]``.

    Does **not** validate integer counts — caller must set ``input_is_log1p=True`` only when
    the chosen matrix is actually log1p-normalized expression.
    """
    work = adata.copy()
    if counts_layer:
        if counts_layer not in work.layers:
            raise ValueError(
                f"scVI log1p path: layer {counts_layer!r} not found. "
                f"Available layers: {list(work.layers.keys())!r}."
            )
        return work, counts_layer
    return work, "X"


def encode(
    adata: ad.AnnData,
    *,
    n_latent: int = 10,
    n_layers: int = 2,
    max_epochs: int = 400,
    gene_likelihood: Optional[str] = None,
    input_is_log1p: bool = False,
    log1p_gene_likelihood: str = "normal",
    counts_layer: Optional[str] = "counts",
    batch_key: Optional[str] = None,
    dummy_batch_key: str = "_scvi_batch",
    train_kwargs: Optional[dict[str, Any]] = None,
    model_save_dir: Optional[Path] = None,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Train SCVI on ``adata`` and return ``get_latent_representation()`` for all cells.

    Args:
        adata: Single-dataset object (e.g. merged control + gt).
        n_latent: SCVI latent size.
        n_layers: SCVI encoder depth.
        max_epochs: Training epochs (override via ``train_kwargs['max_epochs']`` if set).
        gene_likelihood: SCVI ``gene_likelihood`` on the **counts** path; ``None``
            means ``\"nb\"``. When ``input_is_log1p=True``, the model uses
            ``log1p_gene_likelihood`` instead; if you **explicitly** set this argument
            to ``\"nb\"`` or ``\"zinb\"``, a ``ValueError`` is raised (see below).
        input_is_log1p: If ``True``, train on log1p expression in ``X`` or
            ``layers[counts_layer]`` using ``log1p_gene_likelihood`` (default
            ``\"normal\"``). **Do not** pass count-style ``gene_likelihood`` (``nb`` /
            ``zinb``) — that raises ``ValueError``. See ``docs/encoder_impl/scvi_baseline.md``.
        log1p_gene_likelihood: When ``input_is_log1p=True``, the decoder likelihood
            passed to ``SCVI`` (default ``\"normal\"``). Ignored when
            ``input_is_log1p=False``.
        counts_layer: Name of layer with raw counts, or ``None`` to force using ``X``
            after integer-ish validation (discouraged for production). On the log1p
            path, the same parameter selects the **expression matrix** (layer or ``X``);
            naming is kept for API compatibility.
        batch_key: Existing ``obs`` column for batch; if ``None``, create
            ``dummy_batch_key`` with a single category.
        dummy_batch_key: Column name for single-batch fallback.
        train_kwargs: Extra kwargs forwarded to ``model.train`` (e.g. ``accelerator``).
        model_save_dir: If set, ``model.save(dir, overwrite=True)`` after training.
        seed: scVI / pytorch seed via ``scvi.settings.seed``.

    Returns:
        ``(latent, meta)`` with latent ``(n_obs, n_latent)``.
    """
    try:
        import scvi
        from scvi.model import SCVI
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "scVI baseline requires scvi-tools. Use SCFM_SCLDM_PYTHON or an environment "
            "with scvi-tools pinned compatibly with this benchmark."
        ) from e

    scvi.settings.seed = int(seed)

    if input_is_log1p:
        if gene_likelihood is not None and str(gene_likelihood).lower() in ("nb", "zinb"):
            raise ValueError(
                "With input_is_log1p=True, Negative Binomial / ZINB decoders (gene_likelihood='nb' or "
                "'zinb') are inconsistent: scVI models those as count data, but log1p-normalized "
                "expression is continuous and not integer counts. Omit gene_likelihood and use "
                "log1p_gene_likelihood='normal' (Gaussian head), or supply true raw counts and call "
                "with input_is_log1p=False. "
                "See docs/encoder_impl/scvi_baseline.md (section «log1p 输入») and "
                "docs/encoder_impl/README.md (log1p 数据流审计)."
            )
        gene_likelihood_effective = str(log1p_gene_likelihood)
        work, layer_resolved = _ensure_log1p_matrix(adata, counts_layer)
        log1p_strategy = (
            "normal_likelihood"
            if str(log1p_gene_likelihood).lower() == "normal"
            else str(log1p_gene_likelihood)
        )
        if layer_resolved == "X":
            data_source = "X_log1p"
        else:
            data_source = "counts_layer"
    else:
        gene_likelihood_effective = gene_likelihood if gene_likelihood is not None else "nb"
        work, layer_resolved = _ensure_counts_matrix(adata, counts_layer)
        log1p_strategy = None
        if layer_resolved == "X":
            data_source = "X_counts"
        else:
            data_source = "counts_layer"

    if batch_key is None:
        bk = dummy_batch_key
        work.obs[bk] = "batch0"
    else:
        bk = batch_key
        if bk not in work.obs:
            raise KeyError(f"batch_key {bk!r} not in adata.obs")

    setup_kw: dict[str, Any] = {"batch_key": bk}
    if layer_resolved == "X":
        SCVI.setup_anndata(work, layer=None, **setup_kw)
    else:
        SCVI.setup_anndata(work, layer=layer_resolved, **setup_kw)

    model = SCVI(
        work,
        n_layers=int(n_layers),
        n_latent=int(n_latent),
        gene_likelihood=gene_likelihood_effective,
    )

    tr = dict(train_kwargs or {})
    tr.setdefault("accelerator", "auto")
    tr.setdefault("devices", "auto")
    me = int(tr.pop("max_epochs", max_epochs))
    model.train(max_epochs=me, **tr)

    latent = np.asarray(model.get_latent_representation(), dtype=np.float32)

    if model_save_dir is not None:
        model_save_dir = Path(model_save_dir)
        model_save_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(model_save_dir), overwrite=True)

    meta: dict[str, Any] = {
        "encoder_role": "ExpressionOnlyEncoder",
        "fit_scope": "dataset",
        "fit_method": "scvi",
        "force_pert_effective": False,
        "pert_source": None,
        "n_latent": int(n_latent),
        "n_layers": int(n_layers),
        "max_epochs": me,
        "input_is_log1p": bool(input_is_log1p),
        "gene_likelihood": gene_likelihood_effective,
        "log1p_strategy": log1p_strategy,
        "data_source": data_source,
        "counts_layer": layer_resolved if layer_resolved != "X" else None,
        "counts_from_X": layer_resolved == "X",
        "batch_key": bk,
        "n_obs": int(work.n_obs),
        "n_vars": int(work.n_vars),
        "model_saved_to": str(model_save_dir.resolve()) if model_save_dir else None,
    }
    return latent, meta
