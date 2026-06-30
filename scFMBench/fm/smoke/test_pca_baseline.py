#!/usr/bin/env python3
"""Smoke: dataset-fitted PCA on one full DE5000_bench h5ad with raw X recovery."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    lb = Path(__file__).resolve().parents[1]
    if str(lb) not in sys.path:
        sys.path.insert(0, str(lb))

    import paths

    bench_adata = Path(
        os.environ.get(
            "LATENT_BENCH_SMOKE_ADAMSON_BENCH",
            str(paths.data_root() / "raw" / "DE5000_bench" / "Adamson.h5ad"),
        )
    )
    raw_adata = Path(
        os.environ.get(
            "LATENT_BENCH_SMOKE_ADAMSON_RAW",
            str(paths.data_root() / "raw" / "DE5000" / "Adamson.h5ad"),
        )
    )
    if not bench_adata.is_file():
        print(f"SKIP: no {bench_adata}", file=sys.stderr)
        return 0
    if not raw_adata.is_file():
        print(f"SKIP: no {raw_adata}", file=sys.stderr)
        return 0

    import scanpy as sc

    from adapters.dataset_fitted_io import attach_expression_from_h5ad
    from adapters.pca_baseline.encoder import encode

    bench = sc.read_h5ad(str(bench_adata))
    full = attach_expression_from_h5ad(bench, raw_adata)

    n_comp = 32
    emb, meta = encode(full, n_components=n_comp)
    assert meta.get("fit_method") == "pca"
    assert meta.get("fit_scope") == "dataset"
    assert meta.get("fit_target") == "all_input_cells"
    assert meta.get("encoder_role") == "ExpressionOnlyEncoder"
    assert meta.get("force_pert_effective") is False
    assert meta.get("pert_source") is None
    n_actual = meta["n_components_actual"]
    assert emb.shape == (full.n_obs, n_actual)
    assert np.isfinite(emb).all()

    print("pca_baseline smoke PASSED", emb.shape, "explained_sum=", round(meta["explained_variance_ratio_sum"], 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
