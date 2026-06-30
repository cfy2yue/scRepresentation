#!/usr/bin/env python3
"""Smoke: dataset-fitted scVI — synthetic control+gt merge, 1 epoch CPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> int:
    lb = Path(__file__).resolve().parents[1]
    if str(lb) not in sys.path:
        sys.path.insert(0, str(lb))

    try:
        import scvi  # noqa: F401
    except ImportError:
        print("SKIP: scvi-tools not installed", file=sys.stderr)
        return 0

    import anndata as ad
    from scipy import sparse

    from adapters.scvi_baseline.encoder import encode

    rng = np.random.default_rng(0)
    n0, n1, g = 64, 64, 48
    x0 = rng.poisson(1.2, size=(n0, g)).astype(np.float32)
    x1 = rng.poisson(2.0, size=(n1, g)).astype(np.float32)
    a0 = ad.AnnData(sparse.csr_matrix(x0))
    a0.obs["dataset_fitted_split"] = "control"
    a1 = ad.AnnData(sparse.csr_matrix(x1))
    a1.obs["dataset_fitted_split"] = "perturbed_gt"
    merged = ad.concat([a0, a1], join="inner", index_unique="-")
    merged.layers["counts"] = merged.X.copy()

    emb, meta = encode(
        merged,
        n_latent=6,
        n_layers=1,
        max_epochs=1,
        train_kwargs={
            "accelerator": "cpu",
            "devices": 1,
            "enable_progress_bar": False,
        },
    )
    assert meta.get("fit_method") == "scvi"
    assert meta.get("fit_scope") == "dataset"
    assert meta.get("encoder_role") == "ExpressionOnlyEncoder"
    assert meta.get("force_pert_effective") is False
    assert emb.shape == (merged.n_obs, 6)
    assert np.isfinite(emb).all()

    print("scvi_baseline smoke PASSED", emb.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
