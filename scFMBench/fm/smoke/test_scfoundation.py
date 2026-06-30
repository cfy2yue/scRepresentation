#!/usr/bin/env python3
"""Minimal scFoundation adapter smoke test (scfoundation conda).

Requires GPU for practical runtime; uses Adamson DE5000 X from full h5ad.
"""

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

    ckpt = os.environ.get(
        "LATENT_BENCH_SCFOUNDATION_CKPT",
        str(paths.pretrained_root() / "scFoundation" / "models.ckpt"),
    )
    if not Path(ckpt).is_file():
        print(f"SKIP: no checkpoint at {ckpt}", file=sys.stderr)
        return 0

    import scanpy as sc

    from adapters.scfoundation.encoder import encode

    adamson = Path(
        os.environ.get(
            "LATENT_BENCH_SMOKE_ADAMSON",
            str(paths.data_root() / "raw" / "DE5000" / "Adamson.h5ad"),
        )
    )
    if not adamson.is_file():
        print(f"SKIP: no {adamson}", file=sys.stderr)
        return 0

    adata = sc.read_h5ad(str(adamson))
    n = min(4, adata.n_obs)
    sub = adata[:n].copy()
    dev = os.environ.get("LATENT_BENCH_SCFOUNDATION_DEVICE", "cuda")

    emb, meta = encode(sub, checkpoint=ckpt, device=dev, force_pert=False, input_is_log1p=True)
    assert emb.shape == (n, meta["hidden_dim"]), (emb.shape, meta)
    assert np.isfinite(emb).all()
    assert meta.get("encoder_role") == "ExpressionOnlyEncoder"

    # Protected coverage: pick a gene in both Adamson and scFoundation 19264 list
    import pandas as pd

    gtsv = paths.third_party_root() / "scFoundation" / "model" / "OS_scRNA_gene_index.19264.tsv"
    gene_list = list(pd.read_csv(gtsv, sep="\t")["gene_name"])
    vars_set = set(sub.var_names.astype(str))
    common = [g for g in gene_list if g in vars_set]
    if common:
        g0 = common[0]
        one = adata[:1].copy()
        j = int(np.where(one.var_names == g0)[0][0])
        import scipy.sparse as sp

        X = one.X.toarray() if sp.issparse(one.X) else np.asarray(one.X).copy()
        X[0, j] = 0.0
        one.X = X
        one.obsm["pert_var_idx"] = np.array([[j]], dtype=np.int64)
        e_on, m_on = encode(one, checkpoint=ckpt, device=dev, force_pert=True, input_is_log1p=True)
        e_off, m_off = encode(one, checkpoint=ckpt, device=dev, force_pert=False, input_is_log1p=True)
        assert m_on["force_pert_effective"] and not m_off["force_pert_effective"]
        assert float(np.linalg.norm(e_on - e_off)) > 1e-6

    print("scfoundation smoke test PASSED", emb.shape, meta.get("hidden_dim"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
