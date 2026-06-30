#!/usr/bin/env python3
"""Minimal CellNavi adapter smoke test.

Requires:
  - cellnavi conda/venv (user-provided)
  - ``pretrain_weights.pth`` and ``Nichenet/graph.pkl`` (often absent from minimal ``third_party/CellNavi`` mirror)

Assets (recommended single layout under the CoupledFM repo)::

  ``<COUPLEDFM>/pretrained/cellnavi/data/pretrain/pretrain_weights.pth``
  ``<COUPLEDFM>/pretrained/cellnavi/data/Nichenet/graph.pkl``
  (``gene_name.txt`` / ``node2idx.json`` should live alongside under the same ``data/`` tree.)

If ``LATENT_BENCH_CELLNAVI_*`` are unset, this script fills ckpt / graph paths from that tree when the files exist (same policy as ``adapters/cellnavi/encoder.encode`` defaults).

Env:
  ``LATENT_BENCH_CELLNAVI_CKPT`` — path to pretrain_weights.pth
  ``LATENT_BENCH_CELLNAVI_GRAPH_PKL`` — path to graph.pkl
  ``LATENT_BENCH_SMOKE_H5AD`` — AnnData with gene symbols in var_names (default: /tmp/adamson_smoke.h5ad)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    lb = Path(__file__).resolve().parents[1]  # fm/
    if str(lb) not in sys.path:
        sys.path.insert(0, str(lb))
    import paths

    pdata = paths.pretrained_root() / "cellnavi" / "data"

    ckpt = os.environ.get("LATENT_BENCH_CELLNAVI_CKPT", "")
    if not ckpt:
        p = pdata / "pretrain" / "pretrain_weights.pth"
        if p.is_file():
            ckpt = str(p)
    gpkl = os.environ.get("LATENT_BENCH_CELLNAVI_GRAPH_PKL", "")
    if not gpkl:
        p = pdata / "Nichenet" / "graph.pkl"
        if p.is_file():
            gpkl = str(p)
    if not ckpt or not Path(ckpt).is_file():
        print("SKIP: set LATENT_BENCH_CELLNAVI_CKPT to pretrain_weights.pth", file=sys.stderr)
        return 0
    if not gpkl or not Path(gpkl).is_file():
        print("SKIP: set LATENT_BENCH_CELLNAVI_GRAPH_PKL to NicheNet graph.pkl", file=sys.stderr)
        return 0

    h5ad = os.environ.get("LATENT_BENCH_SMOKE_H5AD", "/tmp/adamson_smoke.h5ad")
    if not Path(h5ad).is_file():
        print(f"SKIP: no AnnData at {h5ad}", file=sys.stderr)
        return 0

    import scanpy as sc

    from adapters.cellnavi.encoder import encode

    adata = sc.read_h5ad(h5ad)
    n = min(8, adata.n_obs)
    sub = adata[:n].copy()

    emb, meta = encode(
        sub,
        checkpoint=ckpt,
        graph_pkl=gpkl,
        force_pert=bool("pert_var_idx" in sub.obsm),
        input_is_log1p=True,
    )
    assert emb.shape == (n, meta["hidden_dim"]), (emb.shape, meta)
    assert np.isfinite(emb).all()
    assert meta.get("encoder_role") == "ExpressionOnlyEncoder"
    print("cellnavi smoke test PASSED", emb.shape, meta.get("force_pert_effective"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
