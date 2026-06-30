#!/usr/bin/env python3
"""Sanity-check ``control_scfoundation`` / ``gt_scfoundation`` AnnData outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.tools.scfoundation_embedding.common import coupled_fm_root
from model.tools.scldm_embedding.common import discover_raw_stems


def _check_one(path: Path) -> list[str]:
    import anndata as ad

    errs: list[str] = []
    try:
        a = ad.read_h5ad(path)
    except Exception as e:
        return [f"{path}: read failed: {e}"]
    if "emb" not in a.obsm:
        errs.append(f"{path}: missing obsm['emb']")
        return errs
    emb = a.obsm["emb"]
    if emb.shape[0] != a.n_obs:
        errs.append(f"{path}: emb rows {emb.shape[0]} != n_obs {a.n_obs}")
    if "scfoundation_embedding" not in a.uns:
        errs.append(f"{path}: missing uns['scfoundation_embedding']")
    else:
        meta = a.uns["scfoundation_embedding"]
        for k in ("encoder", "checkpoint", "gene_tsv", "embedding_dim"):
            if k not in meta:
                errs.append(f"{path}: uns['scfoundation_embedding'] missing {k!r}")
        if "embedding_dim" in meta and emb.ndim == 2 and int(meta["embedding_dim"]) != emb.shape[1]:
            errs.append(
                f"{path}: embedding_dim {meta['embedding_dim']} != obsm emb width {emb.shape[1]}"
            )
    return errs


def main() -> None:
    root = coupled_fm_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--latent-root",
        default=str(root / "data" / "latent_data" / "scfoundation"),
        type=Path,
    )
    p.add_argument("--raw-genepert", default=str(root / "data" / "raw" / "genepert_DE5000"), type=Path)
    p.add_argument("--raw-chemical", default=str(root / "data" / "raw" / "chemicalpert_DE5000"), type=Path)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--require-gt", action="store_true", help="fail if ``gt_scfoundation`` file missing")
    args = p.parse_args()

    raw_dirs = (args.raw_genepert.expanduser().resolve(), args.raw_chemical.expanduser().resolve())
    stems = args.datasets if args.datasets else discover_raw_stems(*raw_dirs)
    latent = args.latent_root.expanduser().resolve()

    all_errs: list[str] = []
    for ds in stems:
        cpath = latent / "control_scfoundation" / f"{ds}.h5ad"
        gpath = latent / "gt_scfoundation" / f"{ds}.h5ad"
        if not cpath.is_file():
            all_errs.append(f"missing control_scfoundation: {cpath}")
            continue
        all_errs.extend(_check_one(cpath))
        if args.require_gt:
            if not gpath.is_file():
                all_errs.append(f"missing gt_scfoundation: {gpath}")
            else:
                all_errs.extend(_check_one(gpath))

    if all_errs:
        for line in all_errs:
            print(line, file=sys.stderr)
        raise SystemExit(1)
    print("inspect ok", latent)


if __name__ == "__main__":
    main()
