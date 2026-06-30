#!/usr/bin/env python3
"""Sanity-check ``control_scldm`` / ``gt_scldm`` scLDM AnnData outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.tools.scldm_embedding.common import coupled_fm_root, discover_raw_stems


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
    if "scldm_embedding" not in a.uns:
        errs.append(f"{path}: missing uns['scldm_embedding']")
    else:
        meta = a.uns["scldm_embedding"]
        for k in (
            "encoder",
            "checkpoint",
            "config",
            "gene_parquet",
            "source_path",
            "batch_size",
            "latent_dim",
            "mode",
            "meta",
        ):
            if k not in meta:
                errs.append(f"{path}: uns['scldm_embedding'] missing {k!r}")
        ld = meta.get("latent_dim")
        if emb.ndim == 2 and ld is not None and int(ld) != emb.shape[1]:
            errs.append(
                f"{path}: latent_dim {ld} != obsm emb width {emb.shape[1]}"
            )
        nested = meta.get("meta")
        if nested is None or not isinstance(nested, dict):
            errs.append(f"{path}: uns['scldm_embedding']['meta'] must be a dict")
        if emb.ndim != 2:
            errs.append(f"{path}: emb must be 2D array, got shape {emb.shape}")
    return errs


def main() -> None:
    root = coupled_fm_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--latent-root",
        default=str(root / "data" / "latent_data" / "scldm"),
        type=Path,
    )
    p.add_argument("--raw-genepert", default=str(root / "data" / "raw" / "genepert_DE5000"), type=Path)
    p.add_argument("--raw-chemical", default=str(root / "data" / "raw" / "chemicalpert_DE5000"), type=Path)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--require-gt", action="store_true", help="fail if ``gt_scldm`` missing")
    args = p.parse_args()

    raw_dirs = (args.raw_genepert.expanduser(), args.raw_chemical.expanduser())
    stems = args.datasets if args.datasets else discover_raw_stems(*raw_dirs)
    latent_root = args.latent_root.expanduser().resolve()

    errors: list[str] = []
    for ds in stems:
        cp = latent_root / "control_scldm" / f"{ds}.h5ad"
        gp = latent_root / "gt_scldm" / f"{ds}.h5ad"
        if not cp.is_file():
            errors.append(f"missing control_scldm: {cp}")
            continue
        errors.extend(_check_one(cp))
        if args.require_gt:
            if not gp.is_file():
                errors.append(f"missing gt_scldm: {gp}")
            else:
                errors.extend(_check_one(gp))

    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print("OK:", len(stems), "datasets checked under", latent_root)


if __name__ == "__main__":
    main()
