#!/usr/bin/env python3
"""Sanity-check ``control_stack`` / ``gt_stack`` AnnData outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.tools.stack_embedding.common import coupled_fm_root, discover_raw_stems


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
    if "stack_embedding" not in a.uns:
        errs.append(f"{path}: missing uns['stack_embedding']")
    else:
        meta = a.uns["stack_embedding"]
        for k in ("encoder", "checkpoint", "genelist", "embedding_dim"):
            if k not in meta:
                errs.append(f"{path}: uns['stack_embedding'] missing {k!r}")
        if "embedding_dim" in meta and emb.ndim == 2 and int(meta["embedding_dim"]) != emb.shape[1]:
            errs.append(
                f"{path}: embedding_dim {meta['embedding_dim']} != obsm emb width {emb.shape[1]}"
            )
    return errs


def main() -> None:
    root = coupled_fm_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--biflow-dir",
        default=str(root / "data" / "latent_data" / "stack"),
        type=Path,
    )
    p.add_argument(
        "--raw-genepert",
        default=str(root / "data" / "raw" / "genepert_DE5000"),
        type=Path,
    )
    p.add_argument(
        "--raw-chemical",
        default=str(root / "data" / "raw" / "chemicalpert_DE5000"),
        type=Path,
    )
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--require-gt", action="store_true", help="fail if ``gt_stack`` file missing")
    args = p.parse_args()

    raw_dirs = (args.raw_genepert.expanduser(), args.raw_chemical.expanduser())
    stems = args.datasets if args.datasets else discover_raw_stems(*raw_dirs)
    biflow = args.biflow_dir.expanduser().resolve()

    all_errs: list[str] = []
    for ds in stems:
        cpath = biflow / "control_stack" / f"{ds}.h5ad"
        gpath = biflow / "gt_stack" / f"{ds}.h5ad"
        if not cpath.is_file():
            all_errs.append(f"missing control_stack: {cpath}")
            continue
        all_errs.extend(_check_one(cpath))
        if args.require_gt:
            if not gpath.is_file():
                all_errs.append(f"missing gt_stack: {gpath}")
            else:
                all_errs.extend(_check_one(gpath))

    if all_errs:
        print("\n".join(all_errs))
        sys.exit(1)
    print("OK:", len(stems), "datasets checked under", biflow)


if __name__ == "__main__":
    main()
