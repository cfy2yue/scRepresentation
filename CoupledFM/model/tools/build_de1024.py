#!/usr/bin/env python3
"""Precompute per-dataset DE gene lists (default K=1024) for minibatch OT on raw space.

Logic matches ``datasets/scripts/create_DE5000_shorten.py`` (Wilcoxon ctrl vs perturbed,
pct_expressed filter, subsample for DE). Only writes JSON gene-name lists; does not
rewrite h5ad.

Usage:
  python tools/build_de1024.py
  python tools/build_de1024.py --datasets Adamson Schmidt
  python tools/build_de1024.py --k 1024 --force
  RAW_BIFLOW_DIR=/path/to/biFlow_data python tools/build_de1024.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.sparse import issparse

from model import paths

K_DEFAULT = 1024
PCT_EXPRESSED = 0.05
DE_SUBSAMPLE = 50_000

_BIFLOW = Path(os.environ.get("RAW_BIFLOW_DIR", str(paths.biflow_dir())))
_OUT = Path(os.environ.get("RAW_DE_DIR", str(paths.de_dir())))


def pick_top_de(cc: ad.AnnData, gt: ad.AnnData, k: int) -> list[str]:
    """DE genes on merged cc+gt (cc = all control), same spirit as create_DE5000_shorten."""
    gt_set = set(gt.var_names)
    common = [g for g in cc.var_names if g in gt_set]
    if not common:
        return []

    cc2 = cc[:, common].copy()
    gt2 = gt[:, common].copy()
    cc2.obs["perturbation"] = "control"

    adata = ad.concat([cc2, gt2], axis=0)

    X = adata.X
    if issparse(X):
        n_cells_per_gene = np.asarray((X > 0).sum(axis=0)).ravel()
    else:
        n_cells_per_gene = (X > 0).sum(axis=0)

    n_obs = adata.n_obs
    min_cells = max(1, int(n_obs * PCT_EXPRESSED))
    keep_expressed = n_cells_per_gene >= min_cells
    gene_names = np.array(adata.var_names)

    if keep_expressed.sum() == 0:
        adata_sub = adata.copy()
        gene_sub = gene_names
    else:
        adata_sub = adata[:, keep_expressed].copy()
        gene_sub = np.array(adata_sub.var_names)

    n_obs_sub = adata_sub.n_obs
    if n_obs_sub > DE_SUBSAMPLE:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_obs_sub, DE_SUBSAMPLE, replace=False)
        adata_de = adata_sub[idx].copy()
    else:
        adata_de = adata_sub

    ctrl_mask = adata_de.obs["perturbation"].astype(str) == "control"
    n_ctrl = int(ctrl_mask.sum())
    n_pert = int((~ctrl_mask).sum())
    if n_ctrl == 0 or n_pert == 0:
        raise RuntimeError(f"control={n_ctrl} perturbed={n_pert}, cannot run DE")

    adata_de.obs["_group"] = np.where(ctrl_mask, "control", "perturbed")
    adata_de.obs["_group"] = adata_de.obs["_group"].astype("category")

    sc.tl.rank_genes_groups(
        adata_de,
        groupby="_group",
        reference="control",
        method="wilcoxon",
        n_genes=min(k, adata_de.n_vars),
        rankby_abs=True,
        use_raw=False,
    )
    de_df = sc.get.rank_genes_groups_df(adata_de, group="perturbed")
    gene_list = de_df["names"].dropna().astype(str).tolist()
    return [g for g in gene_list if g in gene_sub][:k]


def _resolve_cc_gt_pair(biflow: Path, ds: str) -> tuple[Path, Path] | None:
    """Return (control_h5ad, gt_h5ad) for canonical or stack layout."""
    cc = biflow / "control_center" / f"{ds}.h5ad"
    gt = biflow / "gt" / f"{ds}.h5ad"
    if cc.is_file() and gt.is_file():
        return cc, gt
    cc = biflow / "control_stack" / f"{ds}.h5ad"
    gt = biflow / "gt_stack" / f"{ds}.h5ad"
    if cc.is_file() and gt.is_file():
        return cc, gt
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cc_dir = _BIFLOW / "control_center"
    gt_dir = _BIFLOW / "gt"
    cc_dir_stack = _BIFLOW / "control_stack"
    gt_dir_stack = _BIFLOW / "gt_stack"
    _OUT.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        datasets = args.datasets
    else:
        seen: set[str] = set()
        for cdir, gdir in (
            (cc_dir, gt_dir),
            (cc_dir_stack, gt_dir_stack),
        ):
            if not cdir.is_dir() or not gdir.is_dir():
                continue
            for p in cdir.glob("*.h5ad"):
                if (gdir / p.name).is_file():
                    seen.add(p.stem)
        datasets = sorted(seen)

    print(
        f"[build_de1024] biflow={_BIFLOW}  out={_OUT}  k={args.k}  n_ds={len(datasets)}",
        flush=True,
    )

    for ds in datasets:
        out_p = _OUT / f"{ds}.json"
        if out_p.exists() and not args.force:
            print(f"  [{ds}] exists, skip (--force to overwrite)")
            continue
        pair = _resolve_cc_gt_pair(_BIFLOW, ds)
        if pair is None:
            print(f"  [{ds}] missing paired control/gt (center or stack layout), skip")
            continue
        cc_p, gt_p = pair
        if not cc_p.exists() or not gt_p.exists():
            print(f"  [{ds}] missing cc or gt, skip")
            continue
        cc = ad.read_h5ad(cc_p)
        gt = ad.read_h5ad(gt_p)
        try:
            if "perturbation" not in gt.obs.columns:
                print(f"  [{ds}] gt has no perturbation column, skip")
                continue
            genes = pick_top_de(cc, gt, args.k)
        except Exception as e:
            print(f"  [{ds}] DE failed: {e}")
            continue
        finally:
            del cc, gt

        if not genes:
            print(f"  [{ds}] no genes selected, skip")
            continue
        out_p.write_text(json.dumps(genes, indent=2), encoding="utf-8")
        print(f"  [{ds}] {len(genes)} genes → {out_p.name}", flush=True)

    print("[build_de1024] done", flush=True)


if __name__ == "__main__":
    main()
