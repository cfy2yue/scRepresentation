#!/usr/bin/env python3
"""Build per-dataset Atlas UMAP figures: GT / raw / PCA-128 (3 figures per dataset).

Usage (from scFM root)::

    python benchmark/cli/build_atlas_umap_panels.py [--datasets Blood Heart ...]

Each dataset produces three pdf+png pairs::

    output/figures/atlas_umap/atlas_umap_<DS>_gt.{pdf,png}
    output/figures/atlas_umap/atlas_umap_<DS>_raw.{pdf,png}
    output/figures/atlas_umap/atlas_umap_<DS>_pca128.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parents[2] / "fm"))

from benchmark.plot import atlas_umap as AU
from benchmark.plot import style as ST
import paths


def main() -> int:
    default_ds = tuple(AU.ATLAS_DATASETS.keys())
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scfm-root", type=Path, default=_HERE.parents[2])
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <scfm-root>/output/figures/atlas_umap",
    )
    ap.add_argument(
        "--atlas-ts",
        type=Path,
        default=None,
        help="Default: <scfm-parent>/data/raw/atlas_TS",
    )
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=list(default_ds),
        choices=list(default_ds),
        metavar="DATASET",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=list(ST.ALL_MODELS),
        metavar="MODEL",
    )
    ap.add_argument("--label-col", type=str, default=AU.DEFAULT_LABEL_COL)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--force-umap", action="store_true")
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.3)
    args = ap.parse_args()

    scfm = args.scfm_root.resolve()
    out_dir = (args.out_dir or paths.output_root() / "figures" / "atlas_umap").resolve()
    atlas_ts = args.atlas_ts.resolve() if args.atlas_ts else None

    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
            by_ds = {o["dataset"]: o for o in prev.get("outputs", [])}
        except (json.JSONDecodeError, KeyError, TypeError):
            by_ds = {}
    else:
        by_ds = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped_summary: dict = {}
    for ds in args.datasets:
        result = AU.build_dataset_figures(
            ds,
            out_dir,
            models=args.models,
            label_col=args.label_col,
            force_umap=args.force_umap,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            max_workers=max(1, args.workers),
            atlas_ts=atlas_ts,
        )
        by_ds[ds] = result
        ds_skipped = {
            "raw": result["raw"]["skipped"],
            "pca128": result["pca128"]["skipped"],
        }
        if any(ds_skipped.values()):
            skipped_summary[ds] = ds_skipped
            print(f"[{ds}] skipped models:")
            for sp, items in ds_skipped.items():
                if items:
                    print(f"  [{sp}]")
                    for it in items:
                        print(f"    - {it}")

    manifest = {
        "scfm_root": str(scfm),
        "out_dir": str(out_dir),
        "datasets_requested": list(args.datasets),
        "models": args.models,
        "label_col": args.label_col,
        "outputs": [by_ds[k] for k in sorted(by_ds.keys())],
    }
    if skipped_summary:
        manifest["skipped_summary"] = skipped_summary
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
