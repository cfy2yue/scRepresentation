#!/usr/bin/env python3
"""
Export dataset-fitted baseline latents (PCA or scVI) for one dataset.

Either:
  - ``--biflow-dir`` + ``--dataset-stem`` → load merged control+gt via
    ``adapters.dataset_fitted_io.load_biflow_merged_anndata``, or
  - ``--adata`` → use a single h5ad (already merged).

Outputs (under ``--out-dir`` or default ``<data/scFM>/output/dataset_fitted/...``):

  ``latent.npy``, ``meta.json``; scVI may also write ``model/`` if requested.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

FM_ROOT = Path(__file__).resolve().parents[1]
if str(FM_ROOT) not in sys.path:
    sys.path.insert(0, str(FM_ROOT))
import paths


def _json_safe(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> int:
    lb = Path(__file__).resolve().parents[1]
    if str(lb) not in sys.path:
        sys.path.insert(0, str(lb))

    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--adata",
        type=Path,
        help="Path to one dataset h5ad (for PCA this can be a full DE5000_bench/<dataset>.h5ad)",
    )
    src.add_argument(
        "--biflow-dir",
        type=Path,
        help="biFlow root with control_center/ and gt/",
    )
    p.add_argument(
        "--dataset-stem",
        type=str,
        default=None,
        help="Dataset file stem when using --biflow-dir (e.g. Adamson)",
    )
    p.add_argument(
        "--baseline",
        choices=("pca", "scvi"),
        required=True,
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <scFM>/output/dataset_fitted/<baseline>/<stem>/)",
    )
    # PCA
    p.add_argument("--pca-n-components", type=int, default=50)
    p.add_argument(
        "--pca-expression-from-h5ad",
        type=Path,
        default=None,
        help="Optional h5ad with expression matrix for PCA when --adata is metadata-only (e.g. DE5000_bench with X=null).",
    )
    p.add_argument(
        "--pca-expression-source-layer",
        type=str,
        default=None,
        help="Layer name in the PCA expression h5ad (default: use that file's X).",
    )
    # scVI
    p.add_argument("--scvi-n-latent", type=int, default=10)
    p.add_argument("--scvi-max-epochs", type=int, default=400)
    p.add_argument(
        "--scvi-batch-size",
        type=int,
        default=None,
        help="scvi-tools train batch_size (default 128). Larger values better utilize GPU.",
    )
    p.add_argument(
        "--scvi-counts-layer",
        type=str,
        default="counts",
        help="Layer name for raw counts (required for scVI unless you use counts-only h5ad; see docs).",
    )
    p.add_argument(
        "--scvi-batch-key",
        type=str,
        default=None,
        help="obs column for batch; if omitted, a dummy single-batch column is created.",
    )
    p.add_argument(
        "--scvi-save-model-dir",
        type=Path,
        default=None,
        help="If set, save trained SCVI model under this directory.",
    )
    p.add_argument(
        "--scvi-input-is-log1p",
        action="store_true",
        help="Train SCVI on log1p expression (X or --scvi-counts-layer) with Gaussian decoder "
        "(gene_likelihood from --scvi-log1p-gene-likelihood; default normal). No pseudo-counts.",
    )
    p.add_argument(
        "--scvi-log1p-use-x",
        action="store_true",
        help="With --scvi-input-is-log1p, read expression from adata.X (ignore layers counts name). "
        "When false, defaults to layer --scvi-counts-layer.",
    )
    p.add_argument(
        "--scvi-log1p-gene-likelihood",
        type=str,
        default="normal",
        help="Decoder likelihood when --scvi-input-is-log1p (default: normal).",
    )
    p.add_argument(
        "--scvi-counts-from-h5ad",
        type=Path,
        default=None,
        help="Optional h5ad with raw counts (aligned obs/var). Writes layers[--scvi-counts-layer].",
    )
    p.add_argument(
        "--scvi-counts-source-layer",
        type=str,
        default=None,
        help="Layer name in counts h5ad (default: use that file's X).",
    )
    args = p.parse_args()

    if args.biflow_dir is not None:
        if not args.dataset_stem:
            p.error("--dataset-stem is required with --biflow-dir")
        from adapters.dataset_fitted_io import load_biflow_merged_anndata

        adata = load_biflow_merged_anndata(args.biflow_dir, args.dataset_stem)
        stem = args.dataset_stem
    else:
        import scanpy as sc

        adata = sc.read_h5ad(str(args.adata))
        stem = args.adata.stem

    scfm = lb.parent
    if args.out_dir is None:
        args.out_dir = paths.output_root() / "dataset_fitted" / args.baseline / stem
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.baseline == "pca" and args.pca_expression_from_h5ad is not None:
        from adapters.dataset_fitted_io import attach_expression_from_h5ad

        adata = attach_expression_from_h5ad(
            adata,
            args.pca_expression_from_h5ad,
            source_layer=args.pca_expression_source_layer,
        )

    if args.baseline == "scvi" and args.scvi_counts_from_h5ad is not None:
        from adapters.dataset_fitted_io import attach_counts_from_h5ad

        adata = attach_counts_from_h5ad(
            adata,
            args.scvi_counts_from_h5ad,
            source_layer=args.scvi_counts_source_layer,
            target_layer=args.scvi_counts_layer,
        )

    if args.baseline == "pca":
        from adapters.pca_baseline.encoder import encode as encode_pca

        latent, meta = encode_pca(adata, n_components=args.pca_n_components)
        if args.pca_expression_from_h5ad is not None:
            meta["expression_recovered_from"] = str(args.pca_expression_from_h5ad.resolve())
            if args.pca_expression_source_layer is not None:
                meta["expression_recovered_source_layer"] = args.pca_expression_source_layer
    else:
        from adapters.scvi_baseline.encoder import encode as encode_scvi

        clayer = args.scvi_counts_layer
        if args.scvi_input_is_log1p and args.scvi_log1p_use_x:
            clayer = None
        train_kwargs: dict[str, Any] = {}
        if args.scvi_batch_size is not None:
            train_kwargs["batch_size"] = int(args.scvi_batch_size)
        latent, meta = encode_scvi(
            adata,
            n_latent=args.scvi_n_latent,
            max_epochs=args.scvi_max_epochs,
            counts_layer=clayer,
            batch_key=args.scvi_batch_key,
            model_save_dir=args.scvi_save_model_dir,
            input_is_log1p=args.scvi_input_is_log1p,
            log1p_gene_likelihood=args.scvi_log1p_gene_likelihood,
            train_kwargs=train_kwargs or None,
        )
        if args.scvi_counts_from_h5ad is not None:
            meta["counts_recovered_from"] = str(args.scvi_counts_from_h5ad.resolve())
            if args.scvi_counts_source_layer is not None:
                meta["counts_recovered_source_layer"] = args.scvi_counts_source_layer

    import numpy as np

    align = adata.uns.get("dataset_fitted_align")
    if align is not None:
        meta["dataset_fitted_align"] = {
            k: int(v) if isinstance(v, (int, np.integer)) else v for k, v in dict(align).items()
        }

    np.save(args.out_dir / "latent.npy", latent)
    with open(args.out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(meta), f, indent=2, ensure_ascii=False)

    print("Wrote", args.out_dir / "latent.npy", latent.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
