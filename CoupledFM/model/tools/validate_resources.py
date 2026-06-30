#!/usr/bin/env python3
"""Validate the standalone scdfm dataset/pretrain resource layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from model import paths


GENE_DATASETS = [
    "Adamson",
    "DixitRegev2016_K562_TFs_High_MOI",
    "Frangieh",
    "GasperiniShendure2019_lowMOI",
    "Jiang_IFNB",
    "Jiang_IFNG",
    "Jiang_INS",
    "Jiang_TGFB",
    "Jiang_TNFA",
    "Nadig_hepg2",
    "Nadig_jurket",
    "NormanWeissman2019_filtered",
    "Papalexi",
    "ReplogleWeissman2022_K562_gwps",
    "Replogle_RPE1essential",
    "Schmidt",
    "TianActivation",
    "TianInhibition",
    "Wessels",
]


def _cache_files(root: Path) -> list[Path]:
    return [root / "gene_embeddings.npy", root / "gene_index.tsv", root / "manifest.json"]


def _expected_tree() -> str:
    return """Expected layout:
<delivery_root>/
  model/
  dataset/
    biFlow_data/
      control_stack/{dataset}.h5ad
      gt_stack/{dataset}.h5ad
      control_center_stack/{dataset}.h5ad
      split_seed42.json
    cellgene_census/processed/
      tissue_metainfo.csv
      celltype_metainfo.csv
      kidney/kidney_top6000var.h5ad
      ...
  pretrainckpt/
    cellnavi/data/
      gene_name.txt
      Nichenet/node2idx.json
      Nichenet/graph.pkl
      pretrain/pretrain_weights.pth
    genepert_cache/
      cellnavi_embed_gene/{gene_embeddings.npy,gene_index.tsv,manifest.json}
      scgpt_embed_gene/{gene_embeddings.npy,gene_index.tsv,manifest.json}
"""


def _required_paths(mode: str, datasets: list[str]) -> list[Path]:
    req: list[Path] = [
        paths.gene_name_path(),
        paths.nichenet_node2idx_path(),
        paths.nichenet_graph_pkl_path(),
    ]
    if mode in {"all", "pretrain", "local-smoke"}:
        req.extend(
            [
                paths.cellnavi_pretrain_ckpt_path(),
                paths.cellgene_processed_dir() / "tissue_metainfo.csv",
            ]
        )
        if mode == "local-smoke":
            req.append(paths.cellgene_processed_dir() / "kidney" / "kidney_top6000var.h5ad")
    if mode in {"all", "coupled", "local-smoke"}:
        req.extend(_cache_files(paths.cellnavi_cache_dir()))
        req.extend(_cache_files(paths.scgpt_cache_dir()))
        for ds in datasets:
            req.append(paths.biflow_dir() / "control_stack" / f"{ds}.h5ad")
            req.append(paths.biflow_dir() / "gt_stack" / f"{ds}.h5ad")
    return req


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=["all", "pretrain", "coupled", "local-smoke"],
        default="all",
        help="Resource subset to validate.",
    )
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Datasets required for coupled validation; defaults to all gene datasets.",
    )
    ap.add_argument("--print-only", action="store_true", help="Print resolved paths and exit 0.")
    args = ap.parse_args(argv)

    datasets = list(args.datasets or GENE_DATASETS)
    print(paths.describe_layout())
    if args.print_only:
        print(_expected_tree())
        return 0

    missing = [p for p in _required_paths(args.mode, datasets) if not p.exists()]
    if missing:
        print("\nMissing required resources:", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        print("\n" + _expected_tree(), file=sys.stderr)
        return 2

    print(f"OK: resources validated for mode={args.mode} datasets={len(datasets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
