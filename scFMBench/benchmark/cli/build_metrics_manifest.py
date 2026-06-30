#!/usr/bin/env python3
"""
Scan ``output/embeddings/<model>/<dataset>/raw`` and write ``output/metrics/run_manifest.jsonl``.

Each line is one embedding export with inferred category, ``--skip`` list, and column mapping
for ``run_metrics_one.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

BENCH_ROOT = Path(__file__).resolve().parents[1]
SCFM_ROOT = BENCH_ROOT.parent
FM_ROOT = SCFM_ROOT / "fm"
sys.path.insert(0, str(FM_ROOT))
import paths


def _infer_out_dir(emb_dir: Path) -> Path:
    raw = emb_dir.resolve()
    if raw.name != "raw":
        raise ValueError(f"expected .../raw, got {raw}")
    dataset_id = raw.parent.name
    model = raw.parent.parent.name
    p = raw.parent
    output_root = paths.output_root().resolve()
    while p != p.parent:
        if p == output_root or p.name in {"output", "scFM_output"}:
            return p / "metrics" / model / dataset_id
        p = p.parent
    return output_root / "metrics" / model / dataset_id


def _load_meta(emb_dir: Path) -> Dict[str, Any]:
    with open(emb_dir / "meta.json") as f:
        return json.load(f)


def _resolve_obs_path(emb_dir: Path, meta: Dict[str, Any]) -> Path:
    oa = meta.get("obs_artifact")
    if oa:
        cand = emb_dir / str(oa)
        if cand.is_file():
            return cand
    for name in ("obs.parquet", "obs.csv.gz", "obs.csv"):
        p = emb_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"No obs table in {emb_dir}")


def _read_obs_column_names(obs_path: Path) -> List[str]:
    if obs_path.suffix.lower() == ".parquet" or obs_path.name.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq  # type: ignore

            return list(pq.read_schema(str(obs_path)).names)
        except Exception:
            pass
    import pandas as pd

    if obs_path.name.endswith(".csv.gz") or obs_path.suffix.lower() == ".csv":
        return pd.read_csv(obs_path, nrows=0).columns.tolist()
    raise ValueError(f"Unsupported obs path {obs_path}")


def _category_from_meta(meta: Dict[str, Any], dataset_id: str) -> str:
    """Infer dataset category. Tries source_adata first, then canonical_source_adata
    (set by baselines that sanitize h5ad), then dataset_id heuristics.
    """
    candidates = [
        str(meta.get("source_adata") or ""),
        str(meta.get("canonical_source_adata") or ""),
    ]
    for src in candidates:
        if not src:
            continue
        if "/chempert/" in src or "/chemicalpert_bench/" in src:
            return "chempert"
        if "/genepert/" in src or "/genepert_bench/" in src:
            return "genepert"
        if "/atlas_TS/" in src or "/raw/atlas_TS/" in src:
            return "atlas_TS"
        if "/staging/atlas/" in src:
            return "atlas_staging"
    if dataset_id.startswith("sciplex3_"):
        return "chempert"
    if dataset_id.startswith("TS_") and dataset_id.endswith("_filtered"):
        return "atlas_TS"
    if dataset_id in {
        "Blood", "BoneMarrow", "Heart", "Lung", "LymphNode", "Skin",
        "TS_Immune_xtissue",
    }:
        return "atlas_staging"
    return "unknown"


def _infer_task(
    emb_dir: Path,
    meta: Dict[str, Any],
    cols: Set[str],
) -> Dict[str, Any]:
    dataset_id = emb_dir.resolve().parent.name
    model = emb_dir.resolve().parent.parent.name
    category = _category_from_meta(meta, dataset_id)

    has_atlas_cols = "batch" in cols and "cell_type" in cols
    # Gene-pert bench uses perturbation + is_control; chem bench uses perturbation + control.
    # Some chempert exports also carry an is_control compatibility column, so category/source
    # must decide precedence rather than raw column presence alone.
    has_chem_pert = category == "chempert" and ("perturbation" in cols) and ("control" in cols)
    has_gene_pert = category == "genepert" and ("perturbation" in cols) and ("is_control" in cols)
    if category not in {"chempert", "genepert"}:
        has_chem_pert = ("perturbation" in cols) and ("control" in cols)
        has_gene_pert = not has_chem_pert and ("perturbation" in cols) and ("is_control" in cols)
    has_pert = has_chem_pert or has_gene_pert

    skip: List[str] = []
    if not has_atlas_cols:
        skip.append("atlas")
    if not has_pert:
        skip.append("perturb")

    batch_col = "batch"
    label_col = "cell_type"
    pert_col = "perturbation" if has_pert else "pert"
    if has_chem_pert:
        is_control_col = "control"
    elif has_gene_pert:
        is_control_col = "is_control"
    else:
        is_control_col = "is_control"
    cell_line_col = "cell_line" if "cell_line" in cols else "cell_line"

    out_dir = _infer_out_dir(emb_dir)
    row: Dict[str, Any] = {
        "model": model,
        "dataset_id": dataset_id,
        "category": category,
        "emb_dir": str(emb_dir.resolve()),
        "out_dir": str(out_dir),
        "skip": skip,
        "batch_col": batch_col,
        "label_col": label_col,
        "pert_col": pert_col,
        "is_control_col": is_control_col,
        "cell_line_col": cell_line_col,
        "has_atlas_cols": has_atlas_cols,
        "has_chem_pert_cols": has_chem_pert,
        "has_gene_pert_cols": has_gene_pert,
        "obs_columns_sample": sorted(list(cols))[:24],
    }

    argv: List[str] = [
        "benchmark/cli/run_metrics_one.py",
        "--emb-dir",
        row["emb_dir"],
    ]
    if skip:
        argv.append("--skip")
        argv.extend(sorted(skip))
    argv.extend(
        [
            "--batch-col",
            batch_col,
            "--label-col",
            label_col,
            "--pert-col",
            pert_col,
            "--is-control-col",
            is_control_col,
            "--cell-line-col",
            cell_line_col,
        ]
    )
    row["argv"] = argv
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scfm-root", type=Path, default=SCFM_ROOT)
    ap.add_argument(
        "--out-manifest",
        type=Path,
        default=None,
        help="Default: <scfm-root>/output/metrics/run_manifest.jsonl",
    )
    args = ap.parse_args()

    scfm: Path = args.scfm_root.resolve()
    emb_root = paths.output_root() / "embeddings"
    out_manifest = (
        args.out_manifest
        if args.out_manifest
        else paths.output_root() / "metrics" / "run_manifest.jsonl"
    )
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for meta_path in sorted(emb_root.glob("*/*/raw/meta.json")):
        raw_dir = meta_path.parent
        if not (raw_dir / "latent.npy").is_file():
            continue
        meta = _load_meta(raw_dir)
        obs_path = _resolve_obs_path(raw_dir, meta)
        cols = set(_read_obs_column_names(obs_path))
        rows.append(_infer_task(raw_dir, meta, cols))

    with open(out_manifest, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    print(json.dumps({"manifest": str(out_manifest), "n_tasks": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
