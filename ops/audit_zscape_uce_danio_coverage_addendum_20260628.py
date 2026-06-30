#!/usr/bin/env python3
"""Coverage addendum for the ZSCAPE UCE Danio latent gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_H5AD = ROOT / "reports/zscape_uce_danio_latent_gate_20260628/zscape_uce_danio_128cell_smoke_input.h5ad"
DEFAULT_DE = ROOT / "reports/zscape_expression_latent_biology_preflight_20260628/zscape_expression_de_top_genes.csv"
DEFAULT_OUT = ROOT / "reports/zscape_uce_danio_coverage_addendum_20260628"
PRIMARY_ROWS = ["periderm__noto__24p0h", "periderm__smo__24p0h"]
TOP_KS = [50, 100]


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def top_expression_visibility(adata: ad.AnnData, top_k: int = 1024) -> dict[str, float]:
    counts = adata.layers["counts"].tocsr()
    valid = adata.var["uce_valid"].astype(bool).to_numpy()
    rows = []
    for i in range(counts.shape[0]):
        row = counts.getrow(i)
        if row.nnz == 0:
            continue
        order = np.argsort(-row.data)
        chosen = row.indices[order[: min(top_k, len(order))]]
        total_counts = float(row.data.sum())
        valid_counts = float(row[:, valid].sum())
        rows.append(
            {
                "top_valid_fraction": float(valid[chosen].mean()) if len(chosen) else float("nan"),
                "raw_count_visible_fraction": valid_counts / total_counts if total_counts > 0 else float("nan"),
                "n_nonzero_symbols": int(row.nnz),
            }
        )
    df = pd.DataFrame(rows)
    return {
        "mean_top1024_valid_fraction": float(df["top_valid_fraction"].mean()),
        "min_top1024_valid_fraction": float(df["top_valid_fraction"].min()),
        "mean_raw_count_visible_fraction": float(df["raw_count_visible_fraction"].mean()),
        "min_raw_count_visible_fraction": float(df["raw_count_visible_fraction"].min()),
        "median_nonzero_symbols": float(df["n_nonzero_symbols"].median()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--de-top-genes", type=Path, default=DEFAULT_DE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.h5ad)
    var = adata.var.copy()
    var["symbol_key"] = var["symbol_upper"].astype(str).str.upper()
    var_by_symbol = var.drop_duplicates("symbol_key").set_index("symbol_key")

    de = pd.read_csv(args.de_top_genes)
    de = de[de["row_id"].isin(PRIMARY_ROWS)].copy()
    de["symbol_key"] = de["gene_symbol"].astype(str).str.upper()

    coverage_rows: list[dict[str, Any]] = []
    for row_id in PRIMARY_ROWS:
        sub = de[de["row_id"] == row_id].sort_values("rank_abs_z")
        target = row_id.split("__")[1].split("_")[0].upper()
        for top_k in TOP_KS:
            top = sub.head(top_k).copy()
            joined = top.merge(
                var[["symbol_key", "uce_valid", "uce_chrom_known", "n_ensdarg_collapsed"]],
                on="symbol_key",
                how="left",
            )
            coverage_rows.append(
                {
                    "row_id": row_id,
                    "gene_set": f"de_top{top_k}",
                    "n_symbols": int(joined["symbol_key"].nunique()),
                    "uce_valid_fraction": float(joined["uce_valid"].fillna(False).mean()),
                    "uce_chrom_fraction": float(joined["uce_chrom_known"].fillna(False).mean()),
                    "collapsed_symbol_hits": int((pd.to_numeric(joined["n_ensdarg_collapsed"], errors="coerce").fillna(0) > 1).sum()),
                    "missing_from_h5ad": int(joined["uce_valid"].isna().sum()),
                }
            )
        target_row = var_by_symbol.loc[target] if target in var_by_symbol.index else None
        coverage_rows.append(
            {
                "row_id": row_id,
                "gene_set": f"target_{target.lower()}",
                "n_symbols": 1,
                "uce_valid_fraction": float(bool(target_row is not None and bool(target_row["uce_valid"]))),
                "uce_chrom_fraction": float(bool(target_row is not None and bool(target_row["uce_chrom_known"]))),
                "collapsed_symbol_hits": int(target_row is not None and int(target_row["n_ensdarg_collapsed"]) > 1),
                "missing_from_h5ad": int(target_row is None),
            }
        )

    top_expr = top_expression_visibility(adata)
    high_collapse = var[pd.to_numeric(var["n_ensdarg_collapsed"], errors="coerce").fillna(0) > 1].copy()
    high_collapse = high_collapse.sort_values("n_ensdarg_collapsed", ascending=False).head(50)
    high_collapse.to_csv(args.out_dir / "zscape_uce_danio_high_collapse_symbols.csv")
    pd.DataFrame(coverage_rows).to_csv(args.out_dir / "zscape_uce_danio_de_target_coverage_rows.csv", index=False)

    min_de_top50 = min(
        row["uce_valid_fraction"] for row in coverage_rows if row["gene_set"] == "de_top50"
    )
    targets_ok = all(
        row["uce_valid_fraction"] == 1.0 for row in coverage_rows if row["gene_set"].startswith("target_")
    )
    status = (
        "zscape_uce_danio_coverage_addendum_pass"
        if min_de_top50 >= 0.75 and targets_ok and top_expr["mean_top1024_valid_fraction"] >= 0.75
        else "zscape_uce_danio_coverage_addendum_warn"
    )
    payload = {
        "timestamp": now_cst(),
        "status": status,
        "h5ad": str(args.h5ad),
        "top_expression_visibility": top_expr,
        "coverage_rows": coverage_rows,
        "gate": "pass if both targets are UCE-visible, min DE top50 UCE fraction >=0.75, and mean top1024 valid fraction >=0.75",
    }
    (args.out_dir / "zscape_uce_danio_coverage_addendum_20260628.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    report = args.out_dir / "LATENTFM_ZSCAPE_UCE_DANIO_COVERAGE_ADDENDUM_20260628.md"
    lines = [
        "# LatentFM ZSCAPE UCE Danio Coverage Addendum",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`.",
        "",
        "## Boundary",
        "",
        "- CPU-only coverage addendum for the already-frozen UCE Danio smoke h5ad.",
        "- Does not overwrite smoke input, train, infer, or use GPU.",
        "",
        "## Top-Expression Visibility",
        "",
    ]
    for key, val in top_expr.items():
        lines.append(f"- {key}: `{val}`")
    lines.extend(
        [
            "",
            "## DE/Target Coverage",
            "",
            "| row | gene set | symbols | UCE valid | UCE chrom | collapsed hits | missing |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in coverage_rows:
        lines.append(
            "| {row_id} | {gene_set} | {n_symbols} | {valid:.3f} | {chrom:.3f} | {collapsed} | {missing} |".format(
                row_id=row["row_id"],
                gene_set=row["gene_set"],
                n_symbols=row["n_symbols"],
                valid=row["uce_valid_fraction"],
                chrom=row["uce_chrom_fraction"],
                collapsed=row["collapsed_symbol_hits"],
                missing=row["missing_from_h5ad"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Use this addendum as a provenance guard for the UCE latent smoke.",
            "- A warning here should demote any latent result to exploratory, even if embedding succeeds.",
            "",
            "## Outputs",
            "",
            f"- coverage rows: `{args.out_dir / 'zscape_uce_danio_de_target_coverage_rows.csv'}`",
            f"- high-collapse symbols: `{args.out_dir / 'zscape_uce_danio_high_collapse_symbols.csv'}`",
            f"- JSON: `{args.out_dir / 'zscape_uce_danio_coverage_addendum_20260628.json'}`",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
