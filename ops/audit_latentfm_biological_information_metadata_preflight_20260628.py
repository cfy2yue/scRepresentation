#!/usr/bin/env python3
"""CPU-only metadata preflight for biological information scaling axes.

The goal is to check whether current benchmark h5ad files expose cell type,
subcluster, lineage, tissue, pathway, dose, and perturbation metadata that can
support biologically interpretable information-scaling metrics.

This script opens AnnData files in backed read-only mode and does not train,
infer, read canonical multi, read Track C held-out query, select checkpoints,
or use GPU.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import anndata as ad


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports" / "biological_information_metadata_preflight_20260628"
OUT_MD = ROOT / "reports" / "LATENTFM_BIOLOGICAL_INFORMATION_METADATA_PREFLIGHT_20260628.md"
OUT_JSON = ROOT / "reports" / "latentfm_biological_information_metadata_preflight_20260628.json"
OUT_CSV = OUT_DIR / "h5ad_obs_metadata_inventory.csv"

DATASET_GLOBS = [
    "dataset/raw/genepert_bench/*.h5ad",
    "dataset/raw/chemicalpert_bench/*.h5ad",
]

PATTERNS = {
    "cell_type": ["cell_type", "celltype", "cell type", "annotation", "cell_name", "cell", "ct"],
    "subcluster": ["subcluster", "sub_cluster", "cluster", "leiden", "louvain", "subtype", "sub_type"],
    "lineage_tissue": ["lineage", "tissue", "organ", "germ", "compartment"],
    "perturbation": ["perturb", "condition", "gene", "guide", "drug", "target", "cov"],
    "dose_time": ["dose", "time", "day", "hpf", "stage"],
    "pathway": ["pathway", "moa", "mechanism", "target_class"],
    "batch_sample": ["batch", "sample", "replicate", "donor", "embryo", "hash"],
}


def matches(column: str, needles: list[str]) -> bool:
    lowered = column.lower()
    return any(needle in lowered for needle in needles)


def unique_count(obs, column: str) -> int | str:
    try:
        return int(obs[column].astype(str).nunique(dropna=True))
    except Exception as exc:  # noqa: BLE001 - inventory should continue.
        return f"error:{type(exc).__name__}"


def inspect_h5ad(path: Path) -> dict[str, object]:
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs
    columns = list(obs.columns)
    row: dict[str, object] = {
        "dataset_path": str(path.relative_to(ROOT)),
        "dataset": path.stem,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_obs_columns": len(columns),
        "obs_columns": ";".join(columns),
    }
    for family, needles in PATTERNS.items():
        hits = [c for c in columns if matches(c, needles)]
        row[f"{family}_columns"] = ";".join(hits)
        row[f"{family}_n_columns"] = len(hits)
        row[f"{family}_first_unique"] = unique_count(obs, hits[0]) if hits else ""
    adata.file.close()
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted({p for pattern in DATASET_GLOBS for p in ROOT.glob(pattern)})
    rows = [inspect_h5ad(path) for path in paths]

    fieldnames = [
        "dataset_path",
        "dataset",
        "n_obs",
        "n_vars",
        "n_obs_columns",
        "obs_columns",
    ]
    for family in PATTERNS:
        fieldnames.extend([f"{family}_columns", f"{family}_n_columns", f"{family}_first_unique"])
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "biological_information_metadata_preflight_ready_no_gpu",
        "gpu_authorized": False,
        "n_h5ad_files": len(rows),
        "files_with_cell_type_columns": sum(1 for r in rows if int(r["cell_type_n_columns"]) > 0),
        "files_with_subcluster_columns": sum(1 for r in rows if int(r["subcluster_n_columns"]) > 0),
        "files_with_lineage_tissue_columns": sum(1 for r in rows if int(r["lineage_tissue_n_columns"]) > 0),
        "files_with_pathway_columns": sum(1 for r in rows if int(r["pathway_n_columns"]) > 0),
        "files_with_dose_time_columns": sum(1 for r in rows if int(r["dose_time_n_columns"]) > 0),
        "csv": str(OUT_CSV),
    }
    with OUT_JSON.open("w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)

    top_celltype = [r for r in rows if int(r["cell_type_n_columns"]) > 0][:12]
    top_subcluster = [r for r in rows if int(r["subcluster_n_columns"]) > 0][:12]
    lines = [
        "# LatentFM Biological Information Metadata Preflight",
        "",
        "Timestamp: `2026-06-28 04:32 CST`",
        "",
        "Status: `biological_information_metadata_preflight_ready_no_gpu`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only metadata inventory over current benchmark h5ad files.",
        "- Opens AnnData files in backed read-only mode.",
        "- Does not train, infer, read canonical multi, read Track C held-out query, select checkpoints, or use GPU.",
        "",
        "## Summary",
        "",
        f"- H5AD files audited: `{summary['n_h5ad_files']}`.",
        f"- Files with cell-type-like columns: `{summary['files_with_cell_type_columns']}`.",
        f"- Files with subcluster-like columns: `{summary['files_with_subcluster_columns']}`.",
        f"- Files with lineage/tissue-like columns: `{summary['files_with_lineage_tissue_columns']}`.",
        f"- Files with pathway-like columns: `{summary['files_with_pathway_columns']}`.",
        f"- Files with dose/time-like columns: `{summary['files_with_dose_time_columns']}`.",
        "",
        "## Biological Information Axis",
        "",
        "The target biological scaling x-axis should include cell-type-resolved state information: per-cell-type state diversity, effective subcluster count, rare-state coverage, response heterogeneity, and trajectory/differentiation complexity. If current Track A benchmark files lack enough cell-type/subcluster metadata, this axis should be advanced through ZSCAPE or another annotated atlas rather than inferred from cell-line labels alone.",
        "",
        "## Cell-Type-Like Metadata Examples",
        "",
        "| dataset | n_obs | columns | first unique count |",
        "|---|---:|---|---:|",
    ]
    for row in top_celltype:
        lines.append(
            f"| `{row['dataset']}` | {row['n_obs']} | `{row['cell_type_columns']}` | {row['cell_type_first_unique']} |"
        )
    lines.extend(
        [
            "",
            "## Subcluster-Like Metadata Examples",
            "",
            "| dataset | n_obs | columns | first unique count |",
            "|---|---:|---|---:|",
        ]
    )
    for row in top_subcluster:
        lines.append(
            f"| `{row['dataset']}` | {row['n_obs']} | `{row['subcluster_columns']}` | {row['subcluster_first_unique']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Record biological information content as a required interpretation layer for future scaling claims. The next association gate should not only ask whether an information metric predicts pp/MMD/tails, but also whether it maps to a biological unit such as cell type, subcluster/state diversity, pathway/target family, or dynamic trajectory complexity.",
            "",
            "No GPU is authorized by this metadata preflight.",
            "",
            "## Outputs",
            "",
            f"- CSV: `{OUT_CSV}`",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
