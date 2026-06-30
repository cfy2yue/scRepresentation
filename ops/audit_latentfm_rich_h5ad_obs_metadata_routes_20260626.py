#!/usr/bin/env python3
"""Audit richer local h5ad obs metadata for non-QC artifact routes.

Short CPU task. Reads only `.obs` metadata in backed mode from local h5ad files
that may preserve richer source fields than the small benchmark files. It does
not read expression matrices, checkpoints, canonical multi, Track C query,
train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad


ROOT = Path("/data/cyx/1030/scLatent")
CONDITION_INV = ROOT / "reports/latentfm_condition_level_inventory_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_rich_h5ad_obs_metadata_routes_20260626.json"
OUT_MD = ROOT / "reports/LATENTFM_RICH_H5AD_OBS_METADATA_ROUTES_20260626.md"
OUT_CSV = ROOT / "reports/latentfm_rich_h5ad_obs_metadata_routes_20260626.csv"

DATASET_PATTERNS = {
    "Adamson": ["Adamson.h5ad"],
    "DixitRegev2016_K562_TFs_High_MOI": ["DixitRegev2016_K562_TFs_High_MOI.h5ad"],
    "Frangieh": ["Frangieh.h5ad"],
    "GasperiniShendure2019_lowMOI": [
        "GasperiniShendure2019_lowMOI.h5ad",
        "GasperiniShendure2019_lowMOI__single.h5ad",
        "GasperiniShendure2019_lowMOI__multiple.h5ad",
    ],
    "NormanWeissman2019_filtered": [
        "NormanWeissman2019_filtered.h5ad",
        "NormanWeissman2019_filtered__single.h5ad",
        "NormanWeissman2019_filtered__multiple.h5ad",
    ],
    "Papalexi": ["Papalexi.h5ad"],
}

SEARCH_DIRS = [
    ROOT / "dataset/raw/genepert_DE5000",
    ROOT / "dataset/scFM_data/staging/genepert",
    ROOT / "dataset/raw/genepert_bench",
    ROOT / "dataset/biFlow_data/gt_stack",
    ROOT / "dataset/biFlow_data/control_stack",
]

ROUTE_TOKENS = {
    "guide_reagent": ["guide", "sgrna", "grna", "protospacer", "barcode"],
    "time_maturity": ["time", "timepoint", "hour", "day", "duration"],
    "viability_fitness": ["viability", "fitness", "growth", "essential", "depletion", "coverage"],
    "dose": ["dose", "concentration"],
    "background": ["cell_type", "celltype", "cell_line", "cellline", "cell"],
    "batch_qc": ["batch", "replicate", "library", "sample", "n_genes", "counts", "pct_counts", "mt"],
    "perturbation_label": ["perturb", "condition", "target", "gene"],
}

FORBIDDEN_AS_ARTIFACT = {"perturbation_label", "batch_qc", "background"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def local_conditions() -> dict[str, set[str]]:
    payload = read_json(CONDITION_INV)
    out: dict[str, set[str]] = defaultdict(set)
    for row in payload.get("rows", []):
        out[str(row.get("dataset", ""))].add(str(row.get("condition", "")))
    return out


def candidate_files() -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for dataset, names in DATASET_PATTERNS.items():
        for base in SEARCH_DIRS:
            for name in names:
                path = base / name
                if path.is_file() and path not in seen:
                    seen.add(path)
                    rows.append((dataset, path))
    return rows


def classify_column(column: str) -> list[str]:
    low = column.lower()
    out = []
    for family, tokens in ROUTE_TOKENS.items():
        if any(tok in low for tok in tokens):
            out.append(family)
    return out


def summarize_obs(dataset: str, path: Path, expected: set[str]) -> dict[str, Any]:
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    try:
        adata.file.close()
    except Exception:
        pass
    columns = list(obs.columns)
    route_cols: dict[str, list[str]] = defaultdict(list)
    column_examples: dict[str, list[str]] = {}
    for col in columns:
        families = classify_column(col)
        if families:
            non_na = obs[col].dropna().astype(str)
            examples = [x for x in non_na.unique()[:5]]
            column_examples[col] = examples
            for family in families:
                route_cols[family].append(col)
    condition_overlap = 0
    condition_columns = []
    for col in columns:
        families = classify_column(col)
        if "perturbation_label" not in families:
            continue
        vals = set(obs[col].dropna().astype(str).unique())
        overlap = len(vals & expected)
        if overlap:
            condition_columns.append({"column": col, "overlap": overlap})
            condition_overlap = max(condition_overlap, overlap)
    actionable_families = sorted(
        family
        for family, cols in route_cols.items()
        if cols and family not in FORBIDDEN_AS_ARTIFACT
    )
    actionable_non_qc = sorted(f for f in actionable_families if f not in {"batch_qc", "dose"})
    return {
        "dataset": dataset,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "n_obs": int(obs.shape[0]),
        "n_obs_columns": int(obs.shape[1]),
        "columns": columns,
        "route_columns": {key: cols for key, cols in sorted(route_cols.items())},
        "column_examples": column_examples,
        "condition_columns": sorted(condition_columns, key=lambda x: (-x["overlap"], x["column"]))[:8],
        "condition_overlap_max": condition_overlap,
        "actionable_families": actionable_families,
        "actionable_non_qc_families": actionable_non_qc,
        "has_non_qc_route_candidate": bool(actionable_non_qc),
    }


def main() -> int:
    expected = local_conditions()
    rows = []
    errors = []
    for dataset, path in candidate_files():
        try:
            rows.append(summarize_obs(dataset, path, expected.get(dataset, set())))
        except Exception as exc:  # keep audit fail-soft across heterogeneous files
            errors.append({"dataset": dataset, "path": str(path), "error": repr(exc)})

    family_counts = Counter()
    non_qc_files = []
    for row in rows:
        for family in row["actionable_families"]:
            family_counts[family] += 1
        if row["has_non_qc_route_candidate"]:
            non_qc_files.append(row["path"])

    status = (
        "rich_h5ad_obs_non_qc_source_candidates_found_cpu_extract_next"
        if non_qc_files
        else "rich_h5ad_obs_no_non_qc_artifact_candidates_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": "local h5ad obs metadata only; no expression/checkpoint/canonical multi/Track C query/training/inference/GPU",
        "files_scanned": len(rows),
        "errors": errors,
        "actionable_family_file_counts": dict(sorted(family_counts.items())),
        "non_qc_candidate_files": non_qc_files,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dataset",
            "path",
            "size_bytes",
            "n_obs",
            "n_obs_columns",
            "condition_overlap_max",
            "actionable_families",
            "actionable_non_qc_families",
            "route_columns",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "path": row["path"],
                    "size_bytes": row["size_bytes"],
                    "n_obs": row["n_obs"],
                    "n_obs_columns": row["n_obs_columns"],
                    "condition_overlap_max": row["condition_overlap_max"],
                    "actionable_families": ";".join(row["actionable_families"]),
                    "actionable_non_qc_families": ";".join(row["actionable_non_qc_families"]),
                    "route_columns": json.dumps(row["route_columns"], sort_keys=True),
                }
            )

    lines = [
        "# LatentFM Rich h5ad obs Metadata Route Audit",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M CST')}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- Reads only `.obs` metadata from local richer h5ad files in backed mode.",
        "- Does not read expression matrices, checkpoints, canonical multi, Track C query, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- files scanned: `{len(rows)}`",
        f"- scan errors: `{len(errors)}`",
        f"- actionable family file counts: `{dict(sorted(family_counts.items()))}`",
        f"- non-QC candidate files: `{len(non_qc_files)}`",
        "",
        "## Rows",
        "",
        "| dataset | file | obs rows | columns | condition overlap | actionable families | non-QC families |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['dataset']}` | `{Path(row['path']).relative_to(ROOT)}` | "
            f"{row['n_obs']} | {row['n_obs_columns']} | {row['condition_overlap_max']} | "
            f"`{','.join(row['actionable_families'])}` | "
            f"`{','.join(row['actionable_non_qc_families'])}` |"
        )
    if errors:
        lines += ["", "## Errors", ""]
        for err in errors:
            lines.append(f"- `{err['path']}`: `{err['error']}`")
    lines += [
        "",
        "## Decision",
        "",
    ]
    if non_qc_files:
        lines.append(
            "- Non-QC metadata columns exist locally. Next step is a narrow extraction script that maps only source-safe columns to condition-level `dataset,condition,artifact_value`, then runs the external-artifact preflight."
        )
    else:
        lines.append(
            "- No non-QC local obs metadata route was found. Continue external source acquisition rather than launching GPU."
        )
    lines += [
        "- This audit alone does not authorize GPU.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- rows: `{OUT_CSV}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
