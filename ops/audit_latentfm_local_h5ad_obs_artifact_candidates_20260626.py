#!/usr/bin/env python3
"""Audit local h5ad obs columns for train-only artifact candidates.

Short CPU task. Reads only `.obs` metadata from small local raw/genepert_bench
h5ad files in backed mode. It does not read expression matrices, checkpoints,
canonical multi, Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
CONDITION_INV = ROOT / "reports/latentfm_condition_level_inventory_20260624.json"
OUT_DIR = ROOT / "reports/local_h5ad_obs_artifact_candidates_20260626"
OUT_JSON = ROOT / "reports/latentfm_local_h5ad_obs_artifact_candidates_20260626.json"
OUT_MD = ROOT / "reports/LATENTFM_LOCAL_H5AD_OBS_ARTIFACT_CANDIDATES_20260626.md"
OUT_COLUMNS_CSV = OUT_DIR / "obs_columns.csv"
OUT_ARTIFACT_CSV = OUT_DIR / "candidate_condition_artifacts.csv"

DATASETS = {
    "Adamson": ROOT / "dataset/raw/genepert_bench/Adamson.h5ad",
    "DixitRegev2016_K562_TFs_High_MOI": ROOT / "dataset/raw/genepert_bench/DixitRegev2016_K562_TFs_High_MOI.h5ad",
    "Frangieh": ROOT / "dataset/raw/genepert_bench/Frangieh.h5ad",
    "GasperiniShendure2019_lowMOI": ROOT / "dataset/raw/genepert_bench/GasperiniShendure2019_lowMOI__single.h5ad",
    "NormanWeissman2019_filtered": ROOT / "dataset/raw/genepert_bench/NormanWeissman2019_filtered__single.h5ad",
    "Papalexi": ROOT / "dataset/raw/genepert_bench/Papalexi.h5ad",
}

CONDITION_COLUMN_CANDIDATES = [
    "condition",
    "perturbation",
    "perturbation_name",
    "gene",
    "target",
    "gene_target",
    "covariate",
]

VALUE_FIELD_FAMILIES = {
    "guide_reagent": ["guide", "sgrna", "grna", "guide_identity", "guide_merged", "guide_ids"],
    "coverage_quality": ["coverage", "good_coverage", "n_counts", "total_counts", "n_genes", "pct_counts_mt"],
    "time_maturity": ["time", "timepoint", "day", "hour"],
    "dose": ["dose", "concentration", "perturbation_value"],
    "background": ["cell_type", "cell_line", "celltype"],
    "batch": ["batch", "replicate", "library", "sample"],
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def local_conditions() -> dict[str, set[str]]:
    payload = read_json(CONDITION_INV)
    out: dict[str, set[str]] = defaultdict(set)
    for row in payload.get("rows", []):
        out[str(row.get("dataset", ""))].add(str(row.get("condition", "")))
    return out


def norm_condition(value: Any) -> str:
    text = str(value)
    if text in {"nan", "None"}:
        return ""
    for sep in ["+", "_", "|"]:
        if sep in text and text.count(sep) > 4:
            break
    return text.strip()


def choose_condition_column(obs: pd.DataFrame, expected: set[str]) -> tuple[str | None, int]:
    best_col: str | None = None
    best_overlap = -1
    lower_cols = {col.lower(): col for col in obs.columns}
    candidates = []
    for key in CONDITION_COLUMN_CANDIDATES:
        if key in lower_cols:
            candidates.append(lower_cols[key])
    candidates.extend([col for col in obs.columns if "perturb" in col.lower() or "gene" in col.lower()])
    seen = set()
    for col in candidates:
        if col in seen:
            continue
        seen.add(col)
        vals = {norm_condition(v) for v in obs[col].dropna().astype(str).unique()}
        overlap = len(vals & expected)
        if overlap > best_overlap:
            best_col = col
            best_overlap = overlap
    return best_col, max(best_overlap, 0)


def family_for_column(col: str) -> str | None:
    low = col.lower()
    for family, tokens in VALUE_FIELD_FAMILIES.items():
        if any(tok in low for tok in tokens):
            return family
    return None


def summarize_column(series: pd.Series) -> dict[str, Any]:
    non_na = series.dropna()
    nunique = int(non_na.astype(str).nunique()) if len(non_na) else 0
    sample = [str(x) for x in non_na.astype(str).unique()[:6]]
    numeric = pd.to_numeric(non_na, errors="coerce")
    numeric_frac = float(numeric.notna().mean()) if len(non_na) else 0.0
    return {
        "dtype": str(series.dtype),
        "non_na": int(len(non_na)),
        "nunique": nunique,
        "sample_values": sample,
        "numeric_frac": numeric_frac,
    }


def numeric_artifact_value(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() >= 0.80 and numeric.notna().any():
        return float(numeric.mean())
    counts = Counter(str(x) for x in values.dropna())
    if not counts:
        return None
    total = sum(counts.values())
    # Categorical stability proxy: dominant-category fraction.
    return float(counts.most_common(1)[0][1] / total)


def build_condition_artifacts(
    dataset: str,
    obs: pd.DataFrame,
    condition_col: str,
    expected: set[str],
    candidate_cols: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = obs.dropna(subset=[condition_col]).groupby(condition_col, observed=True)
    for col in candidate_cols:
        family = family_for_column(col)
        if family is None:
            continue
        for condition, group in grouped:
            condition_norm = norm_condition(condition)
            if condition_norm not in expected:
                continue
            value = numeric_artifact_value(group[col])
            if value is None or not math.isfinite(value):
                continue
            rows.append(
                {
                    "artifact": f"local_obs_{family}_{col}",
                    "dataset": dataset,
                    "condition": condition_norm,
                    "artifact_value": value,
                    "source_file": str(DATASETS[dataset]),
                    "source_column": col,
                    "source_condition_column": condition_col,
                    "family": family,
                    "n_cells": int(len(group)),
                }
            )
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conditions = local_conditions()
    column_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    dataset_summaries: list[dict[str, Any]] = []

    for dataset, path in DATASETS.items():
        if not path.exists():
            dataset_summaries.append({"dataset": dataset, "path": str(path), "status": "missing"})
            continue
        adata = ad.read_h5ad(path, backed="r")
        obs = adata.obs.copy()
        try:
            adata.file.close()
        except Exception:
            pass
        expected = conditions.get(dataset, set())
        condition_col, overlap = choose_condition_column(obs, expected)
        candidate_cols = [col for col in obs.columns if family_for_column(col) is not None]
        dataset_artifacts = (
            build_condition_artifacts(dataset, obs, condition_col, expected, candidate_cols)
            if condition_col
            else []
        )
        artifact_rows.extend(dataset_artifacts)
        dataset_summaries.append(
            {
                "dataset": dataset,
                "path": str(path),
                "status": "read_obs",
                "n_obs": int(obs.shape[0]),
                "n_obs_columns": int(obs.shape[1]),
                "condition_column": condition_col,
                "condition_overlap": overlap,
                "expected_conditions": len(expected),
                "candidate_value_columns": candidate_cols,
                "artifact_rows": len(dataset_artifacts),
            }
        )
        for col in obs.columns:
            summary = summarize_column(obs[col])
            column_rows.append(
                {
                    "dataset": dataset,
                    "path": str(path),
                    "column": col,
                    "family": family_for_column(col) or "",
                    **summary,
                }
            )

    with OUT_COLUMNS_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "dataset",
            "path",
            "column",
            "family",
            "dtype",
            "non_na",
            "nunique",
            "numeric_frac",
            "sample_values",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in column_rows:
            out = dict(row)
            out["sample_values"] = ";".join(out["sample_values"])
            writer.writerow(out)

    with OUT_ARTIFACT_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "artifact",
            "dataset",
            "condition",
            "artifact_value",
            "source_file",
            "source_column",
            "source_condition_column",
            "family",
            "n_cells",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in artifact_rows:
            writer.writerow(row)

    family_counts = Counter(row["family"] for row in artifact_rows)
    dataset_artifact_counts = Counter(row["dataset"] for row in artifact_rows)
    condition_artifact_pairs = {(row["dataset"], row["condition"]) for row in artifact_rows}
    status = (
        "local_h5ad_obs_artifact_candidates_ready_cpu_preflight_next"
        if artifact_rows
        else "local_h5ad_obs_artifact_candidates_fail_no_artifact_rows"
    )
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": "local h5ad obs only; no expression/checkpoint/canonical multi/Track C query/training/inference/GPU",
        "dataset_summaries": dataset_summaries,
        "column_csv": str(OUT_COLUMNS_CSV),
        "candidate_artifact_csv": str(OUT_ARTIFACT_CSV),
        "artifact_row_count": len(artifact_rows),
        "artifact_condition_pair_count": len(condition_artifact_pairs),
        "artifact_family_counts": dict(sorted(family_counts.items())),
        "dataset_artifact_counts": dict(sorted(dataset_artifact_counts.items())),
        "decision": (
            "Generated a candidate condition-level artifact table from local raw h5ad obs. "
            "This does not authorize GPU; next step is the existing external-artifact preflight "
            "with overlap, variation, shuffle/source/count/tail controls."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Local h5ad obs Artifact Candidate Audit",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M CST')}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- Reads only `.obs` metadata from small local `dataset/raw/genepert_bench/*.h5ad` files in backed mode.",
        "- Does not read expression matrices, checkpoints, canonical multi, Track C query, train, infer, or use GPU.",
        "- Candidate artifacts are source material for a CPU gate, not launch authorization.",
        "",
        "## Summary",
        "",
        f"- datasets scanned: `{len(DATASETS)}`",
        f"- candidate artifact rows: `{len(artifact_rows)}`",
        f"- condition pairs covered: `{len(condition_artifact_pairs)}`",
        f"- artifact family counts: `{dict(sorted(family_counts.items()))}`",
        "",
        "## Dataset Rows",
        "",
        "| dataset | status | obs rows | condition column | overlap | artifact rows | candidate columns |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for row in dataset_summaries:
        lines.append(
            "| `{dataset}` | `{status}` | `{n_obs}` | `{condition_column}` | `{condition_overlap}` | `{artifact_rows}` | {cols} |".format(
                dataset=row.get("dataset", ""),
                status=row.get("status", ""),
                n_obs=row.get("n_obs", 0),
                condition_column=row.get("condition_column", ""),
                condition_overlap=row.get("condition_overlap", 0),
                artifact_rows=row.get("artifact_rows", 0),
                cols=", ".join(f"`{c}`" for c in row.get("candidate_value_columns", [])[:10]),
            )
        )
    lines += [
        "",
        "## Decision",
        "",
        "- This creates a concrete condition-level candidate artifact table for the scaling/new-artifact route.",
        "- No GPU is authorized yet; run the external artifact preflight next and require overlap, within-dataset variation, shuffle/LODO/source/count controls, and tail/MMD checks.",
        "- If the preflight fails, keep this as source/provenance evidence and do not launch weighted-loss or balancing training.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- obs columns: `{OUT_COLUMNS_CSV}`",
        f"- candidate artifacts: `{OUT_ARTIFACT_CSV}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
