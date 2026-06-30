#!/usr/bin/env python3
"""Build condition-level metadata counts for LatentFM scaling experiments.

Short CPU task. Reads AnnData obs metadata only and aggregates counts by
dataset/bucket/condition/background/type/pathway. No model outputs, canonical
held-out outcomes, or query artifacts are read.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
OBS_SCHEMA_JSON = ROOT / "reports/latentfm_dataset_obs_schema_audit_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_condition_level_inventory_20260624.json"
OUT_TSV = ROOT / "reports/latentfm_condition_level_inventory_20260624.tsv"
OUT_MD = ROOT / "reports/LATENTFM_CONDITION_LEVEL_INVENTORY_20260624.md"


def import_anndata():
    try:
        import anndata as ad  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"anndata import failed: {exc}") from exc
    return ad


def clean_value(value, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return default
    return text


def pick_first(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def infer_columns(columns: list[str], modality: str) -> dict[str, str | None]:
    if modality == "chemical":
        return {
            "condition": pick_first(columns, ["cov_drug_dose_name", "drug_dose_name", "cov_drug", "condition", "cov"]),
            "perturbation": pick_first(columns, ["cov_drug", "target", "condition", "perturbation"]),
            "background": pick_first(columns, ["cell_line", "cell_type", "celltype", "cov"]),
            "perturbation_type": pick_first(columns, ["perturbation_type"]),
            "nperts": pick_first(columns, ["nperts", "n_genes"]),
            "control": pick_first(columns, ["control"]),
            "pathway": pick_first(columns, ["pathway", "pathway_level_1", "pathway_level_2"]),
            "dose": pick_first(columns, ["dose_value", "dose", "cov_drug_dose_name", "drug_dose_name"]),
            "gene": pick_first(columns, ["gene"]),
        }
    return {
        "condition": pick_first(columns, ["perturbation", "condition", "gene", "target"]),
        "perturbation": pick_first(columns, ["perturbation", "gene", "target", "condition"]),
        "background": pick_first(columns, ["cell_line", "cell_type", "celltype", "cov", "tissue_type"]),
        "perturbation_type": pick_first(columns, ["perturbation_type"]),
        "nperts": pick_first(columns, ["nperts", "n_genes"]),
        "control": pick_first(columns, ["control"]),
        "pathway": pick_first(columns, ["pathway", "pathway_level_1", "pathway_level_2"]),
        "dose": pick_first(columns, ["dose_value", "dose"]),
        "gene": pick_first(columns, ["gene", "target"]),
    }


def is_control_row(row: pd.Series, cols: dict[str, str | None]) -> bool:
    for key in ("control", "condition", "perturbation"):
        col = cols.get(key)
        if not col:
            continue
        val = clean_value(row.get(col), "").lower()
        if key == "control" and val in {"true", "1", "yes"}:
            return True
        if val in {"control", "ctrl", "ntc", "non-targeting", "non_targeting", "dmso", "vehicle"}:
            return True
    return False


def audit_file(ad, file_row: dict[str, object]) -> list[dict[str, object]]:
    path = Path(str(file_row["path"]))
    obj = ad.read_h5ad(path, backed="r")
    obs = obj.obs.copy()
    if hasattr(obj, "file") and obj.file is not None:
        try:
            obj.file.close()
        except Exception:
            pass
    cols = infer_columns([str(c) for c in obs.columns], str(file_row["modality"]))
    condition_col = cols.get("condition")
    if condition_col is None:
        raise RuntimeError(f"could not infer condition column for {path}")

    rows = []
    grouped = obs.groupby(condition_col, observed=True, dropna=False)
    for condition, frame in grouped:
        condition_s = clean_value(condition, "unknown")
        first = frame.iloc[0]
        if is_control_row(first, cols):
            continue
        backgrounds = []
        bg_col = cols.get("background")
        if bg_col and bg_col in frame.columns:
            backgrounds = sorted(clean_value(x, "unknown") for x in frame[bg_col].dropna().unique())
        if not backgrounds:
            backgrounds = ["unknown"]
        ptype_col = cols.get("perturbation_type")
        ptype = clean_value(first.get(ptype_col), "unknown") if ptype_col else "unknown"
        nperts_col = cols.get("nperts")
        nperts = clean_value(first.get(nperts_col), "") if nperts_col else ""
        try:
            nperts_i = int(float(nperts)) if nperts else None
        except ValueError:
            nperts_i = None
        pert_col = cols.get("perturbation")
        perturbation = clean_value(first.get(pert_col), condition_s) if pert_col else condition_s
        gene_col = cols.get("gene")
        gene = clean_value(first.get(gene_col), "") if gene_col else ""
        pathway_col = cols.get("pathway")
        pathway = clean_value(first.get(pathway_col), "") if pathway_col else ""
        dose_col = cols.get("dose")
        dose = clean_value(first.get(dose_col), "") if dose_col else ""
        rows.append(
            {
                "dataset": file_row["dataset"],
                "bucket": file_row["bucket"],
                "modality": file_row["modality"],
                "condition": condition_s,
                "perturbation": perturbation,
                "gene": gene,
                "perturbation_type": ptype,
                "nperts": nperts_i,
                "backgrounds": backgrounds,
                "n_backgrounds": len(backgrounds),
                "pathway": pathway,
                "dose": dose,
                "n_cells": int(len(frame)),
                "source_h5ad": str(path),
                "column_mapping": cols,
            }
        )
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_dataset = defaultdict(int)
    cells_by_dataset = defaultdict(int)
    by_modality = Counter()
    by_type = Counter()
    by_bucket = Counter()
    by_background = Counter()
    multi_conditions = 0
    for row in rows:
        by_dataset[str(row["dataset"])] += 1
        cells_by_dataset[str(row["dataset"])] += int(row["n_cells"])
        by_modality[str(row["modality"])] += 1
        by_type[str(row["perturbation_type"])] += 1
        by_bucket[str(row["bucket"])] += 1
        for bg in row["backgrounds"]:
            by_background[str(bg)] += 1
        nperts = row.get("nperts")
        if isinstance(nperts, int) and nperts > 1:
            multi_conditions += 1
        elif "+" in str(row.get("condition", "")):
            multi_conditions += 1

    return {
        "n_condition_rows": len(rows),
        "conditions_by_modality": dict(sorted(by_modality.items())),
        "conditions_by_bucket": dict(sorted(by_bucket.items())),
        "conditions_by_perturbation_type": dict(sorted(by_type.items())),
        "conditions_by_background": dict(sorted(by_background.items())),
        "conditions_by_dataset": dict(sorted(by_dataset.items())),
        "cells_by_dataset": dict(sorted(cells_by_dataset.items())),
        "multi_condition_count_by_obs_or_name": int(multi_conditions),
        "scaling_axes_ready": [
            "dataset count",
            "condition count",
            "cell background",
            "perturbation type",
            "single vs multiple bucket",
            "drug pathway/dose for chemical datasets",
        ],
        "caveats": [
            "backgrounds are inferred from available obs columns and may be unknown for some gene datasets",
            "condition-level rows are metadata only; no outcome metrics are included",
        ],
    }


def write_tsv(rows: list[dict[str, object]]) -> None:
    columns = [
        "dataset",
        "bucket",
        "modality",
        "condition",
        "perturbation",
        "gene",
        "perturbation_type",
        "nperts",
        "backgrounds",
        "n_backgrounds",
        "pathway",
        "dose",
        "n_cells",
        "source_h5ad",
    ]
    with OUT_TSV.open("w", newline="") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            vals = []
            for col in columns:
                value = row.get(col, "")
                if isinstance(value, list):
                    value = ",".join(map(str, value))
                vals.append(str(value).replace("\t", " "))
            f.write("\t".join(vals) + "\n")


def md_dataset_table(summary: dict[str, object]) -> str:
    conds = summary["conditions_by_dataset"]
    cells = summary["cells_by_dataset"]
    lines = ["| dataset | conditions | obs cells |", "|---|---:|---:|"]
    for ds, n in conds.items():
        lines.append(f"| {ds} | {n} | {cells.get(ds, 0)} |")
    return "\n".join(lines)


def main() -> None:
    ad = import_anndata()
    schema = json.loads(OBS_SCHEMA_JSON.read_text())
    rows = []
    for item in schema["rows"]:
        if item.get("status") != "ok":
            continue
        rows.extend(audit_file(ad, item))
    rows = sorted(rows, key=lambda r: (str(r["modality"]), str(r["dataset"]), str(r["bucket"]), str(r["condition"])))
    summary = summarize(rows)
    OUT_JSON.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, sort_keys=True))
    write_tsv(rows)
    OUT_MD.write_text(
        f"""# LatentFM Condition-Level Inventory

Status: `condition_level_inventory_built`

## Boundary

- Short CPU-only metadata aggregation.
- Reads AnnData obs only.
- Does not read model outputs, canonical held-out outcomes, Track C query data,
  active logs, or GPU artifacts.

## Outputs

- JSON: `{OUT_JSON}`
- TSV: `{OUT_TSV}`

## Summary

```json
{json.dumps(summary, indent=2, sort_keys=True)}
```

## Dataset Table

{md_dataset_table(summary)}

## Scaling Interpretation

- We now have metadata rows for `{summary['n_condition_rows']}` non-control
  conditions across dataset/bucket files.
- Ready axes for first GPU smokes: dataset count, condition count,
  perturbation type, cell background where obs exposes it, single-vs-multiple
  bucket, and chemical pathway/dose.
- Before final paper-grade scaling claims, source mapping for entries marked
  candidate/needs-follow-up in the dataset inventory still needs primary-source
  spot checks.
"""
    )
    print(
        json.dumps(
            {
                "status": "condition_level_inventory_built",
                "out_md": str(OUT_MD),
                "out_json": str(OUT_JSON),
                "out_tsv": str(OUT_TSV),
                "n_condition_rows": summary["n_condition_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
