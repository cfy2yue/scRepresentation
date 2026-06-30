#!/usr/bin/env python3
"""Audit h5ad obs schemas for LatentFM scaling-study stratification.

Short CPU task. Reads only AnnData obs metadata from benchmark h5ad files; does
not read model outputs, canonical held-out outcomes, or query artifacts.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
INVENTORY_JSON = ROOT / "reports/latentfm_dataset_scaling_inventory_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_dataset_obs_schema_audit_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_DATASET_OBS_SCHEMA_AUDIT_20260624.md"

CANDIDATE_PATTERNS = {
    "background": ("cell", "cell_line", "celltype", "tissue", "line", "donor", "cov"),
    "perturbation": ("pert", "gene", "condition", "guide", "grna", "sg", "target", "cov"),
    "drug": ("drug", "dose", "pathway", "compound", "cov"),
    "control": ("control", "ctrl", "ntc", "vehicle", "dmso"),
    "multi": ("combo", "multiple", "num", "n_guides", "guide_count", "cov"),
}


def import_anndata():
    try:
        import anndata as ad  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"anndata import failed: {exc}") from exc
    return ad


def load_inventory() -> list[dict[str, object]]:
    data = json.loads(INVENTORY_JSON.read_text())
    return data["rows"]


def unique_output_files(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen = {}
    for row in rows:
        modality = str(row["modality"])
        base = (
            ROOT / "dataset/raw/genepert_bench"
            if modality == "gene"
            else ROOT / "dataset/raw/chemicalpert_bench"
        )
        path = base / str(row["output_file"])
        seen[str(path)] = {
            "dataset": row["dataset"],
            "bucket": row["bucket"],
            "modality": modality,
            "path": path,
        }
    return sorted(seen.values(), key=lambda x: str(x["path"]))


def sample_values(series, max_items: int = 8) -> list[str]:
    values = []
    try:
        raw = series.dropna().astype(str).unique().tolist()
    except Exception:
        raw = []
    for value in raw[:max_items]:
        text = str(value)
        if len(text) > 80:
            text = text[:77] + "..."
        values.append(text)
    return values


def classify_columns(columns: list[str]) -> dict[str, list[str]]:
    lower = {col: col.lower() for col in columns}
    classes = {}
    for label, patterns in CANDIDATE_PATTERNS.items():
        hits = [
            col for col, low in lower.items() if any(pattern in low for pattern in patterns)
        ]
        classes[label] = sorted(hits)
    return classes


def audit_file(ad, item: dict[str, object]) -> dict[str, object]:
    path = Path(item["path"])
    row = {
        "dataset": item["dataset"],
        "bucket": item["bucket"],
        "modality": item["modality"],
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        row["status"] = "missing_h5ad"
        return row

    try:
        obj = ad.read_h5ad(path, backed="r")
        obs = obj.obs
        columns = [str(c) for c in obs.columns]
        row.update(
            {
                "status": "ok",
                "n_obs": int(obj.n_obs),
                "n_vars": int(obj.n_vars),
                "obs_columns": columns,
                "obs_candidate_columns": classify_columns(columns),
                "obs_samples": {
                    col: sample_values(obs[col])
                    for col in columns
                    if any(col in cols for cols in classify_columns(columns).values())
                },
            }
        )
        obj.file.close()
    except Exception as exc:
        row.update({"status": "read_failed", "error": str(exc)})
    return row


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    status_counter = Counter(str(row.get("status")) for row in rows)
    column_counter: Counter[str] = Counter()
    class_counter: dict[str, Counter[str]] = {
        key: Counter() for key in CANDIDATE_PATTERNS
    }
    for row in rows:
        for col in row.get("obs_columns", []):
            column_counter[str(col)] += 1
        for label, cols in row.get("obs_candidate_columns", {}).items():
            for col in cols:
                class_counter[label][str(col)] += 1
    return {
        "n_files": len(rows),
        "status_counts": dict(sorted(status_counter.items())),
        "top_obs_columns": column_counter.most_common(30),
        "candidate_columns_by_class": {
            label: counter.most_common(20) for label, counter in class_counter.items()
        },
        "recommended_scaling_columns": {
            "gene_background": [
                "cell_line",
                "cell_type",
                "cov",
                "condition",
            ],
            "gene_perturbation": [
                "condition",
                "perturbation",
                "gene",
                "guide_id",
                "cov",
            ],
            "chemical_background": ["cell_line", "cov"],
            "chemical_drug_dose_pathway": [
                "cov_drug",
                "cov_drug_dose_name",
                "pathway",
                "pathway_level_1",
                "pathway_level_2",
            ],
        },
        "next_gate": "condition-level inventory using selected obs columns before GPU scaling ablations",
    }


def md_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| dataset | bucket | modality | n_obs | n_vars | key candidate columns | status |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        candidates = []
        for label, cols in row.get("obs_candidate_columns", {}).items():
            if cols:
                candidates.append(f"{label}: {','.join(cols[:5])}")
        cand = "; ".join(candidates)
        if len(cand) > 160:
            cand = cand[:157] + "..."
        lines.append(
            "| {dataset} | {bucket} | {modality} | {n_obs} | {n_vars} | {cand} | {status} |".format(
                dataset=row.get("dataset", ""),
                bucket=row.get("bucket", ""),
                modality=row.get("modality", ""),
                n_obs=row.get("n_obs", ""),
                n_vars=row.get("n_vars", ""),
                cand=cand,
                status=row.get("status", ""),
            )
        )
    return "\n".join(lines)


def main() -> None:
    ad = import_anndata()
    files = unique_output_files(load_inventory())
    rows = [audit_file(ad, item) for item in files]
    summary = summarize(rows)
    OUT_JSON.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, sort_keys=True))
    OUT_MD.write_text(
        f"""# LatentFM Dataset Obs-Schema Audit

Status: `dataset_obs_schema_audit_done`

## Boundary

- Short CPU-only metadata audit.
- Reads only AnnData obs schemas and small unique-value samples.
- Does not read model outputs, canonical held-out outcomes, Track C query data,
  active logs, or GPU artifacts.

## Outputs

- JSON: `{OUT_JSON}`

## Summary

```json
{json.dumps(summary, indent=2, sort_keys=True)}
```

## File Table

{md_table(rows)}

## Immediate Interpretation

- Chemical sci-Plex3 files expose drug/dose/pathway columns directly.
- Gene perturbation files expose candidate perturbation/background/control
  fields, but exact condition-level background split needs a follow-up
  condition-table inventory, because some dataset-level `cell_line` entries are
  mixed backgrounds.
- The next CPU gate should aggregate selected obs columns by dataset/bucket into
  condition-level counts before any GPU scaling ablation is launched.
"""
    )
    print(
        json.dumps(
            {
                "status": "dataset_obs_schema_audit_done",
                "out_md": str(OUT_MD),
                "out_json": str(OUT_JSON),
                "n_files": len(rows),
                "status_counts": summary["status_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
