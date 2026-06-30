#!/usr/bin/env python3
"""CPU-only condition-level metainfo audit for LatentFM scaling law design."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import anndata as ad


ROOT = Path("/data/cyx/1030/scLatent")
H5AD_DIR = ROOT / "dataset/Training_data/scfoundation/gt_scfoundation"
OUT_JSON = ROOT / "reports/latentfm_condition_level_metainfo_scaling_audit_20260624.json"
OUT_CSV = ROOT / "reports/latentfm_condition_level_metainfo_scaling_audit_20260624.csv"
OUT_MD = ROOT / "reports/LATENTFM_CONDITION_LEVEL_METAINFO_SCALING_AUDIT_20260624.md"

PATTERNS = {
    "condition": ("pert", "condition", "gene", "target", "drug", "guide", "combo"),
    "cell_background": ("cell_type", "cell_line", "cellline", "background", "tissue"),
    "perturbation_type": ("perturbation_type", "crispr", "cas", "modality", "pert_type"),
    "dose": ("dose", "dosage", "concentration", "um", "time"),
}


def classify_column(col: str) -> list[str]:
    c = col.lower()
    hits: list[str] = []
    for axis, pats in PATTERNS.items():
        if any(p in c for p in pats):
            hits.append(axis)
    return hits


def top_values(series: Any, limit: int = 8) -> list[dict[str, Any]]:
    values = series.astype(str).fillna("NA").tolist()
    counts = Counter(values)
    return [{"value": k, "n": int(v)} for k, v in counts.most_common(limit)]


def audit_one(path: Path) -> dict[str, Any]:
    ada = ad.read_h5ad(path, backed="r")
    try:
        obs = ada.obs.copy()
        row: dict[str, Any] = {
            "dataset": path.stem,
            "path": str(path),
            "n_obs": int(ada.n_obs),
            "n_vars": int(ada.n_vars),
            "obs_columns": list(map(str, obs.columns)),
            "candidate_columns": {},
        }
        for col in obs.columns:
            axes = classify_column(str(col))
            if not axes:
                continue
            try:
                nunique = int(obs[col].astype(str).nunique(dropna=False))
                examples = top_values(obs[col], limit=8)
            except Exception as exc:  # pragma: no cover - defensive audit script
                nunique = -1
                examples = [{"error": repr(exc)}]
            for axis in axes:
                row["candidate_columns"].setdefault(axis, []).append(
                    {
                        "column": str(col),
                        "n_unique": nunique,
                        "top_values": examples,
                    }
                )
        return row
    finally:
        ada.file.close()


def main() -> int:
    paths = sorted(H5AD_DIR.glob("*.h5ad"))
    rows = [audit_one(path) for path in paths]
    axis_counts = {
        axis: sum(1 for row in rows if row["candidate_columns"].get(axis))
        for axis in PATTERNS
    }
    unresolved = {
        axis: [row["dataset"] for row in rows if not row["candidate_columns"].get(axis)]
        for axis in PATTERNS
    }
    payload = {
        "status": "condition_level_metainfo_scaling_audit_complete",
        "boundary": {
            "cpu_only": True,
            "reads_h5ad_obs_only": True,
            "reads_model_outputs": False,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
            "launches_gpu": False,
        },
        "h5ad_dir": str(H5AD_DIR),
        "n_datasets": len(rows),
        "axis_dataset_counts": axis_counts,
        "unresolved_by_axis": unresolved,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "n_obs",
                "n_vars",
                "axis",
                "column",
                "n_unique",
                "top_values",
            ],
        )
        writer.writeheader()
        for row in rows:
            for axis, cols in row["candidate_columns"].items():
                for col in cols:
                    writer.writerow(
                        {
                            "dataset": row["dataset"],
                            "n_obs": row["n_obs"],
                            "n_vars": row["n_vars"],
                            "axis": axis,
                            "column": col["column"],
                            "n_unique": col["n_unique"],
                            "top_values": "; ".join(
                                f"{v.get('value')}={v.get('n')}" for v in col["top_values"]
                            ),
                        }
                    )

    lines = [
        "# LatentFM Condition-Level Metainfo Scaling Audit",
        "",
        "Status: `condition_level_metainfo_scaling_audit_complete`",
        "",
        "## Boundary",
        "",
        "- CPU-only h5ad obs-column audit.",
        "- Does not read model outputs, canonical metrics, canonical multi, held-out Track C query, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- datasets scanned: `{len(rows)}`",
    ]
    for axis, count in axis_counts.items():
        lines.append(f"- datasets with `{axis}` candidate columns: `{count}`")
    lines.extend(["", "## Unresolved Axes", ""])
    for axis, datasets in unresolved.items():
        preview = ", ".join(f"`{ds}`" for ds in datasets[:12])
        suffix = " ..." if len(datasets) > 12 else ""
        lines.append(f"- `{axis}` unresolved datasets: {preview}{suffix}" if datasets else f"- `{axis}` unresolved datasets: none")
    lines.extend(["", "## Outputs", "", f"- JSON: `{OUT_JSON}`", f"- CSV: `{OUT_CSV}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
