#!/usr/bin/env python3
"""Materialize local SciPlex condition chemical identifiers.

CPU-only metadata extraction from local raw h5ad obs. No expression matrix is
loaded, and no training/inference/GPU/canonical multi/Track C query is used.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
RAW_DIR = ROOT / "dataset" / "raw" / "chemicalpert_bench"
OUT_DIR = ROOT / "reports" / "sciplex_local_chemical_key_table_20260630"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATASETS = ["sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7"]
OBS_COLS = [
    "condition",
    "perturbation",
    "target",
    "pathway",
    "pathway_level_1",
    "pathway_level_2",
    "chembl-ID",
    "SMILES",
    "dose",
    "dose_value",
    "dose_unit",
    "time",
    "cell_line",
    "cell_type",
    "control",
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def norm_values(series: pd.Series) -> str:
    vals = []
    for val in series.dropna().astype(str).unique().tolist():
        val = val.strip()
        if val and val.lower() not in {"nan", "none", "na"}:
            vals.append(val)
    return "|".join(sorted(vals))


def first_nonempty(series: pd.Series) -> str:
    vals = norm_values(series)
    if "|" in vals:
        return vals.split("|")[0]
    return vals


def main() -> int:
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for dataset in DATASETS:
        path = RAW_DIR / f"{dataset}.h5ad"
        adata = ad.read_h5ad(path, backed="r")
        obs = adata.obs[[col for col in OBS_COLS if col in adata.obs.columns]].copy()
        obs["condition"] = obs["condition"].astype(str)
        for condition, sub in obs.groupby("condition", sort=True):
            if condition.lower() in {"control", "ctrl", "dmso", "vehicle"}:
                continue
            row: dict[str, Any] = {
                "dataset": dataset,
                "condition": condition,
                "n_cells": int(len(sub)),
            }
            for col in OBS_COLS:
                if col == "condition" or col not in sub.columns:
                    continue
                row[col.replace("-", "_").replace(" ", "_")] = norm_values(sub[col])
                row[f"{col.replace('-', '_').replace(' ', '_')}_n_unique"] = int(
                    sub[col].dropna().astype(str).nunique()
                )
            rows.append(row)
        inventory.append(
            {
                "dataset": dataset,
                "path": str(path),
                "n_obs": int(adata.n_obs),
                "n_vars": int(adata.n_vars),
                "obs_columns": list(adata.obs.columns),
                "condition_count": int(obs["condition"].nunique()) if "condition" in obs else 0,
            }
        )
        adata.file.close()

    df = pd.DataFrame(rows)
    key_cols = [
        "dataset",
        "condition",
        "n_cells",
        "perturbation",
        "chembl_ID",
        "SMILES",
        "target",
        "pathway",
        "pathway_level_1",
        "pathway_level_2",
        "dose",
        "dose_value",
        "time",
    ]
    for col in key_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[key_cols + [col for col in df.columns if col not in key_cols]]
    out_csv = OUT_DIR / "sciplex_local_chemical_key_table_20260630.csv"
    out_json = OUT_DIR / "sciplex_local_chemical_key_table_20260630.json"
    out_md = OUT_DIR / "LATENTFM_SCIPLEX_LOCAL_CHEMICAL_KEY_TABLE_20260630.md"
    df.to_csv(out_csv, index=False)
    per_dataset = df.groupby("dataset").agg(
        conditions=("condition", "nunique"),
        with_smiles=("SMILES", lambda s: int((s.astype(str).str.len() > 0).sum())),
        with_chembl=("chembl_ID", lambda s: int((s.astype(str).str.len() > 0).sum())),
        unique_smiles=("SMILES", lambda s: int(s[s.astype(str).str.len() > 0].nunique())),
        unique_chembl=("chembl_ID", lambda s: int(s[s.astype(str).str.len() > 0].nunique())),
    ).reset_index()
    payload = {
        "timestamp": now(),
        "status": "sciplex_local_chemical_key_table_complete_no_gpu",
        "boundary": {
            "cpu_metadata_only": True,
            "loads_expression_matrix": False,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
            "uses_gpu": False,
        },
        "inventory": inventory,
        "per_dataset": per_dataset.to_dict(orient="records"),
        "outputs": {"csv": str(out_csv), "json": str(out_json), "markdown": str(out_md)},
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# SciPlex Local Chemical Key Table",
        "",
        f"Created: `{payload['timestamp']}`",
        "",
        "Status: `sciplex_local_chemical_key_table_complete_no_gpu`",
        "",
        "## Boundary",
        "",
        "- CPU metadata-only extraction from local raw h5ad obs.",
        "- No expression matrix loading, training, inference, checkpoint selection, canonical multi selection, Track C query, or GPU.",
        "",
        "## Summary",
        "",
        "| dataset | conditions | with SMILES | with ChEMBL | unique SMILES | unique ChEMBL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in per_dataset.to_dict(orient="records"):
        lines.append(
            f"| `{row['dataset']}` | {row['conditions']} | {row['with_smiles']} | {row['with_chembl']} | "
            f"{row['unique_smiles']} | {row['unique_chembl']} |"
        )
    lines += [
        "",
        "## Use",
        "",
        "Use this table only for source-key harmonization or synonym/identifier mapping preflight. It does not authorize GPU.",
        "",
        "## Outputs",
        "",
        f"- CSV: `{out_csv}`",
        f"- JSON: `{out_json}`",
        f"- Markdown: `{out_md}`",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "csv": str(out_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
