#!/usr/bin/env python3
"""SciPlex explicit time-vector feasibility preflight.

CPU/report-only. Checks whether local SciPlex raw files and LatentFM condition
metadata contain a real time axis that could support a ZSCAPE-inspired temporal
response-vector gate.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
RAW_DIR = ROOT / "dataset/raw/chemicalpert_bench"
COND_META = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
OUT_DIR = ROOT / "reports/sciplex_explicit_time_vector_preflight_20260629"
OUT_MD = OUT_DIR / "LATENTFM_SCIPLEX_EXPLICIT_TIME_VECTOR_PREFLIGHT_20260629.md"
OUT_JSON = OUT_DIR / "latentfm_sciplex_explicit_time_vector_preflight_20260629.json"
OUT_DATASET_CSV = OUT_DIR / "sciplex_time_dataset_inventory.csv"
OUT_CONDITION_CSV = OUT_DIR / "sciplex_time_condition_inventory.csv"


SCIPLEX_FILES = [
    RAW_DIR / "sciplex3_A549.h5ad",
    RAW_DIR / "sciplex3_K562.h5ad",
    RAW_DIR / "sciplex3_MCF7.h5ad",
]


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def as_counts(series: pd.Series, limit: int = 12) -> str:
    counts = series.astype(str).value_counts(dropna=False).head(limit)
    return ";".join(f"{idx}:{int(val)}" for idx, val in counts.items())


def latent_metadata_time_fields() -> dict[str, Any]:
    with COND_META.open(encoding="utf-8") as fh:
        meta = json.load(fh)
    rows: list[dict[str, Any]] = []
    time_like = ("time", "hour", "hr", "duration", "day")
    dose_like = ("dose", "conc", "concentration")
    for dataset in sorted(d for d in meta if d.startswith("sciplex3_")):
        keys_seen: set[str] = set()
        time_keys: set[str] = set()
        dose_keys: set[str] = set()
        for condition, entry in meta[dataset].items():
            for key in entry:
                keys_seen.add(str(key))
                low = str(key).lower()
                if any(token in low for token in time_like):
                    time_keys.add(str(key))
                if any(token in low for token in dose_like):
                    dose_keys.add(str(key))
        rows.append(
            {
                "dataset": dataset,
                "n_conditions": len(meta[dataset]),
                "metadata_keys": sorted(keys_seen),
                "time_like_keys": sorted(time_keys),
                "dose_like_keys": sorted(dose_keys),
            }
        )
    return {"datasets": rows}


def inspect_file(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = path.stem
    if not path.exists():
        return {"dataset": dataset, "path": str(path), "status": "missing"}, []
    adata = ad.read_h5ad(path, backed="r")
    try:
        obs = adata.obs.copy()
    finally:
        adata.file.close()
    time_col = "time" if "time" in obs.columns else None
    dose_col = "dose" if "dose" in obs.columns else "dose_value" if "dose_value" in obs.columns else None
    condition_col = "condition" if "condition" in obs.columns else "perturbation"
    control_col = "control" if "control" in obs.columns else None
    replicate_col = "replicate" if "replicate" in obs.columns else None
    time_values = obs[time_col].astype(str) if time_col else pd.Series(dtype=str)
    dose_values = obs[dose_col].astype(str) if dose_col else pd.Series(dtype=str)
    condition_values = obs[condition_col].astype(str) if condition_col in obs.columns else pd.Series(dtype=str)
    control_mask = obs[control_col].astype(str).isin({"1", "true", "True", "control"}) if control_col else condition_values.eq("control")
    dataset_row = {
        "dataset": dataset,
        "path": str(path.relative_to(ROOT)),
        "status": "ok",
        "n_obs": int(obs.shape[0]),
        "n_vars": int(adata.n_vars),
        "time_col": time_col or "",
        "n_unique_time": int(time_values.nunique()) if time_col else 0,
        "time_counts": as_counts(time_values) if time_col else "",
        "dose_col": dose_col or "",
        "n_unique_dose": int(dose_values.nunique()) if dose_col else 0,
        "dose_counts": as_counts(dose_values) if dose_col else "",
        "n_conditions": int(condition_values.nunique()) if condition_col in obs.columns else 0,
        "n_control_cells": int(control_mask.sum()),
        "replicate_col": replicate_col or "",
        "n_unique_replicate": int(obs[replicate_col].astype(str).nunique()) if replicate_col else 0,
    }

    condition_rows: list[dict[str, Any]] = []
    if condition_col in obs.columns:
        group_cols = [condition_col]
        if time_col:
            group_cols.append(time_col)
        if dose_col:
            group_cols.append(dose_col)
        grouped = obs.groupby(group_cols, observed=True).size().reset_index(name="n_cells")
        for _, row in grouped.iterrows():
            condition_rows.append(
                {
                    "dataset": dataset,
                    "condition": str(row.get(condition_col, "")),
                    "time": str(row.get(time_col, "")) if time_col else "",
                    "dose": str(row.get(dose_col, "")) if dose_col else "",
                    "n_cells": int(row["n_cells"]),
                }
            )
    return dataset_row, condition_rows


def write_report(payload: dict[str, Any], dataset_rows: list[dict[str, Any]], latent_meta: dict[str, Any]) -> None:
    lines = [
        "# LatentFM SciPlex Explicit Time-Vector Preflight",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only metadata feasibility gate.",
        "* Reads local SciPlex raw h5ad obs metadata and LatentFM condition metadata.",
        "* No training, inference, GPU, checkpoint selection, canonical multi, or Track C query use.",
        "",
        "## Dataset Inventory",
        "",
        "| dataset | cells | time values | dose values | conditions | controls | replicates |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for row in dataset_rows:
        lines.append(
            f"| `{row['dataset']}` | `{row.get('n_obs', '')}` | `{row.get('time_counts', '')}` | `{row.get('dose_counts', '')}` | `{row.get('n_conditions', '')}` | `{row.get('n_control_cells', '')}` | `{row.get('n_unique_replicate', '')}` |"
        )
    lines.extend(
        [
            "",
            "## LatentFM Condition Metadata",
            "",
            "| dataset | conditions | time-like keys | dose-like keys |",
            "|---|---:|---|---|",
        ]
    )
    for row in latent_meta["datasets"]:
        lines.append(
            f"| `{row['dataset']}` | `{row['n_conditions']}` | `{','.join(row['time_like_keys']) or 'none'}` | `{','.join(row['dose_like_keys']) or 'none'}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"* Reasons: `{';'.join(payload['reasons']) if payload['reasons'] else 'none'}`.",
            "* The local SciPlex files contain dose gradients at one time point (`24.0`) rather than an explicit time course.",
            "* Current LatentFM SciPlex condition metadata preserves drug identity only, not time/dose as model-visible condition fields.",
            "",
            "## Interpretation",
            "",
            "* Do not run a SciPlex explicit time-vector GPU smoke from the current local artifacts.",
            "* SciPlex remains useful for dose/background/pathway controls, but it does not test the ZSCAPE temporal-response hypothesis without external time-course data or a different dataset.",
            "",
            "## Outputs",
            "",
            f"* Dataset inventory: `{OUT_DATASET_CSV}`",
            f"* Condition inventory: `{OUT_CONDITION_CSV}`",
            f"* JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    for path in SCIPLEX_FILES:
        dataset_row, cond = inspect_file(path)
        dataset_rows.append(dataset_row)
        condition_rows.extend(cond)
    latent_meta = latent_metadata_time_fields()

    reasons: list[str] = []
    if any(row.get("status") != "ok" for row in dataset_rows):
        reasons.append("missing_sciplex_raw_file")
    if not all(int(row.get("n_unique_time", 0)) >= 2 for row in dataset_rows):
        reasons.append("single_timepoint_in_all_local_sciplex_files")
    if not all(row.get("n_unique_dose", 0) >= 2 for row in dataset_rows):
        reasons.append("dose_gradient_missing")
    if any(row["time_like_keys"] for row in latent_meta["datasets"]):
        pass
    else:
        reasons.append("latent_condition_metadata_has_no_time_like_keys")

    status = "sciplex_explicit_time_vector_preflight_fail_no_gpu" if reasons else "sciplex_explicit_time_vector_preflight_pass_no_gpu"
    payload = {
        "created_at": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "reasons": reasons,
        "dataset_rows": dataset_rows,
        "latent_metadata": latent_meta,
        "outputs": {
            "report": str(OUT_MD),
            "json": str(OUT_JSON),
            "dataset_csv": str(OUT_DATASET_CSV),
            "condition_csv": str(OUT_CONDITION_CSV),
        },
        "boundary": "cpu_report_only_sciplex_explicit_time_vector_metadata_preflight_no_training_no_inference_no_gpu",
    }
    pd.DataFrame(dataset_rows).to_csv(OUT_DATASET_CSV, index=False)
    pd.DataFrame(condition_rows).to_csv(OUT_CONDITION_CSV, index=False)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload, dataset_rows, latent_meta)
    print(json.dumps({"status": status, "reasons": reasons, "report": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
