#!/usr/bin/env python3
"""Materialize SciPlex3 dose artifact candidates for LatentFM.

CPU/source-only. Reads local SciPlex obs metadata and train-only split files.
It does not train, infer, read checkpoints, read canonical multi, read Track C
query, or use GPU.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
EMB_ROOT = ROOT / "scFM_output/embeddings/xverse"
BASE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
CAP120_SPLIT = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
OUT_DIR = ROOT / "reports/sciplex_dose_time_artifacts_20260627"
OUT_CSV = OUT_DIR / "sciplex_log_dose_condition_level.csv"
OUT_SUMMARY = ROOT / "reports/LATENTFM_SCIPLEX_DOSE_TIME_ARTIFACTS_20260627.md"
OUT_JSON = ROOT / "reports/latentfm_sciplex_dose_time_artifacts_20260627.json"
MANIFEST = ROOT / "configs/latentfm_sciplex_dose_time_artifact_manifest_20260627.json"

DATASETS = ("sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7")
DOSE_VALUES = (0.001, 0.01, 0.1, 1.0)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def norm_dose(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(out):
        return out
    return None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_split = load_json(BASE_SPLIT)
    cap120_split = load_json(CAP120_SPLIT)
    rows: list[dict[str, Any]] = []
    dataset_summary: list[dict[str, Any]] = []

    for ds in DATASETS:
        obs_path = EMB_ROOT / ds / "raw/obs.parquet"
        if not obs_path.exists():
            raise FileNotFoundError(obs_path)
        obs = pd.read_parquet(obs_path)
        train_drugs = set(str(x) for x in (cap120_split.get(ds) or {}).get("train") or [])
        canonical_ref = set(str(x) for x in (base_split.get(ds) or {}).get("canonical_test_reference") or [])
        train_drugs -= canonical_ref
        kept = 0
        for (drug, dose), group in obs.groupby(["perturbation", "dose"], sort=True):
            dose_value = norm_dose(dose)
            if dose_value is None:
                continue
            drug = str(drug)
            if drug not in train_drugs:
                continue
            if not any(abs(dose_value - allowed) < 1e-12 for allowed in DOSE_VALUES):
                continue
            if str(group["control"].iloc[0]) in {"1", "True", "true"} or drug == "control":
                continue
            background = str(group["cov"].iloc[0])
            time_values = sorted({str(x) for x in group.get("time", pd.Series(dtype=object)).dropna().unique()})
            dose_unit_values = sorted({str(x) for x in group.get("dose_unit", pd.Series(dtype=object)).dropna().unique()})
            rows.append(
                {
                    "dataset": ds,
                    "condition": f"{background}_{drug}_{dose_value:g}",
                    "artifact_value": math.log10(dose_value),
                    "artifact_name": "sciplex_log_dose",
                    "cell_background": background,
                    "drug": drug,
                    "dose": dose_value,
                    "dose_unit": ",".join(dose_unit_values) or "normalized_uM",
                    "log_dose": math.log10(dose_value),
                    "timepoint": ",".join(time_values),
                    "n_cells": int(len(group)),
                    "source": "local_xverse_sciplex3_obs_parquet",
                    "evidence_url": "local:scFM_output/embeddings/xverse/<dataset>/raw/obs.parquet",
                }
            )
            kept += 1
        dataset_summary.append(
            {
                "dataset": ds,
                "train_drugs_after_reference_exclusion": len(train_drugs),
                "dose_condition_rows": kept,
                "timepoints": ",".join(sorted({str(x) for x in obs["time"].dropna().unique()})),
            }
        )

    fields = [
        "dataset",
        "condition",
        "artifact_value",
        "artifact_name",
        "cell_background",
        "drug",
        "dose",
        "dose_unit",
        "log_dose",
        "timepoint",
        "n_cells",
        "source",
        "evidence_url",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "boundary": {
            "cpu_source_only": True,
            "reads_local_obs_metadata_only": True,
            "train_only_cap120_drugs": True,
            "canonical_reference_excluded": True,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
            "note": "Condition key is background_drug_dose. Generic external preflight may have zero overlap with drug-level outcome rows; dose-specific semantic gate is required before GPU.",
        },
        "artifacts": [
            {
                "artifact": "sciplex_log_dose_condition_level",
                "priority": "P0",
                "source_files": [str(OUT_CSV.relative_to(ROOT))],
                "required_columns": ["dataset", "condition", "artifact_value"],
                "minimum_datasets": 3,
                "minimum_overlap_rows": 50,
                "minimum_varying_datasets": 3,
                "promotion_note": "Needs dose-specific background+drug intercept, within-drug dose shuffle, and LODO controls before any GPU.",
            }
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = {
        "status": "sciplex_dose_time_artifacts_materialized_no_gpu",
        "gpu_authorized": False,
        "outputs": {
            "artifact_csv": str(OUT_CSV),
            "manifest": str(MANIFEST),
            "markdown": str(OUT_SUMMARY),
            "json": str(OUT_JSON),
        },
        "dataset_summary": dataset_summary,
        "n_rows": len(rows),
        "dose_values": list(DOSE_VALUES),
        "timepoint_status": "local_obs_time_is_constant_24h_for_all_three_datasets; no independent time artifact materialized",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM SciPlex Dose/Time Artifacts",
        "",
        "Status: `sciplex_dose_time_artifacts_materialized_no_gpu`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/source-only materialization from local SciPlex3 obs metadata.",
        "- Uses cap120 train drugs and excludes canonical reference drugs.",
        "- Does not train, infer, read checkpoints, canonical multi, Track C query, or use GPU.",
        "- `time` is constant 24h locally, so only `sciplex_log_dose` is materialized.",
        "",
        "## Summary",
        "",
        f"- artifact rows: `{len(rows)}`",
        f"- artifact CSV: `{OUT_CSV}`",
        f"- manifest: `{MANIFEST}`",
        "",
        "| dataset | train drugs | dose-condition rows | timepoints |",
        "|---|---:|---:|---|",
    ]
    for row in dataset_summary:
        lines.append(
            f"| `{row['dataset']}` | {row['train_drugs_after_reference_exclusion']} | "
            f"{row['dose_condition_rows']} | `{row['timepoints']}` |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "- This materializes a condition-level dose artifact, not a GPU route.",
        "- Generic external preflight may fail if available outcome proxies are drug-level; dose-specific controls remain required.",
    ]
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(rows), "manifest": str(MANIFEST)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
