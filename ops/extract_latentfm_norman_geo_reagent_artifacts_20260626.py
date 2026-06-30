#!/usr/bin/env python3
"""Extract Norman GEO guide/reagent condition-level artifacts.

Short CPU task. Reads only the downloaded GEO cell-identity metadata CSV.gz and
local condition inventory. It does not read expression matrices, checkpoints,
canonical multi, Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
SOURCE = ROOT / "reports/external_artifact_sources_20260626/norman_geo/GSE133344_filtered_cell_identities.csv.gz"
CONDITION_INV = ROOT / "reports/latentfm_condition_level_inventory_20260624.json"
OUT_DIR = ROOT / "reports/norman_geo_reagent_artifacts_20260626"
OUT_JSON = ROOT / "reports/latentfm_norman_geo_reagent_artifacts_20260626.json"
OUT_MD = ROOT / "reports/LATENTFM_NORMAN_GEO_REAGENT_ARTIFACTS_20260626.md"

DATASET = "NormanWeissman2019_filtered"
ARTIFACTS = {
    "norman_geo_good_coverage_rate": "good_coverage",
    "norman_geo_mean_guide_coverage": "coverage",
    "norman_geo_mean_read_count": "read_count",
    "norman_geo_mean_umi_count": "UMI_count",
}


def load_conditions() -> set[str]:
    payload = json.loads(CONDITION_INV.read_text(encoding="utf-8"))
    return {
        str(row["condition"])
        for row in payload.get("rows", [])
        if row.get("dataset") == DATASET
    }


def clean_guide_identity(value: str) -> str:
    left = value.split("__", 1)[0]
    parts = left.split("_")
    if len(parts) < 2:
        return left
    parts = ["ctrl" if "NegCtrl" in part else part for part in parts[:2]]
    if parts[0] == "ctrl" and parts[1] == "ctrl":
        return "ctrl"
    if parts[0] == "ctrl":
        return parts[1]
    if parts[1] == "ctrl":
        return parts[0]
    return f"{parts[0]}+{parts[1]}"


def canonicalize_to_local(condition: str, local: set[str]) -> str | None:
    if condition in local:
        return condition
    if "+" in condition:
        a, b = condition.split("+", 1)
        rev = f"{b}+{a}"
        if rev in local:
            return rev
    return None


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    local = load_conditions()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    raw_conditions = set()
    with gzip.open(SOURCE, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            merged = clean_guide_identity(str(row.get("guide_identity", "")))
            raw_conditions.add(merged)
            local_cond = canonicalize_to_local(merged, local)
            if local_cond is None:
                continue
            grouped[local_cond].append(row)

    output_files: dict[str, str] = {}
    summaries = []
    for artifact, column in ARTIFACTS.items():
        out = OUT_DIR / f"{artifact}.csv"
        with out.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "dataset",
                "condition",
                "artifact_value",
                "source",
                "source_file",
                "source_column",
                "n_cells",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            values_for_summary = []
            for condition, rows in sorted(grouped.items()):
                if column == "good_coverage":
                    vals = [str(row.get(column, "")).lower() == "true" for row in rows]
                    value = sum(vals) / len(vals) if vals else None
                else:
                    nums = [to_float(row.get(column)) for row in rows]
                    nums = [x for x in nums if x is not None]
                    value = mean(nums) if nums else None
                if value is None:
                    continue
                values_for_summary.append(float(value))
                writer.writerow(
                    {
                        "dataset": DATASET,
                        "condition": condition,
                        "artifact_value": f"{float(value):.10g}",
                        "source": "GEO:GSE133344",
                        "source_file": str(SOURCE),
                        "source_column": column,
                        "n_cells": len(rows),
                    }
                )
        output_files[artifact] = str(out)
        summaries.append(
            {
                "artifact": artifact,
                "source_column": column,
                "rows": len(grouped),
                "value_min": min(values_for_summary) if values_for_summary else None,
                "value_max": max(values_for_summary) if values_for_summary else None,
                "value_mean": mean(values_for_summary) if values_for_summary else None,
                "output": str(out),
            }
        )

    matched = set(grouped)
    payload = {
        "status": "norman_geo_reagent_artifacts_ready_cpu_preflight_next",
        "gpu_authorized": False,
        "boundary": "GEO metadata only; no expression/checkpoint/canonical multi/Track C query/training/inference/GPU",
        "source": str(SOURCE),
        "dataset": DATASET,
        "local_condition_count": len(local),
        "raw_cleaned_condition_count": len(raw_conditions),
        "matched_condition_count": len(matched),
        "unmatched_local_conditions": sorted(local - matched),
        "artifact_outputs": output_files,
        "artifact_summaries": summaries,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Norman GEO Reagent Artifacts",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M CST')}`",
        "",
        "Status: `norman_geo_reagent_artifacts_ready_cpu_preflight_next`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- Reads only downloaded GEO cell identity metadata `GSE133344_filtered_cell_identities.csv.gz`.",
        "- Does not read expression matrices, checkpoints, canonical multi, Track C query, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- local conditions: `{len(local)}`",
        f"- matched conditions: `{len(matched)}`",
        f"- raw cleaned conditions in metadata: `{len(raw_conditions)}`",
        "",
        "| artifact | rows | min | mean | max | output |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['artifact']}` | {row['rows']} | "
            f"{row['value_min'] if row['value_min'] is not None else 'NA'} | "
            f"{row['value_mean'] if row['value_mean'] is not None else 'NA'} | "
            f"{row['value_max'] if row['value_max'] is not None else 'NA'} | `{row['output']}` |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "- This creates real condition-level source artifacts for Norman only.",
        "- It does not authorize GPU because a one-dataset artifact cannot establish a cross-dataset scaling rule.",
        "- Next step: run artifact preflight as a one-dataset source-preview gate, then acquire at least Frangieh/Dixit equivalents if the signal is directionally useful.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- artifact directory: `{OUT_DIR}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
