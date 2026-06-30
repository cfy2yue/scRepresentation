#!/usr/bin/env python3
"""Extract Frangieh figshare/scPerturb reagent artifacts from processed h5ad.

Short CPU task once the source h5ad is downloaded. Reads only `.obs` metadata in
backed mode. It does not read expression matrices, checkpoints, canonical multi,
Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import anndata as ad


ROOT = Path("/data/cyx/1030/scLatent")
SOURCE = ROOT / "reports/external_artifact_sources_20260626/frangieh_figshare/Frangieh_2021.h5ad"
CONDITION_INV = ROOT / "reports/latentfm_condition_level_inventory_20260624.json"
OUT_DIR = ROOT / "reports/frangieh_figshare_reagent_artifacts_20260626"
OUT_JSON = ROOT / "reports/latentfm_frangieh_figshare_reagent_artifacts_20260626.json"
OUT_MD = ROOT / "reports/LATENTFM_FRANGIEH_FIGSHARE_REAGENT_ARTIFACTS_20260626.md"

DATASET = "Frangieh"


def load_conditions() -> set[str]:
    payload = json.loads(CONDITION_INV.read_text(encoding="utf-8"))
    return {
        str(row["condition"])
        for row in payload.get("rows", [])
        if row.get("dataset") == DATASET
    }


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_sgrna(value: Any) -> str:
    text = "" if value is None else str(value)
    # scPerturb notebook used re.sub('[_123]+', '', s); preserve that exact
    # condition naming behavior for compatibility with local condition labels.
    return re.sub(r"[_123]+", "", text)


def choose_condition(obs, local: set[str]) -> tuple[str, dict[str, int]]:
    overlaps: dict[str, int] = {}
    if "perturbation_name" in obs.columns:
        vals = set(obs["perturbation_name"].dropna().astype(str).unique())
        overlaps["perturbation_name"] = len(vals & local)
    if "condition" in obs.columns:
        vals = set(obs["condition"].dropna().astype(str).unique())
        overlaps["condition"] = len(vals & local)
    if "sgRNA" in obs.columns:
        vals = {clean_sgrna(x) for x in obs["sgRNA"].dropna().astype(str).unique()}
        overlaps["sgRNA_cleaned"] = len(vals & local)
    if not overlaps:
        return "", {}
    return max(overlaps, key=overlaps.get), overlaps


def condition_for_row(row: Any, mode: str) -> str:
    if mode == "sgRNA_cleaned":
        return clean_sgrna(row.get("sgRNA"))
    return str(row.get(mode, ""))


def write_artifact(name: str, rows: list[dict[str, Any]], column: str | None, value_fn) -> Path:
    out = OUT_DIR / f"{name}.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        fields = ["dataset", "condition", "artifact_value", "source", "source_file", "source_column", "n_cells"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            values = item["rows"]
            value = value_fn(values)
            if value is None:
                continue
            writer.writerow(
                {
                    "dataset": DATASET,
                    "condition": item["condition"],
                    "artifact_value": f"{float(value):.10g}",
                    "source": "figshare:34013717/scPerturb",
                    "source_file": str(SOURCE),
                    "source_column": column or "n_cells",
                    "n_cells": len(values),
                }
            )
    return out


def numeric_mean(column: str):
    def _fn(rows: list[dict[str, Any]]) -> float | None:
        vals = [to_float(row.get(column)) for row in rows]
        vals = [x for x in vals if x is not None]
        return mean(vals) if vals else None

    return _fn


def assigned_rate(column: str):
    def _fn(rows: list[dict[str, Any]]) -> float | None:
        vals = [str(row.get(column, "")).strip() for row in rows]
        if not vals:
            return None
        good = [v for v in vals if v and v.lower() not in {"nan", "none", "null"}]
        return len(good) / len(vals)

    return _fn


def cell_count(rows: list[dict[str, Any]]) -> float:
    return float(len(rows))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    local = load_conditions()
    if not SOURCE.is_file():
        payload = {
            "status": "frangieh_figshare_source_missing_no_gpu",
            "gpu_authorized": False,
            "source": str(SOURCE),
            "reason": "source h5ad not downloaded yet",
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        OUT_MD.write_text(
            "# LatentFM Frangieh figshare Reagent Artifacts\n\n"
            "Status: `frangieh_figshare_source_missing_no_gpu`\n\n"
            f"Missing source: `{SOURCE}`\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
        return 2

    adata = ad.read_h5ad(SOURCE, backed="r")
    obs = adata.obs.copy()
    try:
        adata.file.close()
    except Exception:
        pass

    mode, overlaps = choose_condition(obs, local)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, row in obs.iterrows():
        condition = condition_for_row(row, mode)
        if condition not in local:
            continue
        grouped[condition].append(row.to_dict())

    artifacts: dict[str, Path] = {}
    if "MOI" in obs.columns:
        artifacts["frangieh_figshare_mean_moi"] = write_artifact(
            "frangieh_figshare_mean_moi", [{"condition": k, "rows": v} for k, v in sorted(grouped.items())], "MOI", numeric_mean("MOI")
        )
    if "UMI_count" in obs.columns:
        artifacts["frangieh_figshare_mean_umi_count"] = write_artifact(
            "frangieh_figshare_mean_umi_count",
            [{"condition": k, "rows": v} for k, v in sorted(grouped.items())],
            "UMI_count",
            numeric_mean("UMI_count"),
        )
    if "sgRNA" in obs.columns:
        artifacts["frangieh_figshare_assigned_sgrna_rate"] = write_artifact(
            "frangieh_figshare_assigned_sgrna_rate",
            [{"condition": k, "rows": v} for k, v in sorted(grouped.items())],
            "sgRNA",
            assigned_rate("sgRNA"),
        )
    artifacts["frangieh_figshare_condition_cell_count"] = write_artifact(
        "frangieh_figshare_condition_cell_count",
        [{"condition": k, "rows": v} for k, v in sorted(grouped.items())],
        None,
        cell_count,
    )

    summaries = []
    for artifact, path in artifacts.items():
        vals = []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                vals.append(float(row["artifact_value"]))
        summaries.append(
            {
                "artifact": artifact,
                "rows": len(vals),
                "value_min": min(vals) if vals else None,
                "value_mean": mean(vals) if vals else None,
                "value_max": max(vals) if vals else None,
                "output": str(path),
            }
        )

    status = (
        "frangieh_figshare_reagent_artifacts_ready_cpu_preflight_next"
        if artifacts and grouped
        else "frangieh_figshare_reagent_artifacts_fail_no_overlap_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": "Frangieh h5ad obs only; no expression/checkpoint/canonical multi/Track C query/training/inference/GPU",
        "source": str(SOURCE),
        "dataset": DATASET,
        "obs_shape": [int(obs.shape[0]), int(obs.shape[1])],
        "obs_columns": list(obs.columns),
        "condition_mode": mode,
        "condition_overlap_candidates": overlaps,
        "local_condition_count": len(local),
        "matched_condition_count": len(grouped),
        "unmatched_local_conditions": sorted(local - set(grouped)),
        "artifact_outputs": {key: str(path) for key, path in artifacts.items()},
        "artifact_summaries": summaries,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Frangieh figshare Reagent Artifacts",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M CST')}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- Reads only `.obs` metadata from downloaded Frangieh processed h5ad in backed mode.",
        "- Does not read expression matrices, checkpoints, canonical multi, Track C query, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- obs shape: `{payload['obs_shape']}`",
        f"- condition mode: `{mode}`",
        f"- local conditions: `{len(local)}`",
        f"- matched conditions: `{len(grouped)}`",
        f"- overlap candidates: `{overlaps}`",
        "",
        "| artifact | rows | min | mean | max | output |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['artifact']}` | {row['rows']} | {row['value_min']} | {row['value_mean']} | {row['value_max']} | `{row['output']}` |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "- This extraction alone does not authorize GPU.",
        "- Next step: combine with Norman source artifacts and run multi-dataset external-artifact preflight/value-signal/tail-MMD controls.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- artifact directory: `{OUT_DIR}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
