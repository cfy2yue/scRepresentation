#!/usr/bin/env python3
"""Extract Dixit figshare/scPerturb reagent artifacts from processed h5ad.

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
SOURCE = ROOT / "reports/external_artifact_sources_20260626/dixit_figshare/Dixit_2016.h5ad"
CONDITION_INV = ROOT / "reports/latentfm_condition_level_inventory_20260624.json"
OUT_DIR = ROOT / "reports/dixit_figshare_reagent_artifacts_20260626"
OUT_JSON = ROOT / "reports/latentfm_dixit_figshare_reagent_artifacts_20260626.json"
OUT_MD = ROOT / "reports/LATENTFM_DIXIT_FIGSHARE_REAGENT_ARTIFACTS_20260626.md"

DATASET = "DixitRegev2016_K562_TFs_High_MOI"


def load_conditions() -> set[str]:
    payload = json.loads(CONDITION_INV.read_text(encoding="utf-8"))
    return {
        str(row["condition"])
        for row in payload.get("rows", [])
        if row.get("dataset") == DATASET and str(row.get("condition")) != "control"
    }


def write_missing(status: str, reason: str) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "gpu_authorized": False,
        "source": str(SOURCE),
        "reason": reason,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(
        "# LatentFM Dixit figshare Reagent Artifacts\n\n"
        f"Status: `{status}`\n\n"
        "GPU authorized: `False`\n\n"
        f"Reason: {reason}\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 2


def token_matches(token: Any, local: set[str]) -> set[str]:
    upper = str(token or "").strip().upper()
    if not upper or "INTERGENIC" in upper or "CONTROL" in upper or upper in {"CTRL", "NEGCTRL", "NAN", "NONE"}:
        return set()
    matched = set()
    for condition in local:
        gene = condition.upper()
        if re.search(rf"(^|[^A-Z0-9])(?:SG)?{re.escape(gene)}([^A-Z0-9]|$)", upper):
            matched.add(condition)
    return matched


def choose_guide_columns(obs, local: set[str]) -> list[str]:
    candidates = []
    for col in obs.columns:
        lower = str(col).lower()
        if lower in {"guide", "sgrna", "guide_identity", "perturbation", "condition", "gene"} or lower.startswith("sg"):
            sample_vals = obs[col].dropna().astype(str).head(2000).tolist()
            hits = sum(1 for value in sample_vals if token_matches(value, local))
            col_hit = len(token_matches(col, local))
            if hits or col_hit:
                candidates.append((hits + col_hit * 1000, col))
    return [col for _, col in sorted(candidates, reverse=True)]


def write_artifact(name: str, values: dict[str, float], source_column: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        fields = ["dataset", "condition", "artifact_value", "source", "source_file", "source_column", "n_cells"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, value in sorted(values.items()):
            writer.writerow(
                {
                    "dataset": DATASET,
                    "condition": condition,
                    "artifact_value": f"{float(value):.10g}",
                    "source": "figshare:34014608/scPerturb",
                    "source_file": str(SOURCE),
                    "source_column": source_column,
                    "n_cells": "",
                }
            )
    return out


def artifact_summary(name: str, path: Path) -> dict[str, object]:
    vals = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            vals.append(float(row["artifact_value"]))
    return {
        "artifact": name,
        "rows": len(vals),
        "value_min": min(vals) if vals else None,
        "value_mean": mean(vals) if vals else None,
        "value_max": max(vals) if vals else None,
        "output": str(path),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE.is_file():
        return write_missing("dixit_figshare_source_missing_no_gpu", "source h5ad not downloaded yet")
    if SOURCE.stat().st_size < 100_000_000:
        return write_missing(
            "dixit_figshare_source_incomplete_no_gpu",
            f"source h5ad exists but is smaller than expected: {SOURCE.stat().st_size} bytes",
        )

    local = load_conditions()
    adata = ad.read_h5ad(SOURCE, backed="r")
    obs = adata.obs.copy()
    try:
        adata.file.close()
    except Exception:
        pass

    guide_columns = choose_guide_columns(obs, local)
    if not guide_columns:
        return write_missing("dixit_figshare_no_guide_columns_no_gpu", "no obs columns map to local Dixit TF conditions")

    primary = guide_columns[0]
    condition_counts: dict[str, int] = defaultdict(int)
    condition_unique_guides: dict[str, set[str]] = defaultdict(set)
    assigned_rows = 0
    for _, row in obs.iterrows():
        row_hits = set()
        for col in guide_columns:
            value = row.get(col)
            hits = token_matches(value, local)
            if hits:
                for hit in hits:
                    condition_unique_guides[hit].add(str(value))
                row_hits |= hits
            # Binary sgRNA indicator columns are often named by the guide.
            col_hits = token_matches(col, local)
            try:
                active = float(row.get(col, 0)) > 0
            except (TypeError, ValueError):
                active = False
            if active and col_hits:
                for hit in col_hits:
                    condition_unique_guides[hit].add(col)
                row_hits |= col_hits
        if row_hits:
            assigned_rows += 1
        for condition in row_hits:
            condition_counts[condition] += 1

    denominator = float(max(1, obs.shape[0]))
    artifacts: dict[str, Path] = {}
    artifacts["dixit_figshare_assigned_cell_count"] = write_artifact(
        "dixit_figshare_assigned_cell_count", {k: float(v) for k, v in condition_counts.items()}, primary
    )
    artifacts["dixit_figshare_assigned_cell_fraction"] = write_artifact(
        "dixit_figshare_assigned_cell_fraction", {k: float(v) / denominator for k, v in condition_counts.items()}, primary
    )
    artifacts["dixit_figshare_unique_guide_count"] = write_artifact(
        "dixit_figshare_unique_guide_count",
        {k: float(len(v)) for k, v in condition_unique_guides.items()},
        ",".join(guide_columns[:20]),
    )

    summaries = [artifact_summary(name, path) for name, path in artifacts.items()]
    status = (
        "dixit_figshare_reagent_artifacts_ready_cpu_preflight_next"
        if any(row["rows"] for row in summaries)
        else "dixit_figshare_reagent_artifacts_fail_no_overlap_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": "Dixit h5ad obs only; no expression/checkpoint/canonical multi/Track C query/training/inference/GPU",
        "source": str(SOURCE),
        "dataset": DATASET,
        "obs_shape": [int(obs.shape[0]), int(obs.shape[1])],
        "obs_columns": list(obs.columns),
        "guide_columns": guide_columns,
        "primary_guide_column": primary,
        "local_condition_count": len(local),
        "matched_condition_count": len(condition_counts),
        "assigned_rows": assigned_rows,
        "artifact_outputs": {key: str(path) for key, path in artifacts.items()},
        "artifact_summaries": summaries,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Dixit figshare Reagent Artifacts",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M CST')}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- Reads only `.obs` metadata from downloaded Dixit processed h5ad in backed mode.",
        "- Does not read expression matrices, checkpoints, canonical multi, Track C query, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- obs shape: `{payload['obs_shape']}`",
        f"- guide columns: `{guide_columns[:20]}`",
        f"- matched conditions: `{len(condition_counts)}` / `{len(local)}`",
        f"- assigned rows: `{assigned_rows}`",
        "",
        "| artifact | rows | min | mean | max | output |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['artifact']}` | {row['rows']} | {row['value_min']} | "
            f"{row['value_mean']} | {row['value_max']} | `{row['output']}` |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "- This creates source-derived Dixit condition-level guide-support artifacts if the processed h5ad preserves usable guide metadata.",
        "- It does not authorize GPU by itself. Combine with Norman/Frangieh and run preflight/value-signal/tail-MMD/source-count controls first.",
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
