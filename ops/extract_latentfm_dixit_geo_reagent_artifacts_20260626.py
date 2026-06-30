#!/usr/bin/env python3
"""Extract Dixit GEO guide-barcode condition-level artifacts.

Short CPU task after the source tar is downloaded. It extracts only small
guide-barcode metadata members from `GSE90063_RAW.tar`; it does not read
expression matrices, checkpoints, canonical multi, Track C query, train, infer,
or use GPU.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import shutil
import sys
import tarfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable


ROOT = Path("/data/cyx/1030/scLatent")
SOURCE_TAR = ROOT / "reports/external_artifact_sources_20260626/dixit_geo/GSE90063_RAW.tar"
EXTRACT_DIR = ROOT / "reports/external_artifact_sources_20260626/dixit_geo/extracted"
CONDITION_INV = ROOT / "reports/latentfm_condition_level_inventory_20260624.json"
OUT_DIR = ROOT / "reports/dixit_geo_reagent_artifacts_20260626"
OUT_JSON = ROOT / "reports/latentfm_dixit_geo_reagent_artifacts_20260626.json"
OUT_MD = ROOT / "reports/LATENTFM_DIXIT_GEO_REAGENT_ARTIFACTS_20260626.md"

DATASET = "DixitRegev2016_K562_TFs_High_MOI"

TARGET_BASENAMES = [
    "GSM2396860_k562_tfs_highmoi_cbc_gbc_dict_strict.csv.gz",
    "GSM2396860_k562_tfs_highmoi_cbc_gbc_dict_lenient.csv.gz",
    "GSM2396860_k562_tfs_highmoi_cellnames.csv.gz",
]

csv.field_size_limit(sys.maxsize)


def load_conditions() -> set[str]:
    payload = json.loads(CONDITION_INV.read_text(encoding="utf-8"))
    return {
        str(row["condition"])
        for row in payload.get("rows", [])
        if row.get("dataset") == DATASET and str(row.get("condition")) != "control"
    }


def write_missing(status: str, reason: str) -> int:
    payload = {
        "status": status,
        "gpu_authorized": False,
        "source": str(SOURCE_TAR),
        "reason": reason,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(
        "# LatentFM Dixit GEO Reagent Artifacts\n\n"
        f"Status: `{status}`\n\n"
        "GPU authorized: `False`\n\n"
        f"Reason: {reason}\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 2


def safe_member_basename(member: tarfile.TarInfo) -> str:
    return Path(member.name).name


def extract_selected_members() -> dict[str, str]:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    selected = set(TARGET_BASENAMES)
    extracted: dict[str, str] = {}
    with tarfile.open(SOURCE_TAR, "r") as tar:
        members_by_base = {safe_member_basename(member): member for member in tar.getmembers()}
        for basename in TARGET_BASENAMES:
            member = members_by_base.get(basename)
            if member is None:
                continue
            out = EXTRACT_DIR / basename
            if not out.is_file() or out.stat().st_size != member.size:
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                with handle, out.open("wb") as dest:
                    shutil.copyfileobj(handle, dest)
            if basename in selected:
                extracted[basename] = str(out)
    return extracted


def iter_csv_rows(path: Path) -> Iterable[list[str]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(handle, dialect)
        for row in reader:
            if row:
                yield [str(cell) for cell in row]


def token_matches(token: str, local: set[str]) -> set[str]:
    upper = token.strip().upper()
    if not upper or "INTERGENIC" in upper or "CONTROL" in upper or upper in {"CTRL", "NEGCTRL"}:
        return set()
    matched = set()
    for condition in local:
        gene = condition.upper()
        if re.search(rf"(^|[^A-Z0-9])(?:SG)?{re.escape(gene)}([^A-Z0-9]|$)", upper):
            matched.add(condition)
    return matched


def summarize_dict(path: Path, local: set[str]) -> dict[str, object]:
    row_count = 0
    matched_row_count = 0
    condition_counts: dict[str, int] = defaultdict(int)
    condition_guides: dict[str, set[str]] = defaultdict(set)
    example_rows = []
    for row in iter_csv_rows(path):
        row_count += 1
        row_matches = set()
        for token in row:
            hits = token_matches(token, local)
            if not hits:
                continue
            for hit in hits:
                condition_guides[hit].add(token)
            row_matches |= hits
        if row_matches:
            matched_row_count += 1
            if len(example_rows) < 5:
                example_rows.append(row[:8])
        for condition in row_matches:
            condition_counts[condition] += 1
    return {
        "source_file": str(path),
        "row_count": row_count,
        "matched_row_count": matched_row_count,
        "condition_counts": dict(condition_counts),
        "condition_unique_guides": {key: len(value) for key, value in condition_guides.items()},
        "example_matched_rows": example_rows,
    }


def write_artifact(name: str, source_file: Path, values: dict[str, float], source_column: str) -> Path:
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
                    "source": "GEO:GSE90063",
                    "source_file": str(source_file),
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
    if not SOURCE_TAR.is_file():
        return write_missing("dixit_geo_source_missing_no_gpu", "source tar not downloaded yet")
    if SOURCE_TAR.stat().st_size < 1_000_000_000:
        return write_missing(
            "dixit_geo_source_incomplete_no_gpu",
            f"source tar exists but is smaller than expected: {SOURCE_TAR.stat().st_size} bytes",
        )

    local = load_conditions()
    try:
        extracted = extract_selected_members()
    except tarfile.TarError as exc:
        return write_missing("dixit_geo_tar_read_fail_no_gpu", f"could not read source tar: {exc}")

    dict_files = {
        "strict": EXTRACT_DIR / "GSM2396860_k562_tfs_highmoi_cbc_gbc_dict_strict.csv.gz",
        "lenient": EXTRACT_DIR / "GSM2396860_k562_tfs_highmoi_cbc_gbc_dict_lenient.csv.gz",
    }
    summaries_by_mode = {}
    artifacts: dict[str, Path] = {}
    for mode, path in dict_files.items():
        if not path.is_file():
            continue
        summary = summarize_dict(path, local)
        summaries_by_mode[mode] = summary
        counts = {key: float(value) for key, value in summary["condition_counts"].items()}
        guides = {key: float(value) for key, value in summary["condition_unique_guides"].items()}
        denominator = float(summary["row_count"] or 1)
        fractions = {key: value / denominator for key, value in counts.items()}
        artifacts[f"dixit_geo_highmoi_{mode}_assigned_row_count"] = write_artifact(
            f"dixit_geo_highmoi_{mode}_assigned_row_count", path, counts, "matched_rows"
        )
        artifacts[f"dixit_geo_highmoi_{mode}_assigned_row_fraction"] = write_artifact(
            f"dixit_geo_highmoi_{mode}_assigned_row_fraction", path, fractions, "matched_rows_fraction"
        )
        artifacts[f"dixit_geo_highmoi_{mode}_unique_guide_count"] = write_artifact(
            f"dixit_geo_highmoi_{mode}_unique_guide_count", path, guides, "unique_guide_tokens"
        )

    status = (
        "dixit_geo_reagent_artifacts_ready_cpu_preflight_next"
        if artifacts and any(row.get("matched_row_count", 0) for row in summaries_by_mode.values())
        else "dixit_geo_reagent_artifacts_fail_no_overlap_no_gpu"
    )
    artifact_summaries = [artifact_summary(name, path) for name, path in artifacts.items()]
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": "Dixit GEO guide-barcode metadata only; no expression/checkpoint/canonical multi/Track C query/training/inference/GPU",
        "source_tar": str(SOURCE_TAR),
        "extracted_files": extracted,
        "dataset": DATASET,
        "local_condition_count": len(local),
        "local_conditions": sorted(local),
        "dict_summaries": summaries_by_mode,
        "artifact_outputs": {key: str(path) for key, path in artifacts.items()},
        "artifact_summaries": artifact_summaries,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Dixit GEO Reagent Artifacts",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M CST')}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- Extracts only small guide-barcode dictionary metadata from `GSE90063_RAW.tar`.",
        "- Does not read expression matrices, checkpoints, canonical multi, Track C query, train, infer, or use GPU.",
        "",
        "## Dict Summary",
        "",
        "| mode | rows | matched rows | matched conditions | source file |",
        "|---|---:|---:|---:|---|",
    ]
    for mode, row in summaries_by_mode.items():
        lines.append(
            f"| `{mode}` | {row['row_count']} | {row['matched_row_count']} | "
            f"{len(row['condition_counts'])} | `{row['source_file']}` |"
        )
    lines += [
        "",
        "## Artifacts",
        "",
        "| artifact | rows | min | mean | max | output |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in artifact_summaries:
        lines.append(
            f"| `{row['artifact']}` | {row['rows']} | {row['value_min']} | "
            f"{row['value_mean']} | {row['value_max']} | `{row['output']}` |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "- This creates source-derived Dixit condition-level guide-support artifacts only if the tar and guide dictionaries are present.",
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
