#!/usr/bin/env python3
"""Materialize Adamson author guide-support artifacts from GEO cell identities.

The source fields are read/UMI/coverage/cell-count style guide assignment
metadata. They are therefore QC/reagent-support diagnostics, not a new
response/effect-size mechanism. CPU/source-only: no training, inference,
canonical multi selection, Track C query, or GPU.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
SRC_DIR = ROOT / "reports/external_artifact_sources_20260627/adamson_gasperini_crispri_scout/adamson_gse90546"
SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
OUT_DIR = ROOT / "reports/adamson_author_guide_support_artifacts_20260627"
OUT_CSV = OUT_DIR / "adamson_author_guide_support_artifacts.csv"
OUT_JSON = ROOT / "reports/latentfm_adamson_author_guide_support_artifacts_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_ADAMSON_AUTHOR_GUIDE_SUPPORT_ARTIFACTS_20260627.md"
MANIFEST = ROOT / "configs/latentfm_adamson_author_guide_support_artifact_manifest_20260627.json"
DATASET = "Adamson"
SOURCE_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2406nnn"
ALIASES = {"PERK": "EIF2AK3", "IRE1": "ERN1"}


def fnum(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_targets(identity: str, local: set[str]) -> list[str]:
    prefix = identity.split("_p", 1)[0]
    prefix = prefix.replace("(mod)", "")
    parts = [ALIASES.get(p, p) for p in prefix.split("_") if p]
    return [p for p in parts if p in local]


def load_split_conditions() -> dict[str, str]:
    payload = json.loads(SPLIT.read_text(encoding="utf-8"))
    out = {}
    for split, items in payload[DATASET].items():
        if isinstance(items, list):
            for condition in items:
                out[str(condition)] = split
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    split_by_condition = load_split_conditions()
    local = set(split_by_condition)
    per_condition: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    source_files = []
    total_rows = 0
    matched_rows = 0
    for path in sorted(SRC_DIR.glob("*_cell_identities.csv.gz")):
        source_files.append(str(path))
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                total_rows += 1
                targets = parse_targets(row.get("guide identity", ""), local)
                if not targets:
                    continue
                read_count = fnum(row.get("read count"))
                umi_count = fnum(row.get("UMI count"))
                coverage = fnum(row.get("coverage"))
                cell_mult = fnum(row.get("number of cells"))
                good = str(row.get("good coverage", "")).lower() == "true"
                for target in targets:
                    matched_rows += 1
                    d = per_condition[target]
                    d["read_count"].append(read_count or 0.0)
                    d["umi_count"].append(umi_count or 0.0)
                    d["coverage"].append(coverage or 0.0)
                    d["number_of_cells_field"].append(cell_mult or 0.0)
                    d["good_coverage"].append(1.0 if good else 0.0)

    rows = []
    for condition, vals in sorted(per_condition.items()):
        artifacts = {
            "adamson_guide_cell_assignments": float(len(vals["read_count"])),
            "adamson_guide_read_count_mean": mean(vals["read_count"]),
            "adamson_guide_read_count_median": median(vals["read_count"]),
            "adamson_guide_umi_count_mean": mean(vals["umi_count"]),
            "adamson_guide_umi_count_median": median(vals["umi_count"]),
            "adamson_guide_coverage_mean": mean(vals["coverage"]),
            "adamson_guide_coverage_median": median(vals["coverage"]),
            "adamson_guide_good_coverage_fraction": mean(vals["good_coverage"]),
            "adamson_guide_number_of_cells_field_mean": mean(vals["number_of_cells_field"]),
        }
        for artifact, value in artifacts.items():
            rows.append(
                {
                    "dataset": DATASET,
                    "condition": condition,
                    "split": split_by_condition[condition],
                    "artifact": artifact,
                    "artifact_value": value,
                    "artifact_role": "qc_reagent_support",
                    "raw_column": artifact.replace("adamson_guide_", ""),
                    "source_files": ";".join(source_files),
                    "source_url": SOURCE_BASE,
                }
            )

    fields = ["dataset", "condition", "split", "artifact", "artifact_value", "artifact_role", "raw_column", "source_files", "source_url"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    artifacts = sorted({r["artifact"] for r in rows})
    payload = {
        "status": "adamson_author_guide_support_artifacts_ready_diagnostic_no_gpu",
        "gpu_authorized": False,
        "dataset": DATASET,
        "total_identity_rows": total_rows,
        "matched_target_rows": matched_rows,
        "local_conditions": len(local),
        "matched_conditions": len(per_condition),
        "missing_conditions": sorted(local - set(per_condition)),
        "rows": len(rows),
        "artifacts": artifacts,
        "source_files": source_files,
        "outputs": {"csv": str(OUT_CSV), "json": str(OUT_JSON), "markdown": str(OUT_MD), "manifest": str(MANIFEST)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MANIFEST.write_text(json.dumps({"status": payload["status"], "artifacts": artifacts, "csv": str(OUT_CSV), "source_files": source_files}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Adamson Author Guide-Support Artifacts 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/source-only materialization from GEO sample-level cell identity files.",
        "- Fields are guide/read/UMI/coverage/cell-count support, so this is diagnostic/QC-reagent support only.",
        "- No training, inference, canonical multi selection, Track C query, or GPU.",
        "",
        "## Summary",
        "",
        f"- total identity rows: `{total_rows}`",
        f"- matched target rows: `{matched_rows}`",
        f"- local conditions: `{len(local)}`",
        f"- matched conditions: `{len(per_condition)}`",
        f"- artifact rows: `{len(rows)}`",
        f"- artifacts: `{artifacts}`",
        f"- missing conditions: `{payload['missing_conditions']}`",
        "",
        "## Outputs",
        "",
        f"- csv: `{OUT_CSV}`",
        f"- manifest: `{MANIFEST}`",
        f"- json: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(rows), "report": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
