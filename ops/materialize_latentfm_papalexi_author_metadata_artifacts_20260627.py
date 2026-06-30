#!/usr/bin/env python3
"""Materialize Papalexi GSE153056 author metadata artifacts.

The small GEO tables expose cell-level guide/hash/count/phase metadata, not an
independent perturbation response/effect-size table. This is therefore a
single-source diagnostic/QC-reagent-support materialization only.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
SRC_DIR = ROOT / "reports/external_artifact_sources_20260627/papalexi_gse153056_scout"
SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
OUT_DIR = ROOT / "reports/papalexi_author_metadata_artifacts_20260627"
OUT_CSV = OUT_DIR / "papalexi_author_metadata_artifacts.csv"
OUT_JSON = ROOT / "reports/latentfm_papalexi_author_metadata_artifacts_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_PAPALEXI_AUTHOR_METADATA_ARTIFACTS_20260627.md"
MANIFEST = ROOT / "configs/latentfm_papalexi_author_metadata_artifact_manifest_20260627.json"
DATASET = "Papalexi"
SOURCE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE153nnn/GSE153056/suppl/"


def fnum(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_split_conditions() -> dict[str, str]:
    payload = json.loads(SPLIT.read_text(encoding="utf-8"))
    out = {}
    for split, items in payload[DATASET].items():
        if isinstance(items, list):
            for condition in items:
                out[str(condition)] = split
    return out


def guide_to_gene(value: str) -> str | None:
    value = str(value or "")
    if not value or value == "Negative":
        return None
    first = value.split("_", 1)[0]
    return re.sub(r"g\d+$", "", first)


def add_num(bucket: dict[str, list[float]], key: str, value: Any) -> None:
    num = fnum(value)
    if num is not None:
        bucket[key].append(num)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    split_by_condition = load_split_conditions()
    local = set(split_by_condition)
    per_condition: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    guide_sets: dict[str, set[str]] = defaultdict(set)
    rep_counts: dict[str, Counter[str]] = defaultdict(Counter)
    phase_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_files = []
    total_rows = 0
    matched_rows = 0

    eccite = SRC_DIR / "GSE153056_ECCITE_metadata.tsv.gz"
    source_files.append(str(eccite))
    with gzip.open(eccite, "rt", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            total_rows += 1
            gene = row.get("gene")
            if gene not in local:
                continue
            if row.get("crispr") not in {"Perturbed", "NT"}:
                continue
            matched_rows += 1
            bucket = per_condition[gene]
            guide_sets[gene].add(row.get("guide_ID", ""))
            rep_counts[gene][row.get("replicate", "")] += 1
            phase_counts[gene][row.get("Phase", "")] += 1
            for key in [
                "nCount_RNA",
                "nFeature_RNA",
                "nCount_HTO",
                "nFeature_HTO",
                "nCount_GDO",
                "nFeature_GDO",
                "nCount_ADT",
                "nFeature_ADT",
                "percent.mito",
                "S.Score",
                "G2M.Score",
            ]:
                add_num(bucket, key, row.get(key))

    arrayed = SRC_DIR / "GSE153056_ECCITE_Arrayed_metadata.tsv.gz"
    source_files.append(str(arrayed))
    with gzip.open(arrayed, "rt", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            total_rows += 1
            gene = guide_to_gene(row.get("GO_cite_classification"))
            if gene not in local:
                continue
            matched_rows += 1
            bucket = per_condition[gene]
            guide_sets[gene].add(row.get("GO_cite_classification", ""))
            for key in [
                "nCount_HTO",
                "nFeature_HTO",
                "nCount_GO_lenti",
                "nFeature_GO_lenti",
                "GO_lenti_margin",
                "nCount_GO_cite",
                "nFeature_GO_cite",
                "GO_cite_margin",
                "nCount_ADT",
                "nFeature_ADT",
                "nCount_RNA",
                "nFeature_RNA",
                "percent.mito",
            ]:
                add_num(bucket, f"arrayed_{key}", row.get(key))

    rows = []
    for condition, vals in sorted(per_condition.items()):
        artifacts: dict[str, float] = {
            "papalexi_author_cell_assignments": float(len(vals.get("nCount_RNA", []))),
            "papalexi_author_unique_guides_observed": float(len({g for g in guide_sets[condition] if g})),
        }
        for key, numbers in vals.items():
            if not numbers:
                continue
            safe_key = key.replace(".", "_")
            artifacts[f"papalexi_author_{safe_key}_mean"] = mean(numbers)
            artifacts[f"papalexi_author_{safe_key}_median"] = median(numbers)
        total_rep = sum(rep_counts[condition].values())
        if total_rep:
            artifacts["papalexi_author_replicate_max_fraction"] = max(rep_counts[condition].values()) / total_rep
        total_phase = sum(phase_counts[condition].values())
        if total_phase:
            for phase in ("G1", "S", "G2M"):
                artifacts[f"papalexi_author_phase_{phase}_fraction"] = phase_counts[condition][phase] / total_phase
        for artifact, value in sorted(artifacts.items()):
            rows.append(
                {
                    "dataset": DATASET,
                    "condition": condition,
                    "split": split_by_condition[condition],
                    "artifact": artifact,
                    "artifact_value": value,
                    "artifact_role": "qc_reagent_support_metadata",
                    "source_files": ";".join(source_files),
                    "source_url": SOURCE_URL,
                }
            )

    fields = ["dataset", "condition", "split", "artifact", "artifact_value", "artifact_role", "source_files", "source_url"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "status": "papalexi_author_metadata_artifacts_ready_diagnostic_no_gpu",
        "gpu_authorized": False,
        "dataset": DATASET,
        "total_metadata_rows": total_rows,
        "matched_rows": matched_rows,
        "local_conditions": len(local),
        "matched_conditions": len(per_condition),
        "missing_conditions": sorted(local - set(per_condition)),
        "artifact_rows": len(rows),
        "artifacts": sorted({r["artifact"] for r in rows}),
        "source_files": source_files,
        "outputs": {"csv": str(OUT_CSV), "json": str(OUT_JSON), "markdown": str(OUT_MD), "manifest": str(MANIFEST)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MANIFEST.write_text(json.dumps({"status": payload["status"], "csv": str(OUT_CSV), "artifacts": payload["artifacts"], "source_files": source_files}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Papalexi Author Metadata Artifacts 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/source-only materialization from GSE153056 metadata TSVs.",
        "- Fields are guide/hash/count/phase metadata, not an independent response/effect-size table.",
        "- No training, inference, canonical multi selection, Track C query, or GPU.",
        "",
        "## Summary",
        "",
        f"- total metadata rows read: `{total_rows}`",
        f"- matched rows: `{matched_rows}`",
        f"- local conditions: `{len(local)}`",
        f"- matched conditions: `{len(per_condition)}`",
        f"- artifact rows: `{len(rows)}`",
        f"- missing conditions: `{payload['missing_conditions']}`",
        "",
        "## Decision",
        "",
        "This is a well-aligned Papalexi small-table source, but it is single-dataset",
        "and QC/reagent-support metadata. It requires a diagnostic preview only and",
        "cannot authorize GPU without a materially new response/effect-size artifact.",
        "",
        "## Outputs",
        "",
        f"- csv: `{OUT_CSV}`",
        f"- manifest: `{MANIFEST}`",
        f"- json: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "artifact_rows": len(rows), "report": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
