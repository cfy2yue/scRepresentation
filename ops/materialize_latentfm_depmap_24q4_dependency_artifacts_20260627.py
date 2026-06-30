#!/usr/bin/env python3
"""Materialize DepMap 24Q4 matched dependency artifacts for LatentFM.

CPU/report-only. Reads public DepMap CRISPRGeneEffect.csv after source
inspection and joins exact gene symbols to the local gene-condition scaffold.
No training, inference, checkpoint selection, canonical multi selection, Track C
query, or GPU.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
SRC_DIR = ROOT / "reports/external_artifact_sources_20260627/depmap_24q4_figshare"
GENE_EFFECT = SRC_DIR / "CRISPRGeneEffect.csv"
MODEL = SRC_DIR / "Model.csv"
SCREEN = SRC_DIR / "CRISPRScreenMap.csv"
SCAFFOLD = ROOT / "reports/depmap_24q4_artifact_scaffold_20260627/depmap_24q4_gene_condition_scaffold.csv"
OUT_DIR = ROOT / "reports/depmap_24q4_dependency_artifacts_20260627"
OUT_CSV = OUT_DIR / "depmap_24q4_dependency_artifacts.csv"
MANIFEST_JSON = ROOT / "configs/latentfm_depmap_24q4_dependency_artifact_manifest_20260627.json"
OUT_JSON = ROOT / "reports/latentfm_depmap_24q4_dependency_artifacts_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_DEPMAP_24Q4_DEPENDENCY_ARTIFACTS_20260627.md"


def norm(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "na", "nan", "none", "<na>"}:
        return ""
    return text


def to_float(value: Any) -> float | None:
    text = norm(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def gene_symbol(column: str) -> str:
    return column.split(" (", 1)[0].strip()


def read_scaffold() -> list[dict[str, str]]:
    with SCAFFOLD.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_gene_effect_for_models(model_ids: set[str]) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    values: dict[str, dict[str, float]] = {}
    duplicate_gene_columns = 0
    with GENE_EFFECT.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        gene_to_idx: dict[str, int] = {}
        for idx, col in enumerate(header[1:], start=1):
            sym = gene_symbol(col)
            if not sym:
                continue
            if sym in gene_to_idx:
                duplicate_gene_columns += 1
                continue
            gene_to_idx[sym] = idx
        for row in reader:
            if not row:
                continue
            model_id = row[0]
            if model_id not in model_ids:
                continue
            rec: dict[str, float] = {}
            for sym, idx in gene_to_idx.items():
                if idx < len(row):
                    val = to_float(row[idx])
                    if val is not None:
                        rec[sym] = val
            values[model_id] = rec
            if len(values) == len(model_ids):
                break
    meta = {
        "gene_columns_unique": len(next(iter(values.values()))) if values else 0,
        "duplicate_gene_columns_skipped": duplicate_gene_columns,
        "model_ids_found": sorted(values),
    }
    return values, meta


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_files = [GENE_EFFECT, MODEL, SCREEN, SCAFFOLD]
    missing = [str(p) for p in source_files if not p.is_file()]
    if missing:
        payload = {
            "status": "depmap_24q4_dependency_artifacts_missing_sources_no_gpu",
            "gpu_authorized": False,
            "missing": missing,
            "outputs": {"json": str(OUT_JSON), "markdown": str(OUT_MD)},
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        OUT_MD.write_text(
            "# LatentFM DepMap 24Q4 Dependency Artifacts 2026-06-27\n\n"
            f"Status: `{payload['status']}`\n\nGPU authorized: `False`\n\n"
            f"Missing sources: `{missing}`\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": payload["status"], "missing": missing}, indent=2))
        return 2

    scaffold = read_scaffold()
    model_ids = {row["depmap_model_id"] for row in scaffold}
    gene_effect, meta = read_gene_effect_for_models(model_ids)
    out_rows: list[dict[str, Any]] = []
    missing_model = 0
    missing_gene = 0
    for row in scaffold:
        model_id = row["depmap_model_id"]
        target = row["target_gene"]
        model_values = gene_effect.get(model_id)
        if model_values is None:
            missing_model += 1
            continue
        raw = model_values.get(target)
        if raw is None:
            missing_gene += 1
            continue
        out_rows.append(
            {
                "dataset": row["dataset"],
                "condition": row["condition"],
                "split": row["split"],
                "cell_background": row["cell_background"],
                "depmap_model_id": model_id,
                "target_gene": target,
                "depmap_gene_effect": raw,
                "artifact": "depmap_24q4_dependency_score",
                "artifact_value": -raw,
                "depmap_gene_effect_raw": raw,
            }
        )

    fields = [
        "dataset",
        "condition",
        "split",
        "cell_background",
        "depmap_model_id",
        "target_gene",
        "artifact",
        "artifact_value",
        "depmap_gene_effect",
        "depmap_gene_effect_raw",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    by_dataset = defaultdict(int)
    by_split = defaultdict(int)
    for row in out_rows:
        by_dataset[row["dataset"]] += 1
        by_split[row["split"]] += 1

    manifest = {
        "status": "depmap_24q4_dependency_artifacts_materialized_no_gpu",
        "artifacts": [
            {
                "artifact": "depmap_24q4_dependency_score",
                "source_files": [str(OUT_CSV)],
                "required_columns": ["dataset", "condition", "artifact_value"],
                "minimum_datasets": 3,
                "minimum_overlap_rows": 50,
                "minimum_varying_datasets": 3,
                "promotion_note": "CPU gate only; higher artifact_value means more negative DepMap gene effect / stronger dependency.",
            }
        ],
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = {
        "status": "depmap_24q4_dependency_artifacts_materialized_no_gpu",
        "gpu_authorized": False,
        "source_dir": str(SRC_DIR),
        "scaffold": str(SCAFFOLD),
        "scaffold_rows": len(scaffold),
        "materialized_rows": len(out_rows),
        "missing_model_rows": missing_model,
        "missing_gene_rows": missing_gene,
        "dataset_counts": dict(sorted(by_dataset.items())),
        "split_counts": dict(sorted(by_split.items())),
        "gene_effect_meta": meta,
        "outputs": {"csv": str(OUT_CSV), "manifest": str(MANIFEST_JSON), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM DepMap 24Q4 Dependency Artifacts 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only materialization of matched DepMap gene dependency artifacts.",
        "- Higher `artifact_value` means stronger dependency (`-CRISPRGeneEffect`).",
        "- Does not train, infer, select checkpoints, read canonical multi for selection, read Track C query, or use GPU.",
        "",
        "## Summary",
        "",
        f"- scaffold rows: `{len(scaffold)}`",
        f"- materialized rows: `{len(out_rows)}`",
        f"- missing model rows: `{missing_model}`",
        f"- missing gene rows: `{missing_gene}`",
        "",
        "| dataset | rows |",
        "|---|---:|",
    ]
    for dataset, count in sorted(by_dataset.items()):
        lines.append(f"| `{dataset}` | {count} |")
    lines += [
        "",
        "## Decision",
        "",
        "This is a source artifact only. Run the strict DepMap CPU gate before any adapter design or GPU smoke.",
        "",
        f"- CSV: `{OUT_CSV}`",
        f"- manifest: `{MANIFEST_JSON}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(out_rows), "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
