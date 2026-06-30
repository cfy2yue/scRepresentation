#!/usr/bin/env python3
"""Materialize Jiang author-DE background artifacts.

CPU/report-only. Parses the Zenodo author DE archive where cytokine/dataset and
regulator are encoded in file paths and cell backgrounds are encoded in columns.
This does not train, infer, read checkpoints, read canonical multi for
selection, read Track C query, or use GPU.
"""

from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
ZIP_PATH = (
    ROOT
    / "reports/external_artifact_sources_20260627/jiang_zenodo_v2_1/DE_results_all_pathway.zip"
)
SCAFFOLD = (
    ROOT
    / "reports/jiang_background_artifact_scaffold_20260627/jiang_condition_background_scaffold.csv"
)
OUT_DIR = ROOT / "reports/jiang_author_de_artifacts_20260627"
BG_CSV = OUT_DIR / "jiang_author_de_background_artifacts.csv"
COND_CSV = OUT_DIR / "jiang_author_de_condition_aggregate_artifacts.csv"
MANIFEST_JSON = ROOT / "configs/latentfm_jiang_author_de_artifact_manifest_20260627.json"
OUT_JSON = ROOT / "reports/latentfm_jiang_author_de_artifacts_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_JIANG_AUTHOR_DE_ARTIFACTS_20260627.md"

CYTOKINE_TO_DATASET = {
    "IFNB": "Jiang_IFNB",
    "IFNG": "Jiang_IFNG",
    "INS": "Jiang_INS",
    "TGFB1": "Jiang_TGFB",
    "TNFA": "Jiang_TNFA",
}

METRICS = [
    "mean_abs_log2fc",
    "mean_signed_log2fc",
    "mean_abs_beta",
    "mean_signed_beta",
    "mean_abs_lfc_neglog10p",
    "sig_frac_p05_abs005",
    "valid_gene_count",
]


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


def safe_neglog10(p: float | None) -> float | None:
    if p is None:
        return None
    return min(-math.log10(max(p, 1e-300)), 50.0)


def read_scaffold() -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    with SCAFFOLD.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["condition"], row["cell_background"])
            rows[key] = row
    return rows


def parse_member_name(name: str) -> tuple[str, str, str] | None:
    parts = name.split("/")
    if len(parts) != 3 or parts[0] != "DE_results_all_pathway":
        return None
    folder, filename = parts[1], parts[2]
    if not folder.startswith("Parse_") or not filename.endswith("_pathway_DE_results.txt"):
        return None
    cytokine = folder.removeprefix("Parse_")
    dataset = CYTOKINE_TO_DATASET.get(cytokine)
    if dataset is None:
        return None
    stem = filename.removesuffix("_pathway_DE_results.txt")
    suffix = f"_{cytokine}"
    if not stem.endswith(suffix):
        return None
    regulator = stem[: -len(suffix)]
    if not regulator:
        return None
    return dataset, cytokine, regulator


def summarize_background(rows: list[dict[str, str]], background: str) -> dict[str, float] | None:
    lfc_key = f"log2FC_{background}"
    beta_key = f"beta_cell_type{background}"
    p_key = f"p_cell_type{background}"
    abs_lfc: list[float] = []
    signed_lfc: list[float] = []
    abs_beta: list[float] = []
    signed_beta: list[float] = []
    weighted_abs: list[float] = []
    sig = 0
    valid = 0
    for row in rows:
        lfc = to_float(row.get(lfc_key))
        beta = to_float(row.get(beta_key))
        pval = to_float(row.get(p_key))
        if lfc is None and beta is None:
            continue
        if lfc is not None:
            valid += 1
            abs_lfc.append(abs(lfc))
            signed_lfc.append(lfc)
            nlp = safe_neglog10(pval)
            if nlp is not None:
                weighted_abs.append(abs(lfc) * nlp)
            if pval is not None and pval < 0.05 and abs(lfc) >= 0.05:
                sig += 1
        if beta is not None:
            abs_beta.append(abs(beta))
            signed_beta.append(beta)
    if valid == 0:
        return None
    return {
        "mean_abs_log2fc": mean(abs_lfc) if abs_lfc else 0.0,
        "mean_signed_log2fc": mean(signed_lfc) if signed_lfc else 0.0,
        "mean_abs_beta": mean(abs_beta) if abs_beta else 0.0,
        "mean_signed_beta": mean(signed_beta) if signed_beta else 0.0,
        "mean_abs_lfc_neglog10p": mean(weighted_abs) if weighted_abs else 0.0,
        "sig_frac_p05_abs005": sig / valid,
        "valid_gene_count": float(valid),
    }


def read_author_table(zh: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    text = zh.read(name).decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text), delimiter=" ", skipinitialspace=True)
    return list(reader)


def aggregate(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean": mean(values),
        "max": max(values),
        "min": min(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scaffold = read_scaffold()
    bg_rows: list[dict[str, Any]] = []
    missed_scaffold = 0
    parsed_tables = 0
    skipped_tables = 0

    with zipfile.ZipFile(ZIP_PATH) as zh:
        for name in zh.namelist():
            parsed = parse_member_name(name)
            if parsed is None:
                continue
            dataset, cytokine, regulator = parsed
            table = read_author_table(zh, name)
            parsed_tables += 1
            any_bg = False
            for background in ("A549", "BXPC3", "HAP1", "HT29", "K562", "MCF7"):
                scaf = scaffold.get((dataset, regulator, background))
                if scaf is None:
                    missed_scaffold += 1
                    continue
                summary = summarize_background(table, background)
                if summary is None:
                    continue
                any_bg = True
                rec = {
                    "dataset": dataset,
                    "cytokine": cytokine,
                    "condition": regulator,
                    "split": scaf["split"],
                    "cell_background": background,
                    "source_member": name,
                }
                rec.update(summary)
                bg_rows.append(rec)
            if not any_bg:
                skipped_tables += 1

    bg_fields = [
        "dataset",
        "cytokine",
        "condition",
        "split",
        "cell_background",
        "source_member",
        *METRICS,
    ]
    with BG_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=bg_fields)
        writer.writeheader()
        writer.writerows(bg_rows)

    by_condition: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in bg_rows:
        by_condition[(row["dataset"], row["cytokine"], row["condition"], row["split"])].append(row)

    cond_rows: list[dict[str, Any]] = []
    for (dataset, cytokine, condition, split), rows in sorted(by_condition.items()):
        for metric in METRICS:
            values = [float(r[metric]) for r in rows if to_float(r.get(metric)) is not None]
            for agg_name, agg_value in aggregate(values).items():
                cond_rows.append(
                    {
                        "dataset": dataset,
                        "cytokine": cytokine,
                        "condition": condition,
                        "split": split,
                        "artifact_metric": metric,
                        "aggregation": agg_name,
                        "artifact": f"jiang_author_de_{metric}_{agg_name}",
                        "artifact_value": agg_value,
                        "background_count": len(rows),
                    }
                )

    cond_fields = [
        "dataset",
        "cytokine",
        "condition",
        "split",
        "artifact_metric",
        "aggregation",
        "artifact",
        "artifact_value",
        "background_count",
    ]
    with COND_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cond_fields)
        writer.writeheader()
        writer.writerows(cond_rows)

    artifacts = sorted({row["artifact"] for row in cond_rows})
    manifest = {
        "status": "jiang_author_de_artifacts_materialized_no_gpu",
        "artifact_family": "jiang_author_de_background_response",
        "source_files": [str(COND_CSV)],
        "background_source_files": [str(BG_CSV)],
        "artifacts": [
            {
                "artifact": artifact,
                "source_files": [str(COND_CSV)],
                "required_columns": ["dataset", "condition", "artifact", "artifact_value"],
                "minimum_datasets": 3,
                "minimum_overlap_rows": 50,
                "minimum_varying_datasets": 3,
                "promotion_note": "CPU gate only; no GPU without Jiang-specific association/shuffle/LODO audit.",
            }
            for artifact in artifacts
        ],
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    dataset_counts = defaultdict(int)
    split_counts = defaultdict(int)
    for row in by_condition:
        dataset_counts[row[0]] += 1
        split_counts[row[3]] += 1
    payload = {
        "status": "jiang_author_de_artifacts_materialized_no_gpu",
        "gpu_authorized": False,
        "source_zip": str(ZIP_PATH),
        "scaffold": str(SCAFFOLD),
        "parsed_tables": parsed_tables,
        "skipped_tables_no_joined_background": skipped_tables,
        "missed_scaffold_background_rows": missed_scaffold,
        "background_rows": len(bg_rows),
        "condition_rows": len(by_condition),
        "long_artifact_rows": len(cond_rows),
        "artifacts": len(artifacts),
        "dataset_condition_counts": dict(sorted(dataset_counts.items())),
        "split_condition_counts": dict(sorted(split_counts.items())),
        "outputs": {
            "background_csv": str(BG_CSV),
            "condition_aggregate_csv": str(COND_CSV),
            "manifest": str(MANIFEST_JSON),
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Jiang Author-DE Artifacts 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only materialization of Jiang author-provided Mixscale DE summaries.",
        "- Parses whitespace-delimited `*_pathway_DE_results.txt` tables from the Zenodo archive.",
        "- Does not train, infer, select checkpoints, read canonical multi for selection, read Track C query, or use GPU.",
        "",
        "## Summary",
        "",
        f"- parsed author tables: `{parsed_tables}`",
        f"- background rows: `{len(bg_rows)}`",
        f"- joined condition rows: `{len(by_condition)}`",
        f"- long artifact rows: `{len(cond_rows)}`",
        f"- artifact variants: `{len(artifacts)}`",
        f"- missed scaffold background rows: `{missed_scaffold}`",
        "",
        "| dataset | joined conditions |",
        "|---|---:|",
    ]
    for dataset, count in sorted(dataset_counts.items()):
        lines.append(f"| `{dataset}` | {count} |")
    lines += [
        "",
        "## Decision",
        "",
        "The archive is schema-valid after whitespace parsing. These rows are an external background-response artifact source, not a training signal by themselves. Next step is a Jiang-specific CPU association/shuffle/LODO gate against frozen xverse anchor outcomes.",
        "",
        f"- background CSV: `{BG_CSV}`",
        f"- condition aggregate CSV: `{COND_CSV}`",
        f"- manifest: `{MANIFEST_JSON}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "condition_rows": len(by_condition), "artifacts": len(artifacts), "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
