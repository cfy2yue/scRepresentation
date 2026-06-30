#!/usr/bin/env python3
"""Summarize existing LatentFM per-condition residual CSVs.

This is a read-only posthoc helper. It does not run model inference, inspect
tmux, tail logs, or query GPUs. It only reads condition_residual_full128_best.csv
files that already exist under the strategy-probe output directories.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT_ROOT = ROOT / "CoupledFM/output/latentfm_runs"
REPORT_MD = ROOT / "reports/LATENTFM_CONDITION_RESIDUAL_AUDIT_20260619.md"
REPORT_CSV = ROOT / "reports/latentfm_condition_residual_audit_20260619.csv"
REPORT_JSON = ROOT / "reports/latentfm_condition_residual_audit_20260619.json"

RUN_ROOTS = {
    "four_run_scfoundation": OUT_ROOT / "scfoundation_strategy_probe_20260619",
    "four_run_stack": OUT_ROOT / "stack_strategy_probe_20260619",
    "expanded_scfoundation": OUT_ROOT / "scfoundation_strategy_probe_expanded_20260619",
    "expanded_stack": OUT_ROOT / "stack_strategy_probe_expanded_20260619",
}

GROUP_ORDER = (
    "test",
    "test_single",
    "test_multi",
    "test_multi_seen",
    "test_multi_unseen1",
    "test_multi_unseen2",
    "family_gene",
    "family_drug",
)


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def as_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def read_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for matrix, root in RUN_ROOTS.items():
        if not root.is_dir():
            continue
        for csv_path in sorted(root.glob("*/posthoc_eval/condition_residual_full128_best.csv")):
            run = csv_path.parents[1].name
            backbone = "scfoundation" if "scfoundation" in matrix else "stack"
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    groups = [g for g in str(row.get("groups", "")).split(",") if g]
                    rows.append(
                        {
                            "matrix": matrix,
                            "backbone": backbone,
                            "run": run,
                            "dataset": row.get("dataset", ""),
                            "condition": row.get("condition", ""),
                            "groups": groups,
                            "perturbation_family": row.get("perturbation_family", ""),
                            "perturbation_type": row.get("perturbation_type", ""),
                            "n_genes": int(row.get("n_genes") or 0),
                            "is_multi": as_bool(row.get("is_multi")),
                            "pred_target_cosine": as_float(row.get("pred_target_cosine")),
                            "pred_target_pearson": as_float(row.get("pred_target_pearson")),
                            "retrieval_rank": as_float(row.get("retrieval_rank")),
                            "retrieval_true_similarity": as_float(row.get("retrieval_true_similarity")),
                            "retrieval_top1": as_bool(row.get("retrieval_top1")),
                            "retrieval_top5": as_bool(row.get("retrieval_top5")),
                            "retrieval_top10": as_bool(row.get("retrieval_top10")),
                            "path": str(csv_path),
                        }
                    )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row_groups = row["groups"] or ["ungrouped"]
        for group in row_groups:
            if group in GROUP_ORDER or group.startswith("family_"):
                buckets[(row["matrix"], row["backbone"], row["run"], group)].append(row)

    summary: list[dict[str, Any]] = []
    for (matrix, backbone, run, group), vals in sorted(buckets.items()):
        cos = [v["pred_target_cosine"] for v in vals if v["pred_target_cosine"] is not None]
        pear = [v["pred_target_pearson"] for v in vals if v["pred_target_pearson"] is not None]
        ranks = [v["retrieval_rank"] for v in vals if v["retrieval_rank"] is not None]
        summary.append(
            {
                "matrix": matrix,
                "backbone": backbone,
                "run": run,
                "group": group,
                "n_conditions": len(vals),
                "mean_cosine": mean(cos) if cos else None,
                "median_cosine": median(cos) if cos else None,
                "mean_pearson": mean(pear) if pear else None,
                "median_pearson": median(pear) if pear else None,
                "median_retrieval_rank": median(ranks) if ranks else None,
                "top1_rate": mean([1.0 if v["retrieval_top1"] else 0.0 for v in vals]) if vals else None,
                "top5_rate": mean([1.0 if v["retrieval_top5"] else 0.0 for v in vals]) if vals else None,
                "top10_rate": mean([1.0 if v["retrieval_top10"] else 0.0 for v in vals]) if vals else None,
            }
        )
    return summary


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "matrix",
        "backbone",
        "run",
        "group",
        "n_conditions",
        "mean_cosine",
        "median_cosine",
        "mean_pearson",
        "median_pearson",
        "median_retrieval_rank",
        "top1_rate",
        "top5_rate",
        "top10_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    complete_runs = sorted({r["run"] for r in source_rows})
    priority = [
        r
        for r in rows
        if r["group"] in {"test", "test_multi_unseen1", "test_multi_unseen2", "family_gene", "family_drug"}
    ]
    priority.sort(
        key=lambda r: (
            str(r["backbone"]),
            str(r["run"]),
            GROUP_ORDER.index(r["group"]) if r["group"] in GROUP_ORDER else 99,
        )
    )

    lines = [
        "# LatentFM Condition Residual Audit 2026-06-19",
        "",
        "This report summarizes existing per-condition residual CSV files only.",
        "It does not run inference, inspect tmux, tail logs, or query GPU state.",
        "",
        "## Coverage",
        "",
        f"- Runs with residual CSVs: `{len(complete_runs)}`",
        f"- Per-condition rows read: `{len(source_rows)}`",
        f"- Summary rows written: `{len(rows)}`",
        f"- CSV: `{REPORT_CSV}`",
        f"- JSON: `{REPORT_JSON}`",
        "",
        "## Priority Groups",
        "",
        "| Backbone | Run | Group | n | mean cosine | median cosine | mean pearson | median rank | top10 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in priority:
        lines.append(
            "| {backbone} | `{run}` | `{group}` | {n_conditions} | {mean_cosine} | "
            "{median_cosine} | {mean_pearson} | {median_retrieval_rank} | {top10_rate} |".format(
                backbone=row["backbone"],
                run=row["run"],
                group=row["group"],
                n_conditions=row["n_conditions"],
                mean_cosine=fmt(row["mean_cosine"]),
                median_cosine=fmt(row["median_cosine"]),
                mean_pearson=fmt(row["mean_pearson"]),
                median_retrieval_rank=fmt(row["median_retrieval_rank"]),
                top10_rate=fmt(row["top10_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "Use this as a diagnostic layer beneath the split/family pp gate. A branch",
            "with better residual retrieval but worse aggregate pp is not automatically",
            "promotable; it indicates that checkpoint selection or objective alignment",
            "may need to move from dataset-level means toward condition-level residual",
            "ranking.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    source_rows = read_rows()
    rows = summarize(source_rows)
    write_csv(REPORT_CSV, rows)
    payload = {
        "source_csv_count": len({r["path"] for r in source_rows}),
        "source_row_count": len(source_rows),
        "summary_row_count": len(rows),
        "rows": rows,
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(REPORT_MD, rows, source_rows)
    print(json.dumps({"report": str(REPORT_MD), "rows": len(rows), "source_rows": len(source_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
