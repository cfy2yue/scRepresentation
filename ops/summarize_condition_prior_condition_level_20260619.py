#!/usr/bin/env python3
"""Condition-level comparison for primary, teacher-dose, and prior correction."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
PRIMARY_CSV = ROOT / "reports/latentfm_condition_residual_full128_20260619/primary_scfoundation.csv"
DOSE_JSON = ROOT / "reports/latentfm_condition_prior_teacher_dose_20260619.json"
PRIOR_CSV = ROOT / "reports/latentfm_prior_correction_eval_20260619.csv"
OUT_CSV = ROOT / "reports/latentfm_condition_prior_condition_level_comparison_20260619.csv"
OUT_JSON = ROOT / "reports/latentfm_condition_prior_condition_level_comparison_20260619.json"
OUT_MD = ROOT / "reports/LATENTFM_CONDITION_PRIOR_CONDITION_LEVEL_COMPARISON_20260619.md"
BASE_RUN_ROOT = ROOT / "CoupledFM/output/latentfm_runs/condition_prior_teacher_probe_20260619"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def group_from_residual(groups: str) -> str | None:
    parts = {part.strip() for part in groups.split(",") if part.strip()}
    for group in ("test_multi_seen", "test_multi_unseen1", "test_multi_unseen2"):
        if group in parts:
            return group
    return None


def residual_index(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in load_csv(path):
        if str(row.get("is_multi", "")).lower() != "true":
            continue
        group = group_from_residual(row.get("groups", ""))
        if not group:
            continue
        item = dict(row)
        item["group"] = group
        out[(row["dataset"], row["condition"])] = item
    return out


def best_teacher_csv() -> tuple[str, Path]:
    payload = json.loads(DOSE_JSON.read_text(encoding="utf-8"))
    best = payload.get("best")
    if isinstance(best, dict):
        best = best.get("run")
    if not best:
        raise RuntimeError(f"dose JSON has no best completed run: {DOSE_JSON}")
    path = BASE_RUN_ROOT / str(best) / "posthoc_eval/condition_residual_full128_best.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(best), path


def prior_best_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in load_csv(path):
        grouped[(row["dataset"], row["condition"])].append(row)

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        base_candidates = [r for r in rows if as_float(r.get("alpha")) == 0.0]
        base = max(base_candidates or rows, key=lambda r: as_float(r.get("pp")) or float("-inf"))
        best = max(rows, key=lambda r: as_float(r.get("pp")) or float("-inf"))
        base_pp = as_float(base.get("pp"))
        best_pp = as_float(best.get("pp"))
        out[key] = {
            "prior_group": best.get("group", ""),
            "prior_base_alpha": as_float(base.get("alpha")),
            "prior_base_k": as_float(base.get("k")),
            "prior_base_pp": base_pp,
            "prior_best_alpha": as_float(best.get("alpha")),
            "prior_best_k": as_float(best.get("k")),
            "prior_best_pp": best_pp,
            "prior_best_pc": as_float(best.get("pc")),
            "prior_delta_pp": None if base_pp is None or best_pp is None else best_pp - base_pp,
            "prior_available": as_float(best.get("prior_available")),
            "prior_n_missing": as_float(best.get("n_missing")),
            "prior_median_knn_similarity": as_float(best.get("median_knn_similarity")),
        }
    return out


def fmt(value: Any, digits: int = 4) -> str:
    val = as_float(value)
    if val is None:
        return "NA"
    return f"{val:.{digits}f}"


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["dataset"]), str(row["group"]))].append(row)

    metrics = [
        "primary_pearson",
        "teacher_pearson",
        "teacher_delta_pearson",
        "prior_base_pp",
        "prior_best_pp",
        "prior_delta_pp",
    ]
    out = []
    for (dataset, group), vals in sorted(by_key.items()):
        item: dict[str, Any] = {"dataset": dataset, "group": group, "n": len(vals)}
        for metric in metrics:
            nums = [as_float(v.get(metric)) for v in vals]
            nums = [v for v in nums if v is not None]
            item[f"mean_{metric}"] = None if not nums else sum(nums) / len(nums)
        out.append(item)
    return out


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        vals = []
        for field in fields:
            value = row.get(field)
            vals.append(fmt(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def main() -> int:
    best_name, teacher_csv = best_teacher_csv()
    primary = residual_index(PRIMARY_CSV)
    teacher = residual_index(teacher_csv)
    prior = prior_best_rows(PRIOR_CSV)

    rows: list[dict[str, Any]] = []
    for key in sorted(set(primary) & set(teacher) & set(prior)):
        p = primary[key]
        t = teacher[key]
        pr = prior[key]
        primary_pearson = as_float(p.get("pred_target_pearson"))
        teacher_pearson = as_float(t.get("pred_target_pearson"))
        row = {
            "dataset": key[0],
            "condition": key[1],
            "group": p["group"],
            "genes": p.get("genes", ""),
            "n_genes": as_float(p.get("n_genes")),
            "primary_pearson": primary_pearson,
            "teacher_run": best_name,
            "teacher_pearson": teacher_pearson,
            "teacher_delta_pearson": None
            if primary_pearson is None or teacher_pearson is None
            else teacher_pearson - primary_pearson,
            **pr,
        }
        rows.append(row)

    fields = [
        "dataset",
        "condition",
        "group",
        "genes",
        "n_genes",
        "primary_pearson",
        "teacher_run",
        "teacher_pearson",
        "teacher_delta_pearson",
        "prior_base_pp",
        "prior_best_pp",
        "prior_delta_pp",
        "prior_best_alpha",
        "prior_best_k",
        "prior_best_pc",
        "prior_available",
        "prior_n_missing",
        "prior_median_knn_similarity",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    groups = aggregate(rows)
    top_teacher = sorted(rows, key=lambda r: as_float(r.get("teacher_delta_pearson")) or float("-inf"), reverse=True)[:15]
    worst_teacher = sorted(rows, key=lambda r: as_float(r.get("teacher_delta_pearson")) or float("inf"))[:15]
    top_prior = sorted(rows, key=lambda r: as_float(r.get("prior_delta_pp")) or float("-inf"), reverse=True)[:15]
    persistent_failures = sorted(
        [r for r in rows if (as_float(r.get("teacher_pearson")) or 0.0) < 0.0 and (as_float(r.get("prior_best_pp")) or 0.0) < 0.0],
        key=lambda r: (as_float(r.get("teacher_pearson")) or 0.0) + (as_float(r.get("prior_best_pp")) or 0.0),
    )[:15]

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "best_teacher_run": best_name,
        "inputs": {
            "primary_csv": str(PRIMARY_CSV),
            "teacher_csv": str(teacher_csv),
            "prior_csv": str(PRIOR_CSV),
        },
        "n_common_conditions": len(rows),
        "group_summary": groups,
        "top_teacher_improved": top_teacher,
        "top_teacher_regressed": worst_teacher,
        "top_prior_rescued": top_prior,
        "persistent_failures": persistent_failures,
        "csv": str(OUT_CSV),
        "report": str(OUT_MD),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_fields = [
        "dataset",
        "group",
        "n",
        "mean_primary_pearson",
        "mean_teacher_pearson",
        "mean_teacher_delta_pearson",
        "mean_prior_base_pp",
        "mean_prior_best_pp",
        "mean_prior_delta_pp",
    ]
    condition_fields = [
        "dataset",
        "condition",
        "group",
        "genes",
        "primary_pearson",
        "teacher_pearson",
        "teacher_delta_pearson",
        "prior_base_pp",
        "prior_best_pp",
        "prior_delta_pp",
        "prior_best_alpha",
    ]
    persistent_failure_lines = (
        markdown_table(persistent_failures, condition_fields)
        if persistent_failures
        else ["No shared condition has both negative teacher Pearson and negative best prior pp."]
    )
    lines = [
        "# LatentFM Condition-Prior Condition-Level Comparison 2026-06-19",
        "",
        f"Generated: {payload['generated']}",
        "",
        "## Purpose",
        "",
        "Compare condition-level behavior for the primary scFoundation branch, the best condition-prior teacher dose, and the no-leakage posthoc train-single prior correction. This is a read-only analysis over existing CSV artifacts.",
        "",
        "## Inputs",
        "",
        f"- Primary residual CSV: `{PRIMARY_CSV}`",
        f"- Best teacher run: `{best_name}`",
        f"- Best teacher residual CSV: `{teacher_csv}`",
        f"- Prior-correction CSV: `{PRIOR_CSV}`",
        "",
        "## Scope",
        "",
        f"- Common multi-condition rows: `{len(rows)}`",
        "- Prior correction currently covers Norman and Wessels multi-condition splits only, so this condition-level comparison is limited to those shared rows.",
        "- Primary/teacher columns use condition-residual Pearson against perturbation means. Prior columns use the prior-correction evaluator's per-condition pp. Treat cross-column magnitudes as directional evidence rather than identical metric definitions.",
        "",
        "## Group Summary",
        "",
        *markdown_table(groups, summary_fields),
        "",
        "## Top Teacher Improvements",
        "",
        *markdown_table(top_teacher, condition_fields),
        "",
        "## Top Teacher Regressions",
        "",
        *markdown_table(worst_teacher, condition_fields),
        "",
        "## Top Prior-Correction Rescues",
        "",
        *markdown_table(top_prior, condition_fields),
        "",
        "## Persistent Failures",
        "",
        *persistent_failure_lines,
        "",
        "## Interpretation",
        "",
        "The teacher dose moves several multi-condition residuals in the right direction, but condition-level gains are uneven and can regress individual conditions. The posthoc prior remains the stronger signal source for many Norman/Wessels conditions, so the next model-design step should focus on explicit composition or condition-response alignment rather than another scalar teacher-weight sweep.",
        "",
        "## Outputs",
        "",
        f"- CSV: `{OUT_CSV}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(OUT_CSV)
    print(OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
