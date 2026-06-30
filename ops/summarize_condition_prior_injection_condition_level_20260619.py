#!/usr/bin/env python3
"""Condition-level comparison for prior teacher with and without head injection."""
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
NO_INJECT_CSV = (
    ROOT
    / "CoupledFM/output/latentfm_runs/condition_prior_teacher_probe_20260619/"
    "scf_prior010_e2_4k/posthoc_eval/condition_residual_full128_best.csv"
)
INJECT_CSV = (
    ROOT
    / "CoupledFM/output/latentfm_runs/condition_prior_teacher_injection_20260619/"
    "scf_prior010_inject_e2_4k/posthoc_eval/condition_residual_full128_best.csv"
)
PRIOR_CSV = ROOT / "reports/latentfm_prior_correction_eval_20260619.csv"
OUT_CSV = ROOT / "reports/latentfm_condition_prior_injection_condition_level_20260619.csv"
OUT_JSON = ROOT / "reports/latentfm_condition_prior_injection_condition_level_20260619.json"
OUT_MD = ROOT / "reports/LATENTFM_CONDITION_PRIOR_INJECTION_CONDITION_LEVEL_20260619.md"


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
            "prior_base_pp": base_pp,
            "prior_best_pp": best_pp,
            "prior_delta_pp": None if base_pp is None or best_pp is None else best_pp - base_pp,
            "prior_best_alpha": as_float(best.get("alpha")),
            "prior_best_k": as_float(best.get("k")),
            "prior_best_pc": as_float(best.get("pc")),
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
        "no_inject_pearson",
        "inject_pearson",
        "inject_delta_vs_primary",
        "inject_delta_vs_no_inject",
        "prior_base_pp",
        "prior_best_pp",
        "prior_delta_pp",
    ]
    out: list[dict[str, Any]] = []
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
    primary = residual_index(PRIMARY_CSV)
    no_inject = residual_index(NO_INJECT_CSV)
    inject = residual_index(INJECT_CSV)
    prior = prior_best_rows(PRIOR_CSV)
    common = sorted(set(primary) & set(no_inject) & set(inject) & set(prior))

    rows: list[dict[str, Any]] = []
    for key in common:
        p = primary[key]
        n = no_inject[key]
        i = inject[key]
        primary_p = as_float(p.get("pred_target_pearson"))
        no_p = as_float(n.get("pred_target_pearson"))
        inj_p = as_float(i.get("pred_target_pearson"))
        rows.append(
            {
                "dataset": key[0],
                "condition": key[1],
                "group": p["group"],
                "genes": p.get("genes", ""),
                "primary_pearson": primary_p,
                "no_inject_pearson": no_p,
                "inject_pearson": inj_p,
                "inject_delta_vs_primary": None if inj_p is None or primary_p is None else inj_p - primary_p,
                "inject_delta_vs_no_inject": None if inj_p is None or no_p is None else inj_p - no_p,
                **prior[key],
            }
        )

    fields = [
        "dataset",
        "condition",
        "group",
        "genes",
        "primary_pearson",
        "no_inject_pearson",
        "inject_pearson",
        "inject_delta_vs_primary",
        "inject_delta_vs_no_inject",
        "prior_base_pp",
        "prior_best_pp",
        "prior_delta_pp",
        "prior_best_alpha",
        "prior_best_k",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    summary = aggregate(rows)
    top_injection_gain = sorted(
        rows, key=lambda r: as_float(r.get("inject_delta_vs_no_inject")) or float("-inf"), reverse=True
    )[:15]
    top_injection_regression = sorted(
        rows, key=lambda r: as_float(r.get("inject_delta_vs_no_inject")) or float("inf")
    )[:15]
    persistent_failures = sorted(
        [
            r
            for r in rows
            if (as_float(r.get("inject_pearson")) or 0.0) < 0.0
            and (as_float(r.get("prior_best_pp")) or 0.0) < 0.0
        ],
        key=lambda r: (as_float(r.get("inject_pearson")) or 0.0) + (as_float(r.get("prior_best_pp")) or 0.0),
    )[:15]

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "primary_csv": str(PRIMARY_CSV),
            "no_inject_csv": str(NO_INJECT_CSV),
            "inject_csv": str(INJECT_CSV),
            "prior_csv": str(PRIOR_CSV),
        },
        "n_common_conditions": len(rows),
        "group_summary": summary,
        "top_injection_gain": top_injection_gain,
        "top_injection_regression": top_injection_regression,
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
        "mean_no_inject_pearson",
        "mean_inject_pearson",
        "mean_inject_delta_vs_primary",
        "mean_inject_delta_vs_no_inject",
        "mean_prior_best_pp",
    ]
    condition_fields = [
        "dataset",
        "condition",
        "group",
        "genes",
        "primary_pearson",
        "no_inject_pearson",
        "inject_pearson",
        "inject_delta_vs_no_inject",
        "prior_best_pp",
    ]
    persistent_lines = (
        markdown_table(persistent_failures, condition_fields)
        if persistent_failures
        else ["No shared condition has both negative injected Pearson and negative best prior pp."]
    )

    lines = [
        "# LatentFM Condition-Prior Injection Condition-Level Comparison 2026-06-19",
        "",
        f"Generated: {payload['generated']}",
        "",
        "## Purpose",
        "",
        "Compare primary scFoundation, the best no-injection teacher dose, explicit head-injection teacher, and the posthoc train-single prior at condition level.",
        "",
        "## Scope",
        "",
        f"- Common multi-condition rows: `{len(rows)}`",
        "- This is limited to Norman/Wessels rows covered by the prior-correction evaluator.",
        "- Primary/no-injection/injection columns are residual Pearson; prior columns are per-condition pp, so use direction and relative changes rather than identical magnitudes.",
        "",
        "## Group Summary",
        "",
        *markdown_table(summary, summary_fields),
        "",
        "## Top Injection Gains Versus No-Injection",
        "",
        *markdown_table(top_injection_gain, condition_fields),
        "",
        "## Top Injection Regressions Versus No-Injection",
        "",
        *markdown_table(top_injection_regression, condition_fields),
        "",
        "## Persistent Failures After Injection",
        "",
        *persistent_lines,
        "",
        "## Interpretation",
        "",
        "Explicit head injection improves the aggregate dose score and unseen2 metrics, but condition-level effects should be checked for dataset-specific tradeoffs. If gains are concentrated in Norman while Wessels persistent failures remain negative, the next architecture should add an interaction/residual path rather than increasing global prior strength.",
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
