#!/usr/bin/env python3
"""Summarize lookahead/trust-region adapter smoke internal eval.

Compares a smoke checkpoint's internal split-group eval JSON against the
frozen xverse_8k_anchor seed42 internal means. This is posthoc only; it does
not train, infer, read canonical multi for selection, or touch Track C query.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
ANCHOR_INTERNAL = (
    REPORTS
    / "latentfm_xverse_8k_seed_ensemble_internal_means_20260627"
    / "seed42_internal_split_group_means_evalseed42.json"
)
RNG_SEED = 20260627


def norm(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def read_conditions(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for group, gdata in data.get("groups", {}).items():
        for row in gdata.get("condition_metrics", []):
            dataset = norm(row.get("dataset"))
            condition = norm(row.get("condition"))
            if group and dataset and condition:
                rows[(group, dataset, condition)] = row
    return rows


def bootstrap_ci_low(rows: list[dict[str, Any]], key: str, *, n_boot: int = 1000) -> float | None:
    datasets = sorted({row["dataset"] for row in rows})
    if len(datasets) < 3:
        return None
    by_dataset = {dataset: [row for row in rows if row["dataset"] == dataset] for dataset in datasets}
    rng = random.Random(RNG_SEED)
    vals: list[float] = []
    for _ in range(n_boot):
        sample: list[dict[str, Any]] = []
        for dataset in [rng.choice(datasets) for _ in datasets]:
            pool = by_dataset[dataset]
            sample.extend(pool[rng.randrange(len(pool))] for _ in range(len(pool)))
        vals.append(mean(float(row[key]) for row in sample))
    vals.sort()
    return vals[int(0.025 * (len(vals) - 1))] if vals else None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--eval-json", type=Path, default=None)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    eval_json = args.eval_json.resolve() if args.eval_json else run_dir / "internal_eval_split_groups.json"
    if not eval_json.is_file():
        raise FileNotFoundError(eval_json)
    anchor = read_conditions(ANCHOR_INTERNAL)
    cand = read_conditions(eval_json)
    rows: list[dict[str, Any]] = []
    for key, crow in sorted(cand.items()):
        arow = anchor.get(key)
        if arow is None:
            continue
        group, dataset, condition = key
        rows.append(
            {
                "group": group,
                "dataset": dataset,
                "condition": condition,
                "anchor_pearson_pert": arow.get("pearson_pert"),
                "candidate_pearson_pert": crow.get("pearson_pert"),
                "delta_pearson_pert": float(crow.get("pearson_pert", 0.0)) - float(arow.get("pearson_pert", 0.0)),
                "anchor_mmd_clamped": arow.get("test_mmd_clamped"),
                "candidate_mmd_clamped": crow.get("test_mmd_clamped"),
                "delta_mmd_clamped": float(crow.get("test_mmd_clamped", 0.0)) - float(arow.get("test_mmd_clamped", 0.0)),
            }
        )
    summaries: list[dict[str, Any]] = []
    for group in sorted({row["group"] for row in rows}):
        sub = [row for row in rows if row["group"] == group]
        per_dataset: dict[str, float] = {}
        for dataset in sorted({row["dataset"] for row in sub}):
            vals = [float(row["delta_pearson_pert"]) for row in sub if row["dataset"] == dataset]
            if vals:
                per_dataset[dataset] = mean(vals)
        summaries.append(
            {
                "group": group,
                "n_joined": len(sub),
                "datasets": len(per_dataset),
                "mean_delta_pearson_pert": mean(float(row["delta_pearson_pert"]) for row in sub) if sub else None,
                "dataset_min_delta_pearson_pert": min(per_dataset.values()) if per_dataset else None,
                "dataset_bootstrap_ci_low": bootstrap_ci_low(sub, "delta_pearson_pert"),
                "mean_delta_mmd_clamped": mean(float(row["delta_mmd_clamped"]) for row in sub) if sub else None,
                "max_dataset_delta_mmd_clamped": max(
                    mean(float(row["delta_mmd_clamped"]) for row in sub if row["dataset"] == dataset)
                    for dataset in per_dataset
                )
                if per_dataset
                else None,
            }
        )

    reasons: list[str] = []
    if len(summaries) < 2:
        reasons.append("missing_internal_groups")
    for summary in summaries:
        if int(summary["n_joined"] or 0) < 50:
            reasons.append(f"{summary['group']}:joined_rows_below_50")
        if float(summary["mean_delta_pearson_pert"] or 0.0) < 0.0:
            reasons.append(f"{summary['group']}:mean_delta_pp_negative")
        if float(summary["dataset_min_delta_pearson_pert"] or 0.0) < -0.02:
            reasons.append(f"{summary['group']}:dataset_min_delta_below_minus_0p02")
        ci = summary.get("dataset_bootstrap_ci_low")
        if ci is None or float(ci) < -0.005:
            reasons.append(f"{summary['group']}:dataset_bootstrap_ci_low_below_minus_0p005")
        if float(summary["mean_delta_mmd_clamped"] or 0.0) > 0.001:
            reasons.append(f"{summary['group']}:mean_mmd_delta_above_0p001")
    status = (
        "lookahead_trust_region_internal_eval_pass_needs_canonical_noharm"
        if not reasons
        else "lookahead_trust_region_internal_eval_fail_close_or_mutate"
    )

    out_dir = run_dir / "posthoc"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = out_dir / "internal_eval_vs_anchor_rows.csv"
    summary_csv = out_dir / "internal_eval_vs_anchor_summary.csv"
    summary_json = out_dir / "internal_eval_vs_anchor_summary.json"
    report_md = out_dir / "LATENTFM_LOOKAHEAD_TRUST_REGION_INTERNAL_EVAL_DECISION.md"
    write_csv(
        rows_csv,
        rows,
        [
            "group",
            "dataset",
            "condition",
            "anchor_pearson_pert",
            "candidate_pearson_pert",
            "delta_pearson_pert",
            "anchor_mmd_clamped",
            "candidate_mmd_clamped",
            "delta_mmd_clamped",
        ],
    )
    write_csv(
        summary_csv,
        summaries,
        [
            "group",
            "n_joined",
            "datasets",
            "mean_delta_pearson_pert",
            "dataset_min_delta_pearson_pert",
            "dataset_bootstrap_ci_low",
            "mean_delta_mmd_clamped",
            "max_dataset_delta_mmd_clamped",
        ],
    )
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": status,
        "reasons": reasons,
        "anchor_internal": str(ANCHOR_INTERNAL),
        "candidate_eval": str(eval_json),
        "rows": str(rows_csv),
        "summary": str(summary_csv),
        "summaries": summaries,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# LatentFM Lookahead Trust-Region Internal Eval Decision",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M %Z')}`",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "Posthoc comparison of the frozen smoke checkpoint internal eval against "
        "the frozen xverse_8k_anchor internal means. No canonical multi or Track C "
        "query is used for selection.",
        "",
        "## Summary",
        "",
        "| group | n | mean pp delta | dataset min | CI low | mean MMD delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| `{s['group']}` | `{s['n_joined']}` | `{s['mean_delta_pearson_pert']}` | "
            f"`{s['dataset_min_delta_pearson_pert']}` | `{s['dataset_bootstrap_ci_low']}` | "
            f"`{s['mean_delta_mmd_clamped']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Reasons: `{reasons}`",
            "",
            "If this passes, the next step is frozen-checkpoint canonical no-harm "
            "evaluation only. It is still not a final model claim.",
            "",
            "## Outputs",
            "",
            f"- Rows: `{rows_csv}`",
            f"- Summary: `{summary_csv}`",
            f"- JSON: `{summary_json}`",
            "",
        ]
    )
    report_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "reasons": reasons, "report": str(report_md)}, indent=2))
    return 0 if not reasons else 2


if __name__ == "__main__":
    raise SystemExit(main())
