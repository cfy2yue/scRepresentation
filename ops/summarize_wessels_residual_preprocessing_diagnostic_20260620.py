#!/usr/bin/env python3
"""CPU-only Wessels residual/preprocessing diagnostic from existing artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


GROUPS = ("test_multi_seen", "test_multi_unseen1", "test_multi_unseen2")


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(value: Any) -> str:
    value = fnum(value)
    return "NA" if value is None else f"{value:.6f}"


def load_condition_metrics(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for group in GROUPS:
        for row in (payload.get("groups", {}).get(group, {}) or {}).get("condition_metrics", []) or []:
            if not isinstance(row, dict):
                continue
            key = (str(row.get("dataset") or ""), str(row.get("condition") or ""))
            out[key] = {"eval_group": group, **row}
    return out


def load_prior_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("adapter") and row["adapter"] != "raw_global":
                continue
            key = ("Wessels", str(row.get("condition") or ""))
            out[key] = row
    return out


def bin3(value: float | None, cuts: tuple[float, float] | None) -> str:
    if value is None or cuts is None:
        return "NA"
    lo, hi = cuts
    if value <= lo:
        return "low"
    if value <= hi:
        return "mid"
    return "high"


def tertile_cuts(values: list[float | None]) -> tuple[float, float] | None:
    vals = sorted(float(v) for v in values if v is not None)
    if len(vals) < 3:
        return None
    return vals[len(vals) // 3], vals[(2 * len(vals)) // 3]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Wessels Residual/Preprocessing Diagnostic",
        "",
        "This CPU-only diagnostic joins Wessels latest baseline condition metrics with train-only additive-prior norm/statistics.",
        "",
        "## Group Summary",
        "",
        "| group | n | mean pp | mean pc | mean MMD | target norm | additive norm ratio | additive Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["group_summary"]:
        lines.append(
            f"| `{row['group']}` | {row['n']} | {fmt(row['mean_pearson_pert'])} | {fmt(row['mean_pearson_ctrl'])} | "
            f"{fmt(row['mean_mmd'])} | {fmt(row['mean_target_norm'])} | {fmt(row['mean_prior_norm_ratio'])} | "
            f"{fmt(row['mean_prior_pearson'])} |"
        )
    lines.extend(
        [
            "",
            "## Unseen2 Bins",
            "",
            "| bin field | bin | n | mean pp | mean MMD | target norm | additive norm ratio | additive Pearson |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["unseen2_bin_summary"]:
        lines.append(
            f"| `{row['bin_field']}` | `{row['bin']}` | {row['n']} | {fmt(row['mean_pearson_pert'])} | "
            f"{fmt(row['mean_mmd'])} | {fmt(row['mean_target_norm'])} | "
            f"{fmt(row['mean_prior_norm_ratio'])} | {fmt(row['mean_prior_pearson'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Status: `{payload['decision']['status']}`",
            "",
            f"Next action: `{payload['decision']['next_action']}`",
            "",
            f"Reason: {payload['decision']['reason']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-latest-json",
        type=Path,
        default=Path(
            "/data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/dataset_upper_bound_20260620/"
            "scf_prior010_upperbound_wessels_4k/posthoc_eval_latest_global_prior/"
            "split_group_eval_latest_ode20_mse1024_mmd1024_stablecaps.json"
        ),
    )
    parser.add_argument(
        "--prior-conditions-csv",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/latentfm_wessels_context_prior_adapter_conditions_20260620.csv"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/latentfm_wessels_residual_preprocessing_diagnostic_20260620.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/LATENTFM_WESSELS_RESIDUAL_PREPROCESSING_DIAGNOSTIC_20260620.md"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/latentfm_wessels_residual_preprocessing_conditions_20260620.csv"),
    )
    args = parser.parse_args()

    metrics = load_condition_metrics(args.baseline_latest_json)
    prior = load_prior_rows(args.prior_conditions_csv)
    rows: list[dict[str, Any]] = []
    for key, metric in metrics.items():
        prow = prior.get(key, {})
        row = {
            "dataset": key[0],
            "condition": key[1],
            "group": metric.get("eval_group"),
            "pearson_pert": fnum(metric.get("pearson_pert")),
            "pearson_ctrl": fnum(metric.get("pearson_ctrl")),
            "test_mmd": fnum(metric.get("test_mmd_clamped", metric.get("test_mmd"))),
            "n_src_eval": fnum(metric.get("n_src_eval")),
            "n_gt_eval": fnum(metric.get("n_gt_eval")),
            "n_genes": fnum(prow.get("n_genes")),
            "target_norm": fnum(prow.get("target_norm")),
            "prior_norm_ratio": fnum(prow.get("norm_ratio")),
            "prior_pearson": fnum(prow.get("pearson")),
            "prior_train_condition_count_sum": fnum(prow.get("prior_train_condition_count_sum")),
        }
        rows.append(row)
    group_summary = []
    for group in GROUPS:
        subset = [row for row in rows if row["group"] == group]
        group_summary.append(
            {
                "group": group,
                "n": len(subset),
                "mean_pearson_pert": mean([row["pearson_pert"] for row in subset]),
                "mean_pearson_ctrl": mean([row["pearson_ctrl"] for row in subset]),
                "mean_mmd": mean([row["test_mmd"] for row in subset]),
                "mean_target_norm": mean([row["target_norm"] for row in subset]),
                "mean_prior_norm_ratio": mean([row["prior_norm_ratio"] for row in subset]),
                "mean_prior_pearson": mean([row["prior_pearson"] for row in subset]),
            }
        )
    unseen2 = [row for row in rows if row["group"] == "test_multi_unseen2"]
    cuts = {
        "target_norm": tertile_cuts([row["target_norm"] for row in unseen2]),
        "prior_norm_ratio": tertile_cuts([row["prior_norm_ratio"] for row in unseen2]),
        "prior_pearson": tertile_cuts([row["prior_pearson"] for row in unseen2]),
    }
    bin_rows = []
    for field, field_cuts in cuts.items():
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in unseen2:
            buckets[bin3(row[field], field_cuts)].append(row)
        for label in ("low", "mid", "high", "NA"):
            subset = buckets.get(label, [])
            if not subset:
                continue
            bin_rows.append(
                {
                    "bin_field": field,
                    "bin": label,
                    "n": len(subset),
                    "mean_pearson_pert": mean([row["pearson_pert"] for row in subset]),
                    "mean_mmd": mean([row["test_mmd"] for row in subset]),
                    "mean_target_norm": mean([row["target_norm"] for row in subset]),
                    "mean_prior_norm_ratio": mean([row["prior_norm_ratio"] for row in subset]),
                    "mean_prior_pearson": mean([row["prior_pearson"] for row in subset]),
                }
            )
    unseen2_pp_by_target = [row for row in bin_rows if row["bin_field"] == "target_norm"]
    spread = None
    if len(unseen2_pp_by_target) >= 2:
        vals = [row["mean_pearson_pert"] for row in unseen2_pp_by_target if row["mean_pearson_pert"] is not None]
        spread = max(vals) - min(vals) if vals else None
    decision = {
        "status": "diagnostic_only_no_simple_norm_rescue",
        "next_action": "prioritize_latent_or_unsupervised_combo_sensitivity_before_more_wessels_prior_training",
        "reason": (
            "condition-level bins can guide failure analysis, but no train multi supervision "
            "or additive/context prior gate justifies more Wessels-specific prior GPU runs"
        ),
        "unseen2_target_norm_pp_spread": spread,
    }
    payload = {
        "baseline_latest_json": str(args.baseline_latest_json),
        "prior_conditions_csv": str(args.prior_conditions_csv),
        "group_summary": group_summary,
        "unseen2_bin_summary": bin_rows,
        "decision": decision,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.out_csv, rows)
    write_md(args.out_md, payload)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "decision": decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
