#!/usr/bin/env python3
"""Paired bootstrap CIs for LatentFM posthoc condition metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "pearson_pert",
    "pearson_ctrl",
    "direct_pearson",
    "test_mmd",
    "test_mmd_biased",
    "test_mmd_clamped",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _condition_rows(payload: dict[str, Any], group: str) -> dict[tuple[str, str], dict[str, Any]]:
    g = payload.get("groups", {}).get(group, {})
    rows = g.get("condition_metrics")
    if not isinstance(rows, list):
        raise ValueError(
            f"{group!r} in {payload.get('checkpoint', '<unknown>')} has no condition_metrics; "
            "rerun posthoc with the updated evaluator before bootstrap."
        )
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        ds = str(row.get("dataset") or "")
        cond = str(row.get("condition") or "")
        if ds and cond:
            out[(ds, cond)] = row
    return out


def _dataset_equal_mean(rows: list[dict[str, Any]], metric: str) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = row.get(metric)
        if val is None:
            continue
        try:
            by_ds[str(row["dataset"])].append(float(val))
        except (KeyError, TypeError, ValueError):
            continue
    ds_vals = [float(np.mean(vals)) for vals in by_ds.values() if vals]
    if not ds_vals:
        return None
    return float(np.mean(ds_vals))


def _paired_delta_ci(
    baseline_rows: dict[tuple[str, str], dict[str, Any]],
    run_rows: dict[tuple[str, str], dict[str, Any]],
    *,
    metric: str,
    seed: int,
    n_boot: int,
) -> dict[str, Any]:
    paired_keys = sorted(set(baseline_rows) & set(run_rows))
    n_pairable_before_metric = len(paired_keys)
    paired = []
    for key in paired_keys:
        b = baseline_rows[key].get(metric)
        r = run_rows[key].get(metric)
        if b is None or r is None:
            continue
        try:
            b_f = float(b)
            r_f = float(r)
        except (TypeError, ValueError):
            continue
        ds, cond = key
        paired.append({"dataset": ds, "condition": cond, "baseline": b_f, "run": r_f, "delta": r_f - b_f})
    if not paired:
        return {
            "metric": metric,
            "status": "missing_metric",
            "n_pairs": 0,
            "n_baseline_conditions": len(baseline_rows),
            "n_run_conditions": len(run_rows),
            "n_pairable_before_metric": n_pairable_before_metric,
            "lower_is_better": metric.startswith("test_mmd"),
        }

    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        by_ds[row["dataset"]].append(row)
    ds_names = sorted(by_ds)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sampled_ds = rng.choice(ds_names, size=len(ds_names), replace=True)
        ds_means = []
        for ds in sampled_ds:
            rows = by_ds[str(ds)]
            picks = rng.integers(0, len(rows), size=len(rows))
            ds_means.append(float(np.mean([rows[int(i)]["delta"] for i in picks])))
        boot.append(float(np.mean(ds_means)))

    baseline_mean = _dataset_equal_mean(
        [{"dataset": r["dataset"], metric: r["baseline"]} for r in paired],
        metric,
    )
    run_mean = _dataset_equal_mean(
        [{"dataset": r["dataset"], metric: r["run"]} for r in paired],
        metric,
    )
    delta = None if baseline_mean is None or run_mean is None else run_mean - baseline_mean
    lo, hi = np.quantile(np.asarray(boot, dtype=float), [0.025, 0.975])
    return {
        "metric": metric,
        "status": "ok",
        "n_pairs": len(paired),
        "n_baseline_conditions": len(baseline_rows),
        "n_run_conditions": len(run_rows),
        "n_pairable_before_metric": n_pairable_before_metric,
        "paired_coverage_vs_baseline": len(paired) / max(len(baseline_rows), 1),
        "paired_coverage_vs_run": len(paired) / max(len(run_rows), 1),
        "n_datasets": len(ds_names),
        "baseline_mean": baseline_mean,
        "run_mean": run_mean,
        "delta": delta,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "bootstrap_replicates": int(n_boot),
        "aggregation": "paired_condition_delta_then_dataset_stratified_bootstrap",
        "lower_is_better": metric.startswith("test_mmd"),
        "warning": (
            "paired_condition_coverage_below_80pct"
            if (
                len(paired) / max(len(baseline_rows), 1) < 0.8
                or len(paired) / max(len(run_rows), 1) < 0.8
            )
            else None
        ),
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# LatentFM Condition-Metric Bootstrap",
        "",
        f"Baseline: `{payload['baseline_json']}`",
        f"Run: `{payload['run_json']}`",
        "",
        "## Results",
        "",
        "| group | metric | direction | n pairs | n datasets | baseline | run | delta | 95% CI | status |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for group in payload["groups"]:
        for row in group["metrics"]:
            lines.append(
                "| {group} | {metric} | {direction} | {n_pairs} | {n_datasets} | {baseline} | {run} | {delta} | {ci} | {status} |".format(
                    group=group["group"],
                    metric=row["metric"],
                    direction="lower is better" if row.get("lower_is_better") else "higher is better",
                    n_pairs=row.get("n_pairs", 0),
                    n_datasets=row.get("n_datasets", 0),
                    baseline=_fmt(row.get("baseline_mean")),
                    run=_fmt(row.get("run_mean")),
                    delta=_fmt(row.get("delta")),
                    ci=(
                        f"[{_fmt(row.get('ci95_low'))}, {_fmt(row.get('ci95_high'))}]"
                        if row.get("status") == "ok"
                        else "NA"
                    ),
                    status=row.get("status"),
                )
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "This is paired on `(dataset, condition)` and bootstraps condition deltas "
            "within dataset strata. It requires posthoc JSONs produced by the updated "
            "LatentFM evaluator with `condition_metrics` present.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(x: Any) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.6f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-json", type=Path, required=True)
    ap.add_argument("--run-json", type=Path, required=True)
    ap.add_argument("--group", action="append", required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260619)
    args = ap.parse_args()
    if int(args.n_boot) <= 0:
        raise ValueError("--n-boot must be positive")

    baseline = _load(args.baseline_json)
    run = _load(args.run_json)
    groups = []
    for group in args.group:
        base_rows = _condition_rows(baseline, group)
        run_rows = _condition_rows(run, group)
        groups.append(
            {
                "group": group,
                "n_baseline_conditions": len(base_rows),
                "n_run_conditions": len(run_rows),
                "n_paired_conditions": len(set(base_rows) & set(run_rows)),
                "metrics": [
                    _paired_delta_ci(
                        base_rows,
                        run_rows,
                        metric=metric,
                        seed=int(args.seed) + i,
                        n_boot=int(args.n_boot),
                    )
                    for i, metric in enumerate(METRICS)
                ],
            }
        )

    out = {
        "baseline_json": str(args.baseline_json),
        "run_json": str(args.run_json),
        "seed": int(args.seed),
        "n_boot": int(args.n_boot),
        "groups": groups,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_md(args.out_md, out)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
