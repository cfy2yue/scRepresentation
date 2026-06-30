#!/usr/bin/env python3
"""Single-run condition bootstrap CIs for LatentFM posthoc JSON files."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_METRICS = (
    "pearson_pert",
    "pearson_ctrl",
    "direct_pearson",
    "test_mmd_clamped",
    "test_mmd_biased",
    "test_mmd",
)
LOWER_IS_BETTER = {"test_mmd_clamped", "test_mmd_biased", "test_mmd"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    try:
        if value is None:
            return None
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except (TypeError, ValueError):
        return None


def condition_rows(payload: dict[str, Any], group: str) -> list[dict[str, Any]]:
    if group == "__top_level__":
        rows = payload.get("condition_metrics") or []
    else:
        rows = (payload.get("groups", {}).get(group, {}) or {}).get("condition_metrics") or []
    return [r for r in rows if isinstance(r, dict) and r.get("dataset") and r.get("condition")]


def bootstrap_metric(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, Any]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = fnum(row.get(metric))
        if val is not None:
            by_ds[str(row["dataset"])].append(val)
    datasets = sorted(ds for ds, vals in by_ds.items() if vals)
    out: dict[str, Any] = {
        "metric": metric,
        "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
        "n_conditions": int(sum(len(by_ds[ds]) for ds in datasets)),
        "n_datasets": int(len(datasets)),
        "mean": None,
        "ci95_low": None,
        "ci95_high": None,
        "status": "ok",
    }
    if not datasets:
        out["status"] = "missing_metric"
        return out
    ds_means = [float(np.mean(by_ds[ds])) for ds in datasets]
    out["mean"] = float(np.mean(ds_means))
    if len(datasets) == 1 and out["n_conditions"] < 2:
        out["status"] = "insufficient_condition_metrics"
        return out
    samples = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        boot_ds = rng.choice(datasets, size=len(datasets), replace=True)
        vals = []
        for ds in boot_ds:
            arr = np.asarray(by_ds[str(ds)], dtype=np.float64)
            idx = rng.integers(0, len(arr), size=len(arr))
            vals.append(float(np.mean(arr[idx])))
        samples[i] = float(np.mean(vals))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    out["ci95_low"] = float(lo)
    out["ci95_high"] = float(hi)
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_md(payload: dict[str, Any], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"Eval JSON: `{payload['eval_json']}`",
        f"Bootstrap: `{payload['n_boot']}` resamples, seed `{payload['seed']}`",
        f"Aggregation: {payload['aggregation']}",
        "",
        "| group | metric | direction | n conds | n datasets | mean | 95% CI | status |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for group in payload["groups"]:
        for row in group["metrics"]:
            ci = f"[{fmt(row.get('ci95_low'))}, {fmt(row.get('ci95_high'))}]"
            lines.append(
                "| {group} | {metric} | {direction} | {ncond} | {nds} | {mean} | {ci} | {status} |".format(
                    group=group["group"],
                    metric=row["metric"],
                    direction=row["direction"],
                    ncond=row.get("n_conditions", 0),
                    nds=row.get("n_datasets", 0),
                    mean=fmt(row.get("mean")),
                    ci=ci,
                    status=row.get("status", "NA"),
                )
            )
    lines += [
        "",
        "Interpretation notes:",
        "- This is a single-run condition bootstrap CI, not a paired model comparison.",
        "- Conditions are resampled within each dataset and dataset means are averaged equally.",
        "- Use paired bootstrap for same-latent checkpoint deltas.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", default=["test", "test_multi_unseen2"])
    parser.add_argument("--top-level-group-name", default="")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", default="LatentFM Single-Run Bootstrap CI")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    payload = load_json(args.eval_json)
    rng = np.random.default_rng(int(args.seed))
    groups = []
    if args.top_level_group_name:
        rows = condition_rows(payload, "__top_level__")
        groups.append(
            {
                "group": str(args.top_level_group_name),
                "metrics": [
                    bootstrap_metric(rows, metric, rng=rng, n_boot=int(args.n_boot))
                    for metric in args.metrics
                ],
            }
        )
    for group in args.groups:
        if str(group).strip().lower() in {"", "none", "__none__", "skip"}:
            continue
        rows = condition_rows(payload, group)
        groups.append(
            {
                "group": group,
                "metrics": [
                    bootstrap_metric(rows, metric, rng=rng, n_boot=int(args.n_boot))
                    for metric in args.metrics
                ],
            }
        )

    out = {
        "eval_json": str(args.eval_json),
        "seed": int(args.seed),
        "n_boot": int(args.n_boot),
        "aggregation": "condition bootstrap within dataset, then equal-dataset mean",
        "groups": groups,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(out, args.title), encoding="utf-8")
    print(json.dumps({"out_md": str(args.out_md), "groups": len(groups)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
