#!/usr/bin/env python3
"""Paired bootstrap for deployable LatentFM condition routes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


HIGHER_IS_BETTER = {"pearson_pert", "pearson_ctrl", "direct_pearson"}
LOWER_IS_BETTER = {"test_mmd_clamped"}


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def fnum(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def routes() -> dict[str, Callable[[dict[str, Any]], bool]]:
    return {
        "candidate_gene_multi": lambda r: truthy(r["is_gene"]) and truthy(r["is_multi"]),
        "candidate_multi_not_drug": lambda r: truthy(r["is_multi"]) and not truthy(r["is_drug"]),
        "candidate_multi_only": lambda r: truthy(r["is_multi"]),
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def equal_dataset_delta(rows: list[dict[str, Any]], route_name: str, metric: str) -> float | None:
    pred = routes()[route_name]
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        base = fnum(row.get(f"{metric}_base"))
        cand = fnum(row.get(f"{metric}_candidate"))
        if base is None or cand is None:
            continue
        routed = cand if pred(row) else base
        by_dataset[str(row["dataset"])].append(routed - base)
    means = [float(np.mean(values)) for values in by_dataset.values() if values]
    if not means:
        return None
    return float(np.mean(means))


def bootstrap_delta(
    rows: list[dict[str, Any]],
    route_name: str,
    metric: str,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, list[float]]:
    pred = routes()[route_name]
    by_dataset: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        base = fnum(row.get(f"{metric}_base"))
        cand = fnum(row.get(f"{metric}_candidate"))
        if base is None or cand is None:
            continue
        routed = cand if pred(row) else base
        by_dataset[str(row["dataset"])].append((base, routed))
    datasets = sorted(ds for ds, values in by_dataset.items() if values)
    if not datasets:
        return float("nan"), []
    observed = equal_dataset_delta(rows, route_name, metric)
    samples: list[float] = []
    for _ in range(n_boot):
        ds_means: list[float] = []
        for dataset in datasets:
            values = by_dataset[dataset]
            idx = rng.integers(0, len(values), size=len(values))
            deltas = [values[int(i)][1] - values[int(i)][0] for i in idx]
            ds_means.append(float(np.mean(deltas)))
        samples.append(float(np.mean(ds_means)))
    return float(observed), samples


def summarize(
    rows: list[dict[str, Any]],
    comparisons: list[str],
    route_names: list[str],
    groups: list[str],
    metrics: list[str],
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for comparison in comparisons:
        comp_rows = [r for r in rows if r.get("comparison") == comparison and truthy(r.get("selected_membership_match"))]
        for route_name in route_names:
            for group in groups:
                group_rows = [r for r in comp_rows if r.get("eval_group") == group]
                if not group_rows:
                    continue
                candidate_n = sum(1 for r in group_rows if routes()[route_name](r))
                for metric in metrics:
                    observed, samples = bootstrap_delta(group_rows, route_name, metric, n_boot, rng)
                    if not samples:
                        continue
                    lo, hi = np.quantile(samples, [0.025, 0.975])
                    if metric in HIGHER_IS_BETTER:
                        p_improve = float(np.mean(np.asarray(samples) > 0.0))
                        p_harm = float(np.mean(np.asarray(samples) < 0.0))
                        direction = "higher_is_better"
                    elif metric in LOWER_IS_BETTER:
                        p_improve = float(np.mean(np.asarray(samples) < 0.0))
                        p_harm = float(np.mean(np.asarray(samples) > 0.0))
                        direction = "lower_is_better"
                    else:
                        p_improve = float("nan")
                        p_harm = float("nan")
                        direction = "unknown"
                    out.append(
                        {
                            "comparison": comparison,
                            "route": route_name,
                            "group": group,
                            "metric": metric,
                            "direction": direction,
                            "n_conditions": len(group_rows),
                            "n_datasets": len({r["dataset"] for r in group_rows}),
                            "candidate_conditions": candidate_n,
                            "delta": observed,
                            "ci95": [float(lo), float(hi)],
                            "p_improve": p_improve,
                            "p_harm": p_harm,
                        }
                    )
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Route Paired Bootstrap",
        "",
        f"Condition CSV: `{payload['condition_csv']}`",
        f"Bootstrap: `{payload['n_boot']}` resamples, seed `{payload['seed']}`",
        "",
        "| comparison | route | group | metric | n | cand n | delta | 95% CI | p improve | p harm |",
        "|---|---|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {comparison} | {route} | {group} | {metric} | {n} | {cn} | {delta} | [{lo}, {hi}] | {pi:.4f} | {ph:.4f} |".format(
                comparison=row["comparison"],
                route=row["route"],
                group=row["group"],
                metric=row["metric"],
                n=row["n_conditions"],
                cn=row["candidate_conditions"],
                delta=fmt(row["delta"]),
                lo=fmt(row["ci95"][0]),
                hi=fmt(row["ci95"][1]),
                pi=row["p_improve"],
                ph=row["p_harm"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Positive deltas improve correlation metrics.",
            "- Negative deltas improve `test_mmd_clamped`.",
            "- This bootstraps routed-vs-anchor deltas by resampling conditions within dataset and averaging datasets equally.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-csv", type=Path, required=True)
    parser.add_argument("--comparisons", nargs="+", required=True)
    parser.add_argument("--routes", nargs="+", default=["candidate_gene_multi"])
    parser.add_argument("--groups", nargs="+", default=["test", "test_multi_unseen2", "family_gene", "family_drug", "structure_single"])
    parser.add_argument("--metrics", nargs="+", default=["pearson_pert", "test_mmd_clamped"])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    rows = summarize(
        load_rows(args.condition_csv),
        args.comparisons,
        args.routes,
        args.groups,
        args.metrics,
        args.n_boot,
        args.seed,
    )
    payload = {
        "condition_csv": str(args.condition_csv),
        "comparisons": args.comparisons,
        "routes": args.routes,
        "groups": args.groups,
        "metrics": args.metrics,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
