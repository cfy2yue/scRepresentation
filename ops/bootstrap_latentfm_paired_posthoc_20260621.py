#!/usr/bin/env python3
"""Paired condition-level bootstrap for LatentFM posthoc JSON files.

Inputs are matched anchor/candidate eval JSON files produced by
``eval_split_groups.py`` or ``eval_condition_families.py``.  The bootstrap
resamples conditions within each dataset and then averages dataset means
equally, matching the evaluation aggregation contract recorded in those JSONs.
"""

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


def condition_table(payload: dict[str, Any], group: str) -> dict[tuple[str, str], dict[str, Any]]:
    g = payload.get("groups", {}).get(group, {})
    rows = g.get("condition_metrics") or []
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        ds = str(row.get("dataset", ""))
        cond = str(row.get("condition", ""))
        if ds and cond:
            out[(ds, cond)] = row
    return out


def selected_fingerprint(payload: dict[str, Any], group: str) -> tuple[str, ...]:
    g = payload.get("groups", {}).get(group, {})
    rows = g.get("selected_conditions") or []
    fp: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            fp.append(f"{row.get('dataset', '')}\t{row.get('condition', '')}")
    return tuple(sorted(fp))


def paired_rows(
    base: dict[str, Any],
    cand: dict[str, Any],
    group: str,
    metric: str,
) -> list[tuple[str, float]]:
    btab = condition_table(base, group)
    ctab = condition_table(cand, group)
    common = sorted(set(btab) & set(ctab))
    rows: list[tuple[str, float]] = []
    for key in common:
        b = fnum(btab[key].get(metric))
        c = fnum(ctab[key].get(metric))
        if b is None or c is None:
            continue
        rows.append((key[0], c - b))
    return rows


def equal_dataset_mean(rows: list[tuple[str, float]]) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for ds, val in rows:
        by_ds[ds].append(float(val))
    means = [float(np.mean(vals)) for vals in by_ds.values() if vals]
    if not means:
        return None
    return float(np.mean(means))


def bootstrap_equal_dataset(
    rows: list[tuple[str, float]],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    by_ds: dict[str, np.ndarray] = {}
    tmp: dict[str, list[float]] = defaultdict(list)
    for ds, val in rows:
        tmp[ds].append(float(val))
    for ds, vals in tmp.items():
        by_ds[ds] = np.asarray(vals, dtype=np.float64)
    samples = np.empty(n_boot, dtype=np.float64)
    datasets = sorted(by_ds)
    for i in range(n_boot):
        ds_means: list[float] = []
        for ds in datasets:
            vals = by_ds[ds]
            idx = rng.integers(0, len(vals), size=len(vals))
            ds_means.append(float(np.mean(vals[idx])))
        samples[i] = float(np.mean(ds_means))
    return samples


def summarize_metric(
    base: dict[str, Any],
    cand: dict[str, Any],
    group: str,
    metric: str,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    rows = paired_rows(base, cand, group, metric)
    selected_match = selected_fingerprint(base, group) == selected_fingerprint(cand, group)
    observed = equal_dataset_mean(rows)
    out: dict[str, Any] = {
        "group": group,
        "metric": metric,
        "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
        "selected_match": selected_match,
        "n_matched_conditions": len(rows),
        "n_matched_datasets": len({ds for ds, _ in rows}),
        "delta_mean": observed,
        "ci95_low": None,
        "ci95_high": None,
        "p_improvement": None,
        "p_harm": None,
    }
    if observed is None or len(rows) < 2 or len({ds for ds, _ in rows}) < 1:
        out["status"] = "insufficient_condition_metrics"
        return out
    boots = bootstrap_equal_dataset(rows, n_boot=n_boot, rng=rng)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    if metric in LOWER_IS_BETTER:
        improve = boots < 0.0
        harm = boots > 0.0
    else:
        improve = boots > 0.0
        harm = boots < 0.0
    out.update(
        {
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "p_improvement": float(np.mean(improve)),
            "p_harm": float(np.mean(harm)),
            "status": "ok",
        }
    )
    return out


def summarize_pair(
    *,
    base_path: Path,
    cand_path: Path,
    groups: list[str],
    metrics: list[str],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    base = load_json(base_path)
    cand = load_json(cand_path)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for group in groups:
        for metric in metrics:
            rows.append(summarize_metric(base, cand, group, metric, n_boot=n_boot, rng=rng))
    return {
        "baseline_json": str(base_path),
        "candidate_json": str(cand_path),
        "n_boot": n_boot,
        "seed": seed,
        "aggregation": "condition bootstrap within dataset, then equal-dataset mean",
        "rows": rows,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_md(payload: dict[str, Any], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"Baseline: `{payload['baseline_json']}`",
        f"Candidate: `{payload['candidate_json']}`",
        f"Bootstrap: `{payload['n_boot']}` resamples, seed `{payload['seed']}`",
        f"Aggregation: {payload['aggregation']}",
        "",
        "| group | metric | direction | selected match | n conds | n datasets | delta | 95% CI | p improve | p harm | status |",
        "|---|---|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in payload["rows"]:
        ci = f"[{fmt(row.get('ci95_low'))}, {fmt(row.get('ci95_high'))}]"
        lines.append(
            "| {group} | {metric} | {direction} | {selected} | {ncond} | {nds} | {delta} | {ci} | {pimp} | {pharm} | {status} |".format(
                group=row["group"],
                metric=row["metric"],
                direction=row["direction"],
                selected=fmt(row.get("selected_match")),
                ncond=row.get("n_matched_conditions", 0),
                nds=row.get("n_matched_datasets", 0),
                delta=fmt(row.get("delta_mean")),
                ci=ci,
                pimp=fmt(row.get("p_improvement")),
                pharm=fmt(row.get("p_harm")),
                status=row.get("status", "NA"),
            )
        )
    lines += [
        "",
        "Interpretation notes:",
        "- Positive deltas improve correlation metrics; negative deltas improve MMD metrics.",
        "- This is paired condition-level uncertainty for matched selected conditions, not a replacement for uncapped evaluation.",
        "- Rows with `selected match = no` should not be used as paired evidence.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", default=["test", "test_multi_unseen2", "family_gene"])
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title", default="LatentFM Paired Bootstrap Summary")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    payload = summarize_pair(
        base_path=args.baseline_json,
        cand_path=args.candidate_json,
        groups=list(args.groups),
        metrics=list(args.metrics),
        n_boot=int(args.n_boot),
        seed=int(args.seed),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload, args.title), encoding="utf-8")
    print(json.dumps({"out_md": str(args.out_md), "rows": len(payload["rows"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
