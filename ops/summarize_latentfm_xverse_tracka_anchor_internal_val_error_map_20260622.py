#!/usr/bin/env python3
"""Summarize xverse anchor internal-val predictions against train-only baselines."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_ANCHOR_EVAL = (
    ROOT
    / "runs/latentfm_xverse_tracka_anchor_internal_val_error_map_20260622/"
    "anchor_internal_val_split_eval.json"
)
DEFAULT_BASELINE_JSON = ROOT / "reports/latentfm_xverse_gene_reliability_router_gate_20260622.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_tracka_anchor_internal_val_error_map_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_TRACKA_ANCHOR_INTERNAL_VAL_ERROR_MAP_20260622.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
BASELINES = ("gene_raw_mean", "dataset_mean", "global_mean", "shrink_k8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def equal_dataset_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = row.get(key)
        if val is not None:
            by_ds[str(row["dataset"])].append(float(val))
    vals = [float(np.mean(v)) for v in by_ds.values() if v]
    return None if not vals else float(np.mean(vals))


def paired_bootstrap(
    rows: list[dict[str, Any]],
    candidate: str,
    baseline: str,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    diffs_by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        a = row.get(candidate)
        b = row.get(baseline)
        if a is not None and b is not None:
            diffs_by_ds[str(row["dataset"])].append(float(a) - float(b))
    datasets = sorted(ds for ds, vals in diffs_by_ds.items() if vals)
    if not datasets:
        return {"status": "missing", "candidate": candidate, "baseline": baseline}
    point = float(np.mean([np.mean(diffs_by_ds[ds]) for ds in datasets]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sample_ds = rng.choice(datasets, size=len(datasets), replace=True)
        means = []
        for ds in sample_ds:
            vals = np.asarray(diffs_by_ds[str(ds)], dtype=np.float64)
            means.append(float(np.mean(rng.choice(vals, size=len(vals), replace=True))))
        boot.append(float(np.mean(means)))
    arr = np.asarray(boot, dtype=np.float64)
    by_dataset = {ds: float(np.mean(vals)) for ds, vals in diffs_by_ds.items()}
    leave_one = {}
    for ds in datasets:
        rest = [d for d in datasets if d != ds]
        if rest:
            leave_one[ds] = float(np.mean([np.mean(diffs_by_ds[d]) for d in rest]))
    return {
        "status": "ok",
        "candidate": candidate,
        "baseline": baseline,
        "n_conditions": int(sum(len(diffs_by_ds[ds]) for ds in datasets)),
        "n_datasets": int(len(datasets)),
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": float(np.mean(arr > 0.0)),
        "p_harm": float(np.mean(arr < 0.0)),
        "by_dataset": by_dataset,
        "leave_one_min": None if not leave_one else float(min(leave_one.values())),
    }


def bucket_count(n: int) -> str:
    if n <= 1:
        return "count_0_1"
    if n <= 4:
        return "count_2_4"
    if n <= 9:
        return "count_5_9"
    return "count_ge10"


def build_rows(anchor_eval: dict[str, Any], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_by_key = {
        (str(r["group"]), str(r["dataset"]), str(r["condition"])): r
        for r in baseline.get("val_condition_rows", [])
    }
    rows = []
    for group in GROUPS:
        grp = (anchor_eval.get("groups") or {}).get(group) or {}
        for metric in grp.get("condition_metrics") or []:
            key = (group, str(metric["dataset"]), str(metric["condition"]))
            base = baseline_by_key.get(key)
            if base is None:
                continue
            item = {
                "group": group,
                "dataset": str(metric["dataset"]),
                "condition": str(metric["condition"]),
                "gene": str(base.get("gene", "")),
                "gene_train_count": int(base.get("gene_train_count", 0)),
                "gene_count_bucket": bucket_count(int(base.get("gene_train_count", 0))),
                "anchor_pearson_pert": metric.get("pearson_pert"),
                "anchor_pearson_ctrl": metric.get("pearson_ctrl"),
                "anchor_mmd_clamped": metric.get("test_mmd_clamped"),
            }
            for name in BASELINES:
                item[name] = base.get(name)
            rows.append(item)
    return rows


def grouped_summary(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, vals in sorted(groups.items()):
        item = {k: v for k, v in zip(keys, key)}
        item["n_conditions"] = len(vals)
        item["anchor_pearson_pert"] = float(np.mean([v["anchor_pearson_pert"] for v in vals if v.get("anchor_pearson_pert") is not None]))
        item["anchor_mmd_clamped"] = float(np.mean([v["anchor_mmd_clamped"] for v in vals if v.get("anchor_mmd_clamped") is not None]))
        for base in BASELINES:
            vals_base = [v[base] for v in vals if v.get(base) is not None]
            item[base] = None if not vals_base else float(np.mean(vals_base))
            item[f"anchor_minus_{base}"] = None if not vals_base else float(
                np.mean([float(v["anchor_pearson_pert"]) - float(v[base]) for v in vals if v.get("anchor_pearson_pert") is not None and v.get(base) is not None])
            )
        out.append(item)
    return out


def condition_extremes(rows: list[dict[str, Any]], n: int) -> dict[str, list[dict[str, Any]]]:
    scored = []
    for row in rows:
        if row.get("anchor_pearson_pert") is None or row.get("gene_raw_mean") is None:
            continue
        item = dict(row)
        item["anchor_minus_gene_raw_mean"] = float(row["anchor_pearson_pert"]) - float(row["gene_raw_mean"])
        item["anchor_minus_dataset_mean"] = float(row["anchor_pearson_pert"]) - float(row["dataset_mean"])
        scored.append(item)
    return {
        "worst_vs_gene_raw_mean": sorted(scored, key=lambda r: r["anchor_minus_gene_raw_mean"])[:n],
        "best_vs_gene_raw_mean": sorted(scored, key=lambda r: r["anchor_minus_gene_raw_mean"], reverse=True)[:n],
        "worst_vs_dataset_mean": sorted(scored, key=lambda r: r["anchor_minus_dataset_mean"])[:n],
    }


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    deltas = {
        (row["group"], row["baseline"]): row
        for row in payload["paired_deltas"]
        if row.get("candidate") == "anchor_pearson_pert" and row.get("status") == "ok"
    }
    for group in GROUPS:
        for baseline in ("gene_raw_mean", "dataset_mean"):
            row = deltas.get((group, baseline), {})
            if row.get("status") != "ok":
                reasons.append(f"{group}_{baseline}_comparison_missing")
                continue
            if float(row.get("delta_mean") or 0.0) < 0.02:
                reasons.append(f"{group}_anchor_not_0p02_better_than_{baseline}")
            if float(row.get("p_harm") if row.get("p_harm") is not None else 1.0) > 0.20:
                reasons.append(f"{group}_anchor_harm_risk_vs_{baseline}")
            for ds, value in (row.get("by_dataset") or {}).items():
                if float(value) < -0.02:
                    reasons.append(f"{group}_{ds}_anchor_delta_below_minus_0p02_vs_{baseline}")
                    break
    status = "anchor_internal_val_map_supports_new_mechanism_search" if not reasons else "anchor_internal_val_map_no_gpu_mechanism"
    action = (
        "derive_train_only_mechanism_before_any_gpu_smoke"
        if not reasons
        else "do_not_launch_gpu_from_anchor_error_map"
    )
    return {"status": status, "action": action, "reasons": reasons}


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Track A Anchor Internal-Val Error Map",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
        "",
        "## Provenance",
        "",
        f"- anchor eval JSON: `{payload['anchor_eval_json']}`",
        f"- baseline JSON: `{payload['baseline_json']}`",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- used_ema: `{payload['used_ema']}`",
        f"- condition rows matched: `{payload['n_rows']}`",
        "- canonical posthoc and held-out multi query are not used for this audit.",
        "",
        "## Absolute Scores",
        "",
        "| group | model | equal-dataset pp/MMD |",
        "|---|---|---:|",
    ]
    for group in GROUPS:
        group_rows = [r for r in payload["condition_rows"] if r["group"] == group]
        lines.append(f"| {group} | `anchor_pearson_pert` | {fmt(equal_dataset_mean(group_rows, 'anchor_pearson_pert'))} |")
        lines.append(f"| {group} | `anchor_mmd_clamped` | {fmt(equal_dataset_mean(group_rows, 'anchor_mmd_clamped'))} |")
        for base in BASELINES:
            lines.append(f"| {group} | `{base}` | {fmt(equal_dataset_mean(group_rows, base))} |")
    lines += [
        "",
        "## Paired Anchor Minus Baseline Deltas",
        "",
        "| group | baseline | n cond | n ds | delta | 95% CI | p improve | p harm | min leave-one-dataset |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in payload["paired_deltas"]:
        ci = row.get("ci95") or [None, None]
        lines.append(
            f"| {row['group']} | `{row['baseline']}` | {row.get('n_conditions', 0)} | "
            f"{row.get('n_datasets', 0)} | {fmt(row.get('delta_mean'))} | "
            f"[{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(row.get('p_improve'))} | "
            f"{fmt(row.get('p_harm'))} | {fmt(row.get('leave_one_min'))} |"
        )
    lines += [
        "",
        "## Dataset Summary",
        "",
        "| group | dataset | n | anchor pp | anchor MMD | gene_raw | dataset_mean | anchor-gene | anchor-dataset |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["dataset_summary"]:
        lines.append(
            f"| {row['group']} | {row['dataset']} | {row['n_conditions']} | "
            f"{fmt(row.get('anchor_pearson_pert'))} | {fmt(row.get('anchor_mmd_clamped'))} | "
            f"{fmt(row.get('gene_raw_mean'))} | {fmt(row.get('dataset_mean'))} | "
            f"{fmt(row.get('anchor_minus_gene_raw_mean'))} | {fmt(row.get('anchor_minus_dataset_mean'))} |"
        )
    lines += ["", "## Gate Reasons", ""]
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{reason}`" for reason in reasons] if reasons else ["- none"])
    lines += [
        "",
        "## Interpretation Rule",
        "",
        "- This audit can identify anchor weaknesses but does not itself train or select a checkpoint.",
        "- If the gate fails, do not launch a GPU smoke from this error map.",
        "- If the gate passes, a separate train-only mechanism must still be specified before any GPU smoke.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-eval-json", type=Path, default=DEFAULT_ANCHOR_EVAL)
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-extreme", type=int, default=12)
    args = parser.parse_args()

    anchor_eval = load_json(args.anchor_eval_json)
    baseline = load_json(args.baseline_json)
    rows = build_rows(anchor_eval, baseline)
    paired = []
    for i, group in enumerate(GROUPS):
        group_rows = [r for r in rows if r["group"] == group]
        for j, base in enumerate(BASELINES):
            item = paired_bootstrap(group_rows, "anchor_pearson_pert", base, n_boot=args.n_boot, seed=args.seed + i * 10 + j)
            item["group"] = group
            paired.append(item)
    payload = {
        "anchor_eval_json": str(args.anchor_eval_json),
        "baseline_json": str(args.baseline_json),
        "checkpoint": anchor_eval.get("checkpoint"),
        "split_file": anchor_eval.get("split_file"),
        "used_ema": anchor_eval.get("used_ema"),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_rows": len(rows),
        "condition_rows": rows,
        "paired_deltas": paired,
        "dataset_summary": grouped_summary(rows, ("group", "dataset")),
        "gene_count_summary": grouped_summary(rows, ("group", "gene_count_bucket")),
        "condition_extremes": condition_extremes(rows, args.n_extreme),
    }
    payload["decision"] = decide(payload)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
