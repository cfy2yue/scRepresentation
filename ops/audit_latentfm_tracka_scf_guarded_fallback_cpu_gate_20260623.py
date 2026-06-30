#!/usr/bin/env python3
"""CPU gate for a Jiang-safe scFoundation Track A guarded fallback.

This script uses only the train-only scFoundation gene-reliability CPU gate.
It does not read canonical test metrics, canonical multi, Track C query, model
posthoc predictions, or held-out outcomes.  The tested policy is deliberately
simple and predeclared.  Two policies are supported:

* dataset_negative: use shrink_k2 except on internal-proxy datasets where
  shrink_k2 does not beat dataset_mean by a small margin.
* jiang_lowcount: use shrink_k2 except on Jiang_IFNG/Jiang_TNFA rows with
  gene_train_count <= a fixed threshold.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_CPU_GATE = ROOT / "reports/latentfm_crosslatent_scfoundation_gene_reliability_router_gate_20260622.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_tracka_scf_guarded_fallback_cpu_gate_20260623.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_TRACKA_SCF_GUARDED_FALLBACK_CPU_GATE_20260623.md"
LOWCOUNT_OUT_JSON = ROOT / "reports/latentfm_tracka_scf_jiang_lowcount_mask_cpu_gate_20260623.json"
LOWCOUNT_OUT_MD = ROOT / "reports/LATENTFM_TRACKA_SCF_JIANG_LOWCOUNT_MASK_CPU_GATE_20260623.md"
GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
FOCUS_DATASETS = ("Jiang_IFNG", "Jiang_TNFA")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def group_rows(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("group") == group]


def dataset_means(rows: list[dict[str, Any]], candidate: str, baseline: str) -> dict[str, dict[str, Any]]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        a = row.get(candidate)
        b = row.get(baseline)
        if a is not None and b is not None:
            by_ds[str(row["dataset"])].append(float(a) - float(b))
    out = {}
    for ds, vals in by_ds.items():
        out[ds] = {"n": len(vals), "delta": float(np.mean(vals))}
    return out


def apply_policy(
    rows: list[dict[str, Any]],
    fallback_datasets: set[str],
    *,
    policy: str,
    lowcount_threshold: int,
    base_model: str,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        ds = str(row["dataset"])
        if policy == "dataset_negative":
            use_fallback = ds in fallback_datasets
        elif policy == "jiang_lowcount":
            use_fallback = ds in FOCUS_DATASETS and int(row.get("gene_train_count") or 0) <= int(lowcount_threshold)
        else:
            raise ValueError(f"unknown policy: {policy}")
        item["guarded_fallback"] = float(row["dataset_mean"] if use_fallback else row[base_model])
        item["fallback_used"] = use_fallback
        out.append(item)
    return out


def equal_dataset_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(key) is not None:
            by_ds[str(row["dataset"])].append(float(row[key]))
    vals = [float(np.mean(v)) for v in by_ds.values() if v]
    return None if not vals else float(np.mean(vals))


def paired_bootstrap(rows: list[dict[str, Any]], candidate: str, baseline: str, *, n_boot: int, seed: int) -> dict[str, Any]:
    diffs_by_ds: dict[str, list[float]] = defaultdict(list)
    better = 0
    total = 0
    for row in rows:
        a = row.get(candidate)
        b = row.get(baseline)
        if a is not None and b is not None:
            diff = float(a) - float(b)
            diffs_by_ds[str(row["dataset"])].append(diff)
            better += int(diff > 0.0)
            total += 1
    datasets = sorted(ds for ds, vals in diffs_by_ds.items() if vals)
    if not datasets:
        return {"status": "missing", "candidate": candidate, "baseline": baseline}
    ds_means = [float(np.mean(diffs_by_ds[ds])) for ds in datasets]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sample_ds = rng.choice(datasets, size=len(datasets), replace=True)
        means = []
        for ds in sample_ds:
            vals = np.asarray(diffs_by_ds[str(ds)], dtype=np.float64)
            means.append(float(np.mean(rng.choice(vals, size=len(vals), replace=True))))
        boot.append(float(np.mean(means)))
    arr = np.asarray(boot)
    leave = {}
    for ds in datasets:
        rest = [d for d in datasets if d != ds]
        if rest:
            leave[ds] = float(np.mean([np.mean(diffs_by_ds[d]) for d in rest]))
    return {
        "status": "ok",
        "candidate": candidate,
        "baseline": baseline,
        "n_conditions": int(sum(len(diffs_by_ds[d]) for d in datasets)),
        "n_datasets": int(len(datasets)),
        "delta_mean": float(np.mean(ds_means)),
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": float(np.mean(arr > 0.0)),
        "p_harm": float(np.mean(arr < 0.0)),
        "better_fraction": None if total == 0 else float(better / total),
        "median_dataset_delta": float(np.median(ds_means)),
        "leave_one_min": min(leave.values()) if leave else None,
    }


def decide(
    group_summaries: list[dict[str, Any]],
    margin: float,
    *,
    policy: str,
    lowcount_threshold: int,
    base_model: str,
    status_prefix: str,
) -> dict[str, Any]:
    reasons = []
    by_group = {row["group"]: row for row in group_summaries}
    for group in GROUPS:
        row = by_group.get(group)
        if row is None:
            reasons.append(f"{group}_missing")
            continue
        fallback = set(row["fallback_datasets"])
        if policy == "dataset_negative":
            for ds in FOCUS_DATASETS:
                if ds not in fallback:
                    reasons.append(f"{group}_{ds}_not_guarded")
        elif policy == "jiang_lowcount":
            for ds in FOCUS_DATASETS:
                if int(row.get("fallback_counts_by_dataset", {}).get(ds, 0)) <= 0:
                    reasons.append(f"{group}_{ds}_has_no_lowcount_guarded_rows")
        paired = {(r["candidate"], r["baseline"]): r for r in row["paired_deltas"]}

        def require(baseline: str, min_delta: float, max_harm: float = 0.20) -> None:
            item = paired.get(("guarded_fallback", baseline)) or {}
            if item.get("status") != "ok":
                reasons.append(f"{group}_vs_{baseline}_missing")
                return
            ci = item.get("ci95") or [0.0, 0.0]
            if float(item.get("delta_mean") or 0.0) < min_delta and float(ci[0]) <= 0.0:
                reasons.append(f"{group}_vs_{baseline}_delta_gate_fail")
            if float(item.get("p_harm") if item.get("p_harm") is not None else 1.0) > max_harm:
                reasons.append(f"{group}_vs_{baseline}_harm_risk")
            if item.get("leave_one_min") is None or float(item["leave_one_min"]) < -0.02:
                reasons.append(f"{group}_vs_{baseline}_leave_one_below_minus_002")

        require("dataset_mean", 0.02)
        require("gene_raw_mean", 0.02)
        require(base_model, 0.001, max_harm=0.05)
    status = f"{status_prefix}_guarded_fallback_cpu_gate_pass_no_gpu_yet" if not reasons else f"{status_prefix}_guarded_fallback_cpu_gate_fail_no_gpu"
    policy_desc = (
        f"fallback_to_dataset_mean_when_dataset_shrink_minus_dataset_mean_lt_{margin}"
        if policy == "dataset_negative"
        else f"fallback_to_dataset_mean_for_jiang_gene_train_count_le_{lowcount_threshold}"
    )
    return {
        "status": status,
        "gpu_authorization": "none",
        "policy": policy_desc,
        "reasons": reasons,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track A Guarded Fallback CPU Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        "GPU authorization: `none`",
        "",
        "## Provenance",
        "",
        f"- CPU gate JSON: `{payload['cpu_gate_json']}`",
        f"- leakage status: `{payload['leakage_status']}`",
        f"- selected base model: `{payload['selected_model']}`",
        f"- policy: `{payload['policy']}`",
        f"- fallback margin: `{payload['fallback_margin']}`",
        f"- lowcount threshold: `{payload['lowcount_threshold']}`",
        "",
        "## Policy",
        "",
    ]
    if payload["policy"] == "dataset_negative":
        lines += [
            f"Use `{payload['base_model']}` unless the dataset-level train-only internal proxy",
            f"`{payload['base_model']} - dataset_mean` is below the margin, then use",
            "`dataset_mean` for that dataset.",
        ]
    else:
        lines += [
            f"Use `{payload['base_model']}` unless the row is from Jiang_IFNG/Jiang_TNFA and",
            "`gene_train_count` is at or below the fixed threshold, then use",
            "`dataset_mean` for that row.",
        ]
    for row in payload["group_summaries"]:
        lines += [
            "",
            f"## {row['group']}",
            "",
            f"- fallback datasets: `{', '.join(row['fallback_datasets'])}`",
            f"- fallback counts by dataset: `{row.get('fallback_counts_by_dataset', {})}`",
            "",
            f"| dataset | n | {payload['base_model']} - dataset_mean | guarded? |",
            "|---|---:|---:|---|",
        ]
        for ds, vals in row["dataset_deltas"].items():
            if ds in row["fallback_datasets"] or ds in FOCUS_DATASETS:
                lines.append(f"| `{ds}` | {vals['n']} | {fmt(vals['delta'])} | `{ds in row['fallback_datasets']}` |")
        lines += [
            "",
            "| candidate | baseline | n cond | n ds | delta | 95% CI | p improve | p harm | median ds delta | leave-one min |",
            "|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
        for item in row["paired_deltas"]:
            ci = item.get("ci95") or [None, None]
            lines.append(
                f"| {item['candidate']} | {item['baseline']} | {item.get('n_conditions', 0)} | "
                f"{item.get('n_datasets', 0)} | {fmt(item.get('delta_mean'))} | "
                f"[{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(item.get('p_improve'))} | "
                f"{fmt(item.get('p_harm'))} | {fmt(item.get('median_dataset_delta'))} | "
                f"{fmt(item.get('leave_one_min'))} |"
            )
    lines += ["", "## Gate Reasons", ""]
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{r}`" for r in reasons] if reasons else ["- none"])
    lines += [
        "",
        "## Interpretation",
        "",
        "- Passing would justify implementation/protocol work for a conservative Track A fallback, not an immediate GPU launch.",
        "- Failing closes this simple guarded fallback and points Track A toward a different no-harm mechanism.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-gate-json", type=Path, default=DEFAULT_CPU_GATE)
    parser.add_argument("--fallback-margin", type=float, default=0.0)
    parser.add_argument("--policy", choices=("dataset_negative", "jiang_lowcount"), default="dataset_negative")
    parser.add_argument("--base-model", default="", help="base prior column; defaults to CPU gate selected_model")
    parser.add_argument("--status-prefix", default="tracka_scf")
    parser.add_argument("--lowcount-threshold", type=int, default=1)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    if args.policy == "jiang_lowcount" and args.out_json == DEFAULT_OUT_JSON and args.out_md == DEFAULT_OUT_MD:
        args.out_json = LOWCOUNT_OUT_JSON
        args.out_md = LOWCOUNT_OUT_MD

    cpu = load(args.cpu_gate_json)
    base_model = str(args.base_model or cpu.get("selected_model") or "shrink_k2")
    val_rows = cpu["val_condition_rows"]
    group_summaries = []
    for gi, group in enumerate(GROUPS):
        rows = group_rows(val_rows, group)
        if rows and base_model not in rows[0]:
            raise KeyError(f"base_model={base_model!r} not present in CPU gate rows")
        deltas = dataset_means(rows, base_model, "dataset_mean")
        if args.policy == "dataset_negative":
            fallback = {ds for ds, vals in deltas.items() if float(vals["delta"]) < args.fallback_margin}
        else:
            fallback = {str(r["dataset"]) for r in rows if str(r["dataset"]) in FOCUS_DATASETS and int(r.get("gene_train_count") or 0) <= args.lowcount_threshold}
        guarded = apply_policy(
            rows,
            fallback,
            policy=args.policy,
            lowcount_threshold=args.lowcount_threshold,
            base_model=base_model,
        )
        fallback_counts: dict[str, int] = defaultdict(int)
        for row in guarded:
            if row["fallback_used"]:
                fallback_counts[str(row["dataset"])] += 1
        paired = [
            paired_bootstrap(guarded, "guarded_fallback", baseline, n_boot=args.n_boot, seed=args.seed + gi * 10 + bi)
            for bi, baseline in enumerate(("dataset_mean", "gene_raw_mean", "global_mean", base_model))
        ]
        score_keys = ("guarded_fallback", base_model, "dataset_mean", "gene_raw_mean", "global_mean")
        group_summaries.append(
            {
                "group": group,
                "fallback_datasets": sorted(fallback),
                "fallback_counts_by_dataset": dict(sorted(fallback_counts.items())),
                "dataset_deltas": dict(sorted(deltas.items())),
                "absolute_scores": {
                    key: equal_dataset_mean(guarded, key)
                    for key in score_keys
                    if guarded and (key in guarded[0] or key == "guarded_fallback")
                },
                "paired_deltas": paired,
            }
        )
    payload = {
        "cpu_gate_json": str(args.cpu_gate_json),
        "leakage_status": "train_only_internal_proxy_only_no_canonical_no_multi_no_query",
        "selected_model": cpu.get("selected_model"),
        "base_model": base_model,
        "policy": args.policy,
        "fallback_margin": float(args.fallback_margin),
        "lowcount_threshold": int(args.lowcount_threshold),
        "group_summaries": group_summaries,
    }
    payload["decision"] = decide(
        group_summaries,
        args.fallback_margin,
        policy=args.policy,
        lowcount_threshold=args.lowcount_threshold,
        base_model=base_model,
        status_prefix=str(args.status_prefix),
    )
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
