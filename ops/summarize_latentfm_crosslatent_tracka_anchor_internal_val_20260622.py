#!/usr/bin/env python3
"""Summarize cross-latent Track A anchor internal-val predictions.

The input anchor eval should be produced by ``model.latent.eval_split_groups``
with an explicit train-only ``--pert-means-file``.  The baseline JSON must be
from the same latent/data_dir and split.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
BASELINES = ("gene_raw_mean", "dataset_mean", "global_mean")


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
                "anchor_pearson_pert": metric.get("pearson_pert"),
                "anchor_pearson_ctrl": metric.get("pearson_ctrl"),
                "anchor_mmd_clamped": metric.get("test_mmd_clamped"),
            }
            for name in BASELINES:
                item[name] = base.get(name)
                if item["anchor_pearson_pert"] is not None and item[name] is not None:
                    item[f"anchor_minus_{name}"] = float(item["anchor_pearson_pert"]) - float(item[name])
            rows.append(item)
    return rows


def grouped_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["dataset"])].append(row)
    out = []
    for (group, dataset), vals in sorted(grouped.items()):
        item = {"group": group, "dataset": dataset, "n_conditions": len(vals)}
        for key in ("anchor_pearson_pert", "anchor_mmd_clamped") + BASELINES:
            nums = [float(v[key]) for v in vals if v.get(key) is not None]
            item[key] = None if not nums else float(np.mean(nums))
        for base in BASELINES:
            key = f"anchor_minus_{base}"
            nums = [float(v[key]) for v in vals if v.get(key) is not None]
            item[key] = None if not nums else float(np.mean(nums))
        out.append(item)
    return out


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
            if any(float(v) < -0.02 for v in (row.get("by_dataset") or {}).values()):
                reasons.append(f"{group}_dataset_level_material_harm_vs_{baseline}")
    status = (
        "crosslatent_anchor_internal_val_candidate"
        if not reasons
        else "crosslatent_anchor_internal_val_not_promotable"
    )
    action = (
        "compare_against_other_latents_and_require_protocol_review_before_training"
        if not reasons
        else "do_not_launch_training_from_this_anchor_internal_val_result"
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
        f"# LatentFM Cross-Latent Track A Anchor Internal-Val: {payload['latent']}",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
        "",
        "## Provenance",
        "",
        f"- latent: `{payload['latent']}`",
        f"- anchor eval JSON: `{payload['anchor_eval_json']}`",
        f"- baseline JSON: `{payload['baseline_json']}`",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- used_ema: `{payload['used_ema']}`",
        f"- means files: `{payload['means_files']}`",
        f"- condition rows matched: `{payload['n_rows']}`",
        "- canonical test, canonical multi, and Track C query are not used for this summary.",
        "",
        "## Absolute Scores",
        "",
        "| group | model | equal-dataset score |",
        "|---|---|---:|",
    ]
    for group in GROUPS:
        group_rows = [r for r in payload["condition_rows"] if r["group"] == group]
        for key in ("anchor_pearson_pert", "anchor_mmd_clamped") + BASELINES:
            lines.append(f"| {group} | `{key}` | {fmt(equal_dataset_mean(group_rows, key))} |")
    lines += [
        "",
        "## Paired Anchor Minus Baseline Deltas",
        "",
        "| group | baseline | n cond | n ds | delta | 95% CI | p improve | p harm | leave-one min |",
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
        "- This summarizes a frozen anchor internal-val posthoc result, not a training run.",
        "- Passing would only nominate this latent/checkpoint for cross-latent comparison and protocol review.",
        "- Any later training still needs a separate hypothesis, resource audit, RUN_STATUS, and frozen gate.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latent", required=True)
    ap.add_argument("--anchor-eval-json", type=Path, required=True)
    ap.add_argument("--baseline-json", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    anchor = load_json(args.anchor_eval_json)
    baseline = load_json(args.baseline_json)
    rows = build_rows(anchor, baseline)
    paired = []
    for group in GROUPS:
        group_rows = [r for r in rows if r["group"] == group]
        for base in BASELINES:
            paired.append(
                {
                    "group": group,
                    **paired_bootstrap(
                        group_rows,
                        "anchor_pearson_pert",
                        base,
                        n_boot=int(args.n_boot),
                        seed=int(args.seed) + len(paired),
                    ),
                }
            )
    payload = {
        "latent": args.latent,
        "anchor_eval_json": str(args.anchor_eval_json),
        "baseline_json": str(args.baseline_json),
        "checkpoint": anchor.get("checkpoint"),
        "split_file": anchor.get("split_file"),
        "used_ema": anchor.get("used_ema"),
        "means_files": anchor.get("means_files"),
        "n_rows": len(rows),
        "condition_rows": rows,
        "dataset_summary": grouped_summary(rows),
        "paired_deltas": paired,
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
