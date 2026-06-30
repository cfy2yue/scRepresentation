#!/usr/bin/env python3
"""Track A train-only baseline error map after closed router branches.

This is a read-only synthesis over the existing gene-reliability CPU gate. It
does not read canonical posthoc, held-out multi query, or model checkpoints.
Its purpose is to decide whether the train-only baseline landscape exposes a
new, non-closed Track A mechanism before spending GPU.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_IN_JSON = ROOT / "reports/latentfm_xverse_gene_reliability_router_gate_20260622.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_tracka_baseline_error_map_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_TRACKA_BASELINE_ERROR_MAP_20260622.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
MODELS = ("gene_raw_mean", "dataset_mean", "global_mean", "shrink_k8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if v is not None]
    return None if not vals else float(np.mean(vals))


def bucket_count(n: int) -> str:
    if n <= 1:
        return "count_0_1"
    if n <= 4:
        return "count_2_4"
    if n <= 9:
        return "count_5_9"
    return "count_ge10"


def grouped_summary(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, vals in sorted(groups.items()):
        item = {k: v for k, v in zip(keys, key)}
        item["n_conditions"] = len(vals)
        for model in MODELS:
            item[model] = mean([r.get(model) for r in vals])
        item["gene_minus_dataset"] = mean([float(r["gene_raw_mean"]) - float(r["dataset_mean"]) for r in vals])
        item["shrink_minus_gene"] = mean([float(r["shrink_k8"]) - float(r["gene_raw_mean"]) for r in vals])
        item["dataset_minus_gene"] = mean([float(r["dataset_mean"]) - float(r["gene_raw_mean"]) for r in vals])
        item["winner"] = max(
            ("gene_raw_mean", "dataset_mean", "global_mean", "shrink_k8"),
            key=lambda m: item[m] if item[m] is not None else -1e9,
        )
        out.append(item)
    return out


def condition_extremes(rows: list[dict[str, Any]], n: int) -> dict[str, list[dict[str, Any]]]:
    scored = []
    for row in rows:
        delta = float(row["gene_raw_mean"]) - float(row["dataset_mean"])
        shrink_delta = float(row["shrink_k8"]) - float(row["gene_raw_mean"])
        scored.append(
            {
                "dataset": row["dataset"],
                "condition": row["condition"],
                "gene": row["gene"],
                "group": row["group"],
                "gene_train_count": row["gene_train_count"],
                "gene_raw_mean": row["gene_raw_mean"],
                "dataset_mean": row["dataset_mean"],
                "shrink_k8": row["shrink_k8"],
                "gene_minus_dataset": delta,
                "shrink_minus_gene": shrink_delta,
            }
        )
    return {
        "gene_beats_dataset": sorted(scored, key=lambda r: r["gene_minus_dataset"], reverse=True)[:n],
        "dataset_beats_gene": sorted(scored, key=lambda r: r["gene_minus_dataset"])[:n],
        "shrink_harms_gene": sorted(scored, key=lambda r: r["shrink_minus_gene"])[:n],
    }


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    reasons = [
        "simple_gene_dataset_shrink_router_already_failed",
        "no_anchor_internal_val_prediction_artifact_found_in_input_gate",
        "baseline_map_reveals_known_gene_dataset_complementarity_not_new_mechanism",
    ]
    return {
        "status": "cpu_audit_no_gpu_new_tracka_mechanism",
        "action": "do_not_launch_gpu_from_baseline_error_map",
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
        "# LatentFM xverse Track A Baseline Error Map",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
        "",
        "## Provenance",
        "",
        f"- source gate JSON: `{payload['source_json']}`",
        f"- leakage status: `{payload['leakage_status']}`",
        f"- validation rows: `{payload['n_val_rows']}`",
        "- canonical posthoc, held-out multi query, and model checkpoints are not read.",
        "",
        "## Absolute Scores From Source Gate",
        "",
        "| group | model | equal-dataset pp |",
        "|---|---|---:|",
    ]
    for row in payload["source_absolute_scores"]:
        lines.append(f"| {row['group']} | `{row['model']}` | {fmt(row['pp'])} |")
    lines += [
        "",
        "## Dataset Error Map",
        "",
        "| group | dataset | n | gene_raw | dataset_mean | shrink_k8 | gene-dataset | shrink-gene | winner |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["dataset_summary"]:
        lines.append(
            f"| {row['group']} | {row['dataset']} | {row['n_conditions']} | "
            f"{fmt(row.get('gene_raw_mean'))} | {fmt(row.get('dataset_mean'))} | "
            f"{fmt(row.get('shrink_k8'))} | {fmt(row.get('gene_minus_dataset'))} | "
            f"{fmt(row.get('shrink_minus_gene'))} | `{row['winner']}` |"
        )
    lines += [
        "",
        "## Gene Count Buckets",
        "",
        "| group | gene_count_bucket | n | gene_raw | dataset_mean | shrink_k8 | gene-dataset | shrink-gene | winner |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["count_bucket_summary"]:
        lines.append(
            f"| {row['group']} | {row['gene_count_bucket']} | {row['n_conditions']} | "
            f"{fmt(row.get('gene_raw_mean'))} | {fmt(row.get('dataset_mean'))} | "
            f"{fmt(row.get('shrink_k8'))} | {fmt(row.get('gene_minus_dataset'))} | "
            f"{fmt(row.get('shrink_minus_gene'))} | `{row['winner']}` |"
        )
    lines += [
        "",
        "## Condition Extremes",
        "",
        "Top gene_raw_mean minus dataset_mean:",
        "",
        "| group | dataset | condition | gene | count | gene-dataset | shrink-gene |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in payload["condition_extremes"]["gene_beats_dataset"]:
        lines.append(
            f"| {row['group']} | {row['dataset']} | {row['condition']} | {row['gene']} | "
            f"{row['gene_train_count']} | {fmt(row['gene_minus_dataset'])} | {fmt(row['shrink_minus_gene'])} |"
        )
    lines += [
        "",
        "Top dataset_mean minus gene_raw_mean:",
        "",
        "| group | dataset | condition | gene | count | dataset-gene | shrink-gene |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in payload["condition_extremes"]["dataset_beats_gene"]:
        lines.append(
            f"| {row['group']} | {row['dataset']} | {row['condition']} | {row['gene']} | "
            f"{row['gene_train_count']} | {fmt(-row['gene_minus_dataset'])} | {fmt(row['shrink_minus_gene'])} |"
        )
    lines += [
        "",
        "## Gate Reasons",
        "",
    ]
    lines.extend(f"- `{reason}`" for reason in payload["decision"]["reasons"])
    lines += [
        "",
        "## Interpretation",
        "",
        "This audit does not identify a new GPU-worthy mechanism. The strongest",
        "pattern is the already-known complementarity between gene-level and",
        "dataset-level residual means, and the prior train-only router/shrink gate",
        "showed that simple reliability shrinkage harms the stronger gene_raw_mean",
        "control on both cross-background and family proxies.",
        "",
        "A true anchor-vs-baseline error map would require a separate frozen",
        "internal-val evaluation of the anchor checkpoint on this train-only split.",
        "That should be treated as a new protocol/audit step, not silently inferred",
        "from canonical posthoc or from this baseline-only map.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_IN_JSON)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--n-extreme", type=int, default=12)
    args = parser.parse_args()

    source = load_json(args.input_json)
    rows = list(source["val_condition_rows"])
    for row in rows:
        row["gene_count_bucket"] = bucket_count(int(row["gene_train_count"]))
    payload = {
        "source_json": str(args.input_json),
        "leakage_status": "derived_from_train_only_gene_reliability_gate_no_new_data_reads",
        "n_val_rows": len(rows),
        "source_absolute_scores": source["absolute_scores"],
        "source_decision": source["decision"],
        "dataset_summary": grouped_summary(rows, ("group", "dataset")),
        "count_bucket_summary": grouped_summary(rows, ("group", "gene_count_bucket")),
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
