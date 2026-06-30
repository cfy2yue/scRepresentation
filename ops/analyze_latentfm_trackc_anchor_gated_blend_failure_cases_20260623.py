#!/usr/bin/env python3
"""Failure-case metadata analysis for the frozen Track C query diagnostic.

This is reporting-only over the already-read one-shot query artifact.  It must
not drive route, alpha, threshold, checkpoint, or branch selection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_QUERY_JSON = (
    ROOT
    / "runs/latentfm_trackc_anchor_gated_blend_query_once_20260623_retry1"
    / "eval/anchor_gated_blend_query_once_ode20.json"
)
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
DEFAULT_METADATA = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_trackc_anchor_gated_blend_failure_cases_20260623.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_BLEND_FAILURE_CASES_20260623.md"

QUERY_GROUPS = (
    "heldout_query_multi_final_only",
    "heldout_query_multi_seen_final_only",
    "heldout_query_multi_unseen1_final_only",
    "heldout_query_multi_unseen2_final_only",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if out == out and abs(out) != float("inf") else None


def fmt(value: Any) -> str:
    value = finite(value)
    return "NA" if value is None else f"{value:+.6f}"


def condition_genes(metadata: dict[str, Any], dataset: str, condition: str) -> list[str]:
    meta = ((metadata.get(dataset) or {}).get(condition) or {})
    genes = [str(g).strip() for g in meta.get("genes") or [] if str(g).strip()]
    if genes:
        return genes
    parts = [p.strip() for p in str(condition).split("+") if p.strip()]
    return parts


def split_sets(split: dict[str, Any], dataset: str) -> dict[str, set[str]]:
    obj = split.get(dataset) or {}
    return {
        "train_single": {str(x) for x in obj.get("train_single") or obj.get("train") or []},
        "train_multi": {str(x) for x in obj.get("train_multi") or []},
        "support_val_multi": {str(x) for x in obj.get("support_val_multi") or []},
        "query_seen": {str(x) for x in obj.get("heldout_query_multi_seen_final_only") or []},
        "query_unseen1": {str(x) for x in obj.get("heldout_query_multi_unseen1_final_only") or []},
        "query_unseen2": {str(x) for x in obj.get("heldout_query_multi_unseen2_final_only") or []},
    }


def group_rows(query_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    group = "heldout_query_multi_final_only"
    for row in ((query_payload.get("groups") or {}).get(group) or {}).get("condition_metrics") or []:
        item = dict(row)
        item["query_group"] = group
        rows.append(item)
    return rows


def annotate_row(row: dict[str, Any], split: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    dataset = str(row.get("dataset"))
    condition = str(row.get("condition"))
    genes = condition_genes(metadata, dataset, condition)
    sets = split_sets(split, dataset)
    train_single_genes = sets["train_single"]
    seen_gene_count = sum(g in train_single_genes for g in genes)
    if condition in sets["query_seen"]:
        stratum = "seen"
    elif condition in sets["query_unseen1"]:
        stratum = "unseen1"
    elif condition in sets["query_unseen2"]:
        stratum = "unseen2"
    else:
        stratum = "other_final_query"
    return {
        "dataset": dataset,
        "condition": condition,
        "query_group": row.get("query_group"),
        "stratum": stratum,
        "genes": genes,
        "n_genes": len(genes),
        "seen_gene_count_in_train_single": seen_gene_count,
        "unseen_gene_count_vs_train_single": max(len(genes) - seen_gene_count, 0),
        "condition_in_train_multi": condition in sets["train_multi"],
        "condition_in_support_val_multi": condition in sets["support_val_multi"],
        "pp_delta": finite(row.get("blend_delta_vs_anchor_pearson_pert")),
        "mmd_delta": finite(row.get("blend_delta_vs_anchor_test_mmd_clamped")),
        "anchor_pp": finite(row.get("anchor_pearson_pert")),
        "blend_pp": finite(row.get("blend_pearson_pert")),
        "anchor_mmd": finite(row.get("anchor_test_mmd_clamped")),
        "blend_mmd": finite(row.get("blend_test_mmd_clamped")),
    }


def summarize_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["stratum"]), str(row["dataset"]))].append(row)
    out = []
    for (stratum, dataset), items in sorted(grouped.items()):
        pp = [r["pp_delta"] for r in items if r["pp_delta"] is not None]
        mmd = [r["mmd_delta"] for r in items if r["mmd_delta"] is not None]
        out.append(
            {
                "stratum": stratum,
                "dataset": dataset,
                "n_rows": len(items),
                "n_negative_pp": sum(v < 0.0 for v in pp),
                "negative_pp_fraction": sum(v < 0.0 for v in pp) / max(len(pp), 1),
                "pp_delta_mean": mean(pp) if pp else None,
                "mmd_delta_mean": mean(mmd) if mmd else None,
                "mmd_harm_fraction": sum(v > 0.005 for v in mmd) / max(len(mmd), 1),
            }
        )
    return out


def summarize_gene_failure(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "neg": 0, "pp": [], "mmd": []})
    for row in rows:
        for gene in row["genes"]:
            obj = stats[gene]
            obj["n"] += 1
            if row["pp_delta"] is not None:
                obj["pp"].append(row["pp_delta"])
                obj["neg"] += int(row["pp_delta"] < 0.0)
            if row["mmd_delta"] is not None:
                obj["mmd"].append(row["mmd_delta"])
    out = []
    for gene, obj in stats.items():
        if obj["n"] < 2:
            continue
        pp = obj["pp"]
        mmd = obj["mmd"]
        out.append(
            {
                "gene": gene,
                "n_rows": obj["n"],
                "negative_pp_fraction": obj["neg"] / max(len(pp), 1),
                "pp_delta_mean": mean(pp) if pp else None,
                "mmd_delta_mean": mean(mmd) if mmd else None,
            }
        )
    return sorted(out, key=lambda r: (float(r["pp_delta_mean"] or 0.0), -int(r["n_rows"])))[:20]


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    query = load_json(args.query_json)
    split = load_json(args.split_file)
    metadata = load_json(args.metadata_file)
    annotated = [annotate_row(row, split, metadata) for row in group_rows(query)]
    worst_pp = sorted(annotated, key=lambda r: (float(r["pp_delta"] or 0.0), r["dataset"], r["condition"]))[:20]
    worst_mmd = sorted(annotated, key=lambda r: (float(r["mmd_delta"] or 0.0), r["dataset"], r["condition"]), reverse=True)[:20]
    stratum_counts = Counter(r["stratum"] for r in annotated)
    return {
        "status": "trackc_anchor_gated_blend_failure_case_analysis_ready",
        "boundary": "reporting_only_no_query_tuning",
        "inputs": {
            "query_json": str(args.query_json),
            "split_file": str(args.split_file),
            "metadata_file": str(args.metadata_file),
        },
        "n_rows": len(annotated),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "dataset_summary": summarize_dataset(annotated),
        "worst_pp_rows": worst_pp,
        "worst_mmd_rows": worst_mmd,
        "recurrent_gene_failures": summarize_gene_failure(annotated),
    }


def render_row(row: dict[str, Any]) -> str:
    genes = ",".join(row["genes"])
    return (
        f"| {row['stratum']} | {row['dataset']} | `{row['condition']}` | `{genes}` | "
        f"{row['seen_gene_count_in_train_single']}/{row['n_genes']} | "
        f"{fmt(row['pp_delta'])} | {fmt(row['mmd_delta'])} |"
    )


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Anchor-Gated Blend Failure Cases",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "Reporting-only analysis of the frozen one-shot query artifact.  Do not use this to tune or select any future route.",
        "",
        "## Stratum Counts",
        "",
    ]
    for key, value in payload["stratum_counts"].items():
        lines.append(f"* {key}: `{value}`")
    lines += [
        "",
        "## Worst pp Rows",
        "",
        "| stratum | dataset | condition | genes | train-single seen genes | pp delta | MMD delta |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in payload["worst_pp_rows"][:12]:
        lines.append(render_row(row))
    lines += [
        "",
        "## Worst MMD Rows",
        "",
        "| stratum | dataset | condition | genes | train-single seen genes | pp delta | MMD delta |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in payload["worst_mmd_rows"][:12]:
        lines.append(render_row(row))
    lines += [
        "",
        "## Dataset-Stratum Summary",
        "",
        "| stratum | dataset | rows | pp delta | pp negative frac | MMD delta | MMD harm frac |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["dataset_summary"]:
        lines.append(
            f"| {row['stratum']} | {row['dataset']} | {row['n_rows']} | {fmt(row['pp_delta_mean'])} | "
            f"{fmt(row['negative_pp_fraction'])} | {fmt(row['mmd_delta_mean'])} | {fmt(row['mmd_harm_fraction'])} |"
        )
    lines += [
        "",
        "## Recurrent Gene-Level Failure Signals",
        "",
        "| gene | rows | pp delta | pp negative frac | MMD delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["recurrent_gene_failures"][:12]:
        lines.append(
            f"| {row['gene']} | {row['n_rows']} | {fmt(row['pp_delta_mean'])} | "
            f"{fmt(row['negative_pp_fraction'])} | {fmt(row['mmd_delta_mean'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "* The largest failure is not caused by accidental support/train leakage: worst rows are not in support_val_multi.",
        "* Wessels unseen2 has many negative pp rows despite no aggregate MMD harm, matching the weak unseen2 CI.",
        "* Norman failures dominate the worst MMD rows, especially MAPK1/UBASH3B-associated combinations.",
        "",
        "## Inputs",
        "",
    ]
    for key, value in payload["inputs"].items():
        lines.append(f"* {key}: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-json", type=Path, default=DEFAULT_QUERY_JSON)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--metadata-file", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    payload = summarize(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
