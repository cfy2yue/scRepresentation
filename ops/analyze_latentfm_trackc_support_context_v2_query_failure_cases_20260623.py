#!/usr/bin/env python3
"""Reporting-only failure analysis for the support-context v2 one-shot query.

This reads already-consumed held-out query artifacts after the frozen query
decision. It must not be used to tune routes, checkpoints, thresholds, features,
or decide whether to evaluate another checkpoint.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_NAME = "xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42"
QUERY_DIR = ROOT / f"reports/latentfm_trackc_support_context_v2_query_once_{RUN_NAME}_20260623/eval"
DEFAULT_ANCHOR = QUERY_DIR / "query_anchor_split_ode20.json"
DEFAULT_CANDIDATE = QUERY_DIR / "query_candidate_split_ode20.json"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2.json"
DEFAULT_METADATA = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_trackc_support_context_v2_query_failure_cases_20260623.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_QUERY_FAILURE_CASES_20260623.md"


QUERY_GROUPS = ("query_multi", "query_multi_seen", "query_multi_unseen1", "query_multi_unseen2")


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
    return [part.strip() for part in str(condition).split("+") if part.strip()]


def split_sets(split: dict[str, Any], dataset: str) -> dict[str, set[str]]:
    obj = split.get(dataset) or {}
    return {
        "train_single": {str(x) for x in obj.get("train_single") or obj.get("train") or []},
        "train_multi": {str(x) for x in obj.get("train_multi") or []},
        "support_val_multi": {str(x) for x in obj.get("support_val_multi") or []},
        "query_seen": {str(x) for x in obj.get("query_multi_seen") or []},
        "query_unseen1": {str(x) for x in obj.get("query_multi_unseen1") or []},
        "query_unseen2": {str(x) for x in obj.get("query_multi_unseen2") or []},
    }


def rows_by_key(payload: dict[str, Any], group: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = ((payload.get("groups") or {}).get(group) or {}).get("condition_metrics") or []
    return {(str(row.get("dataset")), str(row.get("condition"))): row for row in rows}


def stratum_for(condition: str, sets: dict[str, set[str]]) -> str:
    if condition in sets["query_seen"]:
        return "seen"
    if condition in sets["query_unseen1"]:
        return "unseen1"
    if condition in sets["query_unseen2"]:
        return "unseen2"
    return "other_query"


def annotate(
    key: tuple[str, str],
    group: str,
    anchor: dict[str, Any],
    candidate: dict[str, Any],
    split: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    dataset, condition = key
    sets = split_sets(split, dataset)
    genes = condition_genes(metadata, dataset, condition)
    seen_gene_count = sum(g in sets["train_single"] for g in genes)
    anchor_pp = finite(anchor.get("pearson_pert"))
    candidate_pp = finite(candidate.get("pearson_pert"))
    anchor_mmd = finite(anchor.get("test_mmd_clamped"))
    candidate_mmd = finite(candidate.get("test_mmd_clamped"))
    return {
        "dataset": dataset,
        "condition": condition,
        "query_group": group,
        "stratum": stratum_for(condition, sets),
        "genes": genes,
        "n_genes": len(genes),
        "seen_gene_count_in_train_single": seen_gene_count,
        "condition_in_train_multi": condition in sets["train_multi"],
        "condition_in_support_val_multi": condition in sets["support_val_multi"],
        "anchor_pp": anchor_pp,
        "candidate_pp": candidate_pp,
        "pp_delta": None if anchor_pp is None or candidate_pp is None else candidate_pp - anchor_pp,
        "anchor_mmd": anchor_mmd,
        "candidate_mmd": candidate_mmd,
        "mmd_delta": None if anchor_mmd is None or candidate_mmd is None else candidate_mmd - anchor_mmd,
    }


def collect_rows(anchor_payload: dict[str, Any], candidate_payload: dict[str, Any], split: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for group in QUERY_GROUPS:
        anchor_rows = rows_by_key(anchor_payload, group)
        candidate_rows = rows_by_key(candidate_payload, group)
        for key, anchor_row in sorted(anchor_rows.items()):
            candidate_row = candidate_rows.get(key)
            if candidate_row is None:
                continue
            if group == "query_multi":
                seen.add(key)
                rows.append(annotate(key, group, anchor_row, candidate_row, split, metadata))
    return rows


def summarize_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["stratum"], row["dataset"])].append(row)
    out = []
    for (stratum, dataset), items in sorted(grouped.items()):
        pp = [r["pp_delta"] for r in items if r["pp_delta"] is not None]
        mmd = [r["mmd_delta"] for r in items if r["mmd_delta"] is not None]
        out.append(
            {
                "stratum": stratum,
                "dataset": dataset,
                "n_rows": len(items),
                "pp_delta_mean": mean(pp) if pp else None,
                "pp_negative_fraction": sum(v < 0.0 for v in pp) / max(len(pp), 1),
                "mmd_delta_mean": mean(mmd) if mmd else None,
                "mmd_harm_fraction": sum(v > 0.005 for v in mmd) / max(len(mmd), 1),
            }
        )
    return out


def recurrent_gene_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "pp": [], "mmd": []})
    for row in rows:
        for gene in row["genes"]:
            obj = stats[gene]
            obj["n"] += 1
            if row["pp_delta"] is not None:
                obj["pp"].append(row["pp_delta"])
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
                "pp_delta_mean": mean(pp) if pp else None,
                "pp_negative_fraction": sum(v < 0 for v in pp) / max(len(pp), 1),
                "mmd_delta_mean": mean(mmd) if mmd else None,
            }
        )
    return sorted(out, key=lambda r: (float(r["pp_delta_mean"] or 0.0), -int(r["n_rows"])))[:25]


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    anchor = load_json(args.anchor_json)
    candidate = load_json(args.candidate_json)
    split = load_json(args.split_file)
    metadata = load_json(args.metadata_file)
    rows = collect_rows(anchor, candidate, split, metadata)
    worst_pp = sorted(rows, key=lambda r: (float(r["pp_delta"] or 0.0), r["dataset"], r["condition"]))[:25]
    worst_mmd = sorted(rows, key=lambda r: (float(r["mmd_delta"] or 0.0), r["dataset"], r["condition"]), reverse=True)[:25]
    return {
        "status": "trackc_support_context_v2_query_failure_cases_ready",
        "boundary": "reporting_only_do_not_tune_or_select",
        "inputs": {
            "anchor_json": str(args.anchor_json),
            "candidate_json": str(args.candidate_json),
            "split_file": str(args.split_file),
            "metadata_file": str(args.metadata_file),
        },
        "n_rows": len(rows),
        "stratum_counts": dict(sorted(Counter(r["stratum"] for r in rows).items())),
        "dataset_summary": summarize_dataset(rows),
        "worst_pp_rows": worst_pp,
        "worst_mmd_rows": worst_mmd,
        "recurrent_gene_signals": recurrent_gene_summary(rows),
    }


def render_row(row: dict[str, Any]) -> str:
    genes = ",".join(row["genes"])
    return (
        f"| {row['stratum']} | {row['dataset']} | `{row['condition']}` | `{genes}` | "
        f"{row['seen_gene_count_in_train_single']}/{row['n_genes']} | "
        f"{fmt(row['anchor_pp'])} | {fmt(row['candidate_pp'])} | {fmt(row['pp_delta'])} | "
        f"{fmt(row['mmd_delta'])} |"
    )


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Support-Context V2 Query Failure Cases",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "Reporting-only analysis of the already-consumed frozen one-shot query artifact.  Do not use this report to tune, select, or trigger another query.",
        "",
        "## Stratum Counts",
        "",
    ]
    for key, value in payload["stratum_counts"].items():
        lines.append(f"* {key}: `{value}`")
    lines += [
        "",
        "## Worst Pearson Rows",
        "",
        "| stratum | dataset | condition | genes | train-single seen genes | anchor pp | candidate pp | pp delta | MMD delta |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["worst_pp_rows"][:15]:
        lines.append(render_row(row))
    lines += [
        "",
        "## Worst MMD Rows",
        "",
        "| stratum | dataset | condition | genes | train-single seen genes | anchor pp | candidate pp | pp delta | MMD delta |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["worst_mmd_rows"][:15]:
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
            f"{fmt(row['pp_negative_fraction'])} | {fmt(row['mmd_delta_mean'])} | {fmt(row['mmd_harm_fraction'])} |"
        )
    lines += [
        "",
        "## Recurrent Gene Signals",
        "",
        "| gene | rows | pp delta | pp negative frac | MMD delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["recurrent_gene_signals"][:15]:
        lines.append(
            f"| `{row['gene']}` | {row['n_rows']} | {fmt(row['pp_delta_mean'])} | "
            f"{fmt(row['pp_negative_fraction'])} | {fmt(row['mmd_delta_mean'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- This report explains where the frozen v2 query diagnostic remains weak; it is not a selection artifact.",
        "- Negative Pearson rows should be used for failure analysis and manuscript caveats only.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-json", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--candidate-json", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--metadata-file", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    payload = summarize(args)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(args.out_md), "n_rows": payload["n_rows"]}, indent=2))


if __name__ == "__main__":
    main()
