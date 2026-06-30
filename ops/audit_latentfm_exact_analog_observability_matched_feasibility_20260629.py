#!/usr/bin/env python3
"""CPU-only exact/analog observability matched-design feasibility audit.

This script tests whether exact response coverage plus train-time analog
support can define a larger, leakage-safe scaling axis than strict exact
coverage alone. It does not launch training or produce launch-ready splits.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
PARENT_SPLIT = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
COND_META = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
COVERAGE_CSV = ROOT / "reports/exact_response_information_combined_coverage_20260628/exact_response_information_condition_rows.csv"
OUT_DIR = ROOT / "reports/exact_analog_observability_matched_feasibility_20260629"
SEED = 42
MAX_PAIRS_PER_DATASET = 96


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


def condition_type(meta: dict[str, Any]) -> str:
    return str(meta.get("perturbation_type_raw", "unknown") or "unknown")


def gene_list(meta: dict[str, Any]) -> list[str]:
    genes = meta.get("genes", [])
    if not isinstance(genes, list):
        return []
    return [str(g).upper() for g in genes if str(g).strip()]


def gene_count_bin(meta: dict[str, Any]) -> str:
    n = len(gene_list(meta))
    if n <= 1:
        return "single"
    if n == 2:
        return "double"
    return "multi3plus"


def chem_key(meta: dict[str, Any]) -> str:
    value = norm_text(meta.get("chem_obs_value"))
    source = norm_text(meta.get("chem_source"))
    if not value:
        return ""
    return f"{source}|{value}"


def make_train_rows(parent: dict[str, Any], meta: dict[str, Any], covered_train: set[tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, groups in parent.items():
        for condition in groups.get("train", []):
            condition_s = str(condition)
            cond_meta = meta.get(dataset, {}).get(condition_s, {})
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition_s,
                    "condition_norm": norm_text(condition_s),
                    "perturbation_type": condition_type(cond_meta),
                    "gene_count_bin": gene_count_bin(cond_meta),
                    "genes": gene_list(cond_meta),
                    "chem_key": chem_key(cond_meta),
                    "exact_train_covered": (dataset, condition_s) in covered_train,
                }
            )
    return rows


def add_analog_support(rows: list[dict[str, Any]]) -> None:
    """Annotate each train row with support from exact-covered train rows in other datasets."""
    covered_rows = [row for row in rows if row["exact_train_covered"]]

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_gene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_chem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in covered_rows:
        by_condition[row["condition_norm"]].append(row)
        for gene in row["genes"]:
            by_gene[gene].append(row)
        if row["chem_key"]:
            by_chem[row["chem_key"]].append(row)

    for row in rows:
        dataset = row["dataset"]
        same_condition_datasets = {
            src["dataset"]
            for src in by_condition.get(row["condition_norm"], [])
            if src["dataset"] != dataset
        }
        gene_datasets: set[str] = set()
        gene_record_count = 0
        for gene in row["genes"]:
            for src in by_gene.get(gene, []):
                if src["dataset"] == dataset:
                    continue
                gene_datasets.add(src["dataset"])
                gene_record_count += 1
        chem_datasets = {
            src["dataset"]
            for src in by_chem.get(row["chem_key"], [])
            if row["chem_key"] and src["dataset"] != dataset
        }
        analog_datasets = same_condition_datasets | gene_datasets | chem_datasets
        row["same_condition_other_dataset_count"] = len(same_condition_datasets)
        row["gene_support_other_dataset_count"] = len(gene_datasets)
        row["gene_support_other_record_count"] = gene_record_count
        row["chem_support_other_dataset_count"] = len(chem_datasets)
        row["analog_support_dataset_count"] = len(analog_datasets)
        row["observability_ge1"] = bool(row["exact_train_covered"] or len(analog_datasets) >= 1)
        row["observability_ge2"] = bool(row["exact_train_covered"] or len(analog_datasets) >= 2)
        row["observability_ge3"] = bool(row["exact_train_covered"] or len(analog_datasets) >= 3)
        row["analog_nonexact_ge1"] = bool((not row["exact_train_covered"]) and len(analog_datasets) >= 1)
        row["analog_nonexact_ge2"] = bool((not row["exact_train_covered"]) and len(analog_datasets) >= 2)


def stratum_key(row: dict[str, Any], mode: str) -> tuple[str, ...]:
    if mode == "strict":
        return (row["dataset"], row["perturbation_type"], row["gene_count_bin"])
    if mode == "type_only":
        return (row["dataset"], row["perturbation_type"])
    if mode == "dataset_only":
        return (row["dataset"],)
    raise ValueError(f"unknown match mode: {mode}")


def match_design(rows: list[dict[str, Any]], high_key: str, mode: str, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    strata: dict[tuple[str, ...], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"high": [], "low": []})
    for row in rows:
        if bool(row.get(high_key)):
            strata[stratum_key(row, mode)]["high"].append(row)
        else:
            strata[stratum_key(row, mode)]["low"].append(row)

    dataset_counts: dict[str, int] = defaultdict(int)
    pairs: list[dict[str, Any]] = []
    for key, buckets in sorted(strata.items(), key=lambda item: item[0]):
        high = sorted(buckets["high"], key=lambda r: (r["dataset"], r["condition"]))
        low = sorted(buckets["low"], key=lambda r: (r["dataset"], r["condition"]))
        rng.shuffle(high)
        rng.shuffle(low)
        for hi, lo in zip(high, low):
            dataset = hi["dataset"]
            if dataset_counts[dataset] >= MAX_PAIRS_PER_DATASET:
                continue
            dataset_counts[dataset] += 1
            pairs.append(
                {
                    "design": high_key,
                    "match_mode": mode,
                    "dataset": dataset,
                    "perturbation_type": hi["perturbation_type"],
                    "gene_count_bin": hi["gene_count_bin"],
                    "high_condition": hi["condition"],
                    "low_condition": lo["condition"],
                    "high_exact_train_covered": hi["exact_train_covered"],
                    "high_analog_support_dataset_count": hi["analog_support_dataset_count"],
                    "low_exact_train_covered": lo["exact_train_covered"],
                    "low_analog_support_dataset_count": lo["analog_support_dataset_count"],
                }
            )
    dataset_rows = []
    by_dataset = defaultdict(list)
    for pair in pairs:
        by_dataset[pair["dataset"]].append(pair)
    for dataset, dataset_pairs in sorted(by_dataset.items()):
        dataset_rows.append(
            {
                "design": high_key,
                "match_mode": mode,
                "dataset": dataset,
                "matched_pairs": len(dataset_pairs),
                "exact_high_pairs": sum(1 for pair in dataset_pairs if pair["high_exact_train_covered"]),
                "analog_nonexact_high_pairs": sum(
                    1
                    for pair in dataset_pairs
                    if (not pair["high_exact_train_covered"]) and pair["high_analog_support_dataset_count"] > 0
                ),
            }
        )
    return pairs, dataset_rows


def summarize_design(pairs: list[dict[str, Any]], dataset_rows: list[dict[str, Any]], high_key: str, mode: str) -> dict[str, Any]:
    total = len(pairs)
    datasets = len(dataset_rows)
    max_dataset_pairs = max([row["matched_pairs"] for row in dataset_rows] or [0])
    max_dataset_share = max_dataset_pairs / total if total else 0.0
    type_counts = pd.Series([pair["perturbation_type"] for pair in pairs]).value_counts().to_dict() if pairs else {}
    return {
        "design": high_key,
        "match_mode": mode,
        "matched_pairs": total,
        "datasets_with_pairs": datasets,
        "max_dataset_pairs": max_dataset_pairs,
        "max_dataset_share": max_dataset_share,
        "exact_high_pairs": sum(1 for pair in pairs if pair["high_exact_train_covered"]),
        "analog_nonexact_high_pairs": sum(
            1
            for pair in pairs
            if (not pair["high_exact_train_covered"]) and pair["high_analog_support_dataset_count"] > 0
        ),
        "perturbation_type_counts": type_counts,
    }


def copy_matched_split(parent: dict[str, Any], pairs: list[dict[str, Any]], side: str) -> dict[str, Any]:
    out = json.loads(json.dumps(parent))
    selected: dict[str, list[str]] = defaultdict(list)
    column = "high_condition" if side == "high" else "low_condition"
    for pair in pairs:
        selected[pair["dataset"]].append(pair[column])
    for dataset in out:
        out[dataset]["train"] = sorted(set(selected.get(dataset, [])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-split", type=Path, default=PARENT_SPLIT)
    parser.add_argument("--condition-metadata", type=Path, default=COND_META)
    parser.add_argument("--coverage-csv", type=Path, default=COVERAGE_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "LATENTFM_EXACT_ANALOG_OBSERVABILITY_MATCHED_FEASIBILITY_20260629.md"
    out_json = out_dir / "latentfm_exact_analog_observability_matched_feasibility_20260629.json"
    out_condition_csv = out_dir / "condition_observability_rows.csv"
    out_summary_csv = out_dir / "design_summary.csv"
    out_dataset_csv = out_dir / "dataset_match_summary.csv"
    out_pairs_csv = out_dir / "best_candidate_matched_pairs.csv"
    high_split = out_dir / "draft_split_seed42_xverse_exact_or_analog_high_matched_from_cap120_all_v2.json"
    low_split = out_dir / "draft_split_seed42_xverse_exact_or_analog_low_matched_from_cap120_all_v2.json"

    parent = load_json(args.parent_split)
    meta = load_json(args.condition_metadata)
    coverage_df = pd.read_csv(args.coverage_csv)
    parent_train = {
        (str(dataset), str(condition))
        for dataset, groups in parent.items()
        for condition in groups.get("train", [])
    }
    covered_train = {
        (str(row.dataset), str(row.condition))
        for row in coverage_df.itertuples(index=False)
        if (str(row.dataset), str(row.condition)) in parent_train
    }

    rows = make_train_rows(parent, meta, covered_train)
    add_analog_support(rows)
    condition_df = pd.DataFrame(rows)
    condition_df.drop(columns=["genes"], errors="ignore").to_csv(out_condition_csv, index=False)

    design_keys = [
        "exact_train_covered",
        "observability_ge1",
        "observability_ge2",
        "observability_ge3",
        "analog_nonexact_ge1",
        "analog_nonexact_ge2",
    ]
    match_modes = ["strict", "type_only", "dataset_only"]
    all_pairs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    all_dataset_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for design_key in design_keys:
        for mode in match_modes:
            pairs, dataset_rows = match_design(rows, design_key, mode, args.seed)
            all_pairs[(design_key, mode)] = pairs
            all_dataset_rows.extend(dataset_rows)
            summaries.append(summarize_design(pairs, dataset_rows, design_key, mode))

    summary_df = pd.DataFrame(summaries).sort_values(
        ["match_mode", "matched_pairs", "datasets_with_pairs"], ascending=[True, False, False]
    )
    summary_df.to_csv(out_summary_csv, index=False)
    pd.DataFrame(all_dataset_rows).to_csv(out_dataset_csv, index=False)

    strict_candidates = [
        item
        for item in summaries
        if item["match_mode"] == "strict" and item["design"] in {"observability_ge1", "observability_ge2", "observability_ge3"}
    ]
    strict_feasible = [
        item
        for item in strict_candidates
        if item["matched_pairs"] >= 300 and item["datasets_with_pairs"] >= 15 and item["max_dataset_share"] <= 0.35
    ]
    relaxed_feasible = [
        item
        for item in summaries
        if item["match_mode"] != "strict"
        and item["design"] in {"observability_ge1", "observability_ge2", "observability_ge3"}
        and item["matched_pairs"] >= 300
        and item["datasets_with_pairs"] >= 15
    ]
    if strict_feasible:
        status = "exact_analog_observability_strict_feasible_no_gpu"
        best = sorted(strict_feasible, key=lambda x: (x["matched_pairs"], x["datasets_with_pairs"]), reverse=True)[0]
    elif relaxed_feasible:
        status = "exact_analog_observability_relaxed_only_confounded_no_gpu"
        best = sorted(relaxed_feasible, key=lambda x: (x["matched_pairs"], x["datasets_with_pairs"]), reverse=True)[0]
    else:
        status = "exact_analog_observability_insufficient_no_gpu"
        best = sorted(strict_candidates or summaries, key=lambda x: (x["matched_pairs"], x["datasets_with_pairs"]), reverse=True)[0]

    best_pairs = all_pairs[(best["design"], best["match_mode"])]
    pd.DataFrame(best_pairs).to_csv(out_pairs_csv, index=False)
    write_json(high_split, copy_matched_split(parent, best_pairs, "high"))
    write_json(low_split, copy_matched_split(parent, best_pairs, "low"))

    total_train = len(rows)
    exact_count = sum(1 for row in rows if row["exact_train_covered"])
    analog_nonexact_ge1 = sum(1 for row in rows if row["analog_nonexact_ge1"])
    analog_nonexact_ge2 = sum(1 for row in rows if row["analog_nonexact_ge2"])
    obs_ge1 = sum(1 for row in rows if row["observability_ge1"])
    obs_ge2 = sum(1 for row in rows if row["observability_ge2"])
    obs_ge3 = sum(1 for row in rows if row["observability_ge3"])

    payload = {
        "created_at": now_cst(),
        "status": status,
        "boundary": "cpu_report_only_no_gpu_no_checkpoint_selection",
        "parent_split": str(args.parent_split),
        "condition_metadata": str(args.condition_metadata),
        "coverage_csv": str(args.coverage_csv),
        "support_universe": "coverage rows intersected with parent train only",
        "matched_gate": {
            "strict_required_pairs": 300,
            "strict_required_datasets": 15,
            "max_dataset_share": 0.35,
        },
        "total_parent_train_conditions": total_train,
        "exact_train_covered_conditions": exact_count,
        "analog_nonexact_ge1_conditions": analog_nonexact_ge1,
        "analog_nonexact_ge2_conditions": analog_nonexact_ge2,
        "observability_ge1_conditions": obs_ge1,
        "observability_ge2_conditions": obs_ge2,
        "observability_ge3_conditions": obs_ge3,
        "best_candidate": best,
        "all_summaries": summaries,
        "outputs": {
            "markdown": str(out_md),
            "json": str(out_json),
            "condition_observability_rows": str(out_condition_csv),
            "design_summary": str(out_summary_csv),
            "dataset_match_summary": str(out_dataset_csv),
            "best_candidate_pairs": str(out_pairs_csv),
            "best_high_draft_split": str(high_split),
            "best_low_draft_split": str(low_split),
        },
    }
    write_json(out_json, payload)

    display_cols = [
        "design",
        "match_mode",
        "matched_pairs",
        "datasets_with_pairs",
        "max_dataset_share",
        "exact_high_pairs",
        "analog_nonexact_high_pairs",
    ]
    top_rows = summary_df[display_cols].sort_values(
        ["matched_pairs", "datasets_with_pairs"], ascending=False
    ).head(12)
    strict_rows = summary_df[summary_df["match_mode"] == "strict"][display_cols].sort_values(
        ["matched_pairs", "datasets_with_pairs"], ascending=False
    )
    top_dataset_rows = (
        pd.DataFrame(all_dataset_rows)
        .query("design == @best['design'] and match_mode == @best['match_mode']")
        .sort_values("matched_pairs", ascending=False)
        .head(15)
        if all_dataset_rows
        else pd.DataFrame()
    )

    lines = [
        "# LatentFM Exact/Analog Observability Matched Feasibility",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only feasibility audit; no training, inference, GPU launch, canonical multi, Track C query, or checkpoint selection.",
        "* Analog support is computed only from exact-covered conditions that are also in the parent train split, so held-out/test response information is not used to define the axis.",
        "* Draft high/low split JSONs are provenance artifacts for review only; they are not launch-ready.",
        "",
        "## Observability Counts",
        "",
        f"* Parent train conditions: `{total_train}`.",
        f"* Strict exact-covered train conditions: `{exact_count}`.",
        f"* Non-exact conditions with analog support in >=1 other dataset: `{analog_nonexact_ge1}`.",
        f"* Non-exact conditions with analog support in >=2 other datasets: `{analog_nonexact_ge2}`.",
        f"* Exact-or-analog ge1 / ge2 / ge3 totals: `{obs_ge1}` / `{obs_ge2}` / `{obs_ge3}`.",
        "",
        "## Design Summary",
        "",
        "| design | match | pairs | datasets | max dataset share | exact high | analog non-exact high |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in top_rows.to_dict("records"):
        lines.append(
            "| {design} | {match_mode} | {matched_pairs} | {datasets_with_pairs} | {max_dataset_share:.3f} | {exact_high_pairs} | {analog_nonexact_high_pairs} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Strict-Match Focus",
            "",
            "| design | pairs | datasets | max dataset share | exact high | analog non-exact high |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in strict_rows.to_dict("records"):
        lines.append(
            "| {design} | {matched_pairs} | {datasets_with_pairs} | {max_dataset_share:.3f} | {exact_high_pairs} | {analog_nonexact_high_pairs} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Best Candidate",
            "",
            f"* Design: `{best['design']}` with match mode `{best['match_mode']}`.",
            f"* Matched pairs: `{best['matched_pairs']}` across `{best['datasets_with_pairs']}` datasets.",
            f"* Max dataset share: `{best['max_dataset_share']:.3f}`.",
            f"* Exact high pairs: `{best['exact_high_pairs']}`; analog non-exact high pairs: `{best['analog_nonexact_high_pairs']}`.",
            "",
            "## Best Candidate Dataset Balance",
            "",
            "| dataset | pairs | exact high | analog non-exact high |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in top_dataset_rows.to_dict("records"):
        lines.append(
            f"| {row['dataset']} | {int(row['matched_pairs'])} | {int(row['exact_high_pairs'])} | {int(row['analog_nonexact_high_pairs'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"* Status: `{status}`.",
        ]
    )
    if status == "exact_analog_observability_strict_feasible_no_gpu":
        lines.extend(
            [
                "* Exact+analog observability is feasible as a strict matched training-data scaling axis.",
                "* Next gate: independent leakage/provenance audit plus launcher design with matched high/low train splits, identical validation/canonical no-harm, and no canonical multi/Track C query selection.",
            ]
        )
    elif status == "exact_analog_observability_relaxed_only_confounded_no_gpu":
        lines.extend(
            [
                "* Analog support expands the axis only under relaxed matching; this is useful evidence but not launch-ready because gene-count composition can still confound the effect.",
                "* Next gate: either find a stricter analog definition or use this as a covariate/diagnostic rather than a GPU training split.",
            ]
        )
    else:
        lines.extend(
            [
                "* Exact+analog support does not yet yield a sufficiently broad strict matched design.",
                "* Next gate: keep exact/analog observability as a manuscript covariate/failure-map variable, or test a different information axis such as train-set cluster/OT information density.",
            ]
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"* JSON: `{out_json}`",
            f"* Condition rows: `{out_condition_csv}`",
            f"* Design summary: `{out_summary_csv}`",
            f"* Dataset summary: `{out_dataset_csv}`",
            f"* Best candidate pairs: `{out_pairs_csv}`",
            f"* Best high draft split: `{high_split}`",
            f"* Best low draft split: `{low_split}`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "best_candidate": best, "report": str(out_md)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
