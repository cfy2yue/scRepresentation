#!/usr/bin/env python3
"""Draft condition-level high/low information splits for scaling-law v2.

CPU/report-only. Uses exact-covered train conditions from a parent split and
matches high/low response-information conditions within dataset, perturbation
type, and gene-count strata. Draft splits are feasibility artifacts only.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
PARENT_SPLIT = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
COND_META = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
COVERAGE = ROOT / "reports/exact_response_information_combined_coverage_20260628/exact_response_information_condition_rows.csv"
OUT_DIR = ROOT / "reports/scaling_v2_condition_information_draft_splits_20260628"
SEED = 42
MAX_PAIRS_PER_DATASET = 120


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


def condition_type(meta: dict[str, Any]) -> str:
    return str(meta.get("perturbation_type_raw", "unknown") or "unknown")


def gene_count_bin(meta: dict[str, Any]) -> str:
    genes = meta.get("genes", [])
    if not isinstance(genes, list):
        return "unknown"
    if len(genes) <= 1:
        return "single"
    if len(genes) == 2:
        return "double"
    return "multi3plus"


def zscore_by_group(df: pd.DataFrame, col: str, group_col: str = "dataset") -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    for _, idx in df.groupby(group_col).groups.items():
        vals = pd.to_numeric(df.loc[idx, col], errors="coerce")
        med = vals.median()
        mad = (vals - med).abs().median()
        scale = mad if mad and np.isfinite(mad) else vals.std()
        if not scale or not np.isfinite(scale):
            scale = 1.0
        out.loc[idx] = (vals - med) / scale
    return out


def build_condition_table(parent: dict[str, Any], meta: dict[str, Any], cov: pd.DataFrame) -> pd.DataFrame:
    cov = cov.copy()
    cov["condition"] = cov["condition"].astype(str)
    cov["dataset"] = cov["dataset"].astype(str)
    cov = cov.drop_duplicates(subset=["dataset", "condition"])
    parent_rows = []
    for dataset, groups in parent.items():
        for condition in groups.get("train", []):
            cm = meta.get(dataset, {}).get(str(condition), {})
            parent_rows.append(
                {
                    "dataset": str(dataset),
                    "condition": str(condition),
                    "perturbation_type": condition_type(cm),
                    "gene_count_bin": gene_count_bin(cm),
                }
            )
    parent_df = pd.DataFrame(parent_rows)
    df = parent_df.merge(cov, on=["dataset", "condition"], how="inner", validate="one_to_one")
    df["log_response_energy"] = np.log1p(pd.to_numeric(df["response_energy"], errors="coerce"))
    df["hvg_concentration_80"] = 1.0 - pd.to_numeric(df["hvg_k80"], errors="coerce") / pd.to_numeric(df["n_vars"], errors="coerce")
    df["hvg_concentration_90"] = 1.0 - pd.to_numeric(df["hvg_k90"], errors="coerce") / pd.to_numeric(df["n_vars"], errors="coerce")
    df["hvg_advantage_80"] = (pd.to_numeric(df["abundance_k80"], errors="coerce") - pd.to_numeric(df["hvg_k80"], errors="coerce")) / pd.to_numeric(df["n_vars"], errors="coerce")
    df["cell_support_log"] = np.log1p(pd.to_numeric(df["n_pert"], errors="coerce"))
    for col in ["log_response_energy", "hvg_concentration_80", "hvg_concentration_90", "hvg_advantage_80", "cell_support_log"]:
        df[f"z_{col}"] = zscore_by_group(df, col)
    df["info_composite"] = (
        df["z_log_response_energy"].fillna(0)
        + df["z_hvg_concentration_80"].fillna(0)
        + 0.5 * df["z_hvg_advantage_80"].fillna(0)
        + 0.25 * df["z_cell_support_log"].fillna(0)
    )
    return df


def make_pairs(df: pd.DataFrame, axis: str, rng: random.Random) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for dataset, ds in df.groupby("dataset", sort=True):
        dataset_pairs = 0
        for key, group in ds.groupby(["perturbation_type", "gene_count_bin"], sort=True):
            if len(group) < 6:
                continue
            lo_q = group[axis].quantile(1 / 3)
            hi_q = group[axis].quantile(2 / 3)
            lows = group[group[axis] <= lo_q].sort_values(axis, ascending=True).copy()
            highs = group[group[axis] >= hi_q].sort_values(axis, ascending=False).copy()
            high_rows = highs.to_dict("records")
            low_rows = lows.to_dict("records")
            rng.shuffle(high_rows)
            rng.shuffle(low_rows)
            n = min(len(high_rows), len(low_rows), MAX_PAIRS_PER_DATASET - dataset_pairs)
            for hi, lo in zip(high_rows[:n], low_rows[:n]):
                pairs.append(
                    {
                        "axis": axis,
                        "dataset": dataset,
                        "perturbation_type": key[0],
                        "gene_count_bin": key[1],
                        "high_condition": hi["condition"],
                        "low_condition": lo["condition"],
                        "high_value": float(hi[axis]),
                        "low_value": float(lo[axis]),
                        "axis_delta": float(hi[axis] - lo[axis]),
                        "high_response_energy": float(hi["response_energy"]),
                        "low_response_energy": float(lo["response_energy"]),
                        "high_hvg_k80": float(hi["hvg_k80"]),
                        "low_hvg_k80": float(lo["hvg_k80"]),
                    }
                )
            dataset_pairs += n
            if dataset_pairs >= MAX_PAIRS_PER_DATASET:
                break
    return pairs


def write_split(parent: dict[str, Any], pairs: list[dict[str, Any]], out_high: Path, out_low: Path) -> None:
    high = json.loads(json.dumps(parent))
    low = json.loads(json.dumps(parent))
    by_dataset: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"high": [], "low": []})
    for row in pairs:
        by_dataset[row["dataset"]]["high"].append(str(row["high_condition"]))
        by_dataset[row["dataset"]]["low"].append(str(row["low_condition"]))
    for dataset, groups in parent.items():
        high[dataset]["train"] = sorted(set(by_dataset.get(dataset, {}).get("high", [])))
        low[dataset]["train"] = sorted(set(by_dataset.get(dataset, {}).get("low", [])))
    write_json(out_high, high)
    write_json(out_low, low)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--axis", default="info_composite")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    parent = load_json(PARENT_SPLIT)
    meta = load_json(COND_META)
    cov = pd.read_csv(COVERAGE)
    table = build_condition_table(parent, meta, cov)
    rng = random.Random(SEED)
    pairs = make_pairs(table, args.axis, rng)

    table.to_csv(args.out_dir / "condition_information_table.csv", index=False)
    pair_df = pd.DataFrame(pairs)
    pair_df.to_csv(args.out_dir / "condition_information_matched_pairs.csv", index=False)

    high_split = args.out_dir / f"draft_split_seed42_xverse_{args.axis}_high_from_cap120_all_v2.json"
    low_split = args.out_dir / f"draft_split_seed42_xverse_{args.axis}_low_from_cap120_all_v2.json"
    write_split(parent, pairs, high_split, low_split)

    dataset_rows = []
    if len(pair_df):
        for dataset, sub in pair_df.groupby("dataset", sort=True):
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "matched_pairs": int(len(sub)),
                    "mean_axis_delta": float(sub["axis_delta"].mean()),
                    "perturbation_types": int(sub["perturbation_type"].nunique()),
                    "gene_count_bins": int(sub["gene_count_bin"].nunique()),
                }
            )
    dataset_df = pd.DataFrame(dataset_rows)
    dataset_df.to_csv(args.out_dir / "condition_information_dataset_summary.csv", index=False)

    total_pairs = len(pair_df)
    datasets_with_pairs = int(dataset_df["dataset"].nunique()) if len(dataset_df) else 0
    status = "scaling_v2_condition_information_draft_partial_no_gpu"
    if total_pairs >= 300 and datasets_with_pairs >= 8:
        status = "scaling_v2_condition_information_draft_feasible_review_no_gpu"
    payload = {
        "timestamp": now_cst(),
        "status": status,
        "axis": args.axis,
        "parent_split": str(PARENT_SPLIT),
        "covered_parent_train_conditions": int(len(table)),
        "matched_pairs": int(total_pairs),
        "datasets_with_pairs": datasets_with_pairs,
        "max_pairs_per_dataset": MAX_PAIRS_PER_DATASET,
        "high_split": str(high_split),
        "low_split": str(low_split),
        "boundary": "CPU/report-only draft; not launch-ready",
    }
    write_json(args.out_dir / "latentfm_scaling_v2_condition_information_draft_splits_20260628.json", payload)

    top_datasets = dataset_df.sort_values("matched_pairs", ascending=False).head(12) if len(dataset_df) else dataset_df
    report = args.out_dir / "LATENTFM_SCALING_V2_CONDITION_INFORMATION_DRAFT_SPLITS_20260628.md"
    lines = [
        "# LatentFM Scaling V2 Condition-Information Draft Splits",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`.",
        "",
        "## Boundary",
        "",
        "- CPU/report-only draft split feasibility artifact.",
        "- Uses only exact-covered parent train conditions and condition metadata.",
        "- High/low conditions are matched within dataset, perturbation type, and gene-count bin.",
        "- Draft splits are not launch-ready and do not authorize training.",
        "",
        "## Summary",
        "",
        f"- parent split: `{PARENT_SPLIT}`",
        f"- axis: `{args.axis}`",
        f"- exact-covered parent train conditions: `{len(table)}`",
        f"- matched pairs: `{total_pairs}`",
        f"- datasets with pairs: `{datasets_with_pairs}`",
        "",
        "| dataset | matched pairs | mean axis delta | perturbation types | gene-count bins |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in top_datasets.iterrows():
        lines.append(
            f"| {row['dataset']} | {int(row['matched_pairs'])} | {float(row['mean_axis_delta']):.3f} | {int(row['perturbation_types'])} | {int(row['gene_count_bins'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if status.endswith("feasible_review_no_gpu"):
        lines.extend(
            [
                "- This draft has enough condition pairs for external review and leakage/no-harm packet construction.",
                "- It still does not authorize GPU until split provenance, validation design, dual baselines, and no-harm gates are written and audited.",
            ]
        )
    else:
        lines.extend(
            [
                "- This draft is not feasible enough for a GPU packet.",
                "- Keep it as evidence for the next deliberately matched split-family design.",
            ]
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- condition table: `{args.out_dir / 'condition_information_table.csv'}`",
            f"- matched pairs: `{args.out_dir / 'condition_information_matched_pairs.csv'}`",
            f"- dataset summary: `{args.out_dir / 'condition_information_dataset_summary.csv'}`",
            f"- high draft split: `{high_split}`",
            f"- low draft split: `{low_split}`",
            f"- JSON: `{args.out_dir / 'latentfm_scaling_v2_condition_information_draft_splits_20260628.json'}`",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "matched_pairs": total_pairs, "datasets_with_pairs": datasets_with_pairs}, indent=2))


if __name__ == "__main__":
    main()
