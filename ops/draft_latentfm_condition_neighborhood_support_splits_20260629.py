#!/usr/bin/env python3
"""Draft condition-neighborhood high/low support splits.

CPU/report-only. This converts the passed neighborhood-support feasibility
gate into auditable split JSON drafts, without launching training.
"""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
PARENT_SPLIT = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
ROWS_CSV = ROOT / "reports/condition_neighborhood_support_gate_20260629/condition_neighborhood_support_rows.csv"
PAIRS_CSV = ROOT / "reports/condition_neighborhood_support_gate_20260629/condition_neighborhood_support_matched_pairs.csv"
SPLIT_DIR = ROOT / "dataset/biFlow_data/xverse_condition_neighborhood_support_splits_20260629"
OUT_DIR = ROOT / "reports/condition_neighborhood_support_split_draft_20260629"
OUT_MD = OUT_DIR / "LATENTFM_CONDITION_NEIGHBORHOOD_SUPPORT_SPLIT_DRAFT_20260629.md"
OUT_JSON = OUT_DIR / "latentfm_condition_neighborhood_support_split_draft_20260629.json"
OUT_SELECTED = OUT_DIR / "condition_neighborhood_support_selected_pairs.csv"
OUT_BALANCE = OUT_DIR / "condition_neighborhood_support_split_balance.csv"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def load_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def select_unique_pairs(pairs: pd.DataFrame, max_per_dataset: int = 60, max_pairs: int = 260) -> pd.DataFrame:
    work = pairs.copy()
    work["high_key"] = work["high_dataset"].astype(str) + "||" + work["high_condition"].astype(str)
    work["low_key"] = work["low_dataset"].astype(str) + "||" + work["low_condition"].astype(str)
    high_used: set[str] = set()
    low_used: set[str] = set()
    high_ds: Counter[str] = Counter()
    low_ds: Counter[str] = Counter()
    selected: list[pd.Series] = []
    for _, row in work.sort_values(["score_gap", "log_cell_diff"], ascending=[False, True]).iterrows():
        high_key = str(row["high_key"])
        low_key = str(row["low_key"])
        if high_key in high_used or low_key in low_used:
            continue
        if high_ds[str(row["high_dataset"])] >= max_per_dataset:
            continue
        if low_ds[str(row["low_dataset"])] >= max_per_dataset:
            continue
        high_used.add(high_key)
        low_used.add(low_key)
        high_ds[str(row["high_dataset"])] += 1
        low_ds[str(row["low_dataset"])] += 1
        selected.append(row)
        if len(selected) >= max_pairs:
            break
    return pd.DataFrame(selected).drop(columns=["high_key", "low_key"], errors="ignore")


def split_from_conditions(parent: dict[str, Any], conditions: pd.DataFrame) -> dict[str, Any]:
    out = copy.deepcopy(parent)
    by_dataset = conditions.groupby("dataset")["condition"].apply(lambda s: sorted(set(map(str, s)))).to_dict()
    for dataset in out:
        out[dataset]["train"] = by_dataset.get(dataset, [])
    return out


def condition_subset(rows: pd.DataFrame, selected: pd.DataFrame, side: str) -> pd.DataFrame:
    key_col = f"{side}_dataset"
    cond_col = f"{side}_condition"
    keys = set(zip(selected[key_col].astype(str), selected[cond_col].astype(str)))
    mask = rows.apply(lambda r: (str(r["dataset"]), str(r["condition"])) in keys, axis=1)
    return rows[mask].copy()


def smd(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    pooled = math.sqrt((float(np.var(x, ddof=1)) + float(np.var(y, ddof=1))) / 2.0)
    if pooled <= 1e-12:
        return 0.0
    return float((float(np.mean(x)) - float(np.mean(y))) / pooled)


def balance_rows(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    continuous = [
        "neighbor_support_score",
        "cross_dataset_neighbor_count_top20",
        "same_dataset_fraction_top20",
        "best_cross_dataset_cosine",
        "n_gt",
        "response_norm",
        "max_state_entropy",
        "same_target_cross_dataset_total",
    ]
    for col in continuous:
        rows.append(
            {
                "feature": col,
                "high_mean": float(pd.to_numeric(high[col], errors="coerce").mean()),
                "low_mean": float(pd.to_numeric(low[col], errors="coerce").mean()),
                "high_median": float(pd.to_numeric(high[col], errors="coerce").median()),
                "low_median": float(pd.to_numeric(low[col], errors="coerce").median()),
                "smd_high_minus_low": smd(high[col], low[col]),
            }
        )
    rows.append(
        {
            "feature": "exact_response_available_fraction",
            "high_mean": float(high["exact_response_available"].astype(bool).mean()),
            "low_mean": float(low["exact_response_available"].astype(bool).mean()),
            "high_median": float(high["exact_response_available"].astype(bool).median()),
            "low_median": float(low["exact_response_available"].astype(bool).median()),
            "smd_high_minus_low": smd(high["exact_response_available"].astype(int), low["exact_response_available"].astype(int)),
        }
    )
    return pd.DataFrame(rows)


def summarize_counts(df: pd.DataFrame, col: str) -> str:
    return "; ".join(f"{k}:{v}" for k, v in df[col].astype(str).value_counts().head(12).items())


def decide(selected: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, balance: pd.DataFrame) -> tuple[str, list[str], dict[str, Any]]:
    datasets = set(high["dataset"].astype(str)) | set(low["dataset"].astype(str))
    ptypes = set(high["perturbation_type_raw"].astype(str)) | set(low["perturbation_type_raw"].astype(str))
    max_abs_covariate_smd = float(
        balance[~balance["feature"].isin(["neighbor_support_score", "cross_dataset_neighbor_count_top20", "same_dataset_fraction_top20"])]
        ["smd_high_minus_low"]
        .abs()
        .max()
    )
    summary = {
        "n_pairs_unique": int(len(selected)),
        "n_high_conditions": int(len(high)),
        "n_low_conditions": int(len(low)),
        "n_datasets": int(len(datasets)),
        "n_perturbation_types": int(len(ptypes)),
        "median_score_gap": float(selected["score_gap"].median()) if not selected.empty else 0.0,
        "max_abs_covariate_smd_excluding_support_features": max_abs_covariate_smd,
    }
    reasons: list[str] = []
    if summary["n_pairs_unique"] < 300:
        reasons.append("unique_matched_pairs_below_300")
    if summary["n_pairs_unique"] < 180:
        reasons.append("unique_matched_pairs_below_180_minimum")
    if summary["n_datasets"] < 15:
        reasons.append("datasets_below_15")
    if summary["n_perturbation_types"] < 3:
        reasons.append("perturbation_types_below_3")
    if summary["median_score_gap"] < 0.50:
        reasons.append("median_score_gap_below_0p50")
    if summary["max_abs_covariate_smd_excluding_support_features"] > 0.75:
        reasons.append("covariate_smd_above_0p75")
    if not reasons:
        status = "condition_neighborhood_support_split_draft_ready_for_external_audit_no_gpu"
    elif summary["n_pairs_unique"] >= 180 and summary["n_datasets"] >= 15 and summary["n_perturbation_types"] >= 3:
        status = "condition_neighborhood_support_split_draft_200pair_audit_or_mutate_no_gpu"
    else:
        status = "condition_neighborhood_support_split_draft_fail_no_gpu"
    return status, reasons, summary


def write_report(payload: dict[str, Any], balance: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame) -> None:
    summary = payload["summary"]
    lines = [
        "# Condition Neighborhood Support Split Draft",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only split drafting from train-only parent residual-neighborhood support rows.",
        "* Outputs split JSON drafts only; no training, inference, GPU, canonical multi, Track C query, or checkpoint selection.",
        "* Draft splits replace only the parent split `train` lists; existing eval/internal/canonical reference groups are copied from the parent split.",
        "",
        "## Summary",
        "",
        f"* Unique selected high-low pairs: `{summary['n_pairs_unique']}`.",
        f"* High/low train conditions: `{summary['n_high_conditions']}` / `{summary['n_low_conditions']}`.",
        f"* Datasets: `{summary['n_datasets']}`; perturbation types: `{summary['n_perturbation_types']}`.",
        f"* Median support-score gap: `{fmt(summary['median_score_gap'])}`.",
        f"* Max abs covariate SMD excluding support-defining features: `{fmt(summary['max_abs_covariate_smd_excluding_support_features'])}`.",
        f"* Reasons/blockers: `{'; '.join(payload['reasons']) if payload['reasons'] else 'none'}`.",
        "",
        "## Balance Table",
        "",
        "| feature | high mean | low mean | high median | low median | SMD high-low |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in balance.iterrows():
        lines.append(
            f"| `{row['feature']}` | `{fmt(row['high_mean'])}` | `{fmt(row['low_mean'])}` | "
            f"`{fmt(row['high_median'])}` | `{fmt(row['low_median'])}` | `{fmt(row['smd_high_minus_low'])}` |"
        )
    lines.extend(
        [
            "",
            "## Distribution Snapshot",
            "",
            f"* High ptypes: `{summarize_counts(high, 'perturbation_type_raw')}`.",
            f"* Low ptypes: `{summarize_counts(low, 'perturbation_type_raw')}`.",
            f"* High datasets: `{summarize_counts(high, 'dataset')}`.",
            f"* Low datasets: `{summarize_counts(low, 'dataset')}`.",
            "",
            "## Decision",
            "",
        ]
    )
    if payload["status"].endswith("ready_for_external_audit_no_gpu"):
        lines.append("* This draft is ready for external audit before any GPU launch.")
    elif payload["status"].endswith("200pair_audit_or_mutate_no_gpu"):
        lines.append(
            "* This draft is useful for external audit and design iteration, but it falls below the preferred 300 unique-pair gate. Do not launch GPU until the audit accepts a lower-n smoke or a mutation raises unique-pair feasibility."
        )
    else:
        lines.append("* Do not use this draft for GPU. Revise the support/matching definition or pivot.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"* High-support split: `{payload['outputs']['high_split']}`",
            f"* Low-support split: `{payload['outputs']['low_split']}`",
            f"* Selected pairs: `{OUT_SELECTED}`",
            f"* Balance CSV: `{OUT_BALANCE}`",
            f"* JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(ROWS_CSV)
    pairs = pd.read_csv(PAIRS_CSV)
    selected = select_unique_pairs(pairs, max_per_dataset=60, max_pairs=260)
    high = condition_subset(rows, selected, "high")
    low = condition_subset(rows, selected, "low")
    balance = balance_rows(high, low)
    status, reasons, summary = decide(selected, high, low, balance)
    parent = load_json(PARENT_SPLIT)
    high_split = split_from_conditions(parent, high)
    low_split = split_from_conditions(parent, low)
    tag = f"{len(selected)}pair_v1"
    high_path = SPLIT_DIR / f"split_seed42_xverse_condition_neighborhood_high_support_{tag}.json"
    low_path = SPLIT_DIR / f"split_seed42_xverse_condition_neighborhood_low_support_{tag}.json"
    write_json(high_path, high_split)
    write_json(low_path, low_split)
    selected.to_csv(OUT_SELECTED, index=False)
    balance.to_csv(OUT_BALANCE, index=False)
    payload = {
        "created_at": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "reasons": reasons,
        "summary": summary,
        "inputs": {
            "parent_split": str(PARENT_SPLIT),
            "rows": str(ROWS_CSV),
            "pairs": str(PAIRS_CSV),
        },
        "outputs": {
            "report": str(OUT_MD),
            "json": str(OUT_JSON),
            "high_split": str(high_path),
            "low_split": str(low_path),
            "selected_pairs": str(OUT_SELECTED),
            "balance": str(OUT_BALANCE),
        },
        "boundary": "cpu_report_only_split_draft_no_training_no_inference_no_gpu_no_canonical_multi_no_trackc_query",
    }
    write_json(OUT_JSON, payload)
    write_report(payload, balance, high, low)
    print(json.dumps({"status": status, "summary": summary, "high_split": str(high_path), "low_split": str(low_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
