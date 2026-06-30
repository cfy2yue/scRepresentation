#!/usr/bin/env python3
"""ZSCAPE-inspired condition response-neighborhood concordance gate.

This is CPU/report-only. It deliberately does not relaunch the closed
response-residualized support branch. Instead, it tests whether local
cross-dataset support and residual-vector direction agree strongly enough to
define a prospective high/low design that survives an axis-specific
direction-shuffle null before any GPU is considered.
"""

from __future__ import annotations

import argparse
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
DEFAULT_ROWS = ROOT / "reports/condition_neighborhood_support_gate_20260629/condition_neighborhood_support_rows.csv"
DEFAULT_PARENT_SPLIT = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
DEFAULT_NULL_PANEL = ROOT / "reports/condition_neighborhood_response_resid_null_variance_panel_20260629/latentfm_condition_neighborhood_response_resid_null_variance_panel_20260629.json"
DEFAULT_OUT_DIR = ROOT / "reports/zscape_condition_response_neighborhood_gate_20260630"
DEFAULT_SPLIT_DIR = ROOT / "dataset/biFlow_data/xverse_zscape_condition_response_neighborhood_splits_20260630"

CONFIGS = [
    {
        "name": f"q{int(q * 100)}_resp{resp}_cell{cell}_cross{cross}",
        "quantile": q,
        "response_log_caliper": resp,
        "cell_log_caliper": cell,
        "cross_count_caliper": cross,
        "max_pairs": 320,
        "max_per_side_dataset": 50,
        "max_per_dataset_pair": 30,
    }
    for q in [0.20, 0.25, 0.30]
    for resp in [0.35, 0.50]
    for cell in [0.50, 0.75]
    for cross in [4, 8]
]


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def load_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def state_bin(entropy: Any) -> str:
    val = float(entropy) if pd.notna(entropy) else 0.0
    if val <= 1e-8:
        return "zero"
    if val <= 1.0:
        return "low"
    if val <= 2.0:
        return "mid"
    return "high"


def target_bin(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        val = 0.0
    return "has_target_cross" if val > 0 else "no_target_cross"


def support_count_bin(value: Any) -> str:
    try:
        val = int(float(value))
    except (TypeError, ValueError):
        val = 0
    if val <= 2:
        return "le2"
    if val <= 5:
        return "3to5"
    if val <= 10:
        return "6to10"
    return "gt10"


def smd(pos: pd.Series, neg: pd.Series) -> float:
    x = pd.to_numeric(pos, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(neg, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    pooled = math.sqrt((float(np.var(x, ddof=1)) + float(np.var(y, ddof=1))) / 2.0)
    if pooled <= 1e-12:
        return 0.0
    return float((float(np.mean(x)) - float(np.mean(y))) / pooled)


def auc_discriminability(pos: pd.Series, neg: pd.Series) -> float:
    x = pd.to_numeric(pos, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(neg, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    vals = np.concatenate([x, y])
    ranks = pd.Series(vals).rank(method="average").to_numpy(dtype=float)
    rank_x = ranks[: len(x)]
    auc = (float(rank_x.sum()) - len(x) * (len(x) + 1) / 2.0) / (len(x) * len(y))
    return float(max(auc, 1.0 - auc))


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    columns = [np.ones(len(y), dtype=float)]
    for j in range(x.shape[1]):
        col = x[:, j].astype(float)
        fill = float(np.nanmean(col)) if np.isfinite(col).any() else 0.0
        col = np.where(np.isfinite(col), col, fill)
        std = float(np.std(col))
        if std > 1e-12:
            columns.append((col - float(np.mean(col))) / std)
    design = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def zscore(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").astype(float)
    mean = float(vals.mean())
    std = float(vals.std(ddof=0))
    if not math.isfinite(std) or std <= 1e-12:
        return pd.Series(np.zeros(len(vals)), index=vals.index)
    return (vals - mean) / std


def load_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    rows["key"] = rows["dataset"].astype(str) + "||" + rows["condition"].astype(str)
    rows["log_response_norm"] = np.log1p(pd.to_numeric(rows["response_norm"], errors="coerce"))
    rows["log_n_gt"] = np.log1p(pd.to_numeric(rows["n_gt"], errors="coerce"))
    rows["log_n_ctrl"] = np.log1p(pd.to_numeric(rows["n_ctrl"], errors="coerce"))
    rows["exact_bool"] = rows["exact_response_available"].astype(str).str.lower().isin(["true", "1", "yes"])
    rows["state_bin"] = rows["max_state_entropy"].map(state_bin)
    rows["target_availability_bin"] = rows["same_target_cross_dataset_total"].map(target_bin)
    rows["support_count_bin"] = rows["cross_dataset_neighbor_count_top20"].map(support_count_bin)
    rows["directional_alignment"] = pd.to_numeric(rows["mean_top5_cross_dataset_cosine"], errors="coerce")
    rows["directional_alignment"] = rows["directional_alignment"].fillna(
        pd.to_numeric(rows["best_cross_dataset_cosine"], errors="coerce")
    )
    rows = rows[
        pd.to_numeric(rows["cross_dataset_neighbor_count_top20"], errors="coerce").fillna(0) >= 3
    ].copy()
    rows = rows[np.isfinite(pd.to_numeric(rows["directional_alignment"], errors="coerce"))].copy()
    return rows


def prepare_scores(rows: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    work = rows.copy().reset_index(drop=True)
    base_columns = [
        work["log_response_norm"].to_numpy(dtype=float),
        work["log_n_gt"].to_numpy(dtype=float),
        work["log_n_ctrl"].to_numpy(dtype=float),
        work["exact_bool"].astype(float).to_numpy(dtype=float),
        pd.to_numeric(work["max_state_entropy"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        pd.to_numeric(work["same_target_cross_dataset_total"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
    ]
    for ptype in sorted(work["perturbation_type_raw"].astype(str).unique())[1:]:
        base_columns.append((work["perturbation_type_raw"].astype(str).to_numpy() == ptype).astype(float))
    for dataset in sorted(work["dataset"].astype(str).unique())[1:]:
        base_columns.append((work["dataset"].astype(str).to_numpy() == dataset).astype(float))
    base_x = np.column_stack(base_columns)
    support = pd.to_numeric(work["neighbor_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    work["support_resid_score"] = residualize(support, base_x)

    direction_columns = base_columns + [
        np.log1p(pd.to_numeric(work["cross_dataset_neighbor_count_top20"], errors="coerce").fillna(0.0)).to_numpy(dtype=float),
        pd.to_numeric(work["cross_dataset_effective_count_top20"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        pd.to_numeric(work["same_ptype_cross_dataset_neighbor_count_top20"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        pd.to_numeric(work["same_dataset_fraction_top20"], errors="coerce").fillna(1.0).to_numpy(dtype=float),
        work["support_resid_score"].to_numpy(dtype=float),
    ]
    direction = pd.to_numeric(work["directional_alignment"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    work["direction_resid_score"] = residualize(direction, np.column_stack(direction_columns))
    if seed is not None:
        rng = np.random.default_rng(seed)
        shuffled = work["direction_resid_score"].to_numpy(dtype=float).copy()
        for _, idx in work.groupby(
            ["perturbation_type_raw", "exact_bool", "state_bin", "target_availability_bin", "support_count_bin"],
            dropna=False,
        ).groups.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            shuffled[idx_arr] = rng.permutation(shuffled[idx_arr])
        work["direction_resid_score"] = shuffled
    work["support_z"] = zscore(work["support_resid_score"])
    work["direction_z"] = zscore(work["direction_resid_score"])
    work["concordance_score"] = np.minimum(work["support_z"], work["direction_z"])
    work["concordance_sum_score"] = work["support_z"] + work["direction_z"] - 0.25 * (work["support_z"] - work["direction_z"]).abs()
    return work


def build_candidates(scored: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    high_thr = float(scored["concordance_score"].quantile(1.0 - float(config["quantile"])))
    low_thr = float(scored["concordance_score"].quantile(float(config["quantile"])))
    high = scored[scored["concordance_score"] >= high_thr].copy()
    low = scored[scored["concordance_score"] <= low_thr].copy()
    rows: list[dict[str, Any]] = []
    for _, h in high.iterrows():
        sub = low[
            low["perturbation_type_raw"].astype(str).eq(str(h["perturbation_type_raw"]))
            & low["exact_bool"].eq(bool(h["exact_bool"]))
            & low["state_bin"].astype(str).eq(str(h["state_bin"]))
            & low["target_availability_bin"].astype(str).eq(str(h["target_availability_bin"]))
            & low["support_count_bin"].astype(str).eq(str(h["support_count_bin"]))
            & low["dataset"].astype(str).ne(str(h["dataset"]))
        ].copy()
        if sub.empty:
            continue
        sub["response_log_abs_diff"] = (sub["log_response_norm"] - float(h["log_response_norm"])).abs()
        sub["n_gt_log_abs_diff"] = (sub["log_n_gt"] - float(h["log_n_gt"])).abs()
        sub["n_ctrl_log_abs_diff"] = (sub["log_n_ctrl"] - float(h["log_n_ctrl"])).abs()
        sub["cross_count_abs_diff"] = (
            pd.to_numeric(sub["cross_dataset_neighbor_count_top20"], errors="coerce")
            - float(h["cross_dataset_neighbor_count_top20"])
        ).abs()
        sub = sub[
            (sub["response_log_abs_diff"] <= float(config["response_log_caliper"]))
            & (sub["n_gt_log_abs_diff"] <= float(config["cell_log_caliper"]))
            & (sub["n_ctrl_log_abs_diff"] <= float(config["cell_log_caliper"]))
            & (sub["cross_count_abs_diff"] <= float(config["cross_count_caliper"]))
        ].copy()
        if sub.empty:
            continue
        for _, l in sub.iterrows():
            gap = float(h["concordance_score"] - l["concordance_score"])
            if gap <= 0:
                continue
            rows.append(
                {
                    "high_key": h["key"],
                    "low_key": l["key"],
                    "high_dataset": h["dataset"],
                    "low_dataset": l["dataset"],
                    "high_condition": h["condition"],
                    "low_condition": l["condition"],
                    "perturbation_type_raw": h["perturbation_type_raw"],
                    "high_concordance_score": float(h["concordance_score"]),
                    "low_concordance_score": float(l["concordance_score"]),
                    "concordance_gap": gap,
                    "high_support_resid_score": float(h["support_resid_score"]),
                    "low_support_resid_score": float(l["support_resid_score"]),
                    "high_direction_resid_score": float(h["direction_resid_score"]),
                    "low_direction_resid_score": float(l["direction_resid_score"]),
                    "support_z_gap": float(h["support_z"] - l["support_z"]),
                    "direction_z_gap": float(h["direction_z"] - l["direction_z"]),
                    "high_response_norm": float(h["response_norm"]),
                    "low_response_norm": float(l["response_norm"]),
                    "high_n_gt": int(h["n_gt"]),
                    "low_n_gt": int(l["n_gt"]),
                    "high_n_ctrl": int(h["n_ctrl"]),
                    "low_n_ctrl": int(l["n_ctrl"]),
                    "high_max_state_entropy": float(h["max_state_entropy"]),
                    "low_max_state_entropy": float(l["max_state_entropy"]),
                    "high_same_target_cross_dataset_total": float(h["same_target_cross_dataset_total"]),
                    "low_same_target_cross_dataset_total": float(l["same_target_cross_dataset_total"]),
                    "high_cross_count": int(h["cross_dataset_neighbor_count_top20"]),
                    "low_cross_count": int(l["cross_dataset_neighbor_count_top20"]),
                    "high_same_dataset_fraction": float(h["same_dataset_fraction_top20"]),
                    "low_same_dataset_fraction": float(l["same_dataset_fraction_top20"]),
                    "high_exact_bool": bool(h["exact_bool"]),
                    "low_exact_bool": bool(l["exact_bool"]),
                    "response_log_abs_diff": float(l["response_log_abs_diff"]),
                    "n_gt_log_abs_diff": float(l["n_gt_log_abs_diff"]),
                    "n_ctrl_log_abs_diff": float(l["n_ctrl_log_abs_diff"]),
                    "cross_count_abs_diff": float(l["cross_count_abs_diff"]),
                    "dataset_pair": f"{h['dataset']}->{l['dataset']}",
                }
            )
    return pd.DataFrame(rows)


def greedy_select(candidates: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    work = candidates.copy()
    work["objective"] = (
        pd.to_numeric(work["concordance_gap"], errors="coerce")
        + 0.20 * pd.to_numeric(work["support_z_gap"], errors="coerce")
        + 0.20 * pd.to_numeric(work["direction_z_gap"], errors="coerce")
        - 0.25 * pd.to_numeric(work["response_log_abs_diff"], errors="coerce")
        - 0.10 * pd.to_numeric(work["n_gt_log_abs_diff"], errors="coerce")
        - 0.10 * pd.to_numeric(work["n_ctrl_log_abs_diff"], errors="coerce")
        - 0.05 * pd.to_numeric(work["cross_count_abs_diff"], errors="coerce")
    )
    high_used: set[str] = set()
    low_used: set[str] = set()
    high_ds: Counter[str] = Counter()
    low_ds: Counter[str] = Counter()
    pair_dir: Counter[str] = Counter()
    selected: list[pd.Series] = []
    for _, row in work.sort_values(["objective", "concordance_gap"], ascending=[False, False]).iterrows():
        hkey = str(row["high_key"])
        lkey = str(row["low_key"])
        hds = str(row["high_dataset"])
        lds = str(row["low_dataset"])
        dkey = str(row["dataset_pair"])
        if hkey in high_used or lkey in low_used:
            continue
        if high_ds[hds] >= int(config["max_per_side_dataset"]) or low_ds[lds] >= int(config["max_per_side_dataset"]):
            continue
        if pair_dir[dkey] >= int(config["max_per_dataset_pair"]):
            continue
        high_used.add(hkey)
        low_used.add(lkey)
        high_ds[hds] += 1
        low_ds[lds] += 1
        pair_dir[dkey] += 1
        selected.append(row)
        if len(selected) >= int(config["max_pairs"]):
            break
    return pd.DataFrame(selected)


def side_rows(scored: pd.DataFrame, selected: pd.DataFrame, side: str) -> pd.DataFrame:
    keys = set(zip(selected[f"{side}_dataset"].astype(str), selected[f"{side}_condition"].astype(str)))
    mask = scored.apply(lambda r: (str(r["dataset"]), str(r["condition"])) in keys, axis=1)
    return scored[mask].copy()


def balance_rows(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    features = [
        "concordance_score",
        "support_resid_score",
        "direction_resid_score",
        "response_norm",
        "n_gt",
        "n_ctrl",
        "max_state_entropy",
        "same_target_cross_dataset_total",
        "cross_dataset_neighbor_count_top20",
        "cross_dataset_effective_count_top20",
        "same_dataset_fraction_top20",
    ]
    for feature in features:
        rows.append(
            {
                "feature": feature,
                "high_mean": float(pd.to_numeric(high[feature], errors="coerce").mean()),
                "low_mean": float(pd.to_numeric(low[feature], errors="coerce").mean()),
                "high_median": float(pd.to_numeric(high[feature], errors="coerce").median()),
                "low_median": float(pd.to_numeric(low[feature], errors="coerce").median()),
                "smd_high_minus_low": smd(high[feature], low[feature]),
                "auc_discriminability": auc_discriminability(high[feature], low[feature]),
            }
        )
    rows.append(
        {
            "feature": "exact_bool",
            "high_mean": float(high["exact_bool"].astype(bool).mean()),
            "low_mean": float(low["exact_bool"].astype(bool).mean()),
            "high_median": float(high["exact_bool"].astype(bool).median()),
            "low_median": float(low["exact_bool"].astype(bool).median()),
            "smd_high_minus_low": smd(high["exact_bool"].astype(int), low["exact_bool"].astype(int)),
            "auc_discriminability": auc_discriminability(high["exact_bool"].astype(int), low["exact_bool"].astype(int)),
        }
    )
    return pd.DataFrame(rows)


def side_fraction(selected: pd.DataFrame, side: str) -> float:
    if selected.empty:
        return 0.0
    return float(selected[f"{side}_dataset"].astype(str).value_counts(normalize=True).max())


def pair_direction_fraction(selected: pd.DataFrame) -> float:
    if selected.empty:
        return 0.0
    return float(selected["dataset_pair"].astype(str).value_counts(normalize=True).max())


def per_ptype_ready(selected: pd.DataFrame) -> int:
    if selected.empty:
        return 0
    ready = 0
    for _, part in selected.groupby("perturbation_type_raw"):
        if (
            len(part) >= 40
            and part["high_key"].nunique() >= 40
            and part["low_key"].nunique() >= 40
            and part["high_dataset"].nunique() >= 3
            and part["low_dataset"].nunique() >= 3
        ):
            ready += 1
    return ready


def assess(scored: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    candidates = build_candidates(scored, config)
    selected = greedy_select(candidates, config)
    high = side_rows(scored, selected, "high") if not selected.empty else pd.DataFrame()
    low = side_rows(scored, selected, "low") if not selected.empty else pd.DataFrame()
    balance = balance_rows(high, low) if not selected.empty else pd.DataFrame()
    cov_features = [
        "response_norm",
        "n_gt",
        "n_ctrl",
        "max_state_entropy",
        "same_target_cross_dataset_total",
        "cross_dataset_neighbor_count_top20",
        "cross_dataset_effective_count_top20",
        "same_dataset_fraction_top20",
        "exact_bool",
    ]
    cov = balance[balance["feature"].isin(cov_features)] if not balance.empty else pd.DataFrame()
    max_cov_smd = float(cov["smd_high_minus_low"].abs().max()) if not cov.empty else float("nan")
    max_cov_auc = float(cov["auc_discriminability"].max()) if not cov.empty else float("nan")
    summary = {
        **config,
        "n_scored_conditions": int(len(scored)),
        "n_candidates": int(len(candidates)),
        "n_pairs_unique": int(len(selected)),
        "n_high_conditions": int(selected["high_key"].nunique()) if not selected.empty else 0,
        "n_low_conditions": int(selected["low_key"].nunique()) if not selected.empty else 0,
        "n_datasets_total": int(len(set(selected["high_dataset"].astype(str)) | set(selected["low_dataset"].astype(str)))) if not selected.empty else 0,
        "n_perturbation_types": int(selected["perturbation_type_raw"].nunique()) if not selected.empty else 0,
        "n_claim_ready_ptypes": per_ptype_ready(selected),
        "median_concordance_gap": float(selected["concordance_gap"].median()) if not selected.empty else 0.0,
        "median_support_z_gap": float(selected["support_z_gap"].median()) if not selected.empty else 0.0,
        "median_direction_z_gap": float(selected["direction_z_gap"].median()) if not selected.empty else 0.0,
        "max_abs_covariate_smd": max_cov_smd,
        "max_covariate_auc": max_cov_auc,
        "top_dataset_pair_direction_fraction": pair_direction_fraction(selected),
        "top_high_dataset_fraction": side_fraction(selected, "high"),
        "top_low_dataset_fraction": side_fraction(selected, "low"),
    }
    reasons: list[str] = []
    if summary["n_pairs_unique"] < 300:
        reasons.append("pairs_below_300")
    if summary["n_datasets_total"] < 15:
        reasons.append("datasets_below_15")
    if summary["n_perturbation_types"] < 3:
        reasons.append("perturbation_types_below_3")
    if summary["n_claim_ready_ptypes"] < 2:
        reasons.append("claim_ready_ptypes_below_2")
    if math.isfinite(summary["max_abs_covariate_smd"]) and summary["max_abs_covariate_smd"] > 0.15:
        reasons.append("max_covariate_smd_above_0p15")
    if math.isfinite(summary["max_covariate_auc"]) and summary["max_covariate_auc"] > 0.60:
        reasons.append("max_covariate_auc_above_0p60")
    if summary["top_dataset_pair_direction_fraction"] > 0.15:
        reasons.append("dataset_pair_direction_above_15pct")
    if summary["top_high_dataset_fraction"] > 0.25 or summary["top_low_dataset_fraction"] > 0.25:
        reasons.append("side_dataset_fraction_above_25pct")
    summary["reasons"] = ";".join(reasons)
    summary["strict_pass"] = not reasons
    return summary, selected, balance


def run_sweep(base_rows: pd.DataFrame, seed: int | None) -> tuple[pd.DataFrame, dict[str, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    scored = prepare_scores(base_rows, seed=seed)
    summaries: list[dict[str, Any]] = []
    selections: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for config in CONFIGS:
        summary, selected, balance = assess(scored, config)
        if seed is not None:
            summary["null_seed"] = seed
        summaries.append(summary)
        selections[str(summary["name"])] = (selected, balance)
    summary_df = pd.DataFrame(summaries)
    return summary_df, selections, scored


def choose_best(configs: pd.DataFrame) -> pd.Series:
    work = configs.copy()
    work["score"] = (
        1000 * work["strict_pass"].astype(bool).astype(int)
        + work["n_pairs_unique"]
        + 60 * work["n_claim_ready_ptypes"]
        + 20 * work["median_concordance_gap"]
        - 80 * work["max_abs_covariate_smd"].fillna(9)
        - 50 * work["max_covariate_auc"].fillna(1)
        - 20 * work["top_dataset_pair_direction_fraction"].fillna(1)
    )
    return work.sort_values("score", ascending=False).iloc[0]


def split_from_conditions(parent: dict[str, Any], selected: pd.DataFrame, side: str) -> dict[str, Any]:
    out = copy.deepcopy(parent)
    by_dataset = selected.groupby(f"{side}_dataset")[f"{side}_condition"].apply(lambda s: sorted(set(map(str, s)))).to_dict()
    for dataset in out:
        out[dataset]["train"] = by_dataset.get(dataset, [])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--parent-split", type=Path, default=DEFAULT_PARENT_SPLIT)
    parser.add_argument("--null-panel-json", type=Path, default=DEFAULT_NULL_PANEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--null-seeds", type=int, nargs="*", default=[43, 44, 45, 46])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.split_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.rows)
    real_configs, real_selections, scored = run_sweep(rows, seed=None)
    best = choose_best(real_configs)
    selected, balance = real_selections[str(best["name"])]

    null_frames = []
    for seed in args.null_seeds:
        null_configs, _, _ = run_sweep(rows, seed=seed)
        null_frames.append(null_configs)
    null_df = pd.concat(null_frames, ignore_index=True) if null_frames else pd.DataFrame()
    null_strict = int(null_df["strict_pass"].astype(bool).sum()) if not null_df.empty else 0
    null_p95_pairs = float(null_df["n_pairs_unique"].quantile(0.95)) if not null_df.empty else 0.0
    null_p95_gap = float(null_df["median_concordance_gap"].quantile(0.95)) if not null_df.empty else 0.0

    null_panel = load_json(args.null_panel_json) if args.null_panel_json.exists() else {}
    null_decision = null_panel.get("decision", null_panel) if isinstance(null_panel, dict) else {}
    req_cross = null_decision.get("future_axis_required_cross_gap", None)
    req_family = null_decision.get("future_axis_required_family_gap", None)

    real_pass = bool(best["strict_pass"])
    null_pass = null_strict == 0 and null_p95_pairs < 250 and null_p95_gap < 0.50 * float(best["median_concordance_gap"])
    status = (
        "zscape_condition_response_neighborhood_gate_pass_external_audit_no_gpu"
        if real_pass and null_pass
        else "zscape_condition_response_neighborhood_gate_blocks_gpu"
    )
    reasons = []
    if not real_pass:
        reasons.append("real_concordance_design_failed_strict_balance")
    if null_strict > 0:
        reasons.append("direction_shuffle_null_reconstructs_strict_design")
    if null_p95_pairs >= 250:
        reasons.append("direction_shuffle_null_has_many_pairs")
    if null_p95_gap >= 0.50 * float(best["median_concordance_gap"]):
        reasons.append("direction_shuffle_null_gap_too_large")

    parent = load_json(args.parent_split)
    tag = f"concordance_{len(selected)}pair_{best['name']}"
    high_split = args.split_dir / f"split_seed42_xverse_zscape_condition_response_neighborhood_high_{tag}.json"
    low_split = args.split_dir / f"split_seed42_xverse_zscape_condition_response_neighborhood_low_{tag}.json"
    if not selected.empty:
        write_json(high_split, split_from_conditions(parent, selected, "high"))
        write_json(low_split, split_from_conditions(parent, selected, "low"))

    config_csv = args.out_dir / "zscape_condition_response_neighborhood_config_rows.csv"
    selected_csv = args.out_dir / "zscape_condition_response_neighborhood_selected_pairs.csv"
    balance_csv = args.out_dir / "zscape_condition_response_neighborhood_balance.csv"
    null_csv = args.out_dir / "zscape_condition_response_neighborhood_direction_shuffle_null_rows.csv"
    scored_csv = args.out_dir / "zscape_condition_response_neighborhood_scored_rows.csv"
    real_configs.to_csv(config_csv, index=False)
    selected.to_csv(selected_csv, index=False)
    balance.to_csv(balance_csv, index=False)
    null_df.to_csv(null_csv, index=False)
    scored.to_csv(scored_csv, index=False)

    payload = {
        "created_at": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "reasons": reasons,
        "selected_config_name": str(best["name"]),
        "summary": {
            k: (bool(v) if isinstance(v, np.bool_) else float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v)
            for k, v in best.to_dict().items()
            if k != "reasons"
        },
        "real_reasons": str(best.get("reasons", "")),
        "null_summary": {
            "null_seeds": args.null_seeds,
            "strict_pass_configs": null_strict,
            "p95_pairs": null_p95_pairs,
            "p95_median_concordance_gap": null_p95_gap,
        },
        "future_gpu_effect_thresholds_from_prior_null_panel": {
            "required_cross_pp_gap": req_cross,
            "required_family_pp_gap": req_family,
        },
        "inputs": {
            "rows": str(args.rows),
            "parent_split": str(args.parent_split),
            "prior_support_null_panel": str(args.null_panel_json),
        },
        "outputs": {
            "report": str(args.out_dir / "LATENTFM_ZSCAPE_CONDITION_RESPONSE_NEIGHBORHOOD_GATE_20260630.md"),
            "json": str(args.out_dir / "latentfm_zscape_condition_response_neighborhood_gate_20260630.json"),
            "configs": str(config_csv),
            "selected_pairs": str(selected_csv),
            "balance": str(balance_csv),
            "direction_shuffle_null_rows": str(null_csv),
            "scored_rows": str(scored_csv),
            "high_split": str(high_split) if not selected.empty else "",
            "low_split": str(low_split) if not selected.empty else "",
        },
        "boundary": "cpu_report_only_zscape_inspired_condition_response_neighborhood_no_training_no_inference_no_gpu_no_canonical_multi_no_trackc_query",
    }
    json_path = args.out_dir / "latentfm_zscape_condition_response_neighborhood_gate_20260630.json"
    write_json(json_path, payload)

    top = real_configs.sort_values(["strict_pass", "n_pairs_unique"], ascending=[False, False]).head(12)
    report = args.out_dir / "LATENTFM_ZSCAPE_CONDITION_RESPONSE_NEIGHBORHOOD_GATE_20260630.md"
    lines = [
        "# LatentFM ZSCAPE Condition Response-Neighborhood Gate",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only gate over parent train condition-neighborhood rows.",
        "* Tests a new concordance axis: high cross-dataset support residual and high residual-vector direction agreement must coincide.",
        "* Direction-shuffle nulls are run before any GPU route is considered.",
        "* No training, inference, GPU, canonical multi, Track C query, or checkpoint selection.",
        "",
        "## Selected Real Design",
        "",
        f"* Config: `{payload['selected_config_name']}`.",
        f"* Real strict-pass reasons: `{payload['real_reasons'] if payload['real_reasons'] else 'none'}`.",
        f"* Pairs: `{int(best['n_pairs_unique'])}`; datasets: `{int(best['n_datasets_total'])}`; ptypes: `{int(best['n_perturbation_types'])}`; claim-ready ptypes: `{int(best['n_claim_ready_ptypes'])}`.",
        f"* Median concordance/support/direction z gaps: `{fmt(best['median_concordance_gap'])}` / `{fmt(best['median_support_z_gap'])}` / `{fmt(best['median_direction_z_gap'])}`.",
        f"* Max covariate SMD/AUC: `{fmt(best['max_abs_covariate_smd'])}` / `{fmt(best['max_covariate_auc'])}`.",
        "",
        "## Direction-Shuffle Null",
        "",
        f"* Null seeds: `{','.join(map(str, args.null_seeds))}`.",
        f"* Strict-pass null configs: `{null_strict}`.",
        f"* Null p95 pairs: `{fmt(null_p95_pairs)}`.",
        f"* Null p95 median concordance gap: `{fmt(null_p95_gap)}`.",
        f"* Block reasons: `{'; '.join(reasons) if reasons else 'none'}`.",
        "",
        "## Prior Matched-Split Null Calibration",
        "",
        f"* Any future GPU high/low axis must exceed prior required cross pp gap: `{fmt(req_cross)}`.",
        f"* Any future GPU high/low axis must exceed prior required family pp gap: `{fmt(req_family)}`.",
        "",
        "## Config Sweep",
        "",
        "| config | q | pairs | datasets | ptypes | claim ptypes | conc gap | support gap | direction gap | cov SMD | cov AUC | strict | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| `{row['name']}` | `{fmt(row['quantile'], 2)}` | `{int(row['n_pairs_unique'])}` | `{int(row['n_datasets_total'])}` | `{int(row['n_perturbation_types'])}` | `{int(row['n_claim_ready_ptypes'])}` | "
            f"`{fmt(row['median_concordance_gap'])}` | `{fmt(row['median_support_z_gap'])}` | `{fmt(row['median_direction_z_gap'])}` | `{fmt(row['max_abs_covariate_smd'])}` | `{fmt(row['max_covariate_auc'])}` | `{bool(row['strict_pass'])}` | `{row['reasons']}` |"
        )
    lines.extend(["", "## Balance", "", "| feature | high mean | low mean | SMD | AUC |", "|---|---:|---:|---:|---:|"])
    for _, row in balance.iterrows():
        lines.append(
            f"| `{row['feature']}` | `{fmt(row['high_mean'])}` | `{fmt(row['low_mean'])}` | `{fmt(row['smd_high_minus_low'])}` | `{fmt(row['auc_discriminability'])}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "* If status blocks GPU, do not mutate this into a GPU launcher; the null says the admission design is not specific enough.",
            "* If status passes, next step is external audit and a launcher guard, not immediate promotion.",
            "* This is not a repeat of the closed response-residualized support branch because direction concordance and direction-shuffle nulls are mandatory before GPU.",
            "",
            "## Outputs",
            "",
            f"* Config rows: `{config_csv}`",
            f"* Selected pairs: `{selected_csv}`",
            f"* Balance: `{balance_csv}`",
            f"* Direction-shuffle null rows: `{null_csv}`",
            f"* JSON: `{json_path}`",
            f"* High split draft: `{high_split if not selected.empty else ''}`",
            f"* Low split draft: `{low_split if not selected.empty else ''}`",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "reasons": reasons, "report": str(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
