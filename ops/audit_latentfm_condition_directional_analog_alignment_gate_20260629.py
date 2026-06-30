#!/usr/bin/env python3
"""Condition-level directional analog alignment gate.

CPU/report-only. This tests whether train-only residual response *direction*
agreement with cross-dataset analogs is a usable information axis after
removing the closed support/count route and obvious covariates.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
OPS = ROOT / "ops"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

from audit_latentfm_condition_neighborhood_response_residualized_support_gate_20260629 import (  # noqa: E402
    PARENT_SPLIT,
    auc_discriminability,
    load_json,
    load_rows,
    pair_direction_fraction,
    per_ptype_claims,
    side_dataset_fraction,
    smd,
    split_from_conditions,
    write_json,
)


OUT_DIR = ROOT / "reports/condition_directional_analog_alignment_gate_20260629"
SPLIT_DIR = ROOT / "dataset/biFlow_data/xverse_condition_directional_analog_splits_20260629"
OUT_MD = OUT_DIR / "LATENTFM_CONDITION_DIRECTIONAL_ANALOG_ALIGNMENT_GATE_20260629.md"
OUT_JSON = OUT_DIR / "latentfm_condition_directional_analog_alignment_gate_20260629.json"
OUT_CONFIGS = OUT_DIR / "condition_directional_analog_config_rows.csv"
OUT_SELECTED = OUT_DIR / "condition_directional_analog_selected_pairs.csv"
OUT_BALANCE = OUT_DIR / "condition_directional_analog_balance.csv"
OUT_CONTROLS = OUT_DIR / "condition_directional_analog_negative_controls.csv"


CONFIGS = [
    {
        "name": f"q{int(q * 100)}_resp{resp}_cell{cell}_supp{supp}_ds{int(ds)}",
        "quantile": q,
        "response_log_caliper": resp,
        "cell_log_caliper": cell,
        "cross_count_caliper": supp,
        "include_dataset_dummies": ds,
        "max_pairs": 320,
        "max_per_side_dataset": 50,
        "max_per_dataset_pair": 30,
    }
    for q in [0.20, 0.25, 0.30]
    for resp in [0.35, 0.50, 0.75]
    for cell in [0.50, 0.75]
    for supp in [2, 4, 8]
    for ds in [False, True]
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


def residualize_direction(rows: pd.DataFrame, include_dataset_dummies: bool) -> np.ndarray:
    columns: list[np.ndarray] = [
        np.ones(len(rows), dtype=float),
        rows["log_response_norm"].to_numpy(dtype=float),
        rows["log_n_gt"].to_numpy(dtype=float),
        rows["log_n_ctrl"].to_numpy(dtype=float),
        rows["exact_bool"].astype(float).to_numpy(dtype=float),
        pd.to_numeric(rows["max_state_entropy"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        pd.to_numeric(rows["same_target_cross_dataset_total"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        np.log1p(pd.to_numeric(rows["cross_dataset_neighbor_count_top20"], errors="coerce").fillna(0.0)).to_numpy(dtype=float),
        pd.to_numeric(rows["cross_dataset_effective_count_top20"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        pd.to_numeric(rows["same_ptype_cross_dataset_neighbor_count_top20"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        pd.to_numeric(rows["same_dataset_fraction_top20"], errors="coerce").fillna(1.0).to_numpy(dtype=float),
    ]
    for ptype in sorted(rows["perturbation_type_raw"].astype(str).unique())[1:]:
        columns.append((rows["perturbation_type_raw"].astype(str).to_numpy() == ptype).astype(float))
    if include_dataset_dummies:
        for dataset in sorted(rows["dataset"].astype(str).unique())[1:]:
            columns.append((rows["dataset"].astype(str).to_numpy() == dataset).astype(float))
    x = np.column_stack(columns).astype(float)
    y = pd.to_numeric(rows["directional_alignment"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


def prepare_rows(rows: pd.DataFrame, include_dataset_dummies: bool) -> pd.DataFrame:
    work = rows.copy()
    work["directional_alignment"] = pd.to_numeric(
        work["mean_top5_cross_dataset_cosine"], errors="coerce"
    )
    fallback = pd.to_numeric(work["best_cross_dataset_cosine"], errors="coerce")
    work["directional_alignment"] = work["directional_alignment"].fillna(fallback)
    work = work[
        pd.to_numeric(work["cross_dataset_neighbor_count_top20"], errors="coerce").fillna(0) >= 3
    ].copy()
    work = work[np.isfinite(pd.to_numeric(work["directional_alignment"], errors="coerce"))].copy()
    work["support_count_bin"] = work["cross_dataset_neighbor_count_top20"].map(support_count_bin)
    work["directional_alignment_resid_score"] = residualize_direction(work, include_dataset_dummies)
    return work


def build_candidates(rows: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = prepare_rows(rows, bool(config["include_dataset_dummies"]))
    high_thr = float(work["directional_alignment_resid_score"].quantile(1.0 - float(config["quantile"])))
    low_thr = float(work["directional_alignment_resid_score"].quantile(float(config["quantile"])))
    high = work[work["directional_alignment_resid_score"] >= high_thr].copy()
    low = work[work["directional_alignment_resid_score"] <= low_thr].copy()
    candidate_rows: list[dict[str, Any]] = []
    for _, h in high.iterrows():
        sub = low[
            low["perturbation_type_raw"].astype(str).eq(str(h["perturbation_type_raw"]))
            & low["exact_bool"].eq(bool(h["exact_bool"]))
            & low["state_bin"].astype(str).eq(str(h["state_bin"]))
            & low["target_availability_bin"].astype(str).eq(str(h["target_availability_bin"]))
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
            resid_gap = float(h["directional_alignment_resid_score"] - l["directional_alignment_resid_score"])
            raw_gap = float(h["directional_alignment"] - l["directional_alignment"])
            if resid_gap <= 0:
                continue
            candidate_rows.append(
                {
                    "high_key": h["key"],
                    "low_key": l["key"],
                    "high_dataset": h["dataset"],
                    "low_dataset": l["dataset"],
                    "high_condition": h["condition"],
                    "low_condition": l["condition"],
                    "perturbation_type_raw": h["perturbation_type_raw"],
                    "high_directional_alignment": float(h["directional_alignment"]),
                    "low_directional_alignment": float(l["directional_alignment"]),
                    "directional_alignment_gap": raw_gap,
                    "high_directional_resid_score": float(h["directional_alignment_resid_score"]),
                    "low_directional_resid_score": float(l["directional_alignment_resid_score"]),
                    "residual_score_gap": resid_gap,
                    "high_cross_count": int(h["cross_dataset_neighbor_count_top20"]),
                    "low_cross_count": int(l["cross_dataset_neighbor_count_top20"]),
                    "high_cross_effective": float(h["cross_dataset_effective_count_top20"]),
                    "low_cross_effective": float(l["cross_dataset_effective_count_top20"]),
                    "high_raw_support_score": float(h["neighbor_support_score"]),
                    "low_raw_support_score": float(l["neighbor_support_score"]),
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
                    "high_exact_bool": bool(h["exact_bool"]),
                    "low_exact_bool": bool(l["exact_bool"]),
                    "response_log_abs_diff": float(l["response_log_abs_diff"]),
                    "n_gt_log_abs_diff": float(l["n_gt_log_abs_diff"]),
                    "n_ctrl_log_abs_diff": float(l["n_ctrl_log_abs_diff"]),
                    "cross_count_abs_diff": float(l["cross_count_abs_diff"]),
                    "dataset_pair": f"{h['dataset']}->{l['dataset']}",
                }
            )
    return pd.DataFrame(candidate_rows), work


def greedy_select(candidates: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    work = candidates.copy()
    work["objective"] = (
        pd.to_numeric(work["residual_score_gap"], errors="coerce")
        + 0.5 * pd.to_numeric(work["directional_alignment_gap"], errors="coerce")
        - 0.25 * pd.to_numeric(work["response_log_abs_diff"], errors="coerce")
        - 0.10 * pd.to_numeric(work["n_gt_log_abs_diff"], errors="coerce")
        - 0.10 * pd.to_numeric(work["n_ctrl_log_abs_diff"], errors="coerce")
        - 0.03 * pd.to_numeric(work["cross_count_abs_diff"], errors="coerce")
    )
    high_used: set[str] = set()
    low_used: set[str] = set()
    high_ds: Counter[str] = Counter()
    low_ds: Counter[str] = Counter()
    pair_dir: Counter[str] = Counter()
    selected: list[pd.Series] = []
    for _, row in work.sort_values(["objective", "residual_score_gap"], ascending=[False, False]).iterrows():
        high_key = str(row["high_key"])
        low_key = str(row["low_key"])
        hds = str(row["high_dataset"])
        lds = str(row["low_dataset"])
        direction = str(row["dataset_pair"])
        if high_key in high_used or low_key in low_used:
            continue
        if high_ds[hds] >= int(config["max_per_side_dataset"]):
            continue
        if low_ds[lds] >= int(config["max_per_side_dataset"]):
            continue
        if pair_dir[direction] >= int(config["max_per_dataset_pair"]):
            continue
        high_used.add(high_key)
        low_used.add(low_key)
        high_ds[hds] += 1
        low_ds[lds] += 1
        pair_dir[direction] += 1
        selected.append(row)
        if len(selected) >= int(config["max_pairs"]):
            break
    return pd.DataFrame(selected)


def balance_rows(selected: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("directional_alignment", "high_directional_alignment", "low_directional_alignment"),
        ("directional_resid_score", "high_directional_resid_score", "low_directional_resid_score"),
        ("cross_count", "high_cross_count", "low_cross_count"),
        ("cross_effective", "high_cross_effective", "low_cross_effective"),
        ("raw_support_score", "high_raw_support_score", "low_raw_support_score"),
        ("response_norm", "high_response_norm", "low_response_norm"),
        ("n_gt", "high_n_gt", "low_n_gt"),
        ("n_ctrl", "high_n_ctrl", "low_n_ctrl"),
        ("max_state_entropy", "high_max_state_entropy", "low_max_state_entropy"),
        ("same_target_cross_dataset_total", "high_same_target_cross_dataset_total", "low_same_target_cross_dataset_total"),
        ("exact_bool", "high_exact_bool", "low_exact_bool"),
    ]
    rows: list[dict[str, Any]] = []
    for name, hcol, lcol in specs:
        h = selected[hcol].astype(float) if selected[hcol].dtype == bool else pd.to_numeric(selected[hcol], errors="coerce")
        l = selected[lcol].astype(float) if selected[lcol].dtype == bool else pd.to_numeric(selected[lcol], errors="coerce")
        rows.append(
            {
                "feature": name,
                "high_mean": float(h.mean()),
                "low_mean": float(l.mean()),
                "high_median": float(h.median()),
                "low_median": float(l.median()),
                "smd_high_minus_low": smd(h, l),
                "auc_discriminability": auc_discriminability(h, l),
            }
        )
    return pd.DataFrame(rows)


def negative_controls(selected: pd.DataFrame, balance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in [
        "cross_count",
        "cross_effective",
        "raw_support_score",
        "response_norm",
        "n_gt",
        "n_ctrl",
        "max_state_entropy",
        "same_target_cross_dataset_total",
        "exact_bool",
    ]:
        hit = balance[balance["feature"].eq(feature)]
        if hit.empty:
            continue
        auc = float(hit.iloc[0]["auc_discriminability"])
        thresh = 0.70 if feature == "raw_support_score" else 0.65
        rows.append(
            {
                "control": f"{feature}_only_discriminability",
                "value": auc,
                "risk": bool(auc > thresh),
                "notes": f"max(AUC,1-AUC); threshold <= {thresh:.2f}",
            }
        )
    top_pair_frac = pair_direction_fraction(selected)
    high_ds_frac = side_dataset_fraction(selected, "high")
    low_ds_frac = side_dataset_fraction(selected, "low")
    crispri_frac = float((selected["perturbation_type_raw"].astype(str) == "CRISPRi").mean()) if not selected.empty else 0.0
    rows.extend(
        [
            {"control": "top_dataset_pair_direction_fraction", "value": top_pair_frac, "risk": bool(top_pair_frac > 0.15), "notes": "threshold <=0.15"},
            {"control": "top_high_side_dataset_fraction", "value": high_ds_frac, "risk": bool(high_ds_frac > 0.25), "notes": "threshold <=0.25"},
            {"control": "top_low_side_dataset_fraction", "value": low_ds_frac, "risk": bool(low_ds_frac > 0.25), "notes": "threshold <=0.25"},
            {"control": "crispri_pair_fraction", "value": crispri_frac, "risk": bool(crispri_frac > 0.80), "notes": "CRISPRi-only explanation risk"},
        ]
    )
    return pd.DataFrame(rows)


def assess_config(rows: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    candidates, scored_rows = build_candidates(rows, config)
    selected = greedy_select(candidates, config)
    if selected.empty:
        balance = pd.DataFrame()
        controls = pd.DataFrame()
        claims = pd.DataFrame()
    else:
        balance = balance_rows(selected)
        controls = negative_controls(selected, balance)
        claims = per_ptype_claims(selected)
    cov_features = [
        "cross_count",
        "cross_effective",
        "raw_support_score",
        "response_norm",
        "n_gt",
        "n_ctrl",
        "max_state_entropy",
        "same_target_cross_dataset_total",
        "exact_bool",
    ]
    max_cov_smd = (
        float(balance[balance["feature"].isin(cov_features)]["smd_high_minus_low"].abs().max())
        if not balance.empty
        else float("nan")
    )
    support_auc = (
        float(balance.loc[balance["feature"].eq("raw_support_score"), "auc_discriminability"].iloc[0])
        if not balance.empty
        else float("nan")
    )
    response_auc = (
        float(balance.loc[balance["feature"].eq("response_norm"), "auc_discriminability"].iloc[0])
        if not balance.empty
        else float("nan")
    )
    directional_gap = float(selected["directional_alignment_gap"].median()) if not selected.empty else 0.0
    resid_gap = float(selected["residual_score_gap"].median()) if not selected.empty else 0.0
    summary = {
        **config,
        "n_scored_conditions": int(len(scored_rows)),
        "n_candidates": int(len(candidates)),
        "n_pairs_unique": int(len(selected)),
        "n_high_conditions": int(selected["high_key"].nunique()) if not selected.empty else 0,
        "n_low_conditions": int(selected["low_key"].nunique()) if not selected.empty else 0,
        "n_datasets_total": int(len(set(selected["high_dataset"].astype(str)) | set(selected["low_dataset"].astype(str)))) if not selected.empty else 0,
        "n_high_datasets": int(selected["high_dataset"].nunique()) if not selected.empty else 0,
        "n_low_datasets": int(selected["low_dataset"].nunique()) if not selected.empty else 0,
        "n_perturbation_types": int(selected["perturbation_type_raw"].nunique()) if not selected.empty else 0,
        "n_claim_ready_ptypes": int(claims["claim_ready"].sum()) if not claims.empty else 0,
        "median_directional_alignment_gap": directional_gap,
        "median_residual_score_gap": resid_gap,
        "max_abs_covariate_smd": max_cov_smd,
        "raw_support_auc": support_auc,
        "response_norm_auc": response_auc,
        "top_dataset_pair_direction_fraction": pair_direction_fraction(selected),
        "top_high_dataset_fraction": side_dataset_fraction(selected, "high"),
        "top_low_dataset_fraction": side_dataset_fraction(selected, "low"),
        "crispri_pair_fraction": float((selected["perturbation_type_raw"].astype(str) == "CRISPRi").mean()) if not selected.empty else 0.0,
        "any_negative_control_risk": bool(controls["risk"].astype(bool).any()) if not controls.empty else True,
    }
    reasons: list[str] = []
    if summary["n_pairs_unique"] < 250:
        reasons.append("pairs_below_250")
    if summary["n_high_conditions"] < 100 or summary["n_low_conditions"] < 100:
        reasons.append("unique_high_low_below_100")
    if summary["n_datasets_total"] < 12:
        reasons.append("datasets_total_below_12")
    if summary["n_high_datasets"] < 6 or summary["n_low_datasets"] < 6:
        reasons.append("side_datasets_below_6")
    if summary["n_perturbation_types"] < 3:
        reasons.append("perturbation_types_below_3")
    if summary["n_claim_ready_ptypes"] < 2:
        reasons.append("claim_ready_ptypes_below_2")
    if summary["median_directional_alignment_gap"] < 0.03:
        reasons.append("directional_alignment_gap_below_0p03")
    if math.isfinite(summary["max_abs_covariate_smd"]) and summary["max_abs_covariate_smd"] > 0.35:
        reasons.append("max_covariate_smd_above_0p35")
    if math.isfinite(summary["raw_support_auc"]) and summary["raw_support_auc"] > 0.70:
        reasons.append("raw_support_auc_above_0p70")
    if math.isfinite(summary["response_norm_auc"]) and summary["response_norm_auc"] > 0.65:
        reasons.append("response_norm_auc_above_0p65")
    if summary["top_dataset_pair_direction_fraction"] > 0.15:
        reasons.append("dataset_pair_direction_above_15pct")
    if summary["top_high_dataset_fraction"] > 0.25 or summary["top_low_dataset_fraction"] > 0.25:
        reasons.append("side_dataset_fraction_above_25pct")
    if summary["crispri_pair_fraction"] > 0.80:
        reasons.append("crispri_fraction_above_80pct")
    if summary["any_negative_control_risk"]:
        reasons.append("negative_control_risk_present")
    summary["reasons"] = ";".join(reasons)
    summary["strict_pass"] = not reasons
    return summary, selected, balance


def choose_best(configs: pd.DataFrame) -> pd.Series:
    work = configs.copy()
    work["score"] = (
        1000 * work["strict_pass"].astype(bool).astype(int)
        + work["n_pairs_unique"]
        + 40 * work["n_claim_ready_ptypes"]
        + 80 * work["median_directional_alignment_gap"]
        - 60 * work["max_abs_covariate_smd"].fillna(9)
        - 40 * work["raw_support_auc"].fillna(1)
        - 30 * work["response_norm_auc"].fillna(1)
        - 20 * work["top_dataset_pair_direction_fraction"].fillna(1)
        - 10 * work["crispri_pair_fraction"].fillna(1)
    )
    return work.sort_values("score", ascending=False).iloc[0]


def write_report(payload: dict[str, Any], config_rows: pd.DataFrame, selected: pd.DataFrame, balance: pd.DataFrame, controls: pd.DataFrame, claims: pd.DataFrame) -> None:
    top = config_rows.sort_values(["strict_pass", "n_pairs_unique"], ascending=[False, False]).head(14)
    lines = [
        "# Condition Directional Analog Alignment Gate",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only gate over parent train split residual-neighborhood rows.",
        "* Directional score is mean top-5 cross-dataset residual-vector cosine, residualized against response norm, cell counts, exact coverage, state entropy, same-target availability, cross-neighbor count/effective support, same-ptype support, same-dataset fraction, perturbation type, and optional dataset dummies.",
        "* No training, inference, GPU, canonical multi, Track C query, or checkpoint selection.",
        "",
        "## Decision",
        "",
        f"* Selected config: `{payload['selected_config_name']}`.",
        f"* Reasons: `{payload['reasons'] if payload['reasons'] else 'none'}`.",
        f"* Scored conditions: `{payload['summary']['n_scored_conditions']}`.",
        f"* Unique pairs: `{payload['summary']['n_pairs_unique']}`; datasets total/high/low: `{payload['summary']['n_datasets_total']}/{payload['summary']['n_high_datasets']}/{payload['summary']['n_low_datasets']}`.",
        f"* Median raw directional gap: `{fmt(payload['summary']['median_directional_alignment_gap'])}`; residual-score gap: `{fmt(payload['summary']['median_residual_score_gap'])}`.",
        f"* Max covariate SMD: `{fmt(payload['summary']['max_abs_covariate_smd'])}`; support AUC: `{fmt(payload['summary']['raw_support_auc'])}`; response AUC: `{fmt(payload['summary']['response_norm_auc'])}`.",
        "",
        "## Config Sweep",
        "",
        "| config | q | resp cal | cell cal | support cal | dataset residualized | scored | candidates | pairs | datasets | ptypes | claim ptypes | dir gap | cov SMD | support AUC | response AUC | pair-dir frac | strict | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| `{row['name']}` | `{fmt(row['quantile'], 2)}` | `{fmt(row['response_log_caliper'], 2)}` | `{fmt(row['cell_log_caliper'], 2)}` | `{fmt(row['cross_count_caliper'], 0)}` | `{bool(row['include_dataset_dummies'])}` | "
            f"`{int(row['n_scored_conditions'])}` | `{int(row['n_candidates'])}` | `{int(row['n_pairs_unique'])}` | `{int(row['n_datasets_total'])}` | `{int(row['n_perturbation_types'])}` | `{int(row['n_claim_ready_ptypes'])}` | "
            f"`{fmt(row['median_directional_alignment_gap'])}` | `{fmt(row['max_abs_covariate_smd'])}` | `{fmt(row['raw_support_auc'])}` | `{fmt(row['response_norm_auc'])}` | `{fmt(row['top_dataset_pair_direction_fraction'])}` | `{bool(row['strict_pass'])}` | `{row['reasons']}` |"
        )
    lines.extend(["", "## Selected Balance", "", "| feature | high mean | low mean | high median | low median | SMD | AUC |", "|---|---:|---:|---:|---:|---:|---:|"])
    for _, row in balance.iterrows():
        lines.append(
            f"| `{row['feature']}` | `{fmt(row['high_mean'])}` | `{fmt(row['low_mean'])}` | `{fmt(row['high_median'])}` | `{fmt(row['low_median'])}` | `{fmt(row['smd_high_minus_low'])}` | `{fmt(row['auc_discriminability'])}` |"
        )
    lines.extend(["", "## Negative Controls", "", "| control | value | risk | notes |", "|---|---:|---:|---|"])
    for _, row in controls.iterrows():
        lines.append(f"| `{row['control']}` | `{fmt(row['value'])}` | `{bool(row['risk'])}` | {row['notes']} |")
    lines.extend(["", "## Per-Perturbation-Type Claim Readiness", "", "| ptype | pairs | high uniq | low uniq | high datasets | low datasets | claim ready |", "|---|---:|---:|---:|---:|---:|---:|"])
    for _, row in claims.iterrows():
        lines.append(
            f"| `{row['perturbation_type_raw']}` | `{int(row['n_pairs'])}` | `{int(row['n_high_conditions'])}` | `{int(row['n_low_conditions'])}` | `{int(row['n_high_datasets'])}` | `{int(row['n_low_datasets'])}` | `{bool(row['claim_ready'])}` |"
        )
    lines.extend(["", "## Interpretation", ""])
    if payload["status"].endswith("pass_external_audit_no_gpu"):
        lines.append("* The directional analog axis passes this CPU balance gate. It still requires external audit and an axis-specific null/control panel before any GPU smoke.")
    else:
        lines.append("* The directional analog axis does not yet pass CPU balance/control gates. Do not launch GPU from this split.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"* Config rows: `{OUT_CONFIGS}`",
            f"* Selected pairs: `{OUT_SELECTED}`",
            f"* Balance: `{OUT_BALANCE}`",
            f"* Negative controls: `{OUT_CONTROLS}`",
            f"* High split: `{payload['outputs'].get('high_split', '')}`",
            f"* Low split: `{payload['outputs'].get('low_split', '')}`",
            f"* JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    summaries: list[dict[str, Any]] = []
    selections: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for config in CONFIGS:
        summary, selected, balance = assess_config(rows, config)
        summaries.append(summary)
        selections[str(summary["name"])] = (selected, balance)
    config_rows = pd.DataFrame(summaries)
    best = choose_best(config_rows)
    selected, balance = selections[str(best["name"])]
    controls = negative_controls(selected, balance) if not selected.empty else pd.DataFrame()
    claims = per_ptype_claims(selected) if not selected.empty else pd.DataFrame()
    parent = load_json(PARENT_SPLIT)
    tag = f"directional_analog_{len(selected)}pair_{best['name']}"
    high_split_path = SPLIT_DIR / f"split_seed42_xverse_condition_directional_analog_high_{tag}.json"
    low_split_path = SPLIT_DIR / f"split_seed42_xverse_condition_directional_analog_low_{tag}.json"
    if not selected.empty:
        write_json(high_split_path, split_from_conditions(parent, selected, "high_dataset", "high_condition"))
        write_json(low_split_path, split_from_conditions(parent, selected, "low_dataset", "low_condition"))
    config_rows.to_csv(OUT_CONFIGS, index=False)
    selected.to_csv(OUT_SELECTED, index=False)
    balance.to_csv(OUT_BALANCE, index=False)
    controls.to_csv(OUT_CONTROLS, index=False)
    status = (
        "condition_directional_analog_alignment_pass_external_audit_no_gpu"
        if bool(best["strict_pass"])
        else "condition_directional_analog_alignment_fail_no_gpu"
    )
    payload = {
        "created_at": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "selected_config_name": str(best["name"]),
        "summary": {
            k: (bool(v) if isinstance(v, np.bool_) else float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v)
            for k, v in best.to_dict().items()
            if k != "reasons"
        },
        "reasons": str(best.get("reasons", "")),
        "inputs": {
            "support_rows": str(ROOT / "reports/condition_neighborhood_support_gate_20260629/condition_neighborhood_support_rows.csv"),
            "parent_split": str(PARENT_SPLIT),
        },
        "outputs": {
            "report": str(OUT_MD),
            "json": str(OUT_JSON),
            "configs": str(OUT_CONFIGS),
            "selected_pairs": str(OUT_SELECTED),
            "balance": str(OUT_BALANCE),
            "negative_controls": str(OUT_CONTROLS),
            "high_split": str(high_split_path) if not selected.empty else "",
            "low_split": str(low_split_path) if not selected.empty else "",
        },
        "boundary": "cpu_report_only_directional_analog_alignment_no_training_no_inference_no_gpu_no_canonical_multi_no_trackc_query",
    }
    write_json(OUT_JSON, payload)
    write_report(payload, config_rows, selected, balance, controls, claims)
    print(
        json.dumps(
            {
                "status": status,
                "selected_config": str(best["name"]),
                "n_pairs": int(best["n_pairs_unique"]),
                "reasons": str(best.get("reasons", "")),
                "report": str(OUT_MD),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
