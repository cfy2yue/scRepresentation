#!/usr/bin/env python3
"""Response-residualized condition-neighborhood support gate.

CPU/report-only. Tests whether cross-dataset neighborhood support remains a
usable train-set information axis after removing response magnitude and other
major covariates from the support score before high/low split drafting.
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
OUT_DIR = ROOT / "reports/condition_neighborhood_response_residualized_support_gate_20260629"
SPLIT_DIR = ROOT / "dataset/biFlow_data/xverse_condition_neighborhood_support_splits_20260629"
OUT_MD = OUT_DIR / "LATENTFM_CONDITION_NEIGHBORHOOD_RESPONSE_RESIDUALIZED_SUPPORT_GATE_20260629.md"
OUT_JSON = OUT_DIR / "latentfm_condition_neighborhood_response_residualized_support_gate_20260629.json"
OUT_CONFIGS = OUT_DIR / "condition_neighborhood_response_residualized_config_rows.csv"
OUT_SELECTED = OUT_DIR / "condition_neighborhood_response_residualized_selected_pairs.csv"
OUT_BALANCE = OUT_DIR / "condition_neighborhood_response_residualized_balance.csv"
OUT_CONTROLS = OUT_DIR / "condition_neighborhood_response_residualized_negative_controls.csv"


CONFIGS = [
    {
        "name": f"q{int(q * 100)}_resp{resp}_cell{cell}_ds{int(ds)}",
        "quantile": q,
        "response_log_caliper": resp,
        "cell_log_caliper": cell,
        "include_dataset_dummies": ds,
        "max_pairs": 320,
        "max_per_side_dataset": 50,
        "max_per_dataset_pair": 30,
    }
    for q in [0.20, 0.25, 0.30]
    for resp in [0.35, 0.50, 0.75]
    for cell in [0.50, 0.75]
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


def target_availability_bin(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        val = 0.0
    return "has_target_cross" if val > 0 else "no_target_cross"


def smd(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
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


def load_rows() -> pd.DataFrame:
    rows = pd.read_csv(ROWS_CSV)
    rows["key"] = rows["dataset"].astype(str) + "||" + rows["condition"].astype(str)
    rows["log_response_norm"] = np.log1p(pd.to_numeric(rows["response_norm"], errors="coerce"))
    rows["log_n_gt"] = np.log1p(pd.to_numeric(rows["n_gt"], errors="coerce"))
    rows["log_n_ctrl"] = np.log1p(pd.to_numeric(rows["n_ctrl"], errors="coerce"))
    rows["exact_bool"] = rows["exact_response_available"].astype(str).str.lower().isin(["true", "1", "yes"])
    rows["state_bin"] = rows["max_state_entropy"].map(state_bin)
    rows["target_availability_bin"] = rows["same_target_cross_dataset_total"].map(target_availability_bin)
    return rows


def residualized_support(rows: pd.DataFrame, include_dataset_dummies: bool) -> np.ndarray:
    columns: list[np.ndarray] = [
        np.ones(len(rows), dtype=float),
        rows["log_response_norm"].to_numpy(dtype=float),
        rows["log_n_gt"].to_numpy(dtype=float),
        rows["log_n_ctrl"].to_numpy(dtype=float),
        rows["exact_bool"].astype(float).to_numpy(dtype=float),
        pd.to_numeric(rows["max_state_entropy"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        pd.to_numeric(rows["same_target_cross_dataset_total"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
    ]
    for ptype in sorted(rows["perturbation_type_raw"].astype(str).unique())[1:]:
        columns.append((rows["perturbation_type_raw"].astype(str).to_numpy() == ptype).astype(float))
    if include_dataset_dummies:
        for dataset in sorted(rows["dataset"].astype(str).unique())[1:]:
            columns.append((rows["dataset"].astype(str).to_numpy() == dataset).astype(float))
    x = np.column_stack(columns).astype(float)
    y = pd.to_numeric(rows["neighbor_support_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


def build_candidates(rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    work = rows.copy()
    work["support_resid_score"] = residualized_support(work, bool(config["include_dataset_dummies"]))
    high_thr = float(work["support_resid_score"].quantile(1.0 - float(config["quantile"])))
    low_thr = float(work["support_resid_score"].quantile(float(config["quantile"])))
    high = work[work["support_resid_score"] >= high_thr].copy()
    low = work[work["support_resid_score"] <= low_thr].copy()
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
        sub = sub[
            (sub["response_log_abs_diff"] <= float(config["response_log_caliper"]))
            & (sub["n_gt_log_abs_diff"] <= float(config["cell_log_caliper"]))
            & (sub["n_ctrl_log_abs_diff"] <= float(config["cell_log_caliper"]))
        ].copy()
        if sub.empty:
            continue
        for _, l in sub.iterrows():
            gap = float(h["support_resid_score"] - l["support_resid_score"])
            if gap <= 0:
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
                    "high_support_resid_score": float(h["support_resid_score"]),
                    "low_support_resid_score": float(l["support_resid_score"]),
                    "residual_score_gap": gap,
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
        - 0.25 * pd.to_numeric(work["response_log_abs_diff"], errors="coerce")
        - 0.10 * pd.to_numeric(work["n_gt_log_abs_diff"], errors="coerce")
        - 0.10 * pd.to_numeric(work["n_ctrl_log_abs_diff"], errors="coerce")
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


def split_from_conditions(parent: dict[str, Any], side: pd.DataFrame, dataset_col: str, condition_col: str) -> dict[str, Any]:
    out = copy.deepcopy(parent)
    by_dataset = side.groupby(dataset_col)[condition_col].apply(lambda s: sorted(set(map(str, s)))).to_dict()
    for dataset in out:
        out[dataset]["train"] = by_dataset.get(dataset, [])
    return out


def side_dataset_fraction(selected: pd.DataFrame, side: str) -> float:
    if selected.empty:
        return 0.0
    return float(selected[f"{side}_dataset"].astype(str).value_counts(normalize=True).max())


def pair_direction_fraction(selected: pd.DataFrame) -> float:
    if selected.empty:
        return 0.0
    return float(selected["dataset_pair"].astype(str).value_counts(normalize=True).max())


def per_ptype_claims(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ptype, group in selected.groupby("perturbation_type_raw"):
        rows.append(
            {
                "perturbation_type_raw": ptype,
                "n_pairs": int(len(group)),
                "n_high_conditions": int(group["high_key"].nunique()),
                "n_low_conditions": int(group["low_key"].nunique()),
                "n_high_datasets": int(group["high_dataset"].nunique()),
                "n_low_datasets": int(group["low_dataset"].nunique()),
                "claim_ready": bool(
                    len(group) >= 50
                    and group["high_key"].nunique() >= 20
                    and group["low_key"].nunique() >= 20
                    and group["high_dataset"].nunique() >= 3
                    and group["low_dataset"].nunique() >= 3
                ),
            }
        )
    return pd.DataFrame(rows)


def balance_rows(selected: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("support_resid_score", "high_support_resid_score", "low_support_resid_score"),
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
    for feature in ["response_norm", "n_gt", "n_ctrl", "max_state_entropy", "same_target_cross_dataset_total", "exact_bool"]:
        hit = balance[balance["feature"].eq(feature)]
        if hit.empty:
            continue
        auc = float(hit.iloc[0]["auc_discriminability"])
        rows.append(
            {
                "control": f"{feature}_only_discriminability",
                "value": auc,
                "risk": bool(auc > 0.75),
                "notes": "max(AUC,1-AUC) for classifying high vs low residualized support",
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
    cov_features = ["response_norm", "n_gt", "n_ctrl", "max_state_entropy", "same_target_cross_dataset_total", "exact_bool"]
    max_cov_smd = (
        float(balance[balance["feature"].isin(cov_features)]["smd_high_minus_low"].abs().max())
        if not balance.empty
        else float("nan")
    )
    response_auc = (
        float(balance.loc[balance["feature"].eq("response_norm"), "auc_discriminability"].iloc[0])
        if not balance.empty
        else float("nan")
    )
    summary = {
        **config,
        "n_candidates": int(len(candidates)),
        "n_pairs_unique": int(len(selected)),
        "n_high_conditions": int(selected["high_key"].nunique()) if not selected.empty else 0,
        "n_low_conditions": int(selected["low_key"].nunique()) if not selected.empty else 0,
        "n_datasets_total": int(len(set(selected["high_dataset"].astype(str)) | set(selected["low_dataset"].astype(str)))) if not selected.empty else 0,
        "n_high_datasets": int(selected["high_dataset"].nunique()) if not selected.empty else 0,
        "n_low_datasets": int(selected["low_dataset"].nunique()) if not selected.empty else 0,
        "n_perturbation_types": int(selected["perturbation_type_raw"].nunique()) if not selected.empty else 0,
        "n_claim_ready_ptypes": int(claims["claim_ready"].sum()) if not claims.empty else 0,
        "median_residual_score_gap": float(selected["residual_score_gap"].median()) if not selected.empty else 0.0,
        "raw_support_high_mean": float(selected["high_raw_support_score"].mean()) if not selected.empty else 0.0,
        "raw_support_low_mean": float(selected["low_raw_support_score"].mean()) if not selected.empty else 0.0,
        "max_abs_covariate_smd": max_cov_smd,
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
    if math.isfinite(summary["max_abs_covariate_smd"]) and summary["max_abs_covariate_smd"] > 0.35:
        reasons.append("max_covariate_smd_above_0p35")
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
        + 30 * work["n_claim_ready_ptypes"]
        - 50 * work["max_abs_covariate_smd"].fillna(9)
        - 30 * work["response_norm_auc"].fillna(1)
        - 20 * work["top_dataset_pair_direction_fraction"].fillna(1)
        - 10 * work["crispri_pair_fraction"].fillna(1)
    )
    return work.sort_values("score", ascending=False).iloc[0]


def write_report(payload: dict[str, Any], config_rows: pd.DataFrame, selected: pd.DataFrame, balance: pd.DataFrame, controls: pd.DataFrame, claims: pd.DataFrame) -> None:
    top = config_rows.sort_values(["strict_pass", "n_pairs_unique"], ascending=[False, False]).head(14)
    lines = [
        "# Condition Neighborhood Response-Residualized Support Gate",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only redesign of condition-neighborhood support.",
        "* Support score is residualized against response norm, cell counts, exact coverage, state entropy, same-target cross-dataset availability, and perturbation type before split drafting.",
        "* No training, inference, GPU, canonical multi, Track C query, or checkpoint selection.",
        "",
        "## Decision",
        "",
        f"* Selected config: `{payload['selected_config_name']}`.",
        f"* Reasons: `{payload['reasons'] if payload['reasons'] else 'none'}`.",
        f"* Unique pairs: `{payload['summary']['n_pairs_unique']}`; datasets total/high/low: `{payload['summary']['n_datasets_total']}/{payload['summary']['n_high_datasets']}/{payload['summary']['n_low_datasets']}`.",
        f"* Claim-ready perturbation types: `{payload['summary']['n_claim_ready_ptypes']}`.",
        f"* Max covariate SMD: `{fmt(payload['summary']['max_abs_covariate_smd'])}`; response-norm AUC: `{fmt(payload['summary']['response_norm_auc'])}`.",
        f"* Top dataset-pair fraction: `{fmt(payload['summary']['top_dataset_pair_direction_fraction'])}`; CRISPRi fraction: `{fmt(payload['summary']['crispri_pair_fraction'])}`.",
        "",
        "## Config Sweep",
        "",
        "| config | q | resp cal | cell cal | dataset residualized | candidates | pairs | datasets | ptypes | claim ptypes | cov SMD | response AUC | pair-dir frac | CRISPRi frac | strict | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| `{row['name']}` | `{fmt(row['quantile'], 2)}` | `{fmt(row['response_log_caliper'], 2)}` | `{fmt(row['cell_log_caliper'], 2)}` | `{bool(row['include_dataset_dummies'])}` | "
            f"`{int(row['n_candidates'])}` | `{int(row['n_pairs_unique'])}` | `{int(row['n_datasets_total'])}` | `{int(row['n_perturbation_types'])}` | `{int(row['n_claim_ready_ptypes'])}` | "
            f"`{fmt(row['max_abs_covariate_smd'])}` | `{fmt(row['response_norm_auc'])}` | `{fmt(row['top_dataset_pair_direction_fraction'])}` | `{fmt(row['crispri_pair_fraction'])}` | `{bool(row['strict_pass'])}` | `{row['reasons']}` |"
        )
    lines.extend(
        [
            "",
            "## Selected Balance",
            "",
            "| feature | high mean | low mean | high median | low median | SMD | AUC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in balance.iterrows():
        lines.append(
            f"| `{row['feature']}` | `{fmt(row['high_mean'])}` | `{fmt(row['low_mean'])}` | `{fmt(row['high_median'])}` | `{fmt(row['low_median'])}` | `{fmt(row['smd_high_minus_low'])}` | `{fmt(row['auc_discriminability'])}` |"
        )
    lines.extend(
        [
            "",
            "## Negative Controls",
            "",
            "| control | value | risk | notes |",
            "|---|---:|---:|---|",
        ]
    )
    for _, row in controls.iterrows():
        lines.append(f"| `{row['control']}` | `{fmt(row['value'])}` | `{bool(row['risk'])}` | {row['notes']} |")
    lines.extend(
        [
            "",
            "## Per-Perturbation-Type Claim Readiness",
            "",
            "| ptype | pairs | high uniq | low uniq | high datasets | low datasets | claim ready |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in claims.iterrows():
        lines.append(
            f"| `{row['perturbation_type_raw']}` | `{int(row['n_pairs'])}` | `{int(row['n_high_conditions'])}` | `{int(row['n_low_conditions'])}` | `{int(row['n_high_datasets'])}` | `{int(row['n_low_datasets'])}` | `{bool(row['claim_ready'])}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if payload["status"].endswith("pass_external_audit_no_gpu"):
        lines.append("* This response-residualized support axis passes CPU split-balance gates and should go to external audit before any GPU launch.")
    else:
        lines.append("* The response-residualized redesign improved balance but is not yet a launchable split.")
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
    controls = negative_controls(selected, balance)
    claims = per_ptype_claims(selected)
    parent = load_json(PARENT_SPLIT)
    tag = f"response_resid_{len(selected)}pair_{best['name']}"
    high_split_path = SPLIT_DIR / f"split_seed42_xverse_condition_neighborhood_high_support_{tag}.json"
    low_split_path = SPLIT_DIR / f"split_seed42_xverse_condition_neighborhood_low_support_{tag}.json"
    write_json(high_split_path, split_from_conditions(parent, selected, "high_dataset", "high_condition"))
    write_json(low_split_path, split_from_conditions(parent, selected, "low_dataset", "low_condition"))
    config_rows.to_csv(OUT_CONFIGS, index=False)
    selected.to_csv(OUT_SELECTED, index=False)
    balance.to_csv(OUT_BALANCE, index=False)
    controls.to_csv(OUT_CONTROLS, index=False)
    status = (
        "condition_neighborhood_response_residualized_support_pass_external_audit_no_gpu"
        if bool(best["strict_pass"])
        else "condition_neighborhood_response_residualized_support_fail_no_gpu"
    )
    payload = {
        "created_at": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "selected_config_name": str(best["name"]),
        "summary": {k: (bool(v) if isinstance(v, np.bool_) else float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v) for k, v in best.to_dict().items() if k != "reasons"},
        "reasons": str(best.get("reasons", "")),
        "inputs": {"rows": str(ROWS_CSV), "parent_split": str(PARENT_SPLIT)},
        "outputs": {
            "report": str(OUT_MD),
            "json": str(OUT_JSON),
            "configs": str(OUT_CONFIGS),
            "selected_pairs": str(OUT_SELECTED),
            "balance": str(OUT_BALANCE),
            "negative_controls": str(OUT_CONTROLS),
            "high_split": str(high_split_path),
            "low_split": str(low_split_path),
        },
        "boundary": "cpu_report_only_response_residualized_support_split_no_training_no_inference_no_gpu_no_canonical_multi_no_trackc_query",
    }
    write_json(OUT_JSON, payload)
    write_report(payload, config_rows, selected, balance, controls, claims)
    print(json.dumps({"status": status, "selected_config": str(best["name"]), "n_pairs": int(best["n_pairs_unique"]), "reasons": str(best.get("reasons", "")), "report": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
