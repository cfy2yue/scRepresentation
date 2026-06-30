#!/usr/bin/env python3
"""Strict split gate for condition-neighborhood support.

CPU/report-only. Mutates the first neighborhood-support draft after external
audit: global response-norm calipers, cell-count calipers, exact/state matching,
dataset caps, pair-direction caps, and simple negative-control diagnostics.
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
OUT_DIR = ROOT / "reports/condition_neighborhood_support_strict_split_gate_20260629"
SPLIT_DIR = ROOT / "dataset/biFlow_data/xverse_condition_neighborhood_support_splits_20260629"
OUT_MD = OUT_DIR / "LATENTFM_CONDITION_NEIGHBORHOOD_SUPPORT_STRICT_SPLIT_GATE_20260629.md"
OUT_JSON = OUT_DIR / "latentfm_condition_neighborhood_support_strict_split_gate_20260629.json"
OUT_CONFIGS = OUT_DIR / "condition_neighborhood_support_strict_config_rows.csv"
OUT_SELECTED = OUT_DIR / "condition_neighborhood_support_strict_selected_pairs.csv"
OUT_BALANCE = OUT_DIR / "condition_neighborhood_support_strict_balance.csv"
OUT_CONTROLS = OUT_DIR / "condition_neighborhood_support_strict_negative_controls.csv"


CONFIGS: list[dict[str, Any]] = []
for response_log_caliper in [0.35, 0.50, 0.75, 1.00]:
    for cell_log_caliper in [0.50, 0.75, 1.00]:
        for match_state_bin in [True, False]:
            CONFIGS.append(
                {
                    "response_log_caliper": response_log_caliper,
                    "cell_log_caliper": cell_log_caliper,
                    "match_exact": True,
                    "match_state_bin": match_state_bin,
                    "match_target_availability_bin": True,
                    "max_pairs": 320,
                    "max_per_side_dataset": 50,
                    "max_per_dataset_pair": 30,
                }
            )


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


def load_augmented_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(ROWS_CSV)
    pairs = pd.read_csv(PAIRS_CSV)
    rows["key"] = rows["dataset"].astype(str) + "||" + rows["condition"].astype(str)
    rows["log_response_norm"] = np.log1p(pd.to_numeric(rows["response_norm"], errors="coerce"))
    rows["log_n_gt"] = np.log1p(pd.to_numeric(rows["n_gt"], errors="coerce"))
    rows["log_n_ctrl"] = np.log1p(pd.to_numeric(rows["n_ctrl"], errors="coerce"))
    rows["state_bin"] = rows["max_state_entropy"].map(state_bin)
    rows["target_availability_bin"] = rows["same_target_cross_dataset_total"].map(target_availability_bin)
    rows["exact_response_available_bool"] = rows["exact_response_available"].astype(str).str.lower().isin(["true", "1", "yes"])
    lookup = rows.set_index("key")
    pairs = pairs.copy()
    pairs["high_key"] = pairs["high_dataset"].astype(str) + "||" + pairs["high_condition"].astype(str)
    pairs["low_key"] = pairs["low_dataset"].astype(str) + "||" + pairs["low_condition"].astype(str)
    high_cols = [
        "response_norm",
        "log_response_norm",
        "n_gt",
        "n_ctrl",
        "log_n_gt",
        "log_n_ctrl",
        "exact_response_available_bool",
        "state_bin",
        "target_availability_bin",
        "max_state_entropy",
        "same_target_cross_dataset_total",
    ]
    for col in high_cols:
        pairs[f"high_{col}"] = pairs["high_key"].map(lookup[col])
        pairs[f"low_{col}"] = pairs["low_key"].map(lookup[col])
    pairs["response_log_abs_diff"] = (pairs["high_log_response_norm"] - pairs["low_log_response_norm"]).abs()
    pairs["n_gt_log_abs_diff"] = (pairs["high_log_n_gt"] - pairs["low_log_n_gt"]).abs()
    pairs["n_ctrl_log_abs_diff"] = (pairs["high_log_n_ctrl"] - pairs["low_log_n_ctrl"]).abs()
    pairs["dataset_pair"] = pairs["high_dataset"].astype(str) + "->" + pairs["low_dataset"].astype(str)
    return rows, pairs


def filter_pairs(pairs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = pairs.copy()
    out = out[out["response_log_abs_diff"] <= float(config["response_log_caliper"])]
    out = out[out["n_gt_log_abs_diff"] <= float(config["cell_log_caliper"])]
    out = out[out["n_ctrl_log_abs_diff"] <= float(config["cell_log_caliper"])]
    if config.get("match_exact", True):
        out = out[out["high_exact_response_available_bool"].astype(bool) == out["low_exact_response_available_bool"].astype(bool)]
    if config.get("match_state_bin", True):
        out = out[out["high_state_bin"].astype(str) == out["low_state_bin"].astype(str)]
    if config.get("match_target_availability_bin", True):
        out = out[out["high_target_availability_bin"].astype(str) == out["low_target_availability_bin"].astype(str)]
    return out


def greedy_select(candidates: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    work = candidates.copy()
    for col in ["response_log_abs_diff", "n_gt_log_abs_diff", "n_ctrl_log_abs_diff"]:
        iqr = float(work[col].quantile(0.75) - work[col].quantile(0.25))
        med = float(work[col].median())
        denom = iqr if iqr > 1e-12 else 1.0
        work[col + "_robust_z"] = (work[col] - med) / denom
    work["objective"] = (
        pd.to_numeric(work["score_gap"], errors="coerce")
        - 0.35 * work["response_log_abs_diff_robust_z"]
        - 0.15 * work["n_gt_log_abs_diff_robust_z"]
        - 0.15 * work["n_ctrl_log_abs_diff_robust_z"]
    )
    used_high: set[str] = set()
    used_low: set[str] = set()
    high_ds: Counter[str] = Counter()
    low_ds: Counter[str] = Counter()
    pair_dir: Counter[str] = Counter()
    selected: list[pd.Series] = []
    for _, row in work.sort_values(["objective", "score_gap"], ascending=[False, False]).iterrows():
        high_key = str(row["high_key"])
        low_key = str(row["low_key"])
        hds = str(row["high_dataset"])
        lds = str(row["low_dataset"])
        direction = str(row["dataset_pair"])
        if high_key in used_high or low_key in used_low:
            continue
        if high_ds[hds] >= int(config["max_per_side_dataset"]):
            continue
        if low_ds[lds] >= int(config["max_per_side_dataset"]):
            continue
        if pair_dir[direction] >= int(config["max_per_dataset_pair"]):
            continue
        used_high.add(high_key)
        used_low.add(low_key)
        high_ds[hds] += 1
        low_ds[lds] += 1
        pair_dir[direction] += 1
        selected.append(row)
        if len(selected) >= int(config["max_pairs"]):
            break
    return pd.DataFrame(selected)


def subset(rows: pd.DataFrame, selected: pd.DataFrame, side: str) -> pd.DataFrame:
    keys = set(selected[f"{side}_key"].astype(str))
    return rows[rows["key"].isin(keys)].copy()


def balance_rows(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "neighbor_support_score",
        "cross_dataset_neighbor_count_top20",
        "same_dataset_fraction_top20",
        "n_gt",
        "n_ctrl",
        "response_norm",
        "max_state_entropy",
        "same_target_cross_dataset_total",
    ]
    rows: list[dict[str, Any]] = []
    for col in cols:
        rows.append(
            {
                "feature": col,
                "high_mean": float(pd.to_numeric(high[col], errors="coerce").mean()),
                "low_mean": float(pd.to_numeric(low[col], errors="coerce").mean()),
                "high_median": float(pd.to_numeric(high[col], errors="coerce").median()),
                "low_median": float(pd.to_numeric(low[col], errors="coerce").median()),
                "smd_high_minus_low": smd(high[col], low[col]),
                "auc_discriminability": auc_discriminability(high[col], low[col]),
            }
        )
    rows.append(
        {
            "feature": "exact_response_available_fraction",
            "high_mean": float(high["exact_response_available_bool"].astype(bool).mean()),
            "low_mean": float(low["exact_response_available_bool"].astype(bool).mean()),
            "high_median": float(high["exact_response_available_bool"].astype(bool).median()),
            "low_median": float(low["exact_response_available_bool"].astype(bool).median()),
            "smd_high_minus_low": smd(high["exact_response_available_bool"].astype(int), low["exact_response_available_bool"].astype(int)),
            "auc_discriminability": auc_discriminability(high["exact_response_available_bool"].astype(int), low["exact_response_available_bool"].astype(int)),
        }
    )
    return pd.DataFrame(rows)


def split_from_conditions(parent: dict[str, Any], conditions: pd.DataFrame) -> dict[str, Any]:
    out = copy.deepcopy(parent)
    by_dataset = conditions.groupby("dataset")["condition"].apply(lambda s: sorted(set(map(str, s)))).to_dict()
    for dataset in out:
        out[dataset]["train"] = by_dataset.get(dataset, [])
    return out


def side_dataset_fraction(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["dataset"].astype(str).value_counts(normalize=True).max())


def pair_direction_fraction(selected: pd.DataFrame) -> float:
    if selected.empty:
        return 0.0
    return float(selected["dataset_pair"].astype(str).value_counts(normalize=True).max())


def per_ptype_claims(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if selected.empty:
        return pd.DataFrame()
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


def negative_controls(selected: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, balance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    top_pair_frac = pair_direction_fraction(selected)
    top_high_ds_frac = side_dataset_fraction(high)
    top_low_ds_frac = side_dataset_fraction(low)
    crispri_frac = float((selected["perturbation_type_raw"].astype(str) == "CRISPRi").mean()) if not selected.empty else 0.0
    for feature in ["response_norm", "n_gt", "n_ctrl", "exact_response_available_fraction", "max_state_entropy"]:
        hit = balance[balance["feature"].eq(feature)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        rows.append(
            {
                "control": f"{feature}_only_discriminability",
                "value": float(row["auc_discriminability"]),
                "risk": bool(float(row["auc_discriminability"]) > 0.75),
                "notes": "max(AUC,1-AUC) for classifying high vs low support",
            }
        )
    rows.extend(
        [
            {
                "control": "top_dataset_pair_direction_fraction",
                "value": top_pair_frac,
                "risk": bool(top_pair_frac > 0.15),
                "notes": "subagent threshold: no dataset-pair direction >15%",
            },
            {
                "control": "top_high_side_dataset_fraction",
                "value": top_high_ds_frac,
                "risk": bool(top_high_ds_frac > 0.25),
                "notes": "subagent threshold: no dataset >25% of either side",
            },
            {
                "control": "top_low_side_dataset_fraction",
                "value": top_low_ds_frac,
                "risk": bool(top_low_ds_frac > 0.25),
                "notes": "subagent threshold: no dataset >25% of either side",
            },
            {
                "control": "crispri_pair_fraction",
                "value": crispri_frac,
                "risk": bool(crispri_frac > 0.80),
                "notes": "CRISPRi-only explanation risk",
            },
        ]
    )
    return pd.DataFrame(rows)


def assess_config(config: dict[str, Any], rows: pd.DataFrame, pairs: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = filter_pairs(pairs, config)
    selected = greedy_select(candidates, config)
    high = subset(rows, selected, "high") if not selected.empty else pd.DataFrame()
    low = subset(rows, selected, "low") if not selected.empty else pd.DataFrame()
    balance = balance_rows(high, low) if not high.empty and not low.empty else pd.DataFrame()
    controls = negative_controls(selected, high, low, balance) if not balance.empty else pd.DataFrame()
    covariate_features = [
        "n_gt",
        "n_ctrl",
        "response_norm",
        "max_state_entropy",
        "same_target_cross_dataset_total",
        "exact_response_available_fraction",
    ]
    max_abs_cov_smd = (
        float(balance[balance["feature"].isin(covariate_features)]["smd_high_minus_low"].abs().max())
        if not balance.empty
        else float("nan")
    )
    claims = per_ptype_claims(selected)
    n_pairs = int(len(selected))
    summary = {
        **config,
        "n_candidate_pairs_after_filter": int(len(candidates)),
        "n_pairs_unique": n_pairs,
        "n_high_conditions": int(selected["high_key"].nunique()) if not selected.empty else 0,
        "n_low_conditions": int(selected["low_key"].nunique()) if not selected.empty else 0,
        "n_datasets_total": int(len(set(selected["high_dataset"].astype(str)) | set(selected["low_dataset"].astype(str)))) if not selected.empty else 0,
        "n_high_datasets": int(selected["high_dataset"].nunique()) if not selected.empty else 0,
        "n_low_datasets": int(selected["low_dataset"].nunique()) if not selected.empty else 0,
        "n_perturbation_types": int(selected["perturbation_type_raw"].nunique()) if not selected.empty else 0,
        "n_claim_ready_ptypes": int(claims["claim_ready"].sum()) if not claims.empty else 0,
        "median_score_gap": float(selected["score_gap"].median()) if not selected.empty else 0.0,
        "max_abs_covariate_smd": max_abs_cov_smd,
        "top_high_dataset_fraction": side_dataset_fraction(high),
        "top_low_dataset_fraction": side_dataset_fraction(low),
        "top_dataset_pair_direction_fraction": pair_direction_fraction(selected),
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
    if summary["n_claim_ready_ptypes"] < 1:
        reasons.append("no_perturbation_type_claim_ready")
    if math.isfinite(summary["max_abs_covariate_smd"]) and summary["max_abs_covariate_smd"] > 0.75:
        reasons.append("max_covariate_smd_above_0p75")
    if summary["top_high_dataset_fraction"] > 0.25 or summary["top_low_dataset_fraction"] > 0.25:
        reasons.append("side_dataset_fraction_above_25pct")
    if summary["top_dataset_pair_direction_fraction"] > 0.15:
        reasons.append("dataset_pair_direction_above_15pct")
    if summary["crispri_pair_fraction"] > 0.80:
        reasons.append("crispri_fraction_above_80pct")
    if summary["any_negative_control_risk"]:
        reasons.append("negative_control_risk_present")
    summary["reasons"] = ";".join(reasons)
    summary["strict_pass"] = not reasons
    summary["partial_200pair"] = bool(
        summary["n_pairs_unique"] >= 180
        and summary["n_high_conditions"] >= 100
        and summary["n_low_conditions"] >= 100
        and summary["n_datasets_total"] >= 12
        and math.isfinite(summary["max_abs_covariate_smd"])
        and summary["max_abs_covariate_smd"] <= 1.25
    )
    return summary, selected, high, low, balance


def choose_best(config_rows: pd.DataFrame) -> pd.Series:
    work = config_rows.copy()
    work["strict_rank"] = work["strict_pass"].astype(bool).astype(int)
    work["partial_rank"] = work["partial_200pair"].astype(bool).astype(int)
    work["score"] = (
        1000 * work["strict_rank"]
        + 100 * work["partial_rank"]
        + work["n_pairs_unique"]
        - 50 * work["max_abs_covariate_smd"].fillna(99)
        - 20 * work["top_dataset_pair_direction_fraction"].fillna(1)
        - 10 * work["crispri_pair_fraction"].fillna(1)
    )
    return work.sort_values("score", ascending=False).iloc[0]


def write_report(payload: dict[str, Any], config_rows: pd.DataFrame, selected: pd.DataFrame, balance: pd.DataFrame, controls: pd.DataFrame, claims: pd.DataFrame) -> None:
    top_configs = config_rows.sort_values(["strict_pass", "partial_200pair", "n_pairs_unique"], ascending=[False, False, False]).head(12)
    lines = [
        "# Condition Neighborhood Support Strict Split Gate",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only strict split audit over train-only condition-neighborhood rows and candidate pairs.",
        "* No training, inference, GPU, canonical multi, Track C query, or checkpoint selection.",
        "* Mutates the first split draft using response-norm/cell-count calipers, exact/state/target-availability matching, side dataset caps, and dataset-pair caps.",
        "",
        "## Decision",
        "",
        f"* Selected config: `{payload['selected_config_name']}`.",
        f"* Reasons: `{payload['reasons'] if payload['reasons'] else 'none'}`.",
        f"* Unique pairs: `{payload['summary']['n_pairs_unique']}`; datasets total/high/low: `{payload['summary']['n_datasets_total']}/{payload['summary']['n_high_datasets']}/{payload['summary']['n_low_datasets']}`.",
        f"* Max covariate SMD: `{fmt(payload['summary']['max_abs_covariate_smd'])}`.",
        f"* Top dataset-pair fraction: `{fmt(payload['summary']['top_dataset_pair_direction_fraction'])}`.",
        f"* CRISPRi fraction: `{fmt(payload['summary']['crispri_pair_fraction'])}`.",
        "",
        "## Config Sweep",
        "",
        "| response caliper | cell caliper | state match | candidates | unique pairs | datasets | ptypes | cov SMD | pair-dir frac | CRISPRi frac | strict | partial | reasons |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in top_configs.iterrows():
        lines.append(
            f"| `{fmt(row['response_log_caliper'], 2)}` | `{fmt(row['cell_log_caliper'], 2)}` | `{bool(row['match_state_bin'])}` | "
            f"`{int(row['n_candidate_pairs_after_filter'])}` | `{int(row['n_pairs_unique'])}` | `{int(row['n_datasets_total'])}` | "
            f"`{int(row['n_perturbation_types'])}` | `{fmt(row['max_abs_covariate_smd'])}` | "
            f"`{fmt(row['top_dataset_pair_direction_fraction'])}` | `{fmt(row['crispri_pair_fraction'])}` | "
            f"`{bool(row['strict_pass'])}` | `{bool(row['partial_200pair'])}` | `{row['reasons']}` |"
        )
    lines.extend(
        [
            "",
            "## Selected Balance",
            "",
            "| feature | high mean | low mean | high median | low median | SMD | AUC risk |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in balance.iterrows():
        lines.append(
            f"| `{row['feature']}` | `{fmt(row['high_mean'])}` | `{fmt(row['low_mean'])}` | "
            f"`{fmt(row['high_median'])}` | `{fmt(row['low_median'])}` | "
            f"`{fmt(row['smd_high_minus_low'])}` | `{fmt(row['auc_discriminability'])}` |"
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
            f"| `{row['perturbation_type_raw']}` | `{int(row['n_pairs'])}` | `{int(row['n_high_conditions'])}` | "
            f"`{int(row['n_low_conditions'])}` | `{int(row['n_high_datasets'])}` | `{int(row['n_low_datasets'])}` | `{bool(row['claim_ready'])}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if payload["status"].endswith("pass_external_audit_no_gpu"):
        lines.append("* A strict candidate passed CPU balance gates and may go to external audit before any GPU launch.")
    elif payload["status"].endswith("partial_mutate_no_gpu"):
        lines.append("* The branch remains promising but not launchable. The stricter matching improves auditability but does not satisfy the preferred threshold set.")
    else:
        lines.append("* The stricter audit blocks the current support-score split definition from GPU use.")
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
    rows, pairs = load_augmented_pairs()
    summaries: list[dict[str, Any]] = []
    selections: dict[int, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for idx, config in enumerate(CONFIGS):
        summary, selected, high, low, balance = assess_config(config, rows, pairs)
        summary["config_index"] = idx
        summaries.append(summary)
        selections[idx] = (selected, high, low, balance)
    config_rows = pd.DataFrame(summaries)
    best = choose_best(config_rows)
    best_idx = int(best["config_index"])
    selected, high, low, balance = selections[best_idx]
    controls = negative_controls(selected, high, low, balance)
    claims = per_ptype_claims(selected)
    parent = load_json(PARENT_SPLIT)
    high_split_path = ""
    low_split_path = ""
    if not selected.empty:
        tag = f"strict_{len(selected)}pair_cfg{best_idx}"
        high_split_path = str(SPLIT_DIR / f"split_seed42_xverse_condition_neighborhood_high_support_{tag}.json")
        low_split_path = str(SPLIT_DIR / f"split_seed42_xverse_condition_neighborhood_low_support_{tag}.json")
        write_json(Path(high_split_path), split_from_conditions(parent, high))
        write_json(Path(low_split_path), split_from_conditions(parent, low))
    selected.to_csv(OUT_SELECTED, index=False)
    balance.to_csv(OUT_BALANCE, index=False)
    controls.to_csv(OUT_CONTROLS, index=False)
    config_rows.to_csv(OUT_CONFIGS, index=False)
    if bool(best["strict_pass"]):
        status = "condition_neighborhood_support_strict_split_pass_external_audit_no_gpu"
    elif bool(best["partial_200pair"]):
        status = "condition_neighborhood_support_strict_split_partial_mutate_no_gpu"
    else:
        status = "condition_neighborhood_support_strict_split_fail_no_gpu"
    payload = {
        "created_at": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "selected_config_name": f"cfg{best_idx}",
        "selected_config": {k: (bool(v) if isinstance(v, np.bool_) else float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v) for k, v in best.to_dict().items()},
        "summary": {k: (bool(v) if isinstance(v, np.bool_) else float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v) for k, v in best.to_dict().items() if k not in {"reasons"}},
        "reasons": str(best.get("reasons", "")),
        "inputs": {"rows": str(ROWS_CSV), "pairs": str(PAIRS_CSV), "parent_split": str(PARENT_SPLIT)},
        "outputs": {
            "report": str(OUT_MD),
            "json": str(OUT_JSON),
            "configs": str(OUT_CONFIGS),
            "selected_pairs": str(OUT_SELECTED),
            "balance": str(OUT_BALANCE),
            "negative_controls": str(OUT_CONTROLS),
            "high_split": high_split_path,
            "low_split": low_split_path,
        },
        "boundary": "cpu_report_only_strict_split_audit_no_training_no_inference_no_gpu_no_canonical_multi_no_trackc_query",
    }
    write_json(OUT_JSON, payload)
    write_report(payload, config_rows, selected, balance, controls, claims)
    print(
        json.dumps(
            {
                "status": status,
                "selected_config": f"cfg{best_idx}",
                "n_pairs": int(best["n_pairs_unique"]),
                "max_covariate_smd": float(best["max_abs_covariate_smd"]),
                "reasons": str(best.get("reasons", "")),
                "report": str(OUT_MD),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
