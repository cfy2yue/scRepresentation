#!/usr/bin/env python3
"""Null panel for dual-baseline tail-geometry admission gate.

CPU/report-only. This validates the positive train-only tail-geometry
admission axis with pair-label shuffles, within-stratum label shuffles,
cap-sensitivity summaries, split integrity checks, and missingness controls.
No training, inference, checkpoint selection, canonical multi, Track C query,
or GPU is used.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
OPS = ROOT / "ops"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import audit_latentfm_dual_baseline_tail_geometry_admission_gate_20260629 as adm  # noqa: E402


DEFAULT_ADMISSION_DIR = ROOT / "reports/dual_baseline_tail_geometry_admission_gate_20260629"
DEFAULT_OUT_DIR = ROOT / "reports/dual_baseline_tail_geometry_null_panel_20260629"
DEFAULT_SPLIT_DIR = ROOT / "dataset/biFlow_data/xverse_dual_baseline_tail_geometry_null_splits_20260629"
DEFAULT_ADMISSION_JSON = DEFAULT_ADMISSION_DIR / "latentfm_dual_baseline_tail_geometry_admission_gate_20260629.json"
DEFAULT_SELECTED_PAIRS = DEFAULT_ADMISSION_DIR / "dual_baseline_tail_geometry_selected_pairs.csv"
DEFAULT_FEATURE_ROWS = DEFAULT_ADMISSION_DIR / "dual_baseline_tail_geometry_feature_rows.csv"
OUT_JSON_NAME = "latentfm_dual_baseline_tail_geometry_null_panel_20260629.json"
OUT_MD_NAME = "LATENTFM_DUAL_BASELINE_TAIL_GEOMETRY_NULL_PANEL_20260629.md"

TAIL_RAW_COLS = [
    "ctrl_gt_mmd_proxy",
    "tail95_shift",
    "tail99_shift",
    "abs_var_shift",
    "tail_risk_ratio",
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
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def split_items(split: dict[str, Any], group: str) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for dataset, groups in split.items():
        for condition in (groups or {}).get(group, []) or []:
            out.add((str(dataset), str(condition)))
    return out


def nontrain_groups(split: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for groups in split.values():
        for group in groups:
            if str(group) != "train":
                out.add(str(group))
    return out


def write_side_split(parent: dict[str, Any], rows: list[dict[str, Any]], side: str, path: Path) -> None:
    out = copy.deepcopy(parent)
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row[f"placebo_{side}_dataset"])].append(str(row[f"placebo_{side}_condition"]))
    for dataset in out:
        out[dataset]["train"] = sorted(set(by_dataset.get(str(dataset), [])))
    write_json(path, out)


def compare_nontrain(parent: dict[str, Any], child: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for dataset, groups in parent.items():
        cand = child.get(dataset, {})
        for group, values in groups.items():
            if str(group) == "train":
                continue
            if list(values or []) != list(cand.get(group, []) or []):
                reasons.append(f"nontrain changed: {dataset}/{group}")
    return reasons


def validate_split(parent: dict[str, Any], high: dict[str, Any], low: dict[str, Any], n_pairs: int) -> list[str]:
    reasons: list[str] = []
    parent_train = split_items(parent, "train")
    high_train = split_items(high, "train")
    low_train = split_items(low, "train")
    if len(high_train) != n_pairs:
        reasons.append(f"high train count {len(high_train)} != pairs {n_pairs}")
    if len(low_train) != n_pairs:
        reasons.append(f"low train count {len(low_train)} != pairs {n_pairs}")
    if high_train & low_train:
        reasons.append(f"high/low overlap {len(high_train & low_train)}")
    if not high_train <= parent_train:
        reasons.append(f"high outside parent train {len(high_train - parent_train)}")
    if not low_train <= parent_train:
        reasons.append(f"low outside parent train {len(low_train - parent_train)}")
    for group in sorted(nontrain_groups(parent)):
        high_overlap = high_train & split_items(high, group)
        low_overlap = low_train & split_items(low, group)
        if high_overlap:
            reasons.append(f"high train overlaps {group}: {len(high_overlap)}")
        if low_overlap:
            reasons.append(f"low train overlaps {group}: {len(low_overlap)}")
    reasons.extend(compare_nontrain(parent, high))
    reasons.extend(compare_nontrain(parent, low))
    return reasons


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pairshuffle_rows(pairs: pd.DataFrame, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    true_high_in_placebo_high = 0
    placebo_resid_gaps: list[float] = []
    placebo_tail_gaps: list[float] = []
    original_resid_gaps: list[float] = []
    original_tail_gaps: list[float] = []
    high_ds: Counter[str] = Counter()
    low_ds: Counter[str] = Counter()
    ptype_counts: Counter[str] = Counter()
    for pair_id, row in pairs.reset_index(drop=True).iterrows():
        keep = rng.random() < 0.5
        h_resid = float(row["high_tail_resid_score"])
        l_resid = float(row["low_tail_resid_score"])
        h_tail = float(row["high_tail_geometry_score"])
        l_tail = float(row["low_tail_geometry_score"])
        original_resid_gaps.append(float(row["residual_score_gap"]))
        original_tail_gaps.append(float(row["tail_geometry_gap"]))
        ptype_counts[str(row["perturbation_type_raw"])] += 1
        if keep:
            ph_dataset = row["high_dataset"]
            ph_condition = row["high_condition"]
            ph_resid = h_resid
            ph_tail = h_tail
            pl_dataset = row["low_dataset"]
            pl_condition = row["low_condition"]
            pl_resid = l_resid
            pl_tail = l_tail
            true_high_in_placebo_high += 1
        else:
            ph_dataset = row["low_dataset"]
            ph_condition = row["low_condition"]
            ph_resid = l_resid
            ph_tail = l_tail
            pl_dataset = row["high_dataset"]
            pl_condition = row["high_condition"]
            pl_resid = h_resid
            pl_tail = h_tail
        high_ds[str(ph_dataset)] += 1
        low_ds[str(pl_dataset)] += 1
        placebo_resid_gaps.append(ph_resid - pl_resid)
        placebo_tail_gaps.append(ph_tail - pl_tail)
        rows.append(
            {
                **row.to_dict(),
                "pair_id": pair_id,
                "seed": seed,
                "swapped": not keep,
                "placebo_high_dataset": ph_dataset,
                "placebo_high_condition": ph_condition,
                "placebo_high_tail_resid_score": ph_resid,
                "placebo_high_tail_geometry_score": ph_tail,
                "placebo_low_dataset": pl_dataset,
                "placebo_low_condition": pl_condition,
                "placebo_low_tail_resid_score": pl_resid,
                "placebo_low_tail_geometry_score": pl_tail,
                "placebo_residual_score_gap": ph_resid - pl_resid,
                "placebo_tail_geometry_gap": ph_tail - pl_tail,
            }
        )
    mean_orig_resid = float(np.mean(original_resid_gaps)) if original_resid_gaps else 0.0
    mean_orig_tail = float(np.mean(original_tail_gaps)) if original_tail_gaps else 0.0
    mean_placebo_resid = float(np.mean(placebo_resid_gaps)) if placebo_resid_gaps else 0.0
    mean_placebo_tail = float(np.mean(placebo_tail_gaps)) if placebo_tail_gaps else 0.0
    summary = {
        "seed": seed,
        "pairs": int(len(pairs)),
        "true_high_fraction_in_placebo_high": true_high_in_placebo_high / len(pairs) if len(pairs) else 0.0,
        "mean_original_residual_score_gap": mean_orig_resid,
        "mean_placebo_residual_score_gap": mean_placebo_resid,
        "abs_placebo_over_original_residual_gap": abs(mean_placebo_resid) / abs(mean_orig_resid) if mean_orig_resid else None,
        "mean_original_tail_geometry_gap": mean_orig_tail,
        "mean_placebo_tail_geometry_gap": mean_placebo_tail,
        "abs_placebo_over_original_tail_gap": abs(mean_placebo_tail) / abs(mean_orig_tail) if mean_orig_tail else None,
        "top_high_dataset_fraction": max(high_ds.values()) / sum(high_ds.values()) if high_ds else 0.0,
        "top_low_dataset_fraction": max(low_ds.values()) / sum(low_ds.values()) if low_ds else 0.0,
        "high_datasets": len(high_ds),
        "low_datasets": len(low_ds),
        "perturbation_type_counts": dict(ptype_counts),
        "top_high_dataset_counts": dict(high_ds.most_common(12)),
        "top_low_dataset_counts": dict(low_ds.most_common(12)),
    }
    return rows, summary


def run_pairshuffle(
    pairs: pd.DataFrame,
    parent: dict[str, Any],
    seeds: list[int],
    out_dir: Path,
    split_dir: Path,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    generated: list[dict[str, str]] = []
    reasons: list[str] = []
    for seed in seeds:
        rows, summary = pairshuffle_rows(pairs, seed)
        high_path = split_dir / f"split_seed42_xverse_tail_geometry_pairshuffle_seed{seed}_high_320pair.json"
        low_path = split_dir / f"split_seed42_xverse_tail_geometry_pairshuffle_seed{seed}_low_320pair.json"
        csv_path = out_dir / f"pairshuffle_seed{seed}_matched_pairs.csv"
        write_side_split(parent, rows, "high", high_path)
        write_side_split(parent, rows, "low", low_path)
        write_csv(csv_path, rows)
        high = load_json(high_path)
        low = load_json(low_path)
        seed_reasons = validate_split(parent, high, low, len(pairs))
        imbalance = abs(float(summary["true_high_fraction_in_placebo_high"]) - 0.5)
        ratio = summary["abs_placebo_over_original_residual_gap"]
        if imbalance > 0.10:
            seed_reasons.append(f"true-high imbalance {imbalance:.4f} > 0.1000")
        if ratio is None or float(ratio) > 0.10:
            seed_reasons.append(f"abs placebo/original residual gap {ratio} > 0.10")
        if float(summary["top_high_dataset_fraction"]) > 0.20:
            seed_reasons.append("top high dataset fraction > 0.20")
        if float(summary["top_low_dataset_fraction"]) > 0.20:
            seed_reasons.append("top low dataset fraction > 0.20")
        summary["reasons"] = seed_reasons
        summary["pass"] = not seed_reasons
        summary["paths"] = {"high_split": str(high_path), "low_split": str(low_path), "pairs_csv": str(csv_path)}
        summaries.append(summary)
        generated.append({"seed": str(seed), "high_split": str(high_path), "low_split": str(low_path), "pairs_csv": str(csv_path)})
        reasons.extend([f"seed{seed}:{reason}" for reason in seed_reasons])
    abs_gaps = [abs(float(row["mean_placebo_residual_score_gap"])) for row in summaries]
    p95 = float(np.quantile(abs_gaps, 0.95)) if abs_gaps else float("nan")
    if math.isfinite(p95) and p95 >= 0.10:
        reasons.append(f"pairshuffle_abs_residual_gap_p95 {p95:.4f} >= 0.10")
    return {
        "status": "pairshuffle_pass" if not reasons else "pairshuffle_fail",
        "pass": not reasons,
        "reasons": reasons,
        "abs_placebo_residual_gap_p95": p95,
        "seed_summaries": summaries,
        "generated": generated,
    }


def load_support_rows_for_parent(parent: dict[str, Any]) -> pd.DataFrame:
    rows = adm.load_rows()
    parent_train = split_items(parent, "train")
    keys = {f"{ds}||{condition}" for ds, condition in parent_train}
    return rows[rows["key"].astype(str).isin(keys)].copy()


def shuffle_tail_features_within_strata(features: pd.DataFrame, support_rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strata = support_rows[["key", "perturbation_type_raw", "exact_bool", "state_bin", "target_availability_bin"]].copy()
    work = features.merge(strata, on="key", how="left")
    out = work.copy()
    group_cols = ["perturbation_type_raw", "exact_bool", "state_bin", "target_availability_bin"]
    for _, idx in work.groupby(group_cols, dropna=False).groups.items():
        idx_list = list(idx)
        if len(idx_list) <= 1:
            continue
        perm = rng.permutation(idx_list)
        out.loc[idx_list, TAIL_RAW_COLS] = work.loc[perm, TAIL_RAW_COLS].to_numpy()
    return out.drop(columns=[c for c in group_cols if c in out.columns])


def assess_feature_set(support_rows: pd.DataFrame, features: pd.DataFrame, configs: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = support_rows.copy()
    rows.attrs["features"] = features
    summaries: list[dict[str, Any]] = []
    selected_by_config: dict[str, pd.DataFrame] = {}
    for config in configs:
        summary, selected, _, _ = adm.assess_config(rows, config)
        summaries.append(summary)
        selected_by_config[str(config["name"])] = selected
    return pd.DataFrame(summaries), selected_by_config


def run_labelshuffle(
    support_rows: pd.DataFrame,
    features: pd.DataFrame,
    selected_config: str,
    seeds: list[int],
    out_dir: Path,
) -> dict[str, Any]:
    reasons: list[str] = []
    seed_summaries: list[dict[str, Any]] = []
    configs_by_name = {str(c["name"]): c for c in adm.CONFIGS}
    frozen_config = configs_by_name[selected_config]
    for seed in seeds:
        shuffled = shuffle_tail_features_within_strata(features, support_rows, seed)
        frozen_rows, _ = assess_feature_set(support_rows, shuffled, [frozen_config])
        all_rows, _ = assess_feature_set(support_rows, shuffled, adm.CONFIGS)
        frozen = frozen_rows.iloc[0].to_dict()
        best = adm.choose_best(all_rows).to_dict()
        strict_count = int(all_rows["strict_pass"].astype(bool).sum()) if "strict_pass" in all_rows else 0
        all_csv = out_dir / f"labelshuffle_seed{seed}_config_rows.csv"
        all_rows.to_csv(all_csv, index=False)
        summary = {
            "seed": seed,
            "frozen_config": {
                "n_pairs_unique": int(frozen["n_pairs_unique"]),
                "median_tail_geometry_gap": float(frozen["median_tail_geometry_gap"]),
                "median_residual_score_gap": float(frozen["median_residual_score_gap"]),
                "max_abs_covariate_smd": float(frozen["max_abs_covariate_smd"]),
                "strict_pass": bool(frozen["strict_pass"]),
                "reasons": str(frozen["reasons"]),
            },
            "best_config": {
                "name": str(best["name"]),
                "n_pairs_unique": int(best["n_pairs_unique"]),
                "median_tail_geometry_gap": float(best["median_tail_geometry_gap"]),
                "median_residual_score_gap": float(best["median_residual_score_gap"]),
                "max_abs_covariate_smd": float(best["max_abs_covariate_smd"]),
                "strict_pass": bool(best["strict_pass"]),
                "reasons": str(best["reasons"]),
            },
            "strict_pass_count_across_configs": strict_count,
            "config_rows": str(all_csv),
        }
        seed_json = out_dir / f"labelshuffle_seed{seed}_summary.json"
        write_json(seed_json, summary)
        seed_summaries.append(summary)
        if strict_count > 0:
            reasons.append(f"seed{seed}_strict_pass_count_{strict_count}")
    best_resid = [float(s["best_config"]["median_residual_score_gap"]) for s in seed_summaries]
    best_tail = [float(s["best_config"]["median_tail_geometry_gap"]) for s in seed_summaries]
    resid_p95 = float(np.quantile(best_resid, 0.95)) if best_resid else float("nan")
    tail_p95 = float(np.quantile(best_tail, 0.95)) if best_tail else float("nan")
    if math.isfinite(resid_p95) and resid_p95 >= 0.10:
        reasons.append(f"labelshuffle_best_residual_gap_p95 {resid_p95:.4f} >= 0.10")
    if math.isfinite(tail_p95) and tail_p95 >= 0.15:
        reasons.append(f"labelshuffle_best_tail_gap_p95 {tail_p95:.4f} >= 0.15")
    return {
        "status": "labelshuffle_pass" if not reasons else "labelshuffle_fail",
        "pass": not reasons,
        "reasons": reasons,
        "best_residual_gap_p95": resid_p95,
        "best_tail_gap_p95": tail_p95,
        "seed_summaries": seed_summaries,
    }


def missingness_controls(pairs: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, hcol, lcol in [
        ("directional_alignment_missing", "high_directional_alignment", "low_directional_alignment"),
    ]:
        h = pairs[hcol].isna().astype(float)
        l = pairs[lcol].isna().astype(float)
        rows.append(
            {
                "feature": name,
                "high_missing_fraction": float(h.mean()),
                "low_missing_fraction": float(l.mean()),
                "smd_high_minus_low": adm.smd(h, l),
                "auc_discriminability": adm.auc_discriminability(h, l),
            }
        )
    max_auc = max(float(r["auc_discriminability"]) for r in rows) if rows else 0.0
    max_smd = max(abs(float(r["smd_high_minus_low"])) for r in rows) if rows else 0.0
    reasons: list[str] = []
    if max_auc > 0.65:
        reasons.append(f"missingness_auc {max_auc:.4f} > 0.65")
    if max_smd > 0.35:
        reasons.append(f"missingness_smd {max_smd:.4f} > 0.35")
    return {
        "status": "missingness_controls_pass" if not reasons else "missingness_controls_fail",
        "pass": not reasons,
        "reasons": reasons,
        "rows": rows,
    }


def read_cap_summary(cap: int, report_dir: Path) -> dict[str, Any]:
    config_path = report_dir / "dual_baseline_tail_geometry_config_rows.csv"
    json_path = report_dir / "latentfm_dual_baseline_tail_geometry_admission_gate_20260629.json"
    if not config_path.is_file() or not json_path.is_file():
        return {"cap": cap, "status": "missing", "pass": False, "reasons": [f"missing cap{cap} report"]}
    payload = load_json(json_path)
    configs = pd.read_csv(config_path)
    strict = configs[configs["strict_pass"].astype(str).str.lower().isin(["true", "1"])]
    best = configs.sort_values(["strict_pass", "n_pairs_unique"], ascending=[False, False]).iloc[0].to_dict()
    reasons: list[str] = []
    if int(best["n_pairs_unique"]) < 250:
        reasons.append("n_pairs_unique_below_250")
    if float(best["median_residual_score_gap"]) < 0.25:
        reasons.append("median_residual_score_gap_below_0p25")
    if float(best["median_tail_geometry_gap"]) < 0.30:
        reasons.append("median_tail_geometry_gap_below_0p30")
    if float(best["max_abs_covariate_smd"]) > 0.35:
        reasons.append("max_abs_covariate_smd_above_0p35")
    if float(best["raw_support_auc"]) > 0.70:
        reasons.append("raw_support_auc_above_0p70")
    if float(best["response_norm_auc"]) > 0.65:
        reasons.append("response_norm_auc_above_0p65")
    if float(best["directional_auc"]) > 0.70:
        reasons.append("directional_auc_above_0p70")
    if strict.empty and float(best["max_abs_covariate_smd"]) > 0.20:
        reasons.append("no_strict_pass_and_cov_smd_above_0p20")
    return {
        "cap": cap,
        "status": payload.get("status"),
        "pass": not reasons,
        "reasons": reasons,
        "strict_pass_count": int(len(strict)),
        "best": {
            "name": str(best["name"]),
            "n_pairs_unique": int(best["n_pairs_unique"]),
            "median_tail_geometry_gap": float(best["median_tail_geometry_gap"]),
            "median_residual_score_gap": float(best["median_residual_score_gap"]),
            "max_abs_covariate_smd": float(best["max_abs_covariate_smd"]),
            "raw_support_auc": float(best["raw_support_auc"]),
            "response_norm_auc": float(best["response_norm_auc"]),
            "directional_auc": float(best["directional_auc"]),
            "strict_pass": bool(str(best["strict_pass"]).lower() in ["true", "1"]),
        },
        "report_dir": str(report_dir),
    }


def cap_sensitivity(cap_dirs: dict[int, Path]) -> dict[str, Any]:
    rows = [read_cap_summary(cap, path) for cap, path in sorted(cap_dirs.items())]
    reasons: list[str] = []
    for row in rows:
        if not row["pass"]:
            reasons.extend([f"cap{row['cap']}:{reason}" for reason in row.get("reasons", [])])
    return {
        "status": "cap_sensitivity_pass" if not reasons else "cap_sensitivity_fail",
        "pass": not reasons,
        "reasons": reasons,
        "caps": rows,
    }


def parse_cap_dirs(text: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        cap_s, path_s = item.split(":", 1)
        out[int(cap_s)] = Path(path_s)
    return out


def decide(pair: dict[str, Any], label: dict[str, Any], miss: dict[str, Any], cap: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    for block_name, block in [("pairshuffle", pair), ("labelshuffle", label), ("missingness", miss), ("cap_sensitivity", cap)]:
        if not block.get("pass"):
            reasons.append(f"{block_name}_failed")
            reasons.extend([f"{block_name}:{r}" for r in block.get("reasons", [])[:20]])
    status = (
        "dual_baseline_tail_geometry_null_panel_pass_prepare_gpu_smoke"
        if not reasons
        else "dual_baseline_tail_geometry_null_panel_blocks_gpu_smoke"
    )
    return {
        "status": status,
        "gpu_smoke_authorized_next": not reasons,
        "reasons": reasons,
    }


def render(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Dual-Baseline Tail-Geometry Null Panel",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{decision['status']}`",
        "",
        f"GPU smoke authorized next: `{decision['gpu_smoke_authorized_next']}`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only null and sensitivity audit for the train-only tail-geometry admission gate.",
        "* Reuses admission feature rows and selected pairs; no training, inference, checkpoint selection, canonical multi, Track C query, or GPU.",
        "",
        "## Pairshuffle",
        "",
        f"* status: `{payload['pairshuffle']['status']}`",
        f"* p95 abs placebo residual gap: `{fmt(payload['pairshuffle']['abs_placebo_residual_gap_p95'])}`",
        "",
        "| seed | pass | true-high frac | placebo/original resid | top high ds | top low ds | reasons |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["pairshuffle"]["seed_summaries"]:
        lines.append(
            f"| `{row['seed']}` | `{row['pass']}` | `{fmt(row['true_high_fraction_in_placebo_high'])}` | "
            f"`{fmt(row['abs_placebo_over_original_residual_gap'])}` | `{fmt(row['top_high_dataset_fraction'])}` | "
            f"`{fmt(row['top_low_dataset_fraction'])}` | `{';'.join(row['reasons'])}` |"
        )
    lines.extend(
        [
            "",
            "## Label Shuffle",
            "",
            f"* status: `{payload['labelshuffle']['status']}`",
            f"* p95 best residual gap: `{fmt(payload['labelshuffle']['best_residual_gap_p95'])}`",
            f"* p95 best tail gap: `{fmt(payload['labelshuffle']['best_tail_gap_p95'])}`",
            "",
            "| seed | strict pass count | best config | pairs | tail gap | resid gap | strict |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["labelshuffle"]["seed_summaries"]:
        best = row["best_config"]
        lines.append(
            f"| `{row['seed']}` | `{row['strict_pass_count_across_configs']}` | `{best['name']}` | "
            f"`{best['n_pairs_unique']}` | `{fmt(best['median_tail_geometry_gap'])}` | "
            f"`{fmt(best['median_residual_score_gap'])}` | `{best['strict_pass']}` |"
        )
    lines.extend(["", "## Missingness Controls", "", f"* status: `{payload['missingness']['status']}`", "", "| feature | high missing | low missing | SMD | AUC |", "|---|---:|---:|---:|---:|"])
    for row in payload["missingness"]["rows"]:
        lines.append(
            f"| `{row['feature']}` | `{fmt(row['high_missing_fraction'])}` | `{fmt(row['low_missing_fraction'])}` | "
            f"`{fmt(row['smd_high_minus_low'])}` | `{fmt(row['auc_discriminability'])}` |"
        )
    lines.extend(["", "## Cap Sensitivity", "", f"* status: `{payload['cap_sensitivity']['status']}`", "", "| cap | pass | strict count | best config | pairs | tail gap | resid gap | cov SMD | support AUC | response AUC | direction AUC | reasons |", "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for row in payload["cap_sensitivity"]["caps"]:
        best = row.get("best") or {}
        lines.append(
            f"| `{row['cap']}` | `{row['pass']}` | `{row.get('strict_pass_count', '')}` | `{best.get('name', '')}` | "
            f"`{best.get('n_pairs_unique', '')}` | `{fmt(best.get('median_tail_geometry_gap'))}` | "
            f"`{fmt(best.get('median_residual_score_gap'))}` | `{fmt(best.get('max_abs_covariate_smd'))}` | "
            f"`{fmt(best.get('raw_support_auc'))}` | `{fmt(best.get('response_norm_auc'))}` | `{fmt(best.get('directional_auc'))}` | "
            f"`{';'.join(row.get('reasons', []))}` |"
        )
    lines.extend(["", "## Decision", ""])
    if decision["reasons"]:
        lines.append("* Blockers:")
        lines.extend(f"  * `{reason}`" for reason in decision["reasons"][:50])
    else:
        lines.append("* All CPU null/sensitivity checks passed. A bounded GPU smoke may be prepared, still without promotion claims.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"* JSON: `{payload['outputs']['json']}`",
            f"* Pairshuffle splits: `{payload['outputs']['split_dir']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission-json", type=Path, default=DEFAULT_ADMISSION_JSON)
    parser.add_argument("--selected-pairs", type=Path, default=DEFAULT_SELECTED_PAIRS)
    parser.add_argument("--feature-rows", type=Path, default=DEFAULT_FEATURE_ROWS)
    parser.add_argument("--parent-split", type=Path, default=adm.PARENT_SPLIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--pairshuffle-seeds", default="43,44,45,46")
    parser.add_argument("--labelshuffle-seeds", default="43,44,45")
    parser.add_argument(
        "--cap-dirs",
        default=(
            "32:/data/cyx/1030/scLatent/reports/dual_baseline_tail_geometry_admission_gate_cap32_20260629,"
            "64:/data/cyx/1030/scLatent/reports/dual_baseline_tail_geometry_admission_gate_20260629,"
            "128:/data/cyx/1030/scLatent/reports/dual_baseline_tail_geometry_admission_gate_cap128_20260629"
        ),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.split_dir.mkdir(parents=True, exist_ok=True)
    admission = load_json(args.admission_json)
    parent = load_json(args.parent_split)
    pairs = pd.read_csv(args.selected_pairs)
    features = pd.read_csv(args.feature_rows)
    support_rows = load_support_rows_for_parent(parent)

    pair = run_pairshuffle(pairs, parent, parse_ints(args.pairshuffle_seeds), args.out_dir, args.split_dir)
    label = run_labelshuffle(support_rows, features, str(admission["selected_config_name"]), parse_ints(args.labelshuffle_seeds), args.out_dir)
    miss = missingness_controls(pairs)
    cap = cap_sensitivity(parse_cap_dirs(args.cap_dirs))
    decision = decide(pair, label, miss, cap)

    out_json = args.out_dir / OUT_JSON_NAME
    out_md = args.out_dir / OUT_MD_NAME
    payload = {
        "created_at": now_cst(),
        "inputs": {
            "admission_json": str(args.admission_json),
            "admission_json_sha256": sha256(args.admission_json),
            "selected_pairs": str(args.selected_pairs),
            "selected_pairs_sha256": sha256(args.selected_pairs),
            "feature_rows": str(args.feature_rows),
            "feature_rows_sha256": sha256(args.feature_rows),
            "parent_split": str(args.parent_split),
            "parent_split_sha256": sha256(args.parent_split),
        },
        "boundary": {
            "cpu_report_only": True,
            "no_training_or_inference": True,
            "no_checkpoint_selection": True,
            "no_canonical_multi": True,
            "no_trackc_query": True,
            "no_gpu": True,
        },
        "pairshuffle": pair,
        "labelshuffle": label,
        "missingness": miss,
        "cap_sensitivity": cap,
        "decision": decision,
        "outputs": {"json": str(out_json), "report": str(out_md), "split_dir": str(args.split_dir)},
    }
    write_json(out_json, payload)
    out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "report": str(out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
