#!/usr/bin/env python3
"""Matched split-level information-axis gate for LatentFM scaling law v2.

CPU/report-only. It combines existing exact-response and HVG response matrices,
then asks whether any information axis already has enough matched high/low
split evidence to justify a downstream GPU packet. It does not train, infer, or
touch held-out Track C query/canonical multi selection.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
EXACT = ROOT / "reports/exact_response_information_posthoc_combined_20260628/exact_response_information_split_matrix.csv"
HVG = ROOT / "reports/hvg_response_scaling_design_matrix_20260628/hvg_response_scaling_design_matrix.csv"
OUT_DIR = ROOT / "reports/scaling_v2_matched_information_gate_20260628"
OUT_MD = OUT_DIR / "LATENTFM_SCALING_V2_MATCHED_INFORMATION_GATE_20260628.md"
OUT_JSON = OUT_DIR / "latentfm_scaling_v2_matched_information_gate_20260628.json"

AXES = {
    "exact_condition_fraction": "fraction of train conditions with exact raw-expression response-information coverage",
    "exact_hvg_share_top1000_mean": "exact mean response-energy share captured by top1000 HVG",
    "exact_abundance_share_top1000_mean": "exact mean response-energy share captured by top1000 abundance genes",
    "exact_hvg_minus_abundance_top1000_mean": "HVG response share beyond abundance baseline",
    "hvg_condition_exact_fraction": "condition-level HVG exact-measured fraction",
    "hvg_top1000_condition_exact_mean": "exact condition-level top1000 HVG response share",
    "hvg_top1000_group_or_dataset_prior_mean": "group/dataset prior top1000 HVG response share",
    "hvg_top1000_advantage_group_or_dataset_prior_mean": "HVG response advantage over random prior",
}
OUTCOMES = ["cross_pp_delta", "family_pp_delta", "family_mmd_delta", "tail_score"]
CONFOUNDS = [
    "log_n_train_conditions",
    "base_dataset_effective_count",
    "base_background_effective_count",
    "base_perturbation_type_effective_count",
    "base_target_gene_effective_count",
]


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def split_family(name: str) -> str:
    if "gene_only_fixed256_budget64_128_256" in name:
        return "gene_only_fixed256_budget_sweep"
    if "scaling_protocol_breadth" in name:
        return "protocol_breadth"
    if "xverse_trainonly_scaling" in name:
        return "trainonly_scaling_v2"
    if "modality_pathway" in name:
        return "modality_pathway"
    return "other"


def read_matrix() -> pd.DataFrame:
    exact = pd.read_csv(EXACT)
    hvg = pd.read_csv(HVG)
    exact = exact.drop_duplicates(subset=["split_name"]).copy()
    hvg = hvg.drop_duplicates(subset=["split_name"]).copy()
    hvg_cols = [
        "split_name",
        "raw_expression_available_fraction",
        "hvg_dataset_prior_fraction",
        "hvg_condition_exact_fraction",
        "hvg_group_or_dataset_prior_fraction",
        "gene_condition_fraction_from_meta",
        "chemical_condition_fraction_from_meta",
        "hvg_top1000_dataset_exact_mean",
        "hvg_top1000_condition_exact_mean",
        "hvg_top1000_group_or_dataset_prior_mean",
        "hvg_top1000_random_group_or_dataset_prior_mean",
        "hvg_top1000_advantage_group_or_dataset_prior_mean",
        "hvg_top1000_oracle_group_or_dataset_prior_mean",
    ]
    merged = exact.merge(hvg[hvg_cols], on="split_name", how="left", validate="one_to_one")
    merged = merged[merged["has_downstream_outcome"].astype(bool)].copy()
    merged["split_family"] = merged["split_name"].map(split_family)
    merged["log_n_train_conditions"] = np.log1p(pd.to_numeric(merged["n_train_conditions"], errors="coerce"))
    for col in list(AXES) + OUTCOMES + CONFOUNDS:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged.reset_index(drop=True)


def normed_confounds(df: pd.DataFrame) -> pd.DataFrame:
    out = df[CONFOUNDS].copy()
    for col in CONFOUNDS:
        med = out[col].median()
        mad = (out[col] - med).abs().median()
        scale = mad if mad and np.isfinite(mad) else out[col].std()
        if not scale or not np.isfinite(scale):
            scale = 1.0
        out[col] = (out[col] - med) / scale
    return out


def pair_distance(z: pd.DataFrame, i: int, j: int) -> float:
    vals = (z.iloc[i] - z.iloc[j]).to_numpy(dtype=float)
    if np.any(~np.isfinite(vals)):
        return float("inf")
    return float(np.sqrt(np.mean(vals**2)))


def make_pairs(df: pd.DataFrame, axis: str, distance_cutoff: float, require_same_family: bool) -> list[dict[str, Any]]:
    ok = df[axis].notna()
    sub = df[ok].copy().reset_index(drop=True)
    if len(sub) < 6:
        return []
    lo_q = sub[axis].quantile(1 / 3)
    hi_q = sub[axis].quantile(2 / 3)
    low_idx = list(sub.index[sub[axis] <= lo_q])
    high_idx = list(sub.index[sub[axis] >= hi_q])
    z = normed_confounds(sub)
    candidates: list[dict[str, Any]] = []
    for hi in high_idx:
        for lo in low_idx:
            if hi == lo:
                continue
            if require_same_family and sub.loc[hi, "split_family"] != sub.loc[lo, "split_family"]:
                continue
            d = pair_distance(z, hi, lo)
            if d > distance_cutoff:
                continue
            axis_delta = float(sub.loc[hi, axis] - sub.loc[lo, axis])
            if axis_delta <= 0:
                continue
            row: dict[str, Any] = {
                "axis": axis,
                "match_mode": "strict_same_family" if require_same_family else "relaxed_cross_family",
                "high_split": sub.loc[hi, "split_name"],
                "low_split": sub.loc[lo, "split_name"],
                "high_family": sub.loc[hi, "split_family"],
                "low_family": sub.loc[lo, "split_family"],
                "axis_delta": axis_delta,
                "confound_distance": d,
                "high_axis_value": float(sub.loc[hi, axis]),
                "low_axis_value": float(sub.loc[lo, axis]),
            }
            for outcome in OUTCOMES:
                hv = sub.loc[hi, outcome]
                lv = sub.loc[lo, outcome]
                row[f"{outcome}_delta_high_minus_low"] = float(hv - lv) if pd.notna(hv) and pd.notna(lv) else np.nan
            candidates.append(row)
    candidates.sort(key=lambda r: (r["confound_distance"], -r["axis_delta"]))
    used_hi: set[str] = set()
    used_lo: set[str] = set()
    pairs: list[dict[str, Any]] = []
    for row in candidates:
        if row["high_split"] in used_hi or row["low_split"] in used_lo:
            continue
        used_hi.add(row["high_split"])
        used_lo.add(row["low_split"])
        pairs.append(row)
    return pairs


def bootstrap_ci(vals: np.ndarray, seed: int, n_boot: int = 2000) -> tuple[float, float, float]:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = [float(np.mean(rng.choice(vals, size=len(vals), replace=True))) for _ in range(n_boot)]
    arr = np.asarray(boot)
    return float(vals.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def sign_perm_p(vals: np.ndarray, seed: int, n_perm: int = 5000) -> float:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    observed = abs(float(vals.mean()))
    null = []
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(vals), replace=True)
        null.append(abs(float(np.mean(vals * signs))))
    null_arr = np.asarray(null)
    return float((np.sum(null_arr >= observed) + 1) / (n_perm + 1))


def summarize_axis(axis: str, pairs: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    row: dict[str, Any] = {"axis": axis, "match_mode": mode, "n_pairs": len(pairs)}
    for outcome in OUTCOMES:
        vals = np.asarray([p.get(f"{outcome}_delta_high_minus_low", np.nan) for p in pairs], dtype=float)
        mean, lo, hi = bootstrap_ci(vals, seed=20260628 + len(axis) + len(outcome))
        row[f"{outcome}_mean"] = mean
        row[f"{outcome}_ci_low"] = lo
        row[f"{outcome}_ci_high"] = hi
        row[f"{outcome}_p_signperm"] = sign_perm_p(vals, seed=202606280 + len(axis) + len(outcome))
    row["gate_pair_count"] = len(pairs) >= 8
    # For PP deltas higher is better; for MMD lower is better; tail_score higher/less negative is better.
    row["gate_signal"] = bool(
        row["gate_pair_count"]
        and row["cross_pp_delta_ci_low"] > 0
        and row["family_pp_delta_ci_low"] > 0
        and row["family_mmd_delta_ci_high"] <= 0
        and row["tail_score_ci_low"] >= 0
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--strict-distance", type=float, default=1.25)
    parser.add_argument("--relaxed-distance", type=float, default=2.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix = read_matrix()
    matrix.to_csv(args.out_dir / "scaling_v2_split_information_matrix.csv", index=False)

    all_pairs: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for axis in AXES:
        strict = make_pairs(matrix, axis, args.strict_distance, require_same_family=True)
        relaxed = make_pairs(matrix, axis, args.relaxed_distance, require_same_family=False)
        all_pairs.extend(strict)
        all_pairs.extend(relaxed)
        summary_rows.append(summarize_axis(axis, strict, "strict_same_family"))
        summary_rows.append(summarize_axis(axis, relaxed, "relaxed_cross_family"))

    pair_df = pd.DataFrame(all_pairs)
    if pair_df.empty:
        pair_df = pd.DataFrame(columns=["axis", "match_mode"])
    summary_df = pd.DataFrame(summary_rows)
    pair_df.to_csv(args.out_dir / "scaling_v2_matched_information_pairs.csv", index=False)
    summary_df.to_csv(args.out_dir / "scaling_v2_matched_information_summary.csv", index=False)

    gate_pass = bool(summary_df["gate_signal"].any()) if "gate_signal" in summary_df else False
    max_pairs = int(summary_df["n_pairs"].max()) if len(summary_df) else 0
    status = "scaling_v2_matched_information_no_gpu"
    if gate_pass:
        status = "scaling_v2_matched_information_review_packet_no_gpu"
    payload = {
        "timestamp": now_cst(),
        "status": status,
        "n_outcome_splits": int(len(matrix)),
        "n_axes": int(len(AXES)),
        "max_pairs": max_pairs,
        "gate_pass_axes": summary_df[summary_df["gate_signal"]]["axis"].tolist() if len(summary_df) else [],
        "boundary": "CPU/report-only; train-side/posthoc split matrix; no GPU authorization",
        "strict_distance": args.strict_distance,
        "relaxed_distance": args.relaxed_distance,
    }
    (args.out_dir / "latentfm_scaling_v2_matched_information_gate_20260628.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    top = summary_df.sort_values(["gate_signal", "n_pairs"], ascending=[False, False]).head(12)
    lines = [
        "# LatentFM Scaling V2 Matched Information Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`.",
        "",
        "## Boundary",
        "",
        "- CPU/report-only split-level matched information audit.",
        "- Uses existing exact-response and HVG response matrices with frozen downstream outcome rows.",
        "- Does not train, infer, select checkpoints, read Track C query, or use canonical multi for selection.",
        "",
        "## Gate",
        "",
        f"- outcome splits: `{len(matrix)}`",
        f"- axes tested: `{len(AXES)}`",
        f"- max matched pairs in any axis/mode: `{max_pairs}`",
        f"- gate pass axes: `{payload['gate_pass_axes']}`",
        "",
        "| axis | mode | pairs | cross pp mean CI | family pp mean CI | family MMD mean CI | tail mean CI | signal |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for _, row in top.iterrows():
        lines.append(
            "| {axis} | {mode} | {pairs} | {cmean:.4f} [{clo:.4f},{chi:.4f}] | {fmean:.4f} [{flo:.4f},{fhi:.4f}] | {mmean:.4f} [{mlo:.4f},{mhi:.4f}] | {tmean:.4f} [{tlo:.4f},{thi:.4f}] | `{sig}` |".format(
                axis=row["axis"],
                mode=row["match_mode"],
                pairs=int(row["n_pairs"]),
                cmean=row["cross_pp_delta_mean"],
                clo=row["cross_pp_delta_ci_low"],
                chi=row["cross_pp_delta_ci_high"],
                fmean=row["family_pp_delta_mean"],
                flo=row["family_pp_delta_ci_low"],
                fhi=row["family_pp_delta_ci_high"],
                mmean=row["family_mmd_delta_mean"],
                mlo=row["family_mmd_delta_ci_low"],
                mhi=row["family_mmd_delta_ci_high"],
                tmean=row["tail_score_mean"],
                tlo=row["tail_score_ci_low"],
                thi=row["tail_score_ci_high"],
                sig=bool(row["gate_signal"]),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if gate_pass:
        lines.extend(
            [
                "- At least one axis produced a review packet, but this still does not authorize GPU.",
                "- Next step is external audit plus a dedicated split/no-harm launcher packet.",
            ]
        )
    else:
        lines.extend(
            [
                "- No current information axis has enough matched split-level evidence for GPU.",
                "- The matrix is useful as a scaling-law failure map and as requirements for a new deliberately matched split family.",
            ]
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- split matrix: `{args.out_dir / 'scaling_v2_split_information_matrix.csv'}`",
            f"- matched pairs: `{args.out_dir / 'scaling_v2_matched_information_pairs.csv'}`",
            f"- summary: `{args.out_dir / 'scaling_v2_matched_information_summary.csv'}`",
            f"- JSON: `{args.out_dir / 'latentfm_scaling_v2_matched_information_gate_20260628.json'}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "max_pairs": max_pairs, "gate_pass_axes": payload["gate_pass_axes"]}, indent=2))


if __name__ == "__main__":
    main()
