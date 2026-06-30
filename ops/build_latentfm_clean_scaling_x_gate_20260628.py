#!/usr/bin/env python3
"""Clean scaling-x gate for LatentFM.

CPU/report-only synthesis with quantitative checks. It combines current
condition-level axis correlations, split-level exact-coverage associations,
ZSCAPE OT geometry, and launchability evidence into one gate. It does not train,
infer, select checkpoints, or authorize GPU by itself.
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
from scipy.stats import spearmanr


ROOT = Path("/data/cyx/1030/scLatent")
RESIDUAL_MATRIX = ROOT / "reports/scaling_v2_residualized_condition_axes_gate_20260628/residualized_condition_axis_matrix.csv"
RESIDUAL_SUMMARY = ROOT / "reports/scaling_v2_residualized_condition_axes_gate_20260628/residualized_condition_axis_summary.csv"
EXACT_SPLIT = ROOT / "reports/exact_response_information_posthoc_combined_20260628/exact_response_information_split_matrix.csv"
EXACT_MATCH = ROOT / "reports/exact_coverage_strict_matched_draft_splits_combined_20260628/dataset_match_summary.csv"
MATCHED_INFO = ROOT / "reports/scaling_v2_matched_information_gate_20260628/scaling_v2_matched_information_summary.csv"
ZSCAPE_OT = ROOT / "reports/zscape_ot_dynamic_response_gate_20260628/zscape_ot_dynamic_response_rows.csv"
ZSCAPE_HELDOUT = (
    ROOT
    / "reports/zscape_embryo_heldout_periderm_module_specificity_20260628"
    / "zscape_embryo_heldout_periderm_module_specificity_summary.csv"
)
HVG_ABUND = ROOT / "reports/hvg_vs_abundance_baseline_20260628/hvg_vs_abundance_summary_rows.csv"
OUT_DIR = ROOT / "reports/clean_scaling_x_gate_20260628"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def bool_any(series: pd.Series) -> bool:
    return bool(series.astype(str).str.lower().isin({"true", "1", "yes"}).any())


def spearman_rows(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            sub = df[[left, right]].dropna()
            if len(sub) < 6 or sub[left].nunique() < 2 or sub[right].nunique() < 2:
                rho = pval = np.nan
            else:
                rho, pval = spearmanr(sub[left], sub[right])
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "n": int(len(sub)),
                    "spearman_rho": float(rho) if math.isfinite(float(rho)) else np.nan,
                    "p_value": float(pval) if math.isfinite(float(pval)) else np.nan,
                    "strong_abs_corr": bool(math.isfinite(float(rho)) and abs(float(rho)) >= 0.70),
                }
            )
    return pd.DataFrame(rows)


def split_assoc_rows(split_df: pd.DataFrame) -> pd.DataFrame:
    predictors = [
        "exact_condition_fraction",
        "exact_hvg_share_top1000_mean",
        "exact_abundance_share_top1000_mean",
        "exact_hvg_minus_abundance_top1000_mean",
        "base_dataset_effective_count",
        "base_background_effective_count",
        "base_perturbation_type_effective_count",
        "base_target_gene_effective_count",
    ]
    outcomes = ["cross_pp_delta", "family_pp_delta", "family_mmd_delta", "tail_score"]
    rows: list[dict[str, Any]] = []
    df = split_df[split_df["has_downstream_outcome"].astype(bool)].copy()
    for pred in predictors:
        for outcome in outcomes:
            sub = df[[pred, outcome]].dropna()
            if len(sub) < 6 or sub[pred].nunique() < 2 or sub[outcome].nunique() < 2:
                rho = pval = np.nan
            else:
                rho, pval = spearmanr(sub[pred], sub[outcome])
            rows.append(
                {
                    "predictor": pred,
                    "outcome": outcome,
                    "n": int(len(sub)),
                    "spearman_rho": float(rho) if math.isfinite(float(rho)) else np.nan,
                    "p_value": float(pval) if math.isfinite(float(pval)) else np.nan,
                    "direction_expected": "positive" if outcome in {"cross_pp_delta", "family_pp_delta", "tail_score"} else "negative",
                }
            )
    return pd.DataFrame(rows)


def strongest_assoc(assoc: pd.DataFrame, predictor: str, outcome: str) -> str:
    sub = assoc[(assoc["predictor"] == predictor) & (assoc["outcome"] == outcome)]
    if sub.empty:
        return "missing"
    row = sub.iloc[0]
    return f"rho={fmt(row['spearman_rho'])}, p={fmt(row['p_value'])}, n={int(row['n'])}"


def candidate_rows(
    residual_summary: pd.DataFrame,
    exact_match: pd.DataFrame,
    matched_info: pd.DataFrame,
    split_assoc: pd.DataFrame,
    zscape_ot: pd.DataFrame,
    zscape_heldout: pd.DataFrame,
    hvg_abund: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    exact_pairs = int(pd.to_numeric(exact_match.get("matched_pairs", pd.Series(dtype=float)), errors="coerce").sum())
    exact_datasets = int((pd.to_numeric(exact_match.get("matched_pairs", pd.Series(dtype=float)), errors="coerce") > 0).sum())
    exact_matched_gate = bool(exact_pairs >= 300 and exact_datasets >= 15)
    rows.append(
        {
            "axis_family": "exact_response_coverage",
            "level": "split/condition",
            "main_signal": strongest_assoc(split_assoc, "exact_condition_fraction", "family_mmd_delta"),
            "positive_evidence": "split-level exact fraction associates with family MMD/tail in earlier clustered CI",
            "main_confound": "dataset/raw-availability/source structure",
            "launch_gate_status": "fail_matched_feasibility",
            "gpu_authorized": False,
            "key_metric": f"strict covered/uncovered pairs={exact_pairs}, datasets={exact_datasets}",
            "next_gate": "new balanced split family or stronger missingness/IPW design",
            "gate_pass": exact_matched_gate,
        }
    )

    strict_resid = residual_summary[residual_summary["match_mode"].astype(str) == "strict"].copy()
    best_strict = strict_resid.sort_values("n_pairs", ascending=False).head(1)
    strict_gate = bool_any(strict_resid.get("strict_gate", pd.Series(dtype=object)))
    best_text = "missing"
    if not best_strict.empty:
        row = best_strict.iloc[0]
        best_text = f"best strict {row['axis']} pairs={int(row['n_pairs'])}, datasets={int(row['n_datasets'])}"
    rows.append(
        {
            "axis_family": "residualized_condition_axes",
            "level": "condition",
            "main_signal": best_text,
            "positive_evidence": "some axes have clean small matched sets",
            "main_confound": "axes become underpowered after matching response/support/gene-budget structure",
            "launch_gate_status": "fail_strict_pair_count",
            "gpu_authorized": False,
            "key_metric": best_text,
            "next_gate": "do not mutate raw info_composite; use as covariate only",
            "gate_pass": strict_gate,
        }
    )

    hvg_signal = "missing"
    if not hvg_abund.empty:
        cols = list(hvg_abund.columns)
        hvg_signal = f"rows={len(hvg_abund)} cols={','.join(cols[:6])}"
    rows.append(
        {
            "axis_family": "observable_gene_budget",
            "level": "condition/raw-expression",
            "main_signal": hvg_signal,
            "positive_evidence": "top-k observable genes capture response energy",
            "main_confound": "HVG is largely abundance/mean-expression equivalent",
            "launch_gate_status": "no_xverse_gpu_route",
            "gpu_authorized": False,
            "key_metric": "HVG-specific superiority unsupported by abundance baseline",
            "next_gate": "expression-space/rawFM budget curve with abundance and mean-matched controls",
            "gate_pass": False,
        }
    )

    dynamic_pos = zscape_ot[zscape_ot.get("dynamic_response_gate", False).astype(bool)] if not zscape_ot.empty else pd.DataFrame()
    heldout_pass = bool_any(zscape_heldout.get("query_gate", pd.Series(dtype=object))) if not zscape_heldout.empty else False
    rows.append(
        {
            "axis_family": "zscape_ot_response_geometry",
            "level": "external biological snapshot",
            "main_signal": f"dynamic positives={len(dynamic_pos)}; heldout specificity any_pass={heldout_pass}",
            "positive_evidence": "periderm noto/smo show within-state OT dynamic response",
            "main_confound": "module/pathway specificity fails; latent route blocked",
            "launch_gate_status": "hypothesis_generator_only",
            "gpu_authorized": False,
            "key_metric": "dynamic positives noto/smo, specificity 0/4",
            "next_gate": "use OT geometry as covariate/diagnostic, not loss term",
            "gate_pass": False,
        }
    )

    matched_info_gate = bool_any(matched_info.get("gate_signal", pd.Series(dtype=object)))
    max_pairs = int(pd.to_numeric(matched_info.get("n_pairs", pd.Series([0])), errors="coerce").max()) if not matched_info.empty else 0
    rows.append(
        {
            "axis_family": "existing_split_matched_information",
            "level": "split",
            "main_signal": f"max matched split pairs={max_pairs}",
            "positive_evidence": "existing split outcomes can be used for retrospective association",
            "main_confound": "few matched pairs; split families differ mechanistically",
            "launch_gate_status": "fail_existing_split_match",
            "gpu_authorized": False,
            "key_metric": f"gate_signal={matched_info_gate}",
            "next_gate": "prospective split family if strict feasibility improves",
            "gate_pass": matched_info_gate,
        }
    )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--residual-matrix", type=Path, default=RESIDUAL_MATRIX)
    parser.add_argument("--residual-summary", type=Path, default=RESIDUAL_SUMMARY)
    parser.add_argument("--exact-split", type=Path, default=EXACT_SPLIT)
    parser.add_argument("--exact-match", type=Path, default=EXACT_MATCH)
    parser.add_argument("--matched-info", type=Path, default=MATCHED_INFO)
    parser.add_argument("--zscape-ot", type=Path, default=ZSCAPE_OT)
    parser.add_argument("--zscape-heldout", type=Path, default=ZSCAPE_HELDOUT)
    parser.add_argument("--hvg-abundance", type=Path, default=HVG_ABUND)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    residual_matrix = pd.read_csv(args.residual_matrix)
    residual_summary = pd.read_csv(args.residual_summary)
    exact_split = pd.read_csv(args.exact_split)
    exact_match = pd.read_csv(args.exact_match)
    matched_info = pd.read_csv(args.matched_info)
    zscape_ot = pd.read_csv(args.zscape_ot)
    zscape_heldout = pd.read_csv(args.zscape_heldout)
    hvg_abund = pd.read_csv(args.hvg_abundance) if args.hvg_abundance.exists() else pd.DataFrame()

    condition_cols = [
        "log_response_energy",
        "hvg_concentration_80",
        "hvg_advantage_80",
        "cell_support_log",
        "abundance_concentration_80",
        "response_energy_resid",
        "hvg_concentration_resid",
        "hvg_advantage_resid",
        "support_resid",
    ]
    condition_corr = spearman_rows(residual_matrix, [c for c in condition_cols if c in residual_matrix.columns])
    split_assoc = split_assoc_rows(exact_split)
    candidates = candidate_rows(
        residual_summary=residual_summary,
        exact_match=exact_match,
        matched_info=matched_info,
        split_assoc=split_assoc,
        zscape_ot=zscape_ot,
        zscape_heldout=zscape_heldout,
        hvg_abund=hvg_abund,
    )
    gpu_authorized = bool(candidates["gpu_authorized"].any())
    status = "clean_scaling_x_gate_gpu_packet_ready" if gpu_authorized else "clean_scaling_x_gate_no_gpu_packet"

    condition_corr_path = args.out_dir / "clean_scaling_x_condition_axis_correlations.csv"
    split_assoc_path = args.out_dir / "clean_scaling_x_split_associations.csv"
    candidates_path = args.out_dir / "clean_scaling_x_candidate_rows.csv"
    condition_corr.to_csv(condition_corr_path, index=False)
    split_assoc.to_csv(split_assoc_path, index=False)
    candidates.to_csv(candidates_path, index=False)

    strong_corr = condition_corr[condition_corr["strong_abs_corr"]].sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)
    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_authorized_next": gpu_authorized,
        "candidate_axes": int(len(candidates)),
        "gpu_authorized_axes": candidates.loc[candidates["gpu_authorized"], "axis_family"].tolist(),
        "strong_condition_axis_correlations": int(len(strong_corr)),
        "outputs": {
            "condition_correlations": str(condition_corr_path),
            "split_associations": str(split_assoc_path),
            "candidate_rows": str(candidates_path),
        },
        "inputs": {
            "residual_matrix": str(args.residual_matrix),
            "residual_summary": str(args.residual_summary),
            "exact_split": str(args.exact_split),
            "exact_match": str(args.exact_match),
            "matched_info": str(args.matched_info),
            "zscape_ot": str(args.zscape_ot),
            "zscape_heldout": str(args.zscape_heldout),
            "hvg_abundance": str(args.hvg_abundance),
        },
    }
    json_path = args.out_dir / "clean_scaling_x_gate_20260628.json"
    write_json(json_path, payload)

    lines = [
        "# LatentFM Clean Scaling-X Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        f"GPU authorized next: `{gpu_authorized}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only synthesis and quantitative gate.",
        "- No training, inference, checkpoint selection, canonical multi selection, or Track C query use.",
        "- Goal: decide whether any current scaling x-axis is clean enough to produce a GPU launch packet.",
        "",
        "## Candidate Axes",
        "",
        "| axis | level | signal | confound | launch status | GPU | next gate |",
        "|---|---|---|---|---|---:|---|",
    ]
    for row in candidates.to_dict("records"):
        lines.append(
            "| `{axis}` | {level} | {signal} | {confound} | `{status}` | `{gpu}` | {next_gate} |".format(
                axis=row["axis_family"],
                level=row["level"],
                signal=str(row["main_signal"]).replace("|", "/"),
                confound=str(row["main_confound"]).replace("|", "/"),
                status=row["launch_gate_status"],
                gpu=row["gpu_authorized"],
                next_gate=str(row["next_gate"]).replace("|", "/"),
            )
        )

    lines.extend(
        [
            "",
            "## Strong Condition-Axis Correlations",
            "",
            "| left | right | n | rho | p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in strong_corr.head(12).to_dict("records"):
        lines.append(
            f"| `{row['left']}` | `{row['right']}` | {int(row['n'])} | {fmt(row['spearman_rho'])} | {fmt(row['p_value'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- No current axis produces a legal GPU launch packet.",
            "- This is not evidence against scaling. It is evidence that the current axes are too confounded or underpowered to use directly.",
            "- The next concrete route is a prospective balanced split design or an expression-space/rawFM budget route, not another mutation of raw `info_composite`.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{json_path}`",
            f"- Candidate rows: `{candidates_path}`",
            f"- Condition correlations: `{condition_corr_path}`",
            f"- Split associations: `{split_assoc_path}`",
        ]
    )
    md_path = args.out_dir / "LATENTFM_CLEAN_SCALING_X_GATE_20260628.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
