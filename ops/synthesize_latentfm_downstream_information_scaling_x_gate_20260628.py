#!/usr/bin/env python3
"""Synthesize downstream perturbation information-scaling x variables.

CPU/report-only integration of current scaling, exact-coverage, HVG/abundance,
and ZSCAPE dynamic-geometry evidence. It defines which x variables are
supported, confounded, blocked, or ready for a future CPU/GPU packet.
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
EXACT_SPLIT = ROOT / "reports/exact_response_information_posthoc_parent_train_complete_20260628/exact_response_information_split_matrix.csv"
CLUSTER_ASSOC = (
    ROOT
    / "reports/exact_response_information_clustered_ci_parent_train_complete_20260628"
    / "exact_response_information_clustered_association_rows.csv"
)
IPW_PERM = (
    ROOT
    / "reports/exact_response_information_ipw_missingness_parent_train_complete_20260628"
    / "dataset_stratified_permutation_rows.csv"
)
CLEAN_CANDIDATES = (
    ROOT / "reports/clean_scaling_x_gate_parent_train_complete_20260628/clean_scaling_x_candidate_rows.csv"
)
RESIDUAL_SUMMARY = (
    ROOT / "reports/scaling_v2_residualized_condition_axes_gate_20260628/residualized_condition_axis_summary.csv"
)
HVG_ABUND_SUMMARY = ROOT / "reports/hvg_vs_abundance_baseline_20260628/hvg_vs_abundance_summary_rows.csv"
HVG_ABUND_ROWS = ROOT / "reports/hvg_vs_abundance_baseline_20260628/condition_hvg_vs_abundance_rows.csv"
ZSCAPE_SYNTH = ROOT / "reports/zscape_dynamic_information_modeling_gate_20260628/zscape_dynamic_information_row_synthesis.csv"
OUT_DIR = ROOT / "reports/downstream_information_scaling_x_gate_20260628"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


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


def strongest_cluster(cluster: pd.DataFrame, predictor: str, outcome: str) -> dict[str, Any]:
    if cluster.empty:
        return {}
    sub = cluster[(cluster.get("predictor") == predictor) & (cluster.get("outcome") == outcome)]
    if sub.empty:
        return {}
    row = sub.iloc[0].to_dict()
    return row


def perm_p(ipw: pd.DataFrame, outcome: str) -> float:
    if ipw.empty:
        return float("nan")
    sub = ipw[ipw.get("outcome") == outcome]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[0].get("dataset_stratified_perm_p", float("nan")))


def spearman_corr_rows(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="spearman")
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            sub = numeric[[left, right]].dropna()
            rho = corr.loc[left, right] if left in corr.index and right in corr.columns else np.nan
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "n": int(len(sub)),
                    "spearman_rho": float(rho) if pd.notna(rho) else float("nan"),
                    "strong_abs_corr": bool(pd.notna(rho) and abs(float(rho)) >= 0.70),
                }
            )
    return pd.DataFrame(rows)


def hvg_summary_stats(summary: pd.DataFrame, rows: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not summary.empty:
        all_rows = summary[(summary.get("level") == "all") & (summary.get("group") == "__ALL__")]
        for budget in [500, 1000]:
            sub = all_rows[pd.to_numeric(all_rows.get("budget"), errors="coerce") == budget]
            if not sub.empty:
                row = sub.iloc[0]
                out[f"budget{budget}_hvg_minus_abundance_mean"] = float(row.get("hvg_minus_abundance_mean", float("nan")))
                out[f"budget{budget}_overlap_fraction_mean"] = float(row.get("hvg_abundance_overlap_fraction_mean", float("nan")))
    if not rows.empty and {"control_hvg_share", "control_abundance_share"}.issubset(rows.columns):
        sub = rows[["control_hvg_share", "control_abundance_share"]].dropna()
        if len(sub) >= 3:
            out["condition_level_hvg_abundance_spearman"] = float(sub.corr(method="spearman").iloc[0, 1])
            out["condition_level_rows"] = int(len(sub))
    return out


def clean_metric(clean: pd.DataFrame, axis: str, col: str = "key_metric") -> str:
    if clean.empty:
        return "missing"
    sub = clean[clean.get("axis_family") == axis]
    if sub.empty:
        return "missing"
    return str(sub.iloc[0].get(col, "missing"))


def residual_best(residual: pd.DataFrame) -> str:
    if residual.empty:
        return "missing"
    strict = residual[residual.get("match_mode").astype(str) == "strict"].copy()
    if strict.empty or "n_pairs" not in strict.columns:
        return "missing"
    strict["n_pairs_num"] = pd.to_numeric(strict["n_pairs"], errors="coerce")
    row = strict.sort_values("n_pairs_num", ascending=False).iloc[0]
    return f"{row.get('axis')} strict_pairs={int(row.get('n_pairs_num', 0))}, datasets={int(row.get('n_datasets', 0))}"


def zscape_metric(zscape: pd.DataFrame) -> str:
    if zscape.empty:
        return "missing"
    geom = int((zscape.get("geometry_gate", pd.Series(dtype=bool)).astype(bool)).sum())
    spec = int((pd.to_numeric(zscape.get("specificity_pass", pd.Series(dtype=float)), errors="coerce") > 0).sum())
    return f"geometry_positive={geom}, specificity_positive_rows={spec}"


def build_readiness_rows(
    cluster: pd.DataFrame,
    ipw: pd.DataFrame,
    clean: pd.DataFrame,
    residual: pd.DataFrame,
    hvg_stats: dict[str, Any],
    zscape: pd.DataFrame,
) -> pd.DataFrame:
    exact_mmd = strongest_cluster(cluster, "exact_hvg_share_top1000_mean", "family_mmd_delta")
    exact_tail = strongest_cluster(cluster, "exact_condition_fraction", "tail_score")
    hvg_abund_rho = hvg_stats.get("condition_level_hvg_abundance_spearman", float("nan"))
    rows = [
        {
            "x_family": "exact_response_observability",
            "biological_meaning": "how much perturbation response is represented by train-safe exact/analog conditions",
            "current_support": (
                f"family_mmd rho={fmt(exact_mmd.get('rho'))}, "
                f"CI=[{fmt(exact_mmd.get('cluster_boot_ci95_low'))},{fmt(exact_mmd.get('cluster_boot_ci95_high'))}], "
                f"tail rho={fmt(exact_tail.get('rho'))}"
            ),
            "main_blocker": (
                f"matched feasibility/IPW confounding; {clean_metric(clean, 'exact_response_coverage')}; "
                f"dataset-stratified p family_mmd={fmt(perm_p(ipw, 'family_mmd_delta'))}"
            ),
            "evidence_status": "hypothesis_generating_confounded",
            "next_cpu_gate": "prospective matched split or stronger IPW/LODO with independent outcome families",
            "gpu_packet_ready": False,
            "model_translation_if_passes": "curriculum/sampling by exact-or-analog observability",
        },
        {
            "x_family": "hvg_minus_abundance_response_budget",
            "biological_meaning": "whether compact HVG/observable gene budget contains response information beyond abundance",
            "current_support": (
                f"condition hvg-abundance rho={fmt(hvg_abund_rho)}; "
                f"all budget500 hvg-minus-abundance={fmt(hvg_stats.get('budget500_hvg_minus_abundance_mean'))}"
            ),
            "main_blocker": "HVG and abundance are nearly collinear; xVERSE frozen embeddings cannot test gene budget directly",
            "evidence_status": "promising_for_rawfm_not_xverse",
            "next_cpu_gate": "RawFM split/gene-mask readiness plus abundance/mean-matched gene-budget controls",
            "gpu_packet_ready": False,
            "model_translation_if_passes": "RawFM top-k/full-gene budget or train-only gene mask curriculum",
        },
        {
            "x_family": "state_context_support",
            "biological_meaning": "nonredundant cell-state/background coverage rather than raw cell count",
            "current_support": "strong ab initio rationale; not yet materialized as unified train-safe matrix",
            "main_blocker": "state diversity axes must be separated from dataset/source/background identity",
            "evidence_status": "unmaterialized_high_priority_cpu",
            "next_cpu_gate": "condition-level state entropy/effective cluster count matrix with source-family LODO",
            "gpu_packet_ready": False,
            "model_translation_if_passes": "state-balanced sampling and curriculum from high-support to low-support conditions",
        },
        {
            "x_family": "ot_response_geometry",
            "biological_meaning": "state-preserving transport/tangent structure of perturbation response",
            "current_support": zscape_metric(zscape),
            "main_blocker": "current support is external ZSCAPE expression geometry; specificity and latent route fail",
            "evidence_status": "diagnostic_positive_not_constraint",
            "next_cpu_gate": "transfer OT geometry metrics to train-safe perturbation datasets and keep pair-shuffle/wrong controls",
            "gpu_packet_ready": False,
            "model_translation_if_passes": "geometry-aware evaluation/sampling; no generic OT pair-mode relaunch without pair-quality pass",
        },
        {
            "x_family": "residualized_condition_axes",
            "biological_meaning": "condition information independent of response/support/gene-budget covariates",
            "current_support": residual_best(residual),
            "main_blocker": "strict matched pair counts underpowered after residualization",
            "evidence_status": "demoted_as_direct_launch_axis",
            "next_cpu_gate": "use as covariates in unified information design matrix, not another raw info_composite GPU run",
            "gpu_packet_ready": False,
            "model_translation_if_passes": "only after matched feasibility: information-balanced high/low smoke",
        },
        {
            "x_family": "replicate_reliability",
            "biological_meaning": "whether response is reproducible at embryo/donor/replicate level rather than cell bootstrap",
            "current_support": "ZSCAPE module heldout effects positive but specificity fails; vector heldout not yet complete",
            "main_blocker": "replicate-level vector consistency is not yet computed for centroid/OT deltas",
            "evidence_status": "immediate_cpu_gate_needed",
            "next_cpu_gate": "embryo-level vector consistency and leave-one-embryo-out for geometry rows",
            "gpu_packet_ready": False,
            "model_translation_if_passes": "weight by replicate confidence and report replicate-aware uncertainty",
        },
    ]
    return pd.DataFrame(rows)


def write_report(out_dir: Path, readiness: pd.DataFrame, corr_rows: pd.DataFrame, hvg_stats: dict[str, Any]) -> None:
    gpu_ready = bool(readiness["gpu_packet_ready"].any()) if not readiness.empty else False
    high_corr = corr_rows[corr_rows["strong_abs_corr"] == True] if not corr_rows.empty else pd.DataFrame()
    lines: list[str] = []
    lines.append("# Downstream Perturbation Information-Scaling X Gate")
    lines.append("")
    lines.append(f"Timestamp: `{now_cst()}`")
    lines.append("")
    lines.append("Status: `downstream_information_scaling_x_gate_no_gpu_packet`")
    lines.append("")
    lines.append(f"GPU authorized: `{gpu_ready}`")
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("- CPU/report-only synthesis over frozen scaling and ZSCAPE artifacts.")
    lines.append("- Defines downstream perturbation information x variables; raw dataset size is a covariate, not the primary x.")
    lines.append("- No training, inference, canonical multi selection, Track C query, or checkpoint selection.")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("- No current x variable authorizes GPU.")
    lines.append("- The strongest scientific direction is a multiaxis information budget: response observability, gene budget beyond abundance, state/context support, OT geometry, and replicate reliability.")
    lines.append("- The next work is CPU materialization/deconfounding, especially state-support matrices, RawFM observable-budget readiness, and replicate-aware ZSCAPE geometry.")
    lines.append("")
    lines.append("## X Readiness")
    lines.append("")
    cols = [
        "x_family",
        "current_support",
        "main_blocker",
        "evidence_status",
        "next_cpu_gate",
        "model_translation_if_passes",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in readiness.iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in cols) + " |")
    lines.append("")
    lines.append("## Key Confounding Signals")
    lines.append("")
    if hvg_stats:
        lines.append(f"- Condition-level HVG/abundance Spearman: `{fmt(hvg_stats.get('condition_level_hvg_abundance_spearman'))}`.")
        lines.append(f"- Budget 500 HVG-minus-abundance mean: `{fmt(hvg_stats.get('budget500_hvg_minus_abundance_mean'))}`.")
    lines.append(f"- Strong absolute split-axis correlations observed: `{len(high_corr)}`.")
    if not high_corr.empty:
        lines.append("")
        lines.append("| left | right | n | rho |")
        lines.append("|---|---|---:|---:|")
        for _, row in high_corr.head(12).iterrows():
            lines.append(
                f"| {row['left']} | {row['right']} | {int(row['n'])} | {fmt(row['spearman_rho'])} |"
            )
    lines.append("")
    lines.append("## GPU Reopen Criteria")
    lines.append("")
    lines.append("- At least one x survives residualization, matching, clustered CI, LODO, and dataset-stratified permutation.")
    lines.append("- Matched high/low feasibility reaches at least 300 pairs across at least 15 datasets, or a separately justified RawFM budget route passes split/gene-mask readiness.")
    lines.append("- Dual baseline/no-harm is specified before launch: anchor plus source/control or no-change baseline, pp improvement, and MMD/tail no-harm.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- readiness rows: `{out_dir / 'downstream_information_scaling_x_readiness_rows.csv'}`")
    lines.append(f"- split-axis correlations: `{out_dir / 'downstream_information_scaling_x_correlations.csv'}`")
    lines.append(f"- JSON: `{out_dir / 'downstream_information_scaling_x_gate_20260628.json'}`")
    (out_dir / "LATENTFM_DOWNSTREAM_INFORMATION_SCALING_X_GATE_20260628.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    split = read_csv(EXACT_SPLIT)
    cluster = read_csv(CLUSTER_ASSOC)
    ipw = read_csv(IPW_PERM)
    clean = read_csv(CLEAN_CANDIDATES)
    residual = read_csv(RESIDUAL_SUMMARY)
    hvg_summary = read_csv(HVG_ABUND_SUMMARY)
    hvg_rows = read_csv(HVG_ABUND_ROWS)
    zscape = read_csv(ZSCAPE_SYNTH)

    hvg_stats = hvg_summary_stats(hvg_summary, hvg_rows)
    readiness = build_readiness_rows(cluster, ipw, clean, residual, hvg_stats, zscape)

    corr_cols = [
        c
        for c in [
            "exact_condition_fraction",
            "exact_hvg_share_top1000_mean",
            "exact_abundance_share_top1000_mean",
            "exact_hvg_minus_abundance_top1000_mean",
            "base_dataset_effective_count",
            "base_background_effective_count",
            "base_perturbation_type_effective_count",
            "base_target_gene_effective_count",
            "cross_pp_delta",
            "family_pp_delta",
            "family_mmd_delta",
            "tail_score",
        ]
        if c in split.columns
    ]
    corr_rows = spearman_corr_rows(split, corr_cols) if corr_cols else pd.DataFrame()

    readiness_path = args.out_dir / "downstream_information_scaling_x_readiness_rows.csv"
    corr_path = args.out_dir / "downstream_information_scaling_x_correlations.csv"
    readiness.to_csv(readiness_path, index=False)
    corr_rows.to_csv(corr_path, index=False)

    obj = {
        "timestamp": now_cst(),
        "status": "downstream_information_scaling_x_gate_no_gpu_packet",
        "gpu_authorized_next": False,
        "n_x_families": int(len(readiness)),
        "hvg_stats": hvg_stats,
        "strong_split_axis_correlations": int((corr_rows.get("strong_abs_corr", pd.Series(dtype=bool)) == True).sum())
        if not corr_rows.empty
        else 0,
        "outputs": {
            "readiness_rows": str(readiness_path),
            "correlations": str(corr_path),
            "report": str(args.out_dir / "LATENTFM_DOWNSTREAM_INFORMATION_SCALING_X_GATE_20260628.md"),
        },
    }
    write_json(args.out_dir / "downstream_information_scaling_x_gate_20260628.json", obj)
    write_report(args.out_dir, readiness, corr_rows, hvg_stats)


if __name__ == "__main__":
    main()
