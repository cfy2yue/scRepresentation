#!/usr/bin/env python3
"""Build a joint downstream scaling-law admission matrix.

CPU/report-only. This does not train, infer, select checkpoints, read
canonical multi for selection, read Track C query, or use GPU.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "downstream_scaling_law_matrix_20260630"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": str(path)}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fnum(value: Any, default: float | None = None) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(val):
        return default
    return val


def inum(value: Any, default: int | None = None) -> int | None:
    val = fnum(value, None)
    if val is None:
        return default
    return int(val)


def fmt(value: Any, digits: int = 4) -> str:
    val = fnum(value, None)
    if val is None:
        return "NA"
    return f"{val:.{digits}f}"


def parse_reasons(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    return [part for part in str(value).split(";") if part]


def ptype_count(raw: Any) -> int | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw)
    count = 0
    for token in ["CRISPRi", "CRISPRko", "CRISPRa", "drug"]:
        if token in text:
            count += 1
    return count or None


def make_row(
    *,
    axis: str,
    family: str,
    evidence_level: str,
    n_pairs: int | None = None,
    n_rows: int | None = None,
    n_datasets: int | None = None,
    n_perturbation_types: int | None = None,
    max_abs_smd: float | None = None,
    confound_metric: str = "",
    confound_value: float | None = None,
    top_dataset_fraction: float | None = None,
    negative_controls: str = "unknown_or_not_passed",
    observed_signal: str = "not_established",
    dual_baseline_ready: bool = False,
    mmd_noharm_ready: bool = False,
    experimentally_closed: bool = False,
    reasons: list[str] | None = None,
    next_action: str = "",
    why_nonduplicate: str = "",
) -> dict[str, Any]:
    reasons = list(reasons or [])
    if n_pairs is None and n_rows is not None:
        reasons.append("no_matched_highlow_pair_design")
    if n_pairs is not None and n_pairs < 300:
        reasons.append("pairs_below_300_matrix_gate")
    if n_datasets is None or n_datasets < 12:
        reasons.append("datasets_below_12_or_unknown")
    if n_perturbation_types is None or n_perturbation_types < 2:
        reasons.append("perturbation_type_coverage_below_2_or_unknown")
    if max_abs_smd is None or max_abs_smd > 0.15:
        reasons.append("max_abs_smd_gt_0p15_or_unknown")
    if confound_value is None:
        reasons.append("confound_metric_missing")
    elif confound_metric in {"mean_confound_distance"} and confound_value > 0.50:
        reasons.append("mean_confound_distance_gt_0p50")
    elif confound_metric in {"source_js_divergence"} and confound_value > 0.25:
        reasons.append("source_js_gt_0p25")
    elif confound_metric in {"propensity_auc_proxy"} and confound_value > 0.60:
        reasons.append("propensity_auc_or_overlap_proxy_gt_0p60")
    if top_dataset_fraction is not None and top_dataset_fraction > 0.20:
        reasons.append("top_dataset_fraction_gt_0p20")
    if negative_controls != "passed":
        reasons.append("negative_controls_not_passed")
    if not dual_baseline_ready:
        reasons.append("dual_baseline_not_ready")
    if not mmd_noharm_ready:
        reasons.append("mmd_noharm_not_ready")
    if observed_signal != "positive_safe":
        reasons.append("observed_signal_not_positive_safe")
    if experimentally_closed:
        reasons.append("experimentally_closed")

    # De-duplicate while preserving order.
    deduped: list[str] = []
    for reason in reasons:
        if reason and reason not in deduped:
            deduped.append(reason)

    gate_pass = not deduped
    return {
        "axis": axis,
        "family": family,
        "evidence_level": evidence_level,
        "n_pairs": n_pairs,
        "n_rows": n_rows,
        "n_datasets": n_datasets,
        "n_perturbation_types": n_perturbation_types,
        "max_abs_smd": max_abs_smd,
        "confound_metric": confound_metric,
        "confound_value": confound_value,
        "top_dataset_fraction": top_dataset_fraction,
        "negative_controls": negative_controls,
        "observed_signal": observed_signal,
        "dual_baseline_ready": dual_baseline_ready,
        "mmd_noharm_ready": mmd_noharm_ready,
        "experimentally_closed": experimentally_closed,
        "matrix_gate_pass": gate_pass,
        "reasons": ";".join(deduped) if deduped else "none",
        "next_action": next_action,
        "why_nonduplicate": why_nonduplicate,
    }


def main() -> int:
    exact_analog = read_csv(
        REPORTS / "exact_analog_observability_matched_feasibility_20260629" / "design_summary.csv"
    )
    exact_prop = read_csv(
        REPORTS
        / "exact_coverage_propensity_residual_match_gate_20260629"
        / "exact_coverage_propensity_match_summary.csv"
    )
    exact_cross = read_csv(
        REPORTS
        / "exact_coverage_crossdataset_matched_feasibility_20260629"
        / "crossdataset_matched_design_summary.csv"
    )
    nonstatic = read_json(
        REPORTS / "nonstatic_observable_information_gate_20260630" / "nonstatic_observable_information_gate_20260630.json"
    )
    v3_pool = read_json(
        REPORTS / "hvg_advantage_resid_v3_pair_pool_20260630" / "hvg_advantage_resid_v3_pair_pool_20260630.json"
    )
    v3_smoke = read_json(
        REPORTS
        / "hvg_advantage_resid_v3_highlow_smoke_20260630"
        / "latentfm_hvg_advantage_resid_v3_highlow_decision_20260630.json"
    )
    gene_budget = read_json(
        REPORTS
        / "observable_gene_budget_stability_readiness_gate_20260630"
        / "observable_gene_budget_stability_readiness_gate_20260630.json"
    )
    multiaxis = read_json(
        REPORTS
        / "multiaxis_information_scaling_incremental_gate_20260629"
        / "latentfm_multiaxis_information_scaling_incremental_gate_20260629.json"
    )
    cluster = read_json(
        REPORTS
        / "trainset_cluster_density_information_gate_20260629"
        / "latentfm_trainset_cluster_density_information_gate_20260629.json"
    )
    otcov = read_json(
        REPORTS
        / "trainset_ot_coverage_information_gate_20260629"
        / "latentfm_trainset_ot_coverage_information_gate_20260629.json"
    )
    zscape = read_json(
        REPORTS / "zscape_structural_dynamic_scaling_x_20260630" / "zscape_structural_dynamic_scaling_x_20260630.json"
    )
    source_scout = read_json(
        REPORTS / "matched_external_source_scout_20260630" / "matched_external_source_scout_20260630.json"
    )
    dual = read_json(
        REPORTS
        / "tracka_benchmark_control_consolidation_20260630"
        / "tracka_benchmark_control_consolidation_20260630.json"
    )

    rows: list[dict[str, Any]] = []

    if not exact_analog.empty:
        best = exact_analog.sort_values("matched_pairs", ascending=False).iloc[0]
        rows.append(
            make_row(
                axis="exact_or_analog_observability_best",
                family="exact_response_coverage",
                evidence_level="matched_design",
                n_pairs=inum(best.get("matched_pairs")),
                n_datasets=inum(best.get("datasets_with_pairs")),
                n_perturbation_types=ptype_count(best.get("perturbation_type_counts")),
                max_abs_smd=None,
                confound_metric="design_only",
                confound_value=None,
                top_dataset_fraction=fnum(best.get("max_dataset_share")),
                negative_controls="not_applicable_design_only",
                observed_signal="descriptor_only",
                next_action="Keep exact coverage as descriptor; do not relaunch without a source-balanced >=300-pair admission design.",
                why_nonduplicate="It tests exact train response coverage, but current matched design is underpowered for training.",
            )
        )

    if not exact_prop.empty:
        best = exact_prop.sort_values("n_pairs", ascending=False).iloc[0]
        rows.append(
            make_row(
                axis="exact_coverage_propensity_residual",
                family="exact_response_coverage",
                evidence_level="matched_design",
                n_pairs=inum(best.get("n_pairs")),
                n_datasets=inum(best.get("n_total_datasets")),
                n_perturbation_types=None,
                max_abs_smd=fnum(best.get("max_abs_covariate_smd")),
                confound_metric="propensity_auc_proxy",
                confound_value=0.91 if "coverage_propensity_auc_gt_0p90" in str(best.get("reasons")) else None,
                top_dataset_fraction=fnum(best.get("top_dataset_pair_fraction")),
                negative_controls="not_applicable_design_only",
                observed_signal="descriptor_only",
                reasons=parse_reasons(best.get("reasons")),
                next_action="Close as current training axis; poor overlap/propensity separation means it is not a clean scaling x.",
                why_nonduplicate="Residualized exact coverage explicitly tests whether coverage survives propensity/source controls.",
            )
        )

    if not exact_cross.empty:
        best = exact_cross.sort_values("n_pairs", ascending=False).iloc[0]
        rows.append(
            make_row(
                axis="exact_coverage_crossdataset",
                family="exact_response_coverage",
                evidence_level="matched_design",
                n_pairs=inum(best.get("n_pairs")),
                n_datasets=inum(best.get("n_total_datasets")),
                n_perturbation_types=ptype_count(best.get("perturbation_type_counts")),
                max_abs_smd=fnum(best.get("max_abs_covariate_smd")),
                confound_metric="source_js_divergence",
                confound_value=fnum(best.get("source_js_divergence")),
                top_dataset_fraction=fnum(best.get("top_dataset_pair_fraction")),
                negative_controls="not_applicable_design_only",
                observed_signal="descriptor_only",
                reasons=parse_reasons(best.get("reasons")),
                next_action="This is the closest coverage design by size, but source/covariate imbalance must be fixed before GPU.",
                why_nonduplicate="It asks whether cross-dataset scaling can be balanced, not whether more raw rows help.",
            )
        )

    nonstatic_decisions = nonstatic.get("axis_decisions", [])
    if nonstatic_decisions:
        best_ns = max(nonstatic_decisions, key=lambda r: (r.get("near_miss", False), r.get("n_pairs", 0)))
        rows.append(
            make_row(
                axis=f"{best_ns.get('axis')}_{best_ns.get('match_mode')}",
                family="nonstatic_observable_information",
                evidence_level="matched_design",
                n_pairs=inum(best_ns.get("n_pairs")),
                n_datasets=inum(best_ns.get("n_datasets")),
                n_perturbation_types=1 if best_ns.get("top_perturbation_type") else None,
                max_abs_smd=fnum(best_ns.get("max_abs_confound_smd")),
                confound_metric="mean_confound_distance",
                confound_value=fnum(best_ns.get("mean_confound_distance")),
                top_dataset_fraction=fnum(best_ns.get("max_dataset_fraction")),
                negative_controls="not_yet_passed",
                observed_signal="design_near_miss_only",
                reasons=parse_reasons(best_ns.get("reasons")),
                next_action="Do not promote the generic near-miss; v3 tested the strongest HVG-advantage derivative and failed.",
                why_nonduplicate="Nonstatic residual axes are condition-level, but current best still lacks outcome/no-harm support.",
            )
        )

    selected_v3 = None
    for mode in v3_pool.get("mode_summaries", []):
        if mode.get("preferred_candidate"):
            selected_v3 = mode
            break
    if selected_v3:
        checks = v3_smoke.get("decision", {}).get("checks", {})
        rows.append(
            make_row(
                axis="hvg_advantage_resid_v3_selected",
                family="nonstatic_observable_information",
                evidence_level="matched_design_plus_gpu_smoke",
                n_pairs=inum(selected_v3.get("n_pairs")),
                n_datasets=inum(selected_v3.get("n_datasets")),
                n_perturbation_types=2,
                max_abs_smd=fnum(selected_v3.get("max_abs_confound_smd")),
                confound_metric="mean_confound_distance",
                confound_value=fnum(selected_v3.get("mean_confound_distance")),
                top_dataset_fraction=fnum(selected_v3.get("max_dataset_fraction")),
                negative_controls="not_needed_after_real_signal_failed",
                observed_signal=(
                    "failed_negative"
                    if fnum(checks.get("high_minus_low_cross_pp_delta"), 0.0) < 0
                    and fnum(checks.get("high_minus_low_family_pp_delta"), 0.0) < 0
                    else "unknown"
                ),
                experimentally_closed=True,
                reasons=parse_reasons(v3_smoke.get("decision", {}).get("reasons", [])),
                next_action="Closed by real high/low smoke; do not launch placebo or longer v3 run.",
                why_nonduplicate="This was the strongest current nonstatic design and got direct experimental evidence.",
            )
        )

    gb_rows = gene_budget.get("rows", [])
    if gb_rows:
        best_gb = max(gb_rows, key=lambda r: r.get("rows", 0))
        rows.append(
            make_row(
                axis="observable_gene_budget_stability",
                family="observable_gene_budget",
                evidence_level="outcome_join",
                n_rows=inum(best_gb.get("rows")),
                n_datasets=inum(best_gb.get("datasets")),
                n_perturbation_types=None,
                max_abs_smd=None,
                confound_metric="feature_outcome_join",
                confound_value=None,
                negative_controls="mean_matched_split_half_only",
                observed_signal="failed_negative",
                mmd_noharm_ready=fnum(best_gb.get("mean_mmd_delta"), 1.0) <= 0.001,
                reasons=gene_budget.get("reasons", []),
                next_action="Keep as descriptor only; overlap is too small and mean pp is negative.",
                why_nonduplicate="Tests observable gene-budget stability rather than raw HVG count.",
            )
        )

    rows.append(
        make_row(
            axis="multiaxis_incremental_information",
            family="joint_scaling_descriptor",
            evidence_level="association_panel",
            n_rows=inum(multiaxis.get("n_outcome_rows")),
            n_datasets=None,
            negative_controls="source_background_controls_failed_to_find_axis",
            observed_signal="no_incremental_axis",
            reasons=multiaxis.get("reasons", []),
            next_action="Use as negative evidence that current multiaxis descriptors do not add training signal.",
            why_nonduplicate="Joint model over exact, HVG, state/context, and controls; not a single-axis rerun.",
        )
    )

    rows.append(
        make_row(
            axis="cluster_density_information",
            family="generic_diversity",
            evidence_level="association_panel",
            n_rows=inum(cluster.get("n_outcome_rows")),
            n_datasets=None,
            negative_controls="strict_incremental_gate_failed",
            observed_signal="no_passing_axis",
            reasons=cluster.get("reasons", []),
            next_action="Do not rerun generic cluster-density sampling without perturbation-conditioned residual vectors.",
            why_nonduplicate="Tests cell-state cluster density, but current generic form is closed.",
        )
    )

    rows.append(
        make_row(
            axis="ot_coverage_information",
            family="generic_diversity",
            evidence_level="association_panel",
            n_rows=inum(otcov.get("n_outcome_rows")),
            n_datasets=None,
            negative_controls="strict_incremental_gate_failed",
            observed_signal="no_passing_axis",
            reasons=otcov.get("reasons", []),
            next_action="Do not treat generic OT coverage as scaling x; only OT-pair-derived dynamic structure remains interesting.",
            why_nonduplicate="Tests OT distribution coverage directly, not minibatch pair implementation.",
        )
    )

    zd = zscape.get("decision", {})
    rows.append(
        make_row(
            axis="zscape_structural_dynamic_information",
            family="dynamic_biology",
            evidence_level="external_dynamic_descriptor",
            n_rows=inum(zd.get("zscape_rows")),
            n_datasets=1,
            n_perturbation_types=None,
            max_abs_smd=None,
            confound_metric="species_dynamic_specificity",
            confound_value=None,
            negative_controls="specificity_not_passed",
            observed_signal="biology_descriptor_only",
            reasons=zd.get("reasons", []),
            next_action="Continue CPU biology/specificity work; no LatentFM GPU route until human-train analogue exists.",
            why_nonduplicate="Dynamic time-course structure, not raw response magnitude or static train-set size.",
        )
    )

    local_queue = source_scout.get("local_admission_queue", [])
    rows.append(
        make_row(
            axis="matched_external_condition_source",
            family="external_source",
            evidence_level="source_inventory",
            n_rows=0,
            n_datasets=0,
            negative_controls="no_local_artifact",
            observed_signal="not_materialized",
            reasons=["local_admission_candidate_0"],
            next_action="Acquire or verify a genuinely external condition/background keyed table before any GPU gate.",
            why_nonduplicate="Would introduce exogenous condition evidence rather than another closed intrinsic scalar.",
        )
    )

    control_summary = read_csv(
        REPORTS
        / "tracka_benchmark_control_consolidation_20260630"
        / "tracka_benchmark_control_consolidation_control_summary.csv"
    )
    dual_summary = read_csv(
        REPORTS
        / "tracka_benchmark_control_consolidation_20260630"
        / "tracka_benchmark_control_consolidation_dual_candidates.csv"
    )
    ctrl_ci_positive = 0
    ctrl_positive = 0
    if not control_summary.empty:
        ctrl_positive = int((pd.to_numeric(control_summary["ctrl_minus_anchor_pp_dataset_equal"]) > 0).sum())
        ctrl_ci_positive = int((pd.to_numeric(control_summary["ctrl_minus_anchor_pp_dataset_ci_low"]) > 0).sum())
    existing_dual_pass = 0
    if not dual_summary.empty:
        existing_dual_pass = int((pd.to_numeric(dual_summary["pass_groups"], errors="coerce") > 0).sum())

    df = pd.DataFrame(rows)
    pass_count = int(df["matrix_gate_pass"].sum()) if not df.empty else 0
    gpu_authorized = pass_count > 0
    status = (
        "downstream_scaling_law_matrix_gpu_candidate_found"
        if gpu_authorized
        else "downstream_scaling_law_matrix_no_gpu_candidate"
    )

    row_path = OUT_DIR / "downstream_scaling_law_matrix_rows_20260630.csv"
    json_path = OUT_DIR / "downstream_scaling_law_matrix_20260630.json"
    md_path = OUT_DIR / "LATENTFM_DOWNSTREAM_SCALING_LAW_MATRIX_20260630.md"
    df.to_csv(row_path, index=False)

    # Rank actionable next work: source acquisition and source-balanced exact design are
    # the only non-closed routes with a plausible path back to GPU.
    next_candidates = [
        {
            "priority": 1,
            "candidate": "matched_external_condition_source",
            "mode": "source_materialization_cpu",
            "why": "Only route that can introduce genuinely new condition evidence beyond closed intrinsic axes.",
            "gate": ">=50 overlap rows, >=3 datasets/backgrounds, within-dataset variation, LODO/source-block pass, shuffle collapse, MMD no-harm, dual-baseline dominance.",
        },
        {
            "priority": 2,
            "candidate": "source_balanced_exact_or_nonstatic_multivariable_design",
            "mode": "cpu_design_gate",
            "why": "Exact coverage has enough cross-dataset pairs but fails source/covariate balance; nonstatic axes are biologically closer but underpowered or experimentally closed.",
            "gate": ">=300 matched pairs, >=12 datasets, >=2 perturbation types, max SMD <=0.15, source JS <=0.25, negative controls collapse, no-harm outcome signal.",
        },
        {
            "priority": 3,
            "candidate": "zscape_dynamic_specificity_expansion",
            "mode": "cpu_biology_only",
            "why": "Most biological insight; not a LatentFM GPU route until specificity and train-set translation pass.",
            "gate": ">=4 strict rows across >=2 lineages and >=3 specificity-passing rows under heldout/wrong-control tests.",
        },
    ]

    out = {
        "timestamp": now(),
        "status": status,
        "gpu_authorized": gpu_authorized,
        "matrix_gate_pass_count": pass_count,
        "boundary": {
            "cpu_report_only": True,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
            "uses_gpu": False,
        },
        "dual_baseline_context": {
            "source_control_positive_dataset_equal_rows": ctrl_positive,
            "source_control_ci_low_positive_rows": ctrl_ci_positive,
            "existing_dual_baseline_candidate_pass_rows": existing_dual_pass,
            "interpretation": "source/control is a mandatory comparator, not a promoted replacement model.",
        },
        "gate_thresholds": {
            "matched_pairs": ">=300",
            "datasets": ">=12",
            "perturbation_types": ">=2",
            "max_abs_smd": "<=0.15",
            "mean_confound_distance": "<=0.50",
            "source_js_divergence": "<=0.25",
            "top_dataset_fraction": "<=0.20",
            "negative_controls": "must pass/collapse",
            "dual_baseline": "candidate > max(anchor, source/control)",
            "mmd_noharm": "candidate MMD harm <= +0.001 and no unsafe tail",
        },
        "next_candidates": next_candidates,
        "outputs": {"rows": str(row_path), "json": str(json_path), "markdown": str(md_path)},
    }
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Downstream Scaling-Law Matrix 20260630",
        "",
        f"Created: `{out['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        f"GPU authorized: `{gpu_authorized}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only joint admission matrix.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C query, or GPU.",
        "- The matrix asks whether any current scaling x can become a downstream training design, not whether it is an interesting descriptor.",
        "",
        "## Dual-Baseline Context",
        "",
        f"- Source/control has positive dataset-equal pp in `{ctrl_positive}` rows, but CI-low positive rows `{ctrl_ci_positive}`.",
        f"- Existing candidate rows with any dual-baseline pass: `{existing_dual_pass}`.",
        "- Treat source/control as a mandatory comparator, not as a promoted model.",
        "",
        "## Gate Thresholds",
        "",
    ]
    for key, value in out["gate_thresholds"].items():
        lines.append(f"- `{key}`: {value}")

    lines += [
        "",
        "## Candidate Matrix",
        "",
        "| axis | family | evidence | size | balance | observed signal | pass | main blockers |",
        "|---|---|---|---|---|---|---:|---|",
    ]
    for _, row in df.iterrows():
        size = f"pairs={row.get('n_pairs') if pd.notna(row.get('n_pairs')) else 'NA'}, rows={row.get('n_rows') if pd.notna(row.get('n_rows')) else 'NA'}, datasets={row.get('n_datasets') if pd.notna(row.get('n_datasets')) else 'NA'}"
        balance = (
            f"SMD={fmt(row.get('max_abs_smd'))}, {row.get('confound_metric') or 'confound'}={fmt(row.get('confound_value'))}, "
            f"topds={fmt(row.get('top_dataset_fraction'))}"
        )
        blockers = str(row.get("reasons", ""))
        if len(blockers) > 140:
            blockers = blockers[:137] + "..."
        lines.append(
            f"| `{row['axis']}` | `{row['family']}` | `{row['evidence_level']}` | {size} | {balance} | `{row['observed_signal']}` | `{bool(row['matrix_gate_pass'])}` | {blockers} |"
        )

    lines += [
        "",
        "## Decision",
        "",
    ]
    if gpu_authorized:
        lines.append("At least one row passed the matrix gate and should be converted into a strict CPU admission/GPU-smoke launcher after resource audit.")
    else:
        lines.append(
            "No current scaling-law axis is GPU-eligible. The matrix closes current intrinsic scalar/descriptor axes as training routes, while keeping them as manuscript and failure-map evidence."
        )
    lines += [
        "",
        "## Next Candidates",
        "",
    ]
    for cand in next_candidates:
        lines.append(
            f"- `{cand['priority']}` `{cand['candidate']}` ({cand['mode']}): {cand['why']} Gate: {cand['gate']}"
        )

    lines += [
        "",
        "## Outputs",
        "",
        f"- rows: `{row_path}`",
        f"- JSON: `{json_path}`",
        f"- Markdown: `{md_path}`",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"status": status, "gpu_authorized": gpu_authorized, "rows": str(row_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
