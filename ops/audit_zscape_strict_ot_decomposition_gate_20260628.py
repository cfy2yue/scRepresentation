#!/usr/bin/env python3
"""Synthesize strict ZSCAPE OT decomposition evidence into modeling gates.

This is intentionally CPU-only and report-only. It does not recompute large
embeddings; it merges the frozen OT, embryo, module, and HVG-atlas artifacts
that were already produced under their own provenance.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
OT_ROWS = ROOT / "reports/zscape_ot_dynamic_response_gate_20260628/zscape_ot_dynamic_response_rows.csv"
OT_REPORT = ROOT / "reports/zscape_ot_dynamic_response_gate_20260628/LATENTFM_ZSCAPE_OT_DYNAMIC_RESPONSE_GATE_20260628.md"
EMBRYO_ROWS = ROOT / "reports/zscape_embryo_vector_consistency_gate_20260628/zscape_embryo_vector_consistency_rows.csv"
MODULE_QUERY_ROWS = (
    ROOT
    / "reports/zscape_periderm_substate_time_qc_ot_module_gate_20260628"
    / "zscape_periderm_substate_time_qc_module_query_rows.csv"
)
MODULE_PLACEBO_ROWS = (
    ROOT
    / "reports/zscape_periderm_substate_time_qc_ot_module_gate_20260628"
    / "zscape_periderm_substate_time_qc_module_placebo_rows.csv"
)
ATLAS_ROWS = ROOT / "reports/zscape_strict_biological_row_atlas_20260628/zscape_strict_biological_row_atlas.csv"
OUT_DIR = ROOT / "reports/zscape_strict_ot_decomposition_gate_20260628"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def safe_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def fmt(value: Any, digits: int = 4) -> str:
    val = safe_float(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def pass_fraction(values: pd.Series) -> tuple[int, int]:
    if values.empty:
        return 0, 0
    vals = values.map(safe_bool)
    return int(vals.sum()), int(vals.shape[0])


def classify_row(row: pd.Series) -> str:
    dynamic = safe_bool(row.get("dynamic_response_gate"))
    embryo = safe_bool(row.get("embryo_vector_gate"))
    module_all = safe_bool(row.get("module_all_query_gate"))
    specificity_all = safe_bool(row.get("module_all_specificity_gate"))
    state_preserved = safe_bool(row.get("state_preserved_by_threshold"))
    comp_frac = safe_float(row.get("composition_norm_fraction_of_centroid"))
    wrong_time_margin = safe_float(row.get("wrong_time_margin_ot"))
    tangent = safe_bool(row.get("trajectory_alignment_gate"))

    if dynamic and embryo and module_all and specificity_all:
        return "constraint_candidate_ready"
    if dynamic and embryo and state_preserved and tangent:
        return "geometry_replicate_insight_specificity_blocked"
    if comp_frac > 0.25 or wrong_time_margin < 0:
        return "composition_or_time_confounded_comparator"
    if embryo:
        return "replicate_stable_diagnostic_only"
    return "not_supported"


def make_score(row: pd.Series) -> float:
    """A diagnostic score, not a formal scaling law."""
    wt = safe_float(row.get("wrong_time_margin_ot"), 0.0)
    wl = safe_float(row.get("wrong_lineage_margin_ot"), 0.0)
    tangent = max(safe_float(row.get("trajectory_cosine"), 0.0), 0.0)
    within = min(max(safe_float(row.get("within_substate_residual_fraction_of_centroid"), 0.0), 0.0), 2.0)
    comp = min(max(safe_float(row.get("composition_norm_fraction_of_centroid"), 0.0), 0.0), 2.0)
    embryo_ci = max(safe_float(row.get("mean_cosine_ci_low"), 0.0), 0.0)
    module_penalty = 0.0 if safe_bool(row.get("module_all_specificity_gate")) else 0.25
    return float((np.tanh(wt / 2.0) + np.tanh(wl / 10.0) + tangent + embryo_ci + within - comp) - module_penalty)


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ot = read_csv(OT_ROWS)
    embryo = read_csv(EMBRYO_ROWS)
    module = read_csv(MODULE_QUERY_ROWS)
    placebo = read_csv(MODULE_PLACEBO_ROWS)
    atlas = read_csv(ATLAS_ROWS)

    module_summary_rows: list[dict[str, Any]] = []
    for row_id, group in module.groupby("row_id"):
        q_pass, q_total = pass_fraction(group["query_gate"])
        spec_pass, spec_total = pass_fraction(group["specificity_gate"])
        qc_pass, qc_total = pass_fraction(group["qc_residual_gate"])
        sub_pass, sub_total = pass_fraction(group["substate_gate"])
        module_summary_rows.append(
            {
                "row_id": row_id,
                "module_queries": int(q_total),
                "module_query_gates": int(q_pass),
                "module_specificity_gates": int(spec_pass),
                "module_qc_gates": int(qc_pass),
                "module_substate_gates": int(sub_pass),
                "module_all_query_gate": q_total > 0 and q_pass == q_total,
                "module_all_specificity_gate": spec_total > 0 and spec_pass == spec_total,
                "module_all_qc_gate": qc_total > 0 and qc_pass == qc_total,
                "module_all_substate_gate": sub_total > 0 and sub_pass == sub_total,
                "module_min_residual_ci_low": float(group["residual_ci_low"].min()),
                "module_max_wrong_time": float(group["wrong_time_max"].max()),
                "module_max_wrong_lineage_p95": float(group["wrong_lineage_p95"].max()),
                "module_top_terms": " | ".join(str(x) for x in group["top_terms"].dropna().head(2)),
            }
        )
    module_summary = pd.DataFrame(module_summary_rows)

    placebo_summary_rows: list[dict[str, Any]] = []
    if not placebo.empty:
        for row_id, group in placebo.groupby("query_row_id"):
            placebo_summary_rows.append(
                {
                    "row_id": row_id,
                    "placebo_max_directed_diff": float(group["directed_diff"].max()),
                    "placebo_p95_directed_diff": float(group["directed_diff"].quantile(0.95)),
                    "placebo_positive_controls": int((group["directed_diff"] > 0).sum()),
                    "placebo_controls": int(group.shape[0]),
                }
            )
    placebo_summary = pd.DataFrame(placebo_summary_rows)

    cols = [
        "row_id",
        "lineage",
        "target",
        "timepoint",
        "audit_role",
        "n_pseudo_pairs",
        "centroid_response_norm",
        "composition_norm_fraction_of_centroid",
        "within_substate_residual_fraction_of_centroid",
        "substate_jsd",
        "state_preserved_by_threshold",
        "trajectory_alignment_gate",
        "trajectory_cosine",
        "wrong_time_margin_ot",
        "wrong_lineage_margin_ot",
        "dynamic_response_gate",
    ]
    merged = ot[[c for c in cols if c in ot.columns]].merge(
        embryo[
            [
                "row_id",
                "n_perturb_embryos",
                "mean_embryo_cosine",
                "mean_cosine_ci_low",
                "positive_embryo_fraction",
                "embryo_vector_gate",
            ]
        ],
        on="row_id",
        how="left",
    )
    merged = merged.merge(module_summary, on="row_id", how="left")
    merged = merged.merge(placebo_summary, on="row_id", how="left")
    atlas_keep = [
        "row_id",
        "hvg1000_response_energy_share",
        "hvg2000_response_energy_share",
        "hvg4000_response_energy_share",
        "claim_guardrail",
    ]
    merged = merged.merge(atlas[[c for c in atlas_keep if c in atlas.columns]], on="row_id", how="left")

    bool_fill = [
        "module_all_query_gate",
        "module_all_specificity_gate",
        "module_all_qc_gate",
        "module_all_substate_gate",
    ]
    for col in bool_fill:
        if col in merged.columns:
            merged[col] = merged[col].fillna(False).astype(bool)
    numeric_fill = [
        "module_queries",
        "module_query_gates",
        "module_specificity_gates",
        "module_qc_gates",
        "module_substate_gates",
    ]
    for col in numeric_fill:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)

    merged["strict_dynamic_class"] = merged.apply(classify_row, axis=1)
    merged["diagnostic_information_score"] = merged.apply(make_score, axis=1)
    merged["modeling_use"] = merged["strict_dynamic_class"].map(
        {
            "constraint_candidate_ready": "eligible_for_next_constraint_design_gate",
            "geometry_replicate_insight_specificity_blocked": "use_for_scaling_x_and_negative_controls_only",
            "composition_or_time_confounded_comparator": "negative_control_for_magnitude_not_information",
            "replicate_stable_diagnostic_only": "replicate_geometry_diagnostic_only",
            "not_supported": "do_not_use",
        }
    )

    x_rows = [
        {
            "candidate_x": "within_state_dynamic_margin",
            "definition": "within-substate residual fraction plus wrong-time/wrong-lineage OT margins",
            "current_signal": "periderm noto/smo positive; muscle high-effect rows fail because time/composition confounds remain",
            "primary_confound": "substate composition, timepoint, lineage, response magnitude",
            "required_control": "match/subtract composition JSD, response norm, lineage/time, and wrong-time OT",
            "model_use_if_pass": "sampling weight or auxiliary trajectory-consistency diagnostic, not direct loss yet",
        },
        {
            "candidate_x": "embryo_replicate_vector_reliability",
            "definition": "embryo-level perturb-control displacement cosine with global row vector",
            "current_signal": "10/10 pass, so reliable but non-discriminative",
            "primary_confound": "generic snapshot shift and broad row effect",
            "required_control": "must be combined with specificity margins; cannot be standalone x",
            "model_use_if_pass": "minimum reliability filter before using any ZSCAPE-derived row",
        },
        {
            "candidate_x": "module_specificity_margin",
            "definition": "QC-residual module effect above wrong-target/wrong-time/wrong-lineage controls",
            "current_signal": "noto/smo effects are real but 4/4 query gates fail specificity",
            "primary_confound": "shared epithelial stress/oxidative and IF programs",
            "required_control": "held-out embryo module discovery plus wrong-control collapse",
            "model_use_if_pass": "pathway/program constraint candidate; currently blocked",
        },
        {
            "candidate_x": "hvg_response_concentration",
            "definition": "response energy share captured by top 1k/2k/4k HVG budgets",
            "current_signal": "atlas suggests compact response energy, but RawFM Wessels naive response-topk failed controls",
            "primary_confound": "abundance/HVG variance/source and gene-count budget",
            "required_control": "abundance/variance-matched random gene budgets and fixed-step no-selection RawFM protocol",
            "model_use_if_pass": "gene-budget curriculum or observable-gene RawFM benchmark axis",
        },
        {
            "candidate_x": "composition_shift_information",
            "definition": "substate JSD and composition component of OT displacement",
            "current_signal": "useful as confound and biology, not as direct response information for perturb prediction",
            "primary_confound": "cell-state abundance shift masquerading as perturb response",
            "required_control": "separate composition and within-state prediction metrics",
            "model_use_if_pass": "multi-head evaluation: composition vs within-state expression response",
        },
    ]
    x_table = pd.DataFrame(x_rows)

    summary = {
        "timestamp": now_cst(),
        "rows": int(merged.shape[0]),
        "constraint_candidate_ready": int((merged["strict_dynamic_class"] == "constraint_candidate_ready").sum()),
        "geometry_replicate_insight_specificity_blocked": int(
            (merged["strict_dynamic_class"] == "geometry_replicate_insight_specificity_blocked").sum()
        ),
        "composition_or_time_confounded_comparator": int(
            (merged["strict_dynamic_class"] == "composition_or_time_confounded_comparator").sum()
        ),
        "replicate_stable_diagnostic_only": int(
            (merged["strict_dynamic_class"] == "replicate_stable_diagnostic_only").sum()
        ),
        "dynamic_response_gate_rows": int(merged["dynamic_response_gate"].map(safe_bool).sum()),
        "embryo_vector_gate_rows": int(merged["embryo_vector_gate"].map(safe_bool).sum()),
        "module_all_query_gate_rows": int(merged["module_all_query_gate"].sum()),
        "source_reports": {
            "ot_rows": str(OT_ROWS),
            "ot_report": str(OT_REPORT),
            "embryo_rows": str(EMBRYO_ROWS),
            "module_query_rows": str(MODULE_QUERY_ROWS),
            "module_placebo_rows": str(MODULE_PLACEBO_ROWS),
            "atlas_rows": str(ATLAS_ROWS),
        },
    }
    return merged, x_table, module_summary, summary


def write_report(rows: pd.DataFrame, x_table: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# LatentFM ZSCAPE Strict OT Decomposition Gate")
    lines.append("")
    lines.append(f"Timestamp: `{summary['timestamp']}`")
    lines.append("")
    lines.append("Status: `zscape_strict_ot_decomposition_gate_complete_cpu_only`")
    lines.append("")
    lines.append("GPU authorized: `False`")
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("- CPU-only synthesis over frozen ZSCAPE report artifacts.")
    lines.append("- No model training, no GPU, no checkpoint selection, no canonical multi, no Track C query.")
    lines.append("- OT pseudo-pairs are snapshot distribution pairs, not true single-cell lineage pairs.")
    lines.append("- This gate decides whether current ZSCAPE evidence is ready for modeling constraints.")
    lines.append("")
    lines.append("## Integrated Decision")
    lines.append("")
    lines.append(f"- rows integrated: `{summary['rows']}`")
    lines.append(f"- dynamic-response gate rows: `{summary['dynamic_response_gate_rows']}`")
    lines.append(f"- embryo-vector gate rows: `{summary['embryo_vector_gate_rows']}`")
    lines.append(f"- ready model-constraint candidates: `{summary['constraint_candidate_ready']}`")
    lines.append(
        f"- geometry/replicate insight but specificity-blocked rows: "
        f"`{summary['geometry_replicate_insight_specificity_blocked']}`"
    )
    lines.append(
        f"- composition/time-confounded comparators: "
        f"`{summary['composition_or_time_confounded_comparator']}`"
    )
    lines.append("")
    if summary["constraint_candidate_ready"] == 0:
        lines.append(
            "Decision: no current ZSCAPE row is ready to become a LatentFM/RawFM "
            "training constraint. The useful output is a biological/scaling "
            "hypothesis set plus negative controls."
        )
    else:
        lines.append(
            "Decision: at least one row is eligible for a separate constraint-design "
            "gate; this report itself still does not authorize training promotion."
        )
    lines.append("")
    lines.append("## Row Decomposition")
    lines.append("")
    header = (
        "| row | class | comp frac | within frac | traj cos | wt margin | wl margin | "
        "embryo CI | module gates | use |"
    )
    lines.append(header)
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    sort_cols = ["strict_dynamic_class", "diagnostic_information_score"]
    view = rows.sort_values(sort_cols, ascending=[True, False])
    for _, row in view.iterrows():
        module_gates = (
            f"{int(row.get('module_query_gates', 0))}/{int(row.get('module_queries', 0))} "
            f"query; spec={bool(row.get('module_all_specificity_gate', False))}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["row_id"]),
                    str(row["strict_dynamic_class"]),
                    fmt(row.get("composition_norm_fraction_of_centroid")),
                    fmt(row.get("within_substate_residual_fraction_of_centroid")),
                    fmt(row.get("trajectory_cosine")),
                    fmt(row.get("wrong_time_margin_ot")),
                    fmt(row.get("wrong_lineage_margin_ot")),
                    fmt(row.get("mean_cosine_ci_low")),
                    module_gates,
                    str(row["modeling_use"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Candidate Scaling x Variables")
    lines.append("")
    lines.append("| x | current signal | main confound | required control | modeling use if pass |")
    lines.append("|---|---|---|---|---|")
    for _, row in x_table.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["candidate_x"]),
                    str(row["current_signal"]),
                    str(row["primary_confound"]),
                    str(row["required_control"]),
                    str(row["model_use_if_pass"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Biological Interpretation")
    lines.append("")
    lines.append(
        "- The most supported ab initio rule is: stable perturbation response should "
        "be measured as a within-state OT displacement with embryo-vector "
        "reliability and wrong-control margins."
    )
    lines.append(
        "- Periderm `noto/smo` satisfy the geometry/replicate part, but the module "
        "specificity gate fails because matched wrong-target/time controls can "
        "show comparable module effects."
    )
    lines.append(
        "- Mature fast muscle rows are useful negative comparators: large response "
        "magnitude is not the same as high-quality information, because "
        "composition/time components can dominate."
    )
    lines.append(
        "- The model-facing consequence is to split future objectives/evaluations "
        "into composition shift, within-state response, developmental tangent "
        "alignment, and module specificity, rather than optimizing a single "
        "uninterpreted expression distance."
    )
    lines.append("")
    lines.append("## Next Gate")
    lines.append("")
    lines.append(
        "Next CPU gate: construct a structural biological-information x table that "
        "combines within-state OT margin, specificity margin, HVG response "
        "concentration, and support, then residualizes/matches against lineage, "
        "time, response norm, abundance/HVG variance, and source."
    )
    lines.append("")
    lines.append(
        "Next GPU smoke only after that CPU gate passes: fixed-step/no-selection "
        "RawFM or LatentFM training-set/gene-budget smoke with matched random and "
        "abundance controls. Stop if the proposed x does not beat controls or "
        "fails anchor no-harm."
    )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(
        f"- row table: `{OUT_DIR / 'zscape_strict_ot_decomposition_rows.csv'}`"
    )
    lines.append(
        f"- candidate x table: `{OUT_DIR / 'zscape_strict_ot_decomposition_candidate_x.csv'}`"
    )
    lines.append(
        f"- module summary: `{OUT_DIR / 'zscape_strict_ot_decomposition_module_summary.csv'}`"
    )
    lines.append(f"- JSON: `{OUT_DIR / 'zscape_strict_ot_decomposition_gate_20260628.json'}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, x_table, module_summary, summary = build_tables()
    rows.to_csv(OUT_DIR / "zscape_strict_ot_decomposition_rows.csv", index=False)
    x_table.to_csv(OUT_DIR / "zscape_strict_ot_decomposition_candidate_x.csv", index=False)
    module_summary.to_csv(OUT_DIR / "zscape_strict_ot_decomposition_module_summary.csv", index=False)
    (OUT_DIR / "zscape_strict_ot_decomposition_gate_20260628.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    report = write_report(rows, x_table, summary)
    (OUT_DIR / "LATENTFM_ZSCAPE_STRICT_OT_DECOMPOSITION_GATE_20260628.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
