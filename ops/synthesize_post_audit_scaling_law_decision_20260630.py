#!/usr/bin/env python3
"""Integrate post-audit downstream scaling-law evidence.

CPU/report-only. This script summarizes whether current scaling x variables are
solid descriptors and whether any can become a training-actionable gate.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "post_audit_scaling_law_decision_20260630"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(val):
        return None
    return val


def fmt(value: Any, digits: int = 4) -> str:
    val = safe_float(value)
    if val is None:
        return "NA"
    return f"{val:.{digits}f}"


def main() -> int:
    paths = {
        "exact_combined": REPORTS
        / "exact_response_information_combined_claim_report_20260629"
        / "latentfm_exact_response_information_combined_claim_report_20260629.json",
        "exact_analog": REPORTS
        / "exact_analog_observability_matched_feasibility_20260629"
        / "latentfm_exact_analog_observability_matched_feasibility_20260629.json",
        "exact_propensity": REPORTS
        / "exact_coverage_propensity_residual_match_gate_20260629"
        / "latentfm_exact_coverage_propensity_residual_match_gate_20260629.json",
        "exact_crossdataset": REPORTS
        / "exact_coverage_crossdataset_matched_feasibility_20260629"
        / "latentfm_exact_coverage_crossdataset_matched_feasibility_20260629.json",
        "observable_budget": REPORTS
        / "observable_gene_budget_scaling_law_gate_20260630"
        / "latentfm_observable_gene_budget_scaling_law_gate_20260630.json",
        "multiaxis": REPORTS
        / "multiaxis_information_scaling_incremental_gate_20260629"
        / "latentfm_multiaxis_information_scaling_incremental_gate_20260629.json",
        "cluster_density": REPORTS
        / "trainset_cluster_density_information_gate_20260629"
        / "latentfm_trainset_cluster_density_information_gate_20260629.json",
        "ot_coverage": REPORTS
        / "trainset_ot_coverage_information_gate_20260629"
        / "latentfm_trainset_ot_coverage_information_gate_20260629.json",
        "zscape_structural": REPORTS
        / "zscape_structural_dynamic_scaling_x_20260630"
        / "zscape_structural_dynamic_scaling_x_20260630.json",
        "tracka_benchmark": REPORTS
        / "tracka_benchmark_control_consolidation_20260630"
        / "tracka_benchmark_control_consolidation_20260630.json",
    }
    payloads = {key: read_json(path) for key, path in paths.items()}

    exact_summary = pd.read_csv(
        REPORTS / "exact_analog_observability_matched_feasibility_20260629" / "design_summary.csv"
    )
    exact_best_pairs = int(pd.to_numeric(exact_summary["matched_pairs"], errors="coerce").max())
    exact_best_datasets = int(pd.to_numeric(exact_summary["datasets_with_pairs"], errors="coerce").max())

    prop_summary = pd.read_csv(
        REPORTS
        / "exact_coverage_propensity_residual_match_gate_20260629"
        / "exact_coverage_propensity_match_summary.csv"
    )
    prop_best = prop_summary.sort_values("n_pairs", ascending=False).iloc[0]

    cross_summary = pd.read_csv(
        REPORTS
        / "exact_coverage_crossdataset_matched_feasibility_20260629"
        / "crossdataset_matched_design_summary.csv"
    )
    cross_best = cross_summary.sort_values("n_pairs", ascending=False).iloc[0]

    multiaxis_assoc = pd.read_csv(
        REPORTS
        / "multiaxis_information_scaling_incremental_gate_20260629"
        / "multiaxis_information_scaling_incremental_association_rows.csv"
    )
    residual_rows = multiaxis_assoc[
        multiaxis_assoc["predictor_family"].astype(str).str.contains("residual|geometry", case=False, regex=True)
        | multiaxis_assoc["predictor"].astype(str).str.contains("vendi|rank|pairwise", case=False, regex=True)
    ].copy()
    residual_best = residual_rows.sort_values("p_value", ascending=True).head(1)

    observable = payloads["observable_budget"]
    obs_metrics = observable.get("decision_metrics", {})
    zscape = payloads["zscape_structural"]

    axes = [
        {
            "axis": "exact_response_coverage",
            "descriptor_status": "solid_descriptor",
            "training_status": "blocked",
            "evidence": (
                f"exact/analog best strict feasibility {exact_best_pairs} pairs across {exact_best_datasets} datasets; "
                f"propensity best {int(prop_best['n_pairs'])} pairs with propensity AUC/confounding flag; "
                f"cross-dataset best {int(cross_best['n_pairs'])} pairs but source/covariate imbalance fails"
            ),
            "blocker": "matched feasibility, source/covariate imbalance, and source/control dual-baseline requirement",
            "next_action": "keep as manuscript/failure-map descriptor unless a new matched split family reaches >=300 balanced pairs",
        },
        {
            "axis": "observable_gene_budget",
            "descriptor_status": "solid_descriptor",
            "training_status": "blocked",
            "evidence": (
                f"top1000 HVG share {fmt(obs_metrics.get('all_top1000_hvg_share'))}; "
                f"descriptor pass {observable.get('descriptor_pass')}; "
                f"HVG-specific intervention gate {observable.get('hvg_specific_intervention_gate')}"
            ),
            "blocker": "abundance/detection/mean controls explain almost all static HVG advantage",
            "next_action": "only revisit with a nonstatic residualized information axis or a separately justified RawFM budget route",
        },
        {
            "axis": "generic_diversity_vendi_geosketch",
            "descriptor_status": "negative_or_weak",
            "training_status": "blocked",
            "evidence": (
                "multiaxis, cluster-density, OT coverage, residual Vendi/effective-rank style axes have zero passing "
                "incremental gates after exact/count/source/background controls"
            ),
            "blocker": "same-n random controls, source permutations, sparse 17-row outcome table, and unstable LODO",
            "next_action": "do not rerun generic diversity; only perturbation-conditioned residual diversity with new vectors would be nonduplicate",
        },
        {
            "axis": "structural_dynamic_information",
            "descriptor_status": "biology_descriptor",
            "training_status": "blocked",
            "evidence": (
                f"magnitude-not-information supported; response norm vs structural score rho "
                f"{fmt(zscape.get('response_norm_vs_structural_score_spearman'))}; ZSCAPE geometry positives are specificity-blocked"
            ),
            "blocker": "ZSCAPE target/pathway specificity, species-safe latent route, and LatentFM train-set translation gates fail",
            "next_action": "use as biological framing and negative-control taxonomy, not as loss/sampling until a new CPU gate passes",
        },
    ]

    status = "scaling_law_descriptor_solid_training_blocked_no_gpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_DIR / "post_audit_scaling_law_axis_rows_20260630.csv"
    json_path = OUT_DIR / "post_audit_scaling_law_decision_20260630.json"
    md_path = OUT_DIR / "LATENTFM_POST_AUDIT_SCALING_LAW_DECISION_20260630.md"
    pd.DataFrame(axes).to_csv(rows_path, index=False)

    out = {
        "created_at": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "training_actionable_axes": 0,
        "solid_descriptor_axes": ["exact_response_coverage", "observable_gene_budget", "structural_dynamic_information"],
        "blocked_axes": [axis["axis"] for axis in axes],
        "boundary": "cpu_report_only_no_training_no_inference_no_gpu_no_canonical_multi_no_trackc_query",
        "inputs": {key: str(path) for key, path in paths.items()},
        "outputs": {"markdown": str(md_path), "json": str(json_path), "rows": str(rows_path)},
        "axis_rows": axes,
    }
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Post-Audit Scaling-Law Decision 20260630",
        "",
        f"Created: {out['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only synthesis after external scaling audit.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C query, or GPU.",
        "- This distinguishes manuscript-grade descriptors from training-actionable variables.",
        "",
        "## Bottom Line",
        "",
        "- Current scaling evidence is solid as controlled descriptor/negative evidence.",
        "- No current x is training-actionable under the dual-baseline Track A gate.",
        "- The strongest valid law is: raw dataset size and response magnitude are not information; information must be condition-specific, controlled, and no-harm.",
        "",
        "## Axis Decisions",
        "",
        "| axis | descriptor | training | blocker |",
        "|---|---|---|---|",
    ]
    for axis in axes:
        lines.append(
            f"| `{axis['axis']}` | `{axis['descriptor_status']}` | `{axis['training_status']}` | {axis['blocker']} |"
        )
    lines += [
        "",
        "## Next Action",
        "",
        "- Do not launch a scaling GPU smoke now.",
        "- For scaling-law science, package exact coverage and observable-gene budget as descriptors with explicit controls.",
        "- For model improvement, require either a new matched external source or a new nonstatic residualized x that passes balanced CPU admission.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{json_path}`",
        f"- rows: `{rows_path}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "markdown": str(md_path), "json": str(json_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
