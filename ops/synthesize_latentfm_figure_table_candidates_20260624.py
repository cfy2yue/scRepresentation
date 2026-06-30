#!/usr/bin/env python3
"""Create figure/table candidate data for the LatentFM consolidation package."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"


def load_json(name: str) -> dict[str, Any]:
    with (REPORTS / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def get_result_summary(data: dict[str, Any], group: str, control: str = "main") -> dict[str, Any]:
    for row in data.get("results", []):
        if row.get("group") == group and row.get("control") == control:
            return row.get("summary", {})
    return {}


def main() -> None:
    ceiling = load_json("latentfm_tracka_identifiability_ceiling_20260624.json")
    control = load_json("latentfm_control_state_support_gate_20260624.json")
    signed = load_json("latentfm_signed_neighborhood_consistency_gate_20260624.json")
    composite = load_json("latentfm_composite_safe_subset_gate_20260624.json")
    noise = load_json("latentfm_bootstrap_target_noise_gate_20260624.json")
    reliability = load_json("latentfm_trainonly_reliability_condition_gate_20260624.json")
    factorized = load_json("latentfm_factorized_gene_context_gate_20260624.json")
    prototype = load_json("latentfm_perturbation_equivariant_prototype_gate_20260624.json")
    trackc = load_json("latentfm_trackc_v2_family_closure_synthesis_20260624.json")
    ot_signal = load_json("latentfm_ot_pairing_signal_audit_20260624.json")
    ot_quality = load_json("latentfm_ot_pairing_quality_reliability_gate_20260624.json")
    ot_random = load_json("latentfm_xverse_ot_pairmode_random_control_decision_20260624.json")
    ot_hungarian = load_json("latentfm_xverse_ot_pairmode_hungarian_decision_20260624.json")

    oracle_ladder = []
    for row in ceiling.get("rows", []):
        oracle_ladder.append(
            {
                "group": row.get("group"),
                "tier": row.get("tier"),
                "name": row.get("name"),
                "mean_pp_delta": row.get("mean_pp_delta"),
                "ci95_low": row.get("ci95_low"),
                "ci95_high": row.get("ci95_high"),
                "bootstrap_p_harm": row.get("bootstrap_p_harm"),
                "dataset_min_pp_delta": row.get("dataset_min_pp_delta"),
                "mean_mmd_delta": row.get("mean_mmd_delta"),
                "mean_alpha": row.get("mean_alpha"),
            }
        )

    def risk_row(name: str, data: dict[str, Any], group: str, status: str) -> dict[str, Any]:
        summary = get_result_summary(data, group)
        return {
            "branch": name,
            "group": group,
            "status": status,
            "mean_pp_delta": summary.get("mean_pp_delta"),
            "ci95_low": summary.get("ci95_low"),
            "ci95_high": summary.get("ci95_high"),
            "bootstrap_p_harm": summary.get("bootstrap_p_harm"),
            "condition_p_harm": summary.get("condition_p_harm"),
            "dataset_min_pp_delta": summary.get("dataset_min_pp_delta"),
            "mean_mmd_delta": summary.get("mean_mmd_delta"),
        }

    groups = ["internal_val_cross_background_seen_gene_proxy", "internal_val_family_gene_proxy"]
    gain_tail = []
    for group in groups:
        gain_tail.extend(
            [
                risk_row("control_state_support_cap120", control, group, control["decision"]["status"]),
                risk_row("signed_neighborhood", signed, group, signed["decision"]["status"]),
                risk_row("composite_safe_subset", composite, group, composite["decision"]["status"]),
                risk_row("bootstrap_target_noise", noise, group, noise["decision"]["status"]),
                risk_row("reliability_condition_cap120", reliability, group, reliability["decision"]["status"]),
            ]
        )
    for name, data in [
        ("perturbation_equivariant_prototype", prototype),
        ("factorized_gene_context", factorized),
    ]:
        for group in groups:
            for row in data.get("results", []):
                if row.get("control") == "main" and row.get("group") == group:
                    summary = row.get("summary", {})
                    gain_tail.append(
                        {
                            "branch": name,
                            "group": group,
                            "status": data["decision"]["status"],
                            "mean_pp_delta": summary.get("mean_pp_delta"),
                            "ci95_low": summary.get("ci95_low"),
                            "ci95_high": summary.get("ci95_high"),
                            "bootstrap_p_harm": summary.get("bootstrap_p_harm"),
                            "condition_p_harm": None,
                            "dataset_min_pp_delta": summary.get("dataset_min_pp_delta"),
                            "mean_mmd_delta": None,
                        }
                    )
                    break

    trackc_overlap = []
    best = trackc["current_best"]
    trackc_overlap.append(
        {
            "panel": "frozen_query_diagnostic",
            "name": best["route"],
            "pearson_delta": best["query_multi_pearson_delta"],
            "mmd_delta": best["query_multi_mmd_delta"],
            "seen_pearson_delta": best["seen_pearson_delta"],
            "unseen1_pearson_delta": best["unseen1_pearson_delta"],
            "unseen2_pearson_delta": best["unseen2_pearson_delta"],
            "failure_or_scope": "diagnostic_only_not_formal_multi",
        }
    )
    for gate in trackc.get("expansion_gates", []):
        metrics = gate.get("key_metrics", {})
        trackc_overlap.append(
            {
                "panel": "query_free_expansion_gate",
                "name": gate.get("name"),
                "pearson_delta": metrics.get("aggregate_pp")
                or metrics.get("support_pp")
                or metrics.get("zero_overlap_pp"),
                "mmd_delta": metrics.get("aggregate_mmd") or metrics.get("support_mmd"),
                "seen_pearson_delta": metrics.get("norman_pp"),
                "unseen1_pearson_delta": metrics.get("wessels_pp"),
                "unseen2_pearson_delta": metrics.get("zero_overlap_pp"),
                "failure_or_scope": "; ".join(gate.get("reasons", [])),
            }
        )

    ot_rows = ot_signal.get("rows", [])
    ot_summary = ot_signal.get("summary", {})
    ot_table = [
        {
            "item": "pairing_signal_multinomial",
            "status": "mechanism_active",
            "metric_a": "cost_delta_frac_mean",
            "value_a": ot_summary.get("multinomial_cost_delta_frac", {}).get("mean")
            if isinstance(ot_summary.get("multinomial_cost_delta_frac"), dict)
            else mean([r["multinomial_cost_delta_frac"] for r in ot_rows]),
            "metric_b": "unique_gt_frac_mean",
            "value_b": ot_summary.get("multinomial_unique_gt_frac", {}).get("mean")
            if isinstance(ot_summary.get("multinomial_unique_gt_frac"), dict)
            else mean([r["multinomial_unique_gt_frac"] for r in ot_rows]),
            "interpretation": "OT changes coupling but replacement sampling perturbs marginals.",
        },
        {
            "item": "pairing_signal_assignment",
            "status": "mechanism_cleaner",
            "metric_a": "cost_delta_frac_mean",
            "value_a": ot_summary.get("assignment_cost_delta_frac", {}).get("mean")
            if isinstance(ot_summary.get("assignment_cost_delta_frac"), dict)
            else mean([r["assignment_cost_delta_frac"] for r in ot_rows]),
            "metric_b": "delta_rel_error_mean",
            "value_b": ot_summary.get("assignment_delta_rel_error", {}).get("mean")
            if isinstance(ot_summary.get("assignment_delta_rel_error"), dict)
            else mean([r["assignment_delta_rel_error"] for r in ot_rows]),
            "interpretation": "Assignment/Hungarian preserves mini-batch marginals.",
        },
        {
            "item": "random_no_ot_control",
            "status": ot_random["decision"]["status"],
            "metric_a": "passed_runs",
            "value_a": len(ot_random["decision"].get("passed_runs", [])),
            "metric_b": "action",
            "value_b": ot_random["decision"].get("action"),
            "interpretation": "Random/no-OT did not justify default change.",
        },
        {
            "item": "hungarian_gpu_smoke",
            "status": ot_hungarian["decision"]["status"],
            "metric_a": "passed_runs",
            "value_a": len(ot_hungarian["decision"].get("passed_runs", [])),
            "metric_b": "action",
            "value_b": ot_hungarian["decision"].get("action"),
            "interpretation": "Marginal-preserving OT fixed mechanism concern but failed model gate.",
        },
        {
            "item": "pairing_quality_reliability",
            "status": ot_quality["decision"]["status"],
            "metric_a": "material_expected_correlations",
            "value_a": ot_quality["decision"].get("material_expected_correlations"),
            "metric_b": "material_contradictions",
            "value_b": ot_quality["decision"].get("material_contradictions"),
            "interpretation": "Pairing quality did not predict response reliability robustly.",
        },
    ]

    out = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "figure_table_candidates_ready_no_gpu",
        "boundary": {
            "completed_json_reports_only": True,
            "active_logs": False,
            "raw_canonical_or_query": False,
            "gpu": False,
        },
        "oracle_headroom_ladder": oracle_ladder,
        "gain_vs_tail_risk": gain_tail,
        "trackc_overlap_failure": trackc_overlap,
        "ot_wired_no_gain": ot_table,
    }

    json_path = REPORTS / "latentfm_figure_table_candidates_20260624.json"
    oracle_csv = REPORTS / "latentfm_oracle_headroom_ladder_20260624.csv"
    risk_csv = REPORTS / "latentfm_gain_vs_tail_risk_20260624.csv"
    trackc_csv = REPORTS / "latentfm_trackc_overlap_failure_panel_20260624.csv"
    ot_csv = REPORTS / "latentfm_ot_wired_no_gain_panel_20260624.csv"
    md_path = REPORTS / "LATENTFM_FIGURE_TABLE_CANDIDATES_20260624.md"

    json_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(
        oracle_csv,
        oracle_ladder,
        [
            "group",
            "tier",
            "name",
            "mean_pp_delta",
            "ci95_low",
            "ci95_high",
            "bootstrap_p_harm",
            "dataset_min_pp_delta",
            "mean_mmd_delta",
            "mean_alpha",
        ],
    )
    write_csv(
        risk_csv,
        gain_tail,
        [
            "branch",
            "group",
            "status",
            "mean_pp_delta",
            "ci95_low",
            "ci95_high",
            "bootstrap_p_harm",
            "condition_p_harm",
            "dataset_min_pp_delta",
            "mean_mmd_delta",
        ],
    )
    write_csv(
        trackc_csv,
        trackc_overlap,
        [
            "panel",
            "name",
            "pearson_delta",
            "mmd_delta",
            "seen_pearson_delta",
            "unseen1_pearson_delta",
            "unseen2_pearson_delta",
            "failure_or_scope",
        ],
    )
    write_csv(
        ot_csv,
        ot_table,
        ["item", "status", "metric_a", "value_a", "metric_b", "value_b", "interpretation"],
    )

    def fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:+.6f}"
        return str(v)

    lines = [
        "# LatentFM Figure/Table Candidates",
        "",
        "Status: `figure_table_candidates_ready_no_gpu`",
        "",
        "## Boundary",
        "",
        "- Reads completed JSON reports only.",
        "- No active logs, raw canonical/query artifacts, canonical multi selection, training, inference, or GPU.",
        "",
        "## Oracle Headroom Ladder",
        "",
        "| Group | Tier | Name | Mean pp delta | Dataset min | p_harm |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in oracle_ladder:
        if row["group"] == "internal_val_cross_background_seen_gene_proxy":
            lines.append(
                f"| {row['group']} | {row['tier']} | {row['name']} | {fmt(row['mean_pp_delta'])} | {fmt(row['dataset_min_pp_delta'])} | {fmt(row['bootstrap_p_harm'])} |"
            )

    lines.extend(
        [
            "",
            "## Average Gain Vs Tail Risk",
            "",
            "| Branch | Group | Mean pp delta | Dataset min | p_harm | Status |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in gain_tail:
        if row["group"] == "internal_val_cross_background_seen_gene_proxy":
            lines.append(
                f"| {row['branch']} | {row['group']} | {fmt(row['mean_pp_delta'])} | {fmt(row['dataset_min_pp_delta'])} | {fmt(row['bootstrap_p_harm'])} | `{row['status']}` |"
            )

    lines.extend(
        [
            "",
            "## Track C Overlap Failure",
            "",
            "| Panel | Name | pp delta | MMD delta | unseen2 / zero-overlap pp | Scope/failure |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in trackc_overlap:
        lines.append(
            f"| {row['panel']} | {row['name']} | {fmt(row['pearson_delta'])} | {fmt(row['mmd_delta'])} | {fmt(row['unseen2_pearson_delta'])} | {row['failure_or_scope']} |"
        )

    lines.extend(
        [
            "",
            "## OT Wired But No Gain",
            "",
            "| Item | Status | Metric A | Value A | Metric B | Value B | Interpretation |",
            "|---|---|---|---:|---|---:|---|",
        ]
    )
    for row in ot_table:
        lines.append(
            f"| {row['item']} | `{row['status']}` | {row['metric_a']} | {fmt(row['value_a'])} | {row['metric_b']} | {fmt(row['value_b'])} | {row['interpretation']} |"
        )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{json_path}`",
            f"- `{oracle_csv}`",
            f"- `{risk_csv}`",
            f"- `{trackc_csv}`",
            f"- `{ot_csv}`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)
    print(json_path)
    print(oracle_csv)
    print(risk_csv)
    print(trackc_csv)
    print(ot_csv)


if __name__ == "__main__":
    main()
