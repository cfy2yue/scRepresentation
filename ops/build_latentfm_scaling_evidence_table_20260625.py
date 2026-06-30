#!/usr/bin/env python3
"""Build a compact scaling evidence table from completed LatentFM gate reports.

This is a CPU-only synthesis/index. It does not read model checkpoints, canonical
multi, held-out Track C query, train, infer, or launch GPU work.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_CSV = REPORTS / "latentfm_scaling_evidence_table_20260625.csv"
OUT_JSON = REPORTS / "latentfm_scaling_evidence_table_20260625.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_EVIDENCE_TABLE_20260625.md"


def load_json(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    return json.loads(path.read_text())


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:+.6f}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(fmt(v) for v in value) + "]"
    return str(value)


def add_row(
    rows: list[dict[str, Any]],
    *,
    axis: str,
    estimand: str,
    status: str,
    claim_scope: str,
    primary_metric: Any = None,
    secondary_metric: Any = None,
    ci95: Any = None,
    tail_metric: Any = None,
    control_signal: Any = None,
    decision: str = "",
    source_report: str,
    notes: str = "",
) -> None:
    rows.append(
        {
            "axis": axis,
            "estimand": estimand,
            "status": status,
            "claim_scope": claim_scope,
            "primary_metric": fmt(primary_metric),
            "secondary_metric": fmt(secondary_metric),
            "ci95": fmt(ci95),
            "tail_metric": fmt(tail_metric),
            "control_signal": fmt(control_signal),
            "decision": decision,
            "source_report": source_report,
            "notes": notes,
        }
    )


def main() -> None:
    rows: list[dict[str, Any]] = []

    condition_table = load_json("reports/latentfm_scaling_law_condition_table_20260624.json")
    condition_rows = condition_table.get("rows", [])
    datasets = {r.get("dataset") for r in condition_rows if r.get("dataset")}
    source_verified = sum(1 for r in condition_rows if r.get("source_quality") == "source_verified")
    add_row(
        rows,
        axis="provenance",
        estimand="frozen_condition_table",
        status="materialized",
        claim_scope="design_only",
        primary_metric=len(condition_rows),
        secondary_metric=len(datasets),
        control_signal=source_verified,
        decision="use as S0 input for scaling estimands",
        source_report="reports/LATENTFM_SCALING_LAW_CONDITION_TABLE_20260624.md",
        notes="primary/secondary/control are condition rows/datasets/source-verified rows",
    )

    nested3k = load_json("reports/latentfm_true_cell_count_nested_matrix_decision_20260624.json")
    for brow in nested3k.get("matrix_summary", {}).get("budget_rows", []):
        budget = brow.get("budget")
        pp_boot = brow.get("cross_background_pp_condition_bootstrap", {})
        tail = brow.get("cross_background_pp_dataset_tail", {})
        add_row(
            rows,
            axis="true_cell_count",
            estimand=f"nested_3k_budget{budget}",
            status=nested3k.get("status", ""),
            claim_scope="mechanism_only",
            primary_metric=brow.get("cross_background_pp_delta_mean"),
            secondary_metric=brow.get("family_gene_pp_delta_mean"),
            ci95=pp_boot.get("ci95"),
            tail_metric=tail.get("negative_tail_lt_minus_0p020"),
            decision=nested3k.get("action", ""),
            source_report="reports/LATENTFM_TRUE_CELL_COUNT_NESTED_MATRIX_DECISION_20260624.md",
            notes=f"min_dataset={tail.get('min_dataset', {}).get('dataset')}",
        )

    budget128_6k = load_json("reports/latentfm_true_cell_count_budget128_tail_stability_6k_decision_20260625.json")
    for brow in budget128_6k.get("matrix_summary", {}).get("budget_rows", []):
        pp_boot = brow.get("cross_background_pp_condition_bootstrap", {})
        tail = brow.get("cross_background_pp_dataset_tail", {})
        add_row(
            rows,
            axis="true_cell_count",
            estimand="nested_6k_budget128",
            status=budget128_6k.get("status", ""),
            claim_scope="mechanism_only_route_frozen_before_noharm",
            primary_metric=brow.get("cross_background_pp_delta_mean"),
            secondary_metric=brow.get("family_gene_pp_delta_mean"),
            ci95=pp_boot.get("ci95"),
            tail_metric=tail.get("negative_tail_lt_minus_0p020"),
            decision=budget128_6k.get("action", ""),
            source_report="reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_TAIL_STABILITY_6K_DECISION_20260625.md",
            notes=f"min_dataset={tail.get('min_dataset', {}).get('dataset')}",
        )

    ar005 = load_json("reports/latentfm_true_cell_count_budget128_anchor_replay005_6k_decision_20260625.json")
    if not ar005.get("_missing"):
        for brow in ar005.get("matrix_summary", {}).get("budget_rows", []):
            pp_boot = brow.get("cross_background_pp_condition_bootstrap", {})
            tail = brow.get("cross_background_pp_dataset_tail", {})
            add_row(
                rows,
                axis="true_cell_count_repair",
                estimand="budget128_6k_anchor_replay005",
                status=ar005.get("status", ""),
                claim_scope="negative_repair_evidence",
                primary_metric=brow.get("cross_background_pp_delta_mean"),
                secondary_metric=brow.get("family_gene_pp_delta_mean"),
                ci95=pp_boot.get("ci95"),
                tail_metric=tail.get("negative_tail_lt_minus_0p020"),
                decision="close_before_canonical_noharm",
                source_report="reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_ANCHOR_REPLAY005_6K_DECISION_20260625.md",
                notes=f"min_dataset={tail.get('min_dataset', {}).get('dataset')}",
            )

    budget64_6k = load_json("reports/latentfm_true_cell_count_budget64_tail_stability_6k_decision_20260625.json")
    if not budget64_6k.get("_missing"):
        for brow in budget64_6k.get("matrix_summary", {}).get("budget_rows", []):
            pp_boot = brow.get("cross_background_pp_condition_bootstrap", {})
            tail = brow.get("cross_background_pp_dataset_tail", {})
            add_row(
                rows,
                axis="true_cell_count",
                estimand="nested_6k_budget64",
                status=budget64_6k.get("status", ""),
                claim_scope="negative_tail_evidence_curve_expansion_closed",
                primary_metric=brow.get("cross_background_pp_delta_mean"),
                secondary_metric=brow.get("family_gene_pp_delta_mean"),
                ci95=pp_boot.get("ci95"),
                tail_metric=tail.get("negative_tail_lt_minus_0p020"),
                decision="do_not_launch_budget256_from_budget64_gate",
                source_report="reports/LATENTFM_TRUE_CELL_COUNT_BUDGET64_TAIL_STABILITY_6K_DECISION_20260625.md",
                notes=f"min_dataset={tail.get('min_dataset', {}).get('dataset')}",
            )

    noharm = load_json("reports/latentfm_true_cell_count_budget128_6k_canonical_noharm_decision_20260625.json")
    for row in noharm.get("rows", []):
        metrics = row.get("metrics", {})
        add_row(
            rows,
            axis="canonical_noharm",
            estimand=f"budget128_6k_seed{row.get('seed')}",
            status=row.get("gate_status", noharm.get("decision", {}).get("status", "")),
            claim_scope="veto_failed_deployable_promotion_closed",
            primary_metric=metrics.get("cross_background_seen_gene:pearson_pert", {}).get("delta_mean"),
            secondary_metric=metrics.get("family_gene:pearson_pert", {}).get("p_harm"),
            ci95=metrics.get("cross_background_seen_gene:pearson_pert", {}).get("ci95"),
            tail_metric=metrics.get("family_gene:test_mmd_clamped", {}).get("p_harm"),
            decision=noharm.get("decision", {}).get("action", ""),
            source_report="reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_CANONICAL_NOHARM_DECISION_20260625.md",
            notes="primary=cross pp delta; secondary=family pp p_harm; tail=family MMD p_harm",
        )

    protocol = load_json("reports/latentfm_scaling_protocol_matrix_decision_20260624.json")
    for row in protocol.get("rows", []):
        metrics = row.get("metrics", {})
        add_row(
            rows,
            axis="condition_count_or_breadth",
            estimand=row.get("arm", ""),
            status=protocol.get("status", ""),
            claim_scope="diagnostic_only",
            primary_metric=metrics.get("cross_pp_delta_vs_anchor"),
            secondary_metric=metrics.get("family_gene_pp_delta_vs_anchor"),
            tail_metric=metrics.get("family_gene_mmd_delta_vs_anchor"),
            decision=protocol.get("decision", {}).get("action", ""),
            source_report="reports/LATENTFM_SCALING_PROTOCOL_MATRIX_DECISION_20260624.md",
            notes=f"role={row.get('role')}",
        )

    seed_gate = load_json("reports/latentfm_scaling_seed_matched_micro_matrix_gate_20260624.json")
    add_row(
        rows,
        axis="condition_count_seed_robustness",
        estimand="cap60_6k_seed_matched",
        status=seed_gate.get("status", ""),
        claim_scope="negative_evidence",
        primary_metric=seed_gate.get("summary", {}).get("n_pass_internal"),
        secondary_metric=seed_gate.get("summary", {}).get("n_fail_internal"),
        control_signal=seed_gate.get("summary", {}).get("cross_pp_sign_flip"),
        decision=seed_gate.get("next_action", ""),
        source_report="reports/LATENTFM_SCALING_SEED_MATCHED_MICRO_MATRIX_GATE_20260624.md",
        notes="primary/secondary are pass/fail internal run counts",
    )

    source_resolved = load_json("reports/latentfm_scaling_source_resolved_matched_estimand_gate_20260624.json")
    summary = source_resolved.get("summary", {})
    add_row(
        rows,
        axis="background_type_source",
        estimand="source_resolved_matched_estimand",
        status=source_resolved.get("status", ""),
        claim_scope="negative_evidence",
        primary_metric=summary.get("condition_count_pp"),
        secondary_metric=summary.get("background_type_nmi"),
        tail_metric=summary.get("condition_count_dataset_min"),
        decision=source_resolved.get("decision", {}).get("scaling_claim_scope", ""),
        source_report="reports/LATENTFM_SCALING_SOURCE_RESOLVED_MATCHED_ESTIMAND_GATE_20260624.md",
        notes="background/type confounded and dataset tails unsafe",
    )

    target = load_json("reports/latentfm_scaling_target_activity_gate_20260624.json")
    add_row(
        rows,
        axis="target_observability",
        estimand="target_activity_gate",
        status=target.get("status", ""),
        claim_scope="negative_evidence_with_hint",
        primary_metric=target.get("summary", {}).get("spearman_pp_vs_nonzero_fraction"),
        secondary_metric=target.get("summary", {}).get("high_activity_pp_mean"),
        tail_metric=target.get("summary", {}).get("high_activity_dataset_min_pp"),
        decision="no GPU",
        source_report="reports/LATENTFM_SCALING_TARGET_ACTIVITY_GATE_20260624.md",
        notes="report MD contains human-readable key values if JSON schema is sparse",
    )

    chemical = load_json("reports/latentfm_scaling_chemical_holdout_eval_gate_20260624.json")
    for arm in chemical.get("arms", []):
        add_row(
            rows,
            axis="chemical_holdout",
            estimand=arm.get("arm", ""),
            status=chemical.get("status", ""),
            claim_scope="negative_evidence",
            primary_metric=arm.get("pp_delta_vs_anchor"),
            secondary_metric=arm.get("mmd_delta_vs_anchor"),
            tail_metric=arm.get("dataset_min_pp_delta"),
            control_signal=arm.get("negative_dataset_tails_lt_minus_0p02"),
            decision=chemical.get("decision", {}).get("next_action", ""),
            source_report="reports/LATENTFM_SCALING_CHEMICAL_HOLDOUT_EVAL_GATE_20260624.md",
            notes="train-only SciPlex chemical holdout; canonical reference excluded",
        )

    noharm_transfer = load_json("reports/latentfm_scaling_noharm_transfer_calibration_gate_20260624.json")
    candidates = noharm_transfer.get("candidates", [])
    n_pass = sum(1 for c in candidates if c.get("canonical_pass"))
    add_row(
        rows,
        axis="noharm_transfer",
        estimand="internal_pass_like_to_canonical_noharm",
        status=noharm_transfer.get("status", ""),
        claim_scope="veto_only",
        primary_metric=n_pass,
        secondary_metric=len(candidates),
        control_signal=noharm_transfer.get("summary", {}).get("spearman_internal_score_vs_pp_harm"),
        decision="do not authorize GPU from surrogate",
        source_report="reports/LATENTFM_SCALING_NOHARM_TRANSFER_CALIBRATION_GATE_20260624.md",
        notes="primary/secondary are canonical pass count / candidate count",
    )

    fieldnames = [
        "axis",
        "estimand",
        "status",
        "claim_scope",
        "primary_metric",
        "secondary_metric",
        "ci95",
        "tail_metric",
        "control_signal",
        "decision",
        "source_report",
        "notes",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    OUT_JSON.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True))

    lines = [
        "# LatentFM Scaling Evidence Table",
        "",
        "Status: `scaling_evidence_table_materialized_no_gpu`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of completed scaling gate reports.",
        "- Does not read canonical multi, held-out Track C query, checkpoints, train, infer, or use GPU.",
        "- This table indexes evidence; source reports remain authoritative.",
        "",
        "## Outputs",
        "",
        f"- CSV: `{OUT_CSV}`",
        f"- JSON: `{OUT_JSON}`",
        "",
        "## Evidence Rows",
        "",
        "| axis | estimand | status | claim scope | primary | secondary | tail/control | source |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        tail_control = row["tail_metric"] or row["control_signal"]
        lines.append(
            "| {axis} | {estimand} | `{status}` | `{claim_scope}` | {primary_metric} | {secondary_metric} | {tail} | `{source}` |".format(
                axis=row["axis"],
                estimand=row["estimand"],
                status=row["status"],
                claim_scope=row["claim_scope"],
                primary_metric=row["primary_metric"],
                secondary_metric=row["secondary_metric"],
                tail=tail_control,
                source=row["source_report"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Current deployable/default model remains `xverse_8k_anchor`.",
            "- The strongest positive scaling evidence is true-cell/cell-cap budget128 6k internal stability.",
            "- Canonical no-harm failed for budget128 6k, so scaling remains mechanism/training-data evidence unless a future frozen route passes no-harm.",
            "- Background/type/source, target observability, chemical, and no-harm-transfer rows are negative or diagnostic controls, not promotion evidence.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
