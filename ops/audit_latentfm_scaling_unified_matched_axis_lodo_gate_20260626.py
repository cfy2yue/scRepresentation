#!/usr/bin/env python3
"""Build a unified scaling-axis gate from completed CPU reports.

This script is CPU/report-only. It does not read checkpoints, canonical multi,
Track C query outputs, expression matrices, or launch training/inference.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_CSV = REPORTS / "latentfm_scaling_unified_matched_axis_lodo_gate_20260626.csv"
OUT_JSON = REPORTS / "latentfm_scaling_unified_matched_axis_lodo_gate_20260626.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_UNIFIED_MATCHED_AXIS_LODO_GATE_20260626.md"


def load_json(path: str) -> dict:
    with (ROOT / path).open() as f:
        return json.load(f)


def fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(fmt(v) for v in value) + "]"
    return str(value)


def status(passable: bool, positive_hint: bool = False) -> str:
    if passable:
        return "pass_gpu_after_external_review"
    if positive_hint:
        return "mechanism_or_hint_fail_no_gpu"
    return "fail_no_gpu"


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M CST")

    truecell = load_json(
        "reports/latentfm_truecell_scaling_count_tail_completion_gate_20260625.json"
    )
    exposure = load_json(
        "reports/latentfm_scaling_nested_condition_exposure_v2_gate_20260625.json"
    )
    source = load_json(
        "reports/latentfm_scaling_source_resolved_estimand_v2_gate_20260625.json"
    )
    target = load_json(
        "reports/latentfm_target_observability_residual_v3_gate_20260626.json"
    )
    reagent = load_json(
        "reports/latentfm_reagent_read_support_source_block_lodo_gate_20260626.json"
    )

    tc_rows = truecell["rows"]
    tc_best = max(tc_rows, key=lambda r: r.get("cross_pp_mean", -999))
    tc_noharm_failed = bool(truecell["canonical_noharm"].get("all_failed"))
    exp_summary = exposure["summary"]
    src_summary = source["summary"]
    tgt = target["best_policy"]
    reagent_row = reagent["artifact_rows"][0]

    rows = [
        {
            "axis": "true_cell_per_condition_support",
            "evidence": (
                "best budget="
                f"{tc_best.get('budget')} steps={tc_best.get('steps')}; "
                f"cross_pp={fmt(tc_best.get('cross_pp_mean'))}; "
                f"family_pp={fmt(tc_best.get('family_pp_mean'))}; "
                f"canonical_noharm_all_failed={tc_noharm_failed}"
            ),
            "matched_or_lodo_control": truecell["controls"].get("nested_3k_status"),
            "tail_or_noharm": "canonical no-harm failed all 3 seeds",
            "positive_hint": True,
            "gate_status": status(False, True),
            "gpu_authorized": False,
            "next_gpu_trigger": (
                "new non-noop tail-protection policy with nonzero canonical "
                "footprint, internal gain, safe MMD/tails, and frozen canonical "
                "no-harm pass"
            ),
        },
        {
            "axis": "condition_exposure_count",
            "evidence": (
                f"cap120-cap30={fmt(exp_summary.get('cap120_minus_cap30_cross_pp'))}; "
                f"full-cap120={fmt(exp_summary.get('full_minus_cap120_cross_pp'))}; "
                f"dataset_min={fmt(exp_summary.get('mixed_lodo_dataset_min_pp'))}"
            ),
            "matched_or_lodo_control": (
                "mixed LODO; leave-background/type mins "
                f"{fmt(exp_summary.get('mixed_lodo_min_leave_background_pp'))}/"
                f"{fmt(exp_summary.get('mixed_lodo_min_leave_type_pp'))}"
            ),
            "tail_or_noharm": (
                f"negative dataset tails={exp_summary.get('mixed_lodo_negative_dataset_tails')}"
            ),
            "positive_hint": True,
            "gate_status": status(False, True),
            "gpu_authorized": False,
            "next_gpu_trigger": (
                "predeclared nested exposure subset with CI lower > 0, source/"
                "background/type matched controls, and no dataset-tail veto"
            ),
        },
        {
            "axis": "background_source_type_breadth",
            "evidence": (
                f"pp_mean={fmt(src_summary.get('pp_delta_mean'))}; "
                f"dataset_min={fmt(src_summary.get('dataset_min_pp'))}; "
                f"min_background={fmt(src_summary.get('min_background_pp'))}; "
                f"min_type={fmt(src_summary.get('min_type_pp'))}"
            ),
            "matched_or_lodo_control": (
                f"bootstrap_ci={fmt(src_summary.get('bootstrap', {}).get('ci95'))}; "
                f"merged_datasets={src_summary.get('merged_datasets')}"
            ),
            "tail_or_noharm": (
                f"negative tails < -0.02={src_summary.get('negative_tails_lt_minus_0p020')}"
            ),
            "positive_hint": False,
            "gate_status": status(False),
            "gpu_authorized": False,
            "next_gpu_trigger": (
                "source-verified crossed background/type estimand beating "
                "dataset-ID/shuffled-label controls with safe tails"
            ),
        },
        {
            "axis": "target_observability_actionability",
            "evidence": (
                f"policy={tgt.get('policy')}; residual_pp={fmt(tgt.get('residual_pp_mean'))}; "
                f"residual_tail={fmt(tgt.get('dataset_min_residual_pp'))}; "
                f"mmd_max={fmt(tgt.get('mmd_max'))}"
            ),
            "matched_or_lodo_control": (
                f"dataset_bootstrap_ci={fmt(tgt.get('dataset_bootstrap_residual_ci95'))}"
            ),
            "tail_or_noharm": (
                f"hard_harm_frac={fmt(tgt.get('hard_harm_frac'))}; "
                + ",".join(tgt.get("reasons", []))
            ),
            "positive_hint": True,
            "gate_status": status(False, True),
            "gpu_authorized": False,
            "next_gpu_trigger": (
                "target/reliability artifact with permutation p<=0.01, "
                "dataset min >= -0.02, MMD <= 0.001, and hard-harm <= 0.20"
            ),
        },
        {
            "axis": "external_reagent_read_support",
            "evidence": (
                f"observed_high_low={fmt(reagent_row.get('observed_high_low'))}; "
                f"spearman={fmt(reagent_row.get('observed_spearman'))}; "
                f"source_boot_lower={fmt(reagent_row.get('source_block_bootstrap_lower'))}; "
                f"shuffle_p={fmt(reagent_row.get('within_dataset_shuffle_p'))}"
            ),
            "matched_or_lodo_control": (
                f"datasets={len(reagent_row.get('datasets', []))}; "
                f"LODO rows={len(reagent_row.get('lodo_rows', []))}"
            ),
            "tail_or_noharm": ",".join(reagent_row.get("reasons", [])),
            "positive_hint": True,
            "gate_status": status(False, True),
            "gpu_authorized": False,
            "next_gpu_trigger": (
                "new external reliability family distinct from read/QC/source/"
                "guide count with within-dataset shuffle pass and safe MMD/tails"
            ),
        },
        {
            "axis": "perturbation_type_allmodality",
            "evidence": (
                f"allmodality_passing_policies="
                f"{exp_summary.get('allmodality_passing_policies')}; "
                f"status={exp_summary.get('allmodality_status')}"
            ),
            "matched_or_lodo_control": "family-stratified policies consumed by prior gates",
            "tail_or_noharm": "gene/drug family tradeoff and hard-harm gates fail",
            "positive_hint": False,
            "gate_status": status(False),
            "gpu_authorized": False,
            "next_gpu_trigger": (
                "new molecule/type mechanism with simultaneous gene/drug family "
                "benefit and shuffled/type controls collapsed"
            ),
        },
        {
            "axis": "ot_minibatch_pairs",
            "evidence": "prior OT quality/comparison gates did not unlock no-harm",
            "matched_or_lodo_control": "not a scaling law axis without pair-quality artifact",
            "tail_or_noharm": "closed default-off for scaling claims",
            "positive_hint": False,
            "gate_status": status(False),
            "gpu_authorized": False,
            "next_gpu_trigger": (
                "new pair-quality feature predicts failed tails and defines a "
                "bounded non-duplicate route"
            ),
        },
        {
            "axis": "chemical_scaffold_semantics",
            "evidence": "V2 fixed-step launcher is protocol-safe but exact-ACK gated",
            "matched_or_lodo_control": "real Morgan512 seeds before shuffled/random controls",
            "tail_or_noharm": "not a non-ACK scaling route",
            "positive_hint": True,
            "gate_status": "ack_required_no_gpu_from_this_gate",
            "gpu_authorized": False,
            "next_gpu_trigger": (
                "exact chemical V2 ACK plus fresh GPU/CPU/RAM audit; launch "
                "real Morgan512 seed43/44 before controls"
            ),
        },
    ]

    immediate = [r for r in rows if r["gpu_authorized"]]
    result = {
        "timestamp": timestamp,
        "status": "scaling_unified_matched_axis_lodo_gate_no_immediate_gpu",
        "default_model": "xverse_8k_anchor",
        "gpu_authorized": bool(immediate),
        "immediate_gpu_candidate_count": len(immediate),
        "boundary": {
            "cpu_only": True,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "trains_or_infers": False,
            "uses_gpu": False,
        },
        "rows": rows,
    }

    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# LatentFM Scaling Unified Matched Axis LODO Gate",
        "",
        f"Timestamp: `{timestamp}`",
        "",
        "Status: `scaling_unified_matched_axis_lodo_gate_no_immediate_gpu`",
        "",
        "Default/deployable model: `xverse_8k_anchor`",
        "",
        f"Immediate GPU candidate count: `{len(immediate)}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only unified gate over completed scaling reports.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "",
        "## Unified Gate Rows",
        "",
        "| axis | gate status | evidence | matched/LODO/control | tail/no-harm blocker | next GPU trigger |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| `{axis}` | `{gate_status}` | {evidence} | {matched_or_lodo_control} | "
            "{tail_or_noharm} | {next_gpu_trigger} |".format(**r)
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- No scaling axis currently authorizes a non-ACK GPU launch.",
            "- The strongest useful signal remains true-cell/per-condition support, but it is blocked by frozen canonical no-harm.",
            "- If a new external artifact or non-noop true-cell repair passes this style of CPU gate, immediately launch bounded GPU smokes under the current temporary cap.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- CSV: `{OUT_CSV}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
