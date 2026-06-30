#!/usr/bin/env python3
"""Nested condition-count/exposure v2 gate for LatentFM scaling.

CPU-only synthesis gate. It reads completed scaling reports and the frozen S0
provenance table, then decides whether the condition-count/exposure scaling
branch has a non-duplicate, leakage-safe GPU candidate. It does not read model
checkpoints, canonical multi, held-out Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_scaling_nested_condition_exposure_v2_gate_20260625.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_NESTED_CONDITION_EXPOSURE_V2_GATE_20260625.md"

INPUTS = {
    "s0": REPORTS / "latentfm_scaling_s0_provenance_freeze_20260625.json",
    "s0_tsv": REPORTS / "latentfm_scaling_s0_provenance_freeze_20260625.tsv",
    "evidence_table": REPORTS / "latentfm_scaling_evidence_table_20260625.json",
    "count_smokes": REPORTS / "latentfm_xverse_scaling_count_smokes_decision_20260624.json",
    "protocol_matrix": REPORTS / "latentfm_scaling_protocol_matrix_decision_20260624.json",
    "highthroughput": REPORTS / "latentfm_scaling_highthroughput_smokes_decision_20260624.json",
    "seed_micro": REPORTS / "latentfm_scaling_seed_matched_micro_matrix_gate_20260624.json",
    "mixed_lodo": REPORTS / "latentfm_scaling_mixed_effect_lodo_condition_count_gate_20260624.json",
    "truecell_128_6k": REPORTS / "latentfm_true_cell_count_budget128_tail_stability_6k_decision_20260625.json",
    "truecell_128_noharm": REPORTS / "latentfm_true_cell_count_budget128_6k_canonical_noharm_decision_20260625.json",
    "allmod_family": REPORTS / "latentfm_allmodality_family_stratified_protocol_gate_20260625.json",
    "chemical_v2_ack": REPORTS / "LATENTFM_CHEMICAL_V2_LAUNCH_ACK_EXTERNAL_AUDIT_HERSCHEL_20260625.md",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"_missing": True, "_path": str(path)}
    with path.open() as f:
        return json.load(f)


def as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def fmt(value: Any) -> str:
    val = as_float(value)
    if val is not None:
        return f"{val:+.6f}"
    return "NA" if value is None else str(value)


def count_s0_rows(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"rows": 0, "datasets": 0, "source_verified": 0, "resolved": 0}
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, dialect="excel-tab")
        rows = list(reader)
    return {
        "rows": len(rows),
        "datasets": len({r.get("dataset") for r in rows if r.get("dataset")}),
        "source_verified": sum(1 for r in rows if r.get("source_quality") == "source_verified"),
        "resolved": sum(
            1
            for r in rows
            if r.get("scaling_claim_inclusion") == "s0_resolved_for_gene_or_nonchemical_axes"
        ),
        "chemical_unresolved": sum(
            1 for r in rows if "chemical_scaffold_unresolved" in (r.get("exclusion_reason") or "")
        ),
    }


def row_metric(row: dict[str, Any], *keys: str) -> Any:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def main() -> int:
    s0 = load_json(INPUTS["s0"])
    count = load_json(INPUTS["count_smokes"])
    protocol = load_json(INPUTS["protocol_matrix"])
    ht = load_json(INPUTS["highthroughput"])
    seed_micro = load_json(INPUTS["seed_micro"])
    mixed = load_json(INPUTS["mixed_lodo"])
    truecell = load_json(INPUTS["truecell_128_6k"])
    noharm = load_json(INPUTS["truecell_128_noharm"])
    allmod = load_json(INPUTS["allmod_family"])

    s0_summary = s0.get("summary") or {}
    if not s0_summary:
        s0_summary = count_s0_rows(INPUTS["s0_tsv"])

    count_rows = {r.get("run"): r for r in count.get("rows", [])}
    cap30 = count_rows.get("xverse_scaling_cap30_all_3k_seed42", {})
    cap120 = count_rows.get("xverse_scaling_cap120_all_3k_seed42", {})
    full = count_rows.get("xverse_scaling_full_trainonly_3k_seed42", {})
    cap120_minus_cap30 = row_metric(
        count.get("decision", {}), "gate_checks", "cap120_crossbg_pp_minus_cap30"
    )
    cap120_minus_anchor = row_metric(
        count.get("decision", {}), "gate_checks", "cap120_crossbg_pp_minus_anchor"
    )
    full_minus_cap120 = row_metric(
        count.get("full_extension_decision", {}), "gate_checks", "full_crossbg_pp_minus_cap120"
    )
    if full_minus_cap120 is None:
        full_minus_cap120 = None
        cap120_cross = as_float(row_metric(cap120, "metrics", "cross_pp_delta_vs_anchor"))
        full_cross = as_float(row_metric(full, "metrics", "cross_pp_delta_vs_anchor"))
        if cap120_cross is not None and full_cross is not None:
            full_minus_cap120 = full_cross - cap120_cross

    protocol_passed = (protocol.get("decision") or {}).get("passed") or []
    protocol_failed = (protocol.get("decision") or {}).get("failed") or []
    breadth_rows = [
        r
        for r in protocol.get("rows", [])
        if str(r.get("arm") or "").startswith("breadth_")
    ]
    breadth_cross_values = [
        as_float((r.get("metrics") or {}).get("cross_pp_delta_vs_anchor")) for r in breadth_rows
    ]
    breadth_all_negative = all(v is not None and v < 0.0 for v in breadth_cross_values)

    seed_rows = seed_micro.get("seed_rows") or []
    seed_sign_flip = bool((seed_micro.get("summary") or {}).get("cross_pp_sign_flip"))
    seed_fail_status = str(seed_micro.get("status") or "").endswith("fail_no_gpu")
    canonical_fail_rows = noharm.get("rows") or []
    canonical_failed_all = bool(canonical_fail_rows) and all(
        str(r.get("gate_status") or r.get("status") or "").startswith("candidate_gate_fail")
        or str(r.get("gate") or "").startswith("candidate_gate_fail")
        for r in canonical_fail_rows
    )
    truecell_budget_rows = (truecell.get("matrix_summary") or {}).get("budget_rows") or []
    truecell_best = truecell_budget_rows[0] if truecell_budget_rows else {}

    mixed_summary = mixed.get("summary") or {}
    mixed_ci = mixed_summary.get("bootstrap_dataset_mean_ci95") or mixed_summary.get("condition_weighted_bootstrap_ci95")
    mixed_dataset_min = mixed_summary.get("dataset_min_pp")
    if mixed_dataset_min is None:
        vals = [as_float(r.get("pp_delta_mean")) for r in mixed.get("dataset_rows", [])]
        vals = [v for v in vals if v is not None]
        mixed_dataset_min = min(vals) if vals else None
    mixed_negative_tails = mixed_summary.get("n_dataset_tails_lt_minus_0p020")
    if mixed_negative_tails is None:
        mixed_negative_tails = sum(
            1 for r in mixed.get("dataset_rows", []) if (as_float(r.get("pp_delta_mean")) or 0.0) < -0.02
        )
    leave_bg = mixed.get("leave_one_background") or []
    leave_type = mixed.get("leave_one_perturbation_type") or []
    min_leave_bg = min(
        (v for v in (as_float(r.get("pp_delta_mean")) for r in leave_bg) if v is not None),
        default=None,
    )
    min_leave_type = min(
        (v for v in (as_float(r.get("pp_delta_mean")) for r in leave_type) if v is not None),
        default=None,
    )

    allmod_status = allmod.get("status")
    allmod_passing = (allmod.get("decision") or {}).get("passing_policies")
    if allmod_passing is None:
        allmod_passing = 0 if str(allmod_status).endswith("fail_no_gpu") else None

    reasons: list[str] = []
    if (s0_summary.get("n_rows") or s0_summary.get("rows") or 0) <= 0:
        reasons.append("s0_provenance_missing")
    if as_float(cap120_minus_cap30) is None or as_float(cap120_minus_cap30) < 0.005:
        reasons.append("cap120_minus_cap30_below_materiality")
    if as_float(full_minus_cap120) is None or as_float(full_minus_cap120) < 0.0:
        reasons.append("full_exposure_not_better_than_moderate_exposure")
    if not protocol_passed:
        reasons.append("no_protocol_condition_count_arm_passed_internal")
    if breadth_all_negative:
        reasons.append("background_breadth_arms_negative")
    if seed_sign_flip or seed_fail_status:
        reasons.append("seed_matched_condition_count_sign_flip_or_fail")
    if canonical_failed_all:
        reasons.append("frozen_canonical_noharm_failed_all_truecell_budget128_seeds")
    if as_float(mixed_dataset_min) is None or as_float(mixed_dataset_min) < -0.02:
        reasons.append("mixed_lodo_dataset_tail_below_minus_0p020")
    if int(mixed_negative_tails or 0) > 0:
        reasons.append("mixed_lodo_negative_dataset_tails_present")
    if min_leave_bg is None or min_leave_bg < 0.0:
        reasons.append("leave_background_signal_not_positive")
    if min_leave_type is None or min_leave_type < 0.0:
        reasons.append("leave_type_signal_not_positive")
    if str(allmod_status).endswith("fail_no_gpu"):
        reasons.append("allmodality_family_stratified_gate_failed")

    gpu_authorized = False
    status = (
        "scaling_nested_condition_exposure_v2_pass_external_review_before_gpu"
        if not reasons
        else "scaling_nested_condition_exposure_v2_fail_no_gpu"
    )

    next_actions = [
        "do_not_launch_condition_count_or_background_breadth_gpu_from_current_evidence",
        "use S0 provenance for figure/failure-map package",
        "build a new CPU gate only if it adds a non-noop tail-protection mechanism or a source-resolved matched estimand",
        "chemical V2 remains the nearest GPU route but requires exact protocol ACK before launch",
    ]

    payload = {
        "status": status,
        "gpu_authorized": gpu_authorized,
        "boundary": {
            "cpu_only": True,
            "reads_completed_reports": True,
            "reads_model_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "inputs": {k: str(v) for k, v in INPUTS.items()},
        "summary": {
            "s0": s0_summary,
            "cap120_minus_cap30_cross_pp": cap120_minus_cap30,
            "cap120_minus_anchor_cross_pp": cap120_minus_anchor,
            "full_minus_cap120_cross_pp": full_minus_cap120,
            "protocol_passed_arms": protocol_passed,
            "protocol_failed_count": len(protocol_failed),
            "breadth_all_negative": breadth_all_negative,
            "seed_rows": seed_rows,
            "seed_sign_flip": seed_sign_flip,
            "truecell_budget128_6k_cross_pp_mean": truecell_best.get("cross_background_pp_delta_mean"),
            "truecell_budget128_6k_family_pp_mean": truecell_best.get("family_gene_pp_delta_mean"),
            "truecell_budget128_canonical_failed_all_seeds": canonical_failed_all,
            "mixed_lodo_ci95": mixed_ci,
            "mixed_lodo_dataset_min_pp": mixed_dataset_min,
            "mixed_lodo_negative_dataset_tails": mixed_negative_tails,
            "mixed_lodo_min_leave_background_pp": min_leave_bg,
            "mixed_lodo_min_leave_type_pp": min_leave_type,
            "allmodality_status": allmod_status,
            "allmodality_passing_policies": allmod_passing,
        },
        "reasons": reasons,
        "next_actions": next_actions,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# LatentFM Scaling Nested Condition/Exposure V2 Gate",
        "",
        f"Status: `{status}`",
        f"GPU authorized: `{gpu_authorized}`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of completed scaling reports and S0 provenance.",
        "- Does not read model checkpoints, canonical multi, Track C held-out query, train, infer, or use GPU.",
        "",
        "## Key Signals",
        "",
        "| signal | value |",
        "|---|---:|",
        f"| S0 rows / datasets / source-verified | `{s0_summary.get('n_rows') or s0_summary.get('rows')}` / `{s0_summary.get('n_datasets') or s0_summary.get('datasets')}` / `{s0_summary.get('n_source_verified') or s0_summary.get('source_verified')}` |",
        f"| cap120 - cap30 cross pp | `{fmt(cap120_minus_cap30)}` |",
        f"| cap120 - anchor cross pp | `{fmt(cap120_minus_anchor)}` |",
        f"| full - cap120 cross pp | `{fmt(full_minus_cap120)}` |",
        f"| protocol passed arms | `{protocol_passed}` |",
        f"| background breadth arms all negative | `{breadth_all_negative}` |",
        f"| seed sign flip | `{seed_sign_flip}` |",
        f"| true-cell budget128 6k cross/family pp | `{fmt(truecell_best.get('cross_background_pp_delta_mean'))}` / `{fmt(truecell_best.get('family_gene_pp_delta_mean'))}` |",
        f"| true-cell budget128 canonical no-harm failed all seeds | `{canonical_failed_all}` |",
        f"| mixed LODO dataset min / negative tails | `{fmt(mixed_dataset_min)}` / `{mixed_negative_tails}` |",
        f"| mixed LODO min leave-background/type pp | `{fmt(min_leave_bg)}` / `{fmt(min_leave_type)}` |",
        f"| all-modality family gate | `{allmod_status}` |",
        "",
        "## Decision",
        "",
        f"- reasons: `{reasons}`",
        f"- next actions: `{next_actions}`",
        "",
        "## Interpretation",
        "",
        "The current scaling evidence supports a mechanism/failure-map claim, not a new GPU training route.",
        "Moderate exposure and true-cell budget remain useful mainline design signals, but current condition-count/background/type expansion is seed-unstable, tail-unsafe, or vetoed by frozen no-harm.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
