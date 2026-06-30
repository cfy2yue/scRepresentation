#!/usr/bin/env python3
"""Nature-Methods-style evidence matrix gate for LatentFM scaling.

CPU/report-only synthesis. It reads completed reports and gates, but does not
launch experiments, read active logs, use canonical multi for selection, or read
Track C query artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_scaling_law_evidence_matrix_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_LAW_EVIDENCE_MATRIX_GATE_20260624.md"


def load_json(name: str) -> dict[str, Any]:
    path = REPORTS / name
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def status(payload: dict[str, Any]) -> str:
    decision = payload.get("decision")
    if isinstance(decision, dict):
        return str(decision.get("status") or decision.get("overall_status") or payload.get("status") or "unknown")
    return str(payload.get("status") or ("missing" if payload.get("missing") else "unknown"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def row_by_arm(payload: dict[str, Any], arm: str) -> dict[str, Any]:
    for row in payload.get("rows") or []:
        if row.get("arm") == arm or row.get("role") == arm or row.get("name") == arm or row.get("run") == arm:
            return row
    return {}


def count_metrics(count_payload: dict[str, Any]) -> dict[str, Any]:
    checks = ((count_payload.get("decision") or {}).get("gate_checks") or {})
    full = ((count_payload.get("full_extension_decision") or {}).get("gate_checks") or {})
    type_bal = ((count_payload.get("type_balance_extension_decision") or {}).get("gate_checks") or {})
    return {
        "cap120_minus_cap30": checks.get("cap120_crossbg_pp_minus_cap30"),
        "cap120_minus_anchor": checks.get("cap120_crossbg_pp_minus_anchor"),
        "cap120_family_minus_anchor": checks.get("cap120_family_pp_minus_anchor"),
        "cap120_family_mmd": checks.get("cap120_family_mmd_minus_anchor"),
        "full_minus_cap120": full.get("full_crossbg_pp_minus_cap120"),
        "type_bal_minus_cap120": type_bal.get("type_balanced_crossbg_pp_minus_cap120"),
    }


def main() -> int:
    count = load_json("latentfm_xverse_scaling_count_smokes_decision_20260624.json")
    count_canon = load_json("latentfm_xverse_scaling_canonical_noharm_decision_20260624.json")
    matrix = load_json("latentfm_scaling_protocol_matrix_decision_20260624.json")
    high = load_json("latentfm_scaling_highthroughput_smokes_decision_20260624.json")
    high_canon = load_json("latentfm_scaling_highthroughput_canonical_noharm_decision_20260624.json")
    target = load_json("latentfm_scaling_target_gene_coverage_protocol_gate_20260624.json")
    strata = load_json("latentfm_scaling_source_verified_background_type_strata_gate_20260624.json")
    noharm = load_json("latentfm_scaling_noharm_stabilizer_design_gate_20260624.json")
    matched = load_json("latentfm_matched_dataset_breadth_gate_20260624.json")
    randomcount = load_json("latentfm_randomcount_mmd_preservation_gate_20260624.json")
    pathway = load_json("latentfm_modality_pathway_mmd_preservation_smoke_decision_20260624.json")
    training = load_json("latentfm_training_data_normalization_closure_20260624.json")

    cm = count_metrics(count)
    seed43 = row_by_arm(high, "seed_robustness")
    pathway_row = row_by_arm(pathway, "pathway_mmd_preservation")

    axes = [
        {
            "axis": "condition_count_moderate_exposure",
            "support": "narrow_train_only_internal",
            "claim_scope": "diagnostic mechanism only",
            "key_evidence": (
                f"cap120-cap30 pp {fmt(cm['cap120_minus_cap30'])}; "
                f"cap120-anchor pp/family/MMD {fmt(cm['cap120_minus_anchor'])}/"
                f"{fmt(cm['cap120_family_minus_anchor'])}/{fmt(cm['cap120_family_mmd'])}; "
                f"protocol cap60 internal pass {status(matrix)}"
            ),
            "blocker": "seed sensitivity and canonical no-harm failure",
            "gpu_authorized": False,
        },
        {
            "axis": "monotonic_more_data_full_train",
            "support": "negative",
            "claim_scope": "negative evidence",
            "key_evidence": f"full minus cap120 cross pp {fmt(cm['full_minus_cap120'])}",
            "blocker": "full train-only does not improve over moderate cap",
            "gpu_authorized": False,
        },
        {
            "axis": "dataset_background_breadth",
            "support": "negative_or_confounded",
            "claim_scope": "diagnostic/negative",
            "key_evidence": f"matched breadth status {status(matched)}; protocol matrix status {status(matrix)}",
            "blocker": "matched breadth arms fail; many-shallow underperforms few-deep",
            "gpu_authorized": False,
        },
        {
            "axis": "target_gene_coverage",
            "support": "negative_or_diagnostic",
            "claim_scope": "diagnostic only",
            "key_evidence": (
                f"coverage gate {status(target)}; cap120-cap30 pp "
                f"{fmt((target.get('comparisons') or {}).get('cap120_minus_cap30', {}).get('pp_delta_mean'))}; "
                f"rho {fmt((target.get('comparisons') or {}).get('cap120_minus_cap30', {}).get('spearman_pp_vs_coverage_gain_dataset_count'))}"
            ),
            "blocker": "coverage gain not predictive and dataset-tail harm",
            "gpu_authorized": False,
        },
        {
            "axis": "perturbation_type_source_background",
            "support": "negative_or_confounded",
            "claim_scope": "diagnostic only",
            "key_evidence": f"source strata gate {status(strata)}; type-balanced minus cap120 {fmt(cm['type_bal_minus_cap120'])}",
            "blocker": "background/type strata have negative tails and source metadata are dataset-confounded",
            "gpu_authorized": False,
        },
        {
            "axis": "pathway_modality_composition",
            "support": "mixed",
            "claim_scope": "composition diagnostic",
            "key_evidence": (
                f"pathway MMD-preserve status {status(pathway)}; cross pp "
                f"{fmt((pathway_row.get('metrics') or {}).get('cross_pp_delta_vs_anchor'))}; "
                f"family MMD {fmt((pathway_row.get('metrics') or {}).get('family_gene_mmd_delta_vs_anchor'))}; "
                f"randomcount MMD gate {status(randomcount)}"
            ),
            "blocker": "MMD-safe pathway composition is cross-weak; randomcount is MMD-unsafe",
            "gpu_authorized": False,
        },
        {
            "axis": "seed_step_robustness",
            "support": "negative_for_promotion",
            "claim_scope": "stability warning",
            "key_evidence": (
                f"seed43 cross/family/MMD {fmt((seed43.get('metrics') or {}).get('cross_pp_delta_vs_anchor'))}/"
                f"{fmt((seed43.get('metrics') or {}).get('family_gene_pp_delta_vs_anchor'))}/"
                f"{fmt((seed43.get('metrics') or {}).get('family_gene_mmd_delta_vs_anchor'))}"
            ),
            "blocker": "positive seed42 cap60 signal flips negative at seed43",
            "gpu_authorized": False,
        },
        {
            "axis": "noharm_transfer_to_canonical",
            "support": "negative",
            "claim_scope": "veto/failure analysis",
            "key_evidence": (
                f"cap120 canonical {status(count_canon)}; cap60/replay canonical {status(high_canon)}; "
                f"stabilizer gate {status(noharm)}"
            ),
            "blocker": "internal passes repeatedly fail frozen canonical single/family no-harm",
            "gpu_authorized": False,
        },
        {
            "axis": "training_set_strategy_for_mainline",
            "support": "conservative_feedback",
            "claim_scope": "method-design guidance",
            "key_evidence": f"training closure {status(training)}; current default {training.get('current_best', 'xverse_8k_anchor')}",
            "blocker": "simple sampling/normalization/weighted-loss/OT variants closed",
            "gpu_authorized": False,
        },
    ]

    gpu_authorized = any(row["gpu_authorized"] for row in axes)
    status_value = "scaling_law_evidence_matrix_ready_no_gpu"
    payload = {
        "status": status_value,
        "gpu_authorized": gpu_authorized,
        "boundary": {
            "reads_completed_reports_only": True,
            "canonical_noharm_used_as_veto_context": True,
            "canonical_multi_selection": False,
            "trackc_query_read": False,
            "training_or_inference": False,
            "active_logs_read": False,
            "gpu": False,
        },
        "overall_conclusion": {
            "supported_claim": "moderate exposure/condition-count train-only internal diagnostic signal",
            "unsupported_claims": [
                "Nature Methods-level monotonic scaling law",
                "source-verified background breadth scaling",
                "target/gene coverage causal scaling",
                "perturbation-type scaling",
                "deployable scaling model improvement",
            ],
            "shortest_path_to_gpu": "no-harm transfer calibration or seed-matched micro-matrix CPU protocol must pass before one bounded smoke",
        },
        "axes": axes,
        "recommended_next_gates": [
            {
                "name": "noharm_transfer_calibration_gate",
                "priority": 1,
                "hypothesis": "train-only internal strata can predict frozen canonical no-harm failure without using canonical for new selection",
                "promotion": "leave-run-family-out classifier/regression predicts canonical pp/MMD harm and identifies a no-harm-safe train-only surrogate",
                "fail_close": "if internal pass cannot predict no-harm risk, scaling-to-promotion is paused",
            },
            {
                "name": "seed_matched_condition_count_micro_matrix_protocol",
                "priority": 2,
                "hypothesis": "moderate exposure effect is real only if same split/steps and >=3 seeds do not flip sign",
                "promotion": "cross/family pp >= +0.010, family MMD <= +0.001, dataset-min >= -0.020, no seed sign flip",
                "fail_close": "any seed43-like negative run closes count scaling as promotion path",
            },
            {
                "name": "matched_background_type_confound_simulation",
                "priority": 3,
                "hypothesis": "background/type scaling can be separated from dataset identity by source-verified matched subsets",
                "promotion": "shuffled dataset/type controls collapse while real matched subsets retain pp gain with safe tails",
                "fail_close": "if controls do not collapse or tails remain negative, keep background/type as diagnostic only",
            },
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Scaling-Law Evidence Matrix Gate",
        "",
        f"Status: `{status_value}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only synthesis of completed reports.",
        "- Canonical no-harm is used only as veto/context, not as checkpoint selection.",
        "- Does not read canonical multi, Track C query, active logs, train, infer, or use GPU.",
        "",
        "## Overall Conclusion",
        "",
        f"- Supported claim: `{payload['overall_conclusion']['supported_claim']}`.",
        "- Unsupported claims: monotonic scaling law, source-verified background scaling, target/gene coverage scaling, perturbation-type scaling, deployable scaling improvement.",
        f"- GPU authorized: `{gpu_authorized}`.",
        "",
        "## Evidence Matrix",
        "",
        "| axis | support | claim scope | key evidence | blocker | GPU |",
        "|---|---|---|---|---|---|",
    ]
    for row in axes:
        lines.append(
            f"| `{row['axis']}` | `{row['support']}` | {row['claim_scope']} | "
            f"{row['key_evidence']} | {row['blocker']} | `{row['gpu_authorized']}` |"
        )
    lines.extend(["", "## Next Gates", ""])
    for gate in payload["recommended_next_gates"]:
        lines.extend(
            [
                f"### {gate['priority']}. {gate['name']}",
                "",
                f"- hypothesis: {gate['hypothesis']}",
                f"- promotion: {gate['promotion']}",
                f"- fail-close: {gate['fail_close']}",
                "",
            ]
        )
    lines.extend(["## JSON", "", f"`{OUT_JSON}`", ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status_value, "out_md": str(OUT_MD), "gpu_authorized": gpu_authorized}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
