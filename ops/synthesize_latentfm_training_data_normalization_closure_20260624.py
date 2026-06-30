#!/usr/bin/env python3
"""Synthesize closure status for LatentFM training-data/normalization ideas."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"


def load_json(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def decision_status(obj: dict[str, Any]) -> str:
    decision = obj.get("decision")
    if isinstance(decision, dict):
        status = decision.get("status") or decision.get("overall_status")
        if status:
            return str(status)
    return str(obj.get("status", "unknown"))


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def main() -> None:
    scaling_strategy = load_json(
        "reports/latentfm_scaling_training_data_strategy_decision_20260624.json"
    )
    scaling_noharm = load_json(
        "reports/latentfm_scaling_protocol_canonical_noharm_decision_20260624.json"
    )
    soft_noharm = load_json(
        "reports/latentfm_xverse_soft_exposure_canonical_noharm_decision_20260624.json"
    )
    reliability = load_json(
        "reports/latentfm_trainonly_reliability_condition_gate_20260624.json"
    )
    response_norm = load_json(
        "reports/latentfm_xverse_scaling_cap120_response_normalization_gate_20260624.json"
    )
    whitening = load_json("reports/latentfm_xverse_control_whitening_gate_20260624.json")
    nuisance = load_json("reports/latentfm_xverse_nuisance_residual_gate_20260624.json")
    distributional = load_json("reports/latentfm_distributional_mmd_harm_gate_20260624.json")
    ot_synthesis = (REPORTS / "LATENTFM_OT_MINIBATCH_PAIRING_SYNTHESIS_20260624.md").read_text(
        encoding="utf-8"
    )

    response_decisions = response_norm.get("decision", {}).get("decisions", [])
    ds_scale = next(
        (row for row in response_decisions if row.get("mode") == "dataset_scale_ridge"),
        {},
    )
    ds_scale_cross = ds_scale.get("cross_background_delta", {})
    ds_scale_family = ds_scale.get("family_delta", {})

    reliability_rows = reliability.get("decision", {}).get("run_decisions", [])
    cap60_rel = next((row for row in reliability_rows if row.get("run") == "cap60_protocol"), {})
    cap120_rel = next((row for row in reliability_rows if row.get("run") == "cap120"), {})

    branches = [
        {
            "branch": "condition-count / dataset-breadth scaling",
            "status": "closed_for_promotion_diagnostic_for_scaling_story",
            "evidence": "reports/LATENTFM_SCALING_TRAINING_DATA_STRATEGY_DECISION_20260624.md",
            "decisive_result": (
                "cap60/protocol internal signal existed, but frozen canonical no-harm "
                f"closed the promoted path: {decision_status(scaling_noharm)}"
            ),
            "gpu_authorized": False,
            "why_not_next": "A repeat cap/breadth matrix would retest an already failed canonical no-harm branch.",
        },
        {
            "branch": "soft exposure / mild sampling smoothing",
            "status": "closed_after_canonical_noharm",
            "evidence": "reports/LATENTFM_XVERSE_SOFT_EXPOSURE_CANONICAL_NOHARM_DECISION_20260624.md",
            "decisive_result": f"frozen no-harm status {decision_status(soft_noharm)}",
            "gpu_authorized": False,
            "why_not_next": "No-hardcap exposure was the strongest simple sampling variant but failed post-freeze no-harm.",
        },
        {
            "branch": "reliability-weighted robust loss",
            "status": "closed_current_signal",
            "evidence": "reports/LATENTFM_TRAINONLY_RELIABILITY_CONDITION_GATE_20260624.md",
            "decisive_result": (
                "cap60 cross/family pp "
                f"{fmt(cap60_rel.get('cross_mean_pp_delta'))}/{fmt(cap60_rel.get('family_mean_pp_delta'))}; "
                f"cap120 {fmt(cap120_rel.get('cross_mean_pp_delta'))}/{fmt(cap120_rel.get('family_mean_pp_delta'))}; "
                f"status {decision_status(reliability)}"
            ),
            "gpu_authorized": False,
            "why_not_next": "Strict condition-level gate missed +0.010 and failed dataset-min/control requirements.",
        },
        {
            "branch": "response normalization / dataset-scale PCA",
            "status": "diagnostic_only_closed_for_gpu",
            "evidence": "reports/LATENTFM_XVERSE_SCALING_CAP120_RESPONSE_NORMALIZATION_GATE_20260624.md",
            "decisive_result": (
                "dataset-scale cross/family deltas vs raw "
                f"{fmt(ds_scale_cross.get('delta'))}/{fmt(ds_scale_family.get('delta'))}; "
                f"CI {ds_scale_cross.get('ci95')} and status {ds_scale.get('status')}"
            ),
            "gpu_authorized": False,
            "why_not_next": "Average signal is not supported by CI/LODO stability and has family harm risk.",
        },
        {
            "branch": "control/background whitening",
            "status": "closed_no_gpu",
            "evidence": "reports/LATENTFM_XVERSE_CONTROL_WHITENING_GATE_20260624.md",
            "decisive_result": (
                f"raw MMD/residual Spearman {fmt(whitening.get('metrics', {}).get('raw_mmd_residual_spearman'))}; "
                f"whitened {fmt(whitening.get('metrics', {}).get('whitened_mmd_residual_spearman'))}; "
                f"status {decision_status(whitening)}"
            ),
            "gpu_authorized": False,
            "why_not_next": "Whitening did not improve alignment with MMD harm over raw residual geometry.",
        },
        {
            "branch": "nuisance-invariant residual routing/loss",
            "status": "closed_no_gpu",
            "evidence": "reports/LATENTFM_XVERSE_NUISANCE_RESIDUAL_GATE_20260624.md",
            "decisive_result": f"status {decision_status(nuisance)}",
            "gpu_authorized": False,
            "why_not_next": "Dataset residual signal exists but is not aligned with MMD harm.",
        },
        {
            "branch": "distributional MMD-risk weighting",
            "status": "closed_no_gpu",
            "evidence": "reports/LATENTFM_DISTRIBUTIONAL_MMD_HARM_GATE_20260624.md",
            "decisive_result": (
                f"cross/family pp {fmt(distributional.get('decision', {}).get('cross_mean_pp_delta'))}/"
                f"{fmt(distributional.get('decision', {}).get('family_mean_pp_delta'))}; "
                f"family MMD {fmt(distributional.get('decision', {}).get('family_mean_mmd_delta'))}; "
                f"status {decision_status(distributional)}"
            ),
            "gpu_authorized": False,
            "why_not_next": "MMD-risk features did not retain pp and controls did not collapse.",
        },
        {
            "branch": "OT minibatch pair weighting/coupling",
            "status": "closed_no_gpu",
            "evidence": "reports/LATENTFM_OT_MINIBATCH_PAIRING_SYNTHESIS_20260624.md",
            "decisive_result": "OT is wired and changes coupling, but random/no-OT and Hungarian one-to-one smokes failed Track A gates.",
            "gpu_authorized": False,
            "why_not_next": "A new OT run needs a condition-level pairing-quality-to-response gate, not another pair-mode sweep.",
            "report_mentions_current_status": "ot_wired_but_no_model_gain_close_current_ot_redesign"
            in ot_synthesis,
        },
    ]

    next_gate_standard = {
        "gpu_authorized_now": False,
        "allowed_next_action": "new_train_only_gate_only_if_mechanism_is_materially_new",
        "must_not_duplicate": [
            "condition-count cap/breadth matrix",
            "hard type/background balancing",
            "simple inverse-frequency sampler or ds_loss sweep",
            "current reliability/response_norm/whitening/MMD-risk proxies",
            "OT pair-mode or cost-function sweep",
            "canonical or held-out Track C query selection",
        ],
        "minimum_gate": [
            "train-only/query-blind inputs with artifact provenance",
            "nested leave-one-dataset or stricter validation",
            "cross-background and family pp gain >= +0.010",
            "dataset-min >= -0.020 and family MMD no-harm",
            "bootstrap/CI and p_harm reported",
            "negative controls collapse: shuffled, inverted/sign, count-only or dataset-identity",
            "predeclared capped GPU smoke and fail-close rule only if gate passes",
        ],
    }

    out = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "training_data_normalization_closure_no_gpu",
        "boundary": {
            "reads_completed_reports_only": True,
            "active_logs": False,
            "raw_canonical_or_query": False,
            "canonical_multi_selection": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "current_best": scaling_strategy.get("decision", {}).get(
            "current_best_deployable_model", "xverse_8k_anchor"
        ),
        "branches": branches,
        "next_gate_standard": next_gate_standard,
        "decision": {
            "gpu_authorized": False,
            "launch_experiments_now": False,
            "reason": "All user-raised training-data/normalization/weighted-loss/OT axes have either failed strict gates or remain diagnostic-only under current evidence.",
        },
    }

    json_path = REPORTS / "latentfm_training_data_normalization_closure_20260624.json"
    md_path = REPORTS / "LATENTFM_TRAINING_DATA_NORMALIZATION_CLOSURE_20260624.md"
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Training-Data / Normalization Closure",
        "",
        "Status: `training_data_normalization_closure_no_gpu`",
        "",
        "## Boundary",
        "",
        "- Reads completed reports only.",
        "- Does not read active logs, raw canonical/query artifacts, use canonical multi for selection, train, infer, or use GPU.",
        "",
        "## Decision",
        "",
        f"- Current deployable/default remains `{out['current_best']}`.",
        "- GPU authorized now: `False`.",
        "- Launch experiments now: `False`.",
        "- Reason: all currently documented training-data, normalization, weighted-loss, and OT axes have failed strict gates or are diagnostic-only.",
        "",
        "## Branch Status",
        "",
        "| Branch | Status | Decisive result | Why not next | Evidence |",
        "|---|---|---|---|---|",
    ]
    for row in branches:
        lines.append(
            f"| {row['branch']} | `{row['status']}` | {row['decisive_result']} | "
            f"{row['why_not_next']} | `{ROOT / row['evidence']}` |"
        )
    lines.extend(
        [
            "",
            "## Next Valid Gate Standard",
            "",
            "A new branch is allowed only if it is materially new and train-only/query-blind.",
            "",
            "Must not duplicate:",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in next_gate_standard["must_not_duplicate"])
    lines.extend(["", "Minimum gate:", ""])
    lines.extend(f"- {item}" for item in next_gate_standard["minimum_gate"])
    lines.extend(["", "## JSON", "", f"`{json_path}`", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
