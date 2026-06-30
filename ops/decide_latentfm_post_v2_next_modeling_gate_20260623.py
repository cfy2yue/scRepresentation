#!/usr/bin/env python3
"""Decide whether any post-v2 modeling branch is ready for GPU launch.

This is a read-only decision helper. It does not launch jobs, read held-out
query for tuning, or authorize GPU work unless a fresh query-free CPU gate is
already present and unconsumed.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"

INPUTS = {
    "v2_reporting_package": REPORTS / "latentfm_trackc_support_context_v2_reporting_package_20260623.json",
    "v2_claim_readiness": REPORTS / "latentfm_trackc_support_context_v2_claim_readiness_audit_20260623.json",
    "post_v2_portfolio": REPORTS / "latentfm_post_support_context_v2_portfolio_decision_20260623.json",
    "next_candidate_review": REPORTS / "LATENTFM_HIGH_THROUGHPUT_NEXT_CANDIDATE_REVIEW_20260623_1045.md",
    "residual_operator_cpu_gate": REPORTS / "latentfm_trackc_residual_operator_cpu_gate_20260623.json",
    "residual_operator_gpu_gate": REPORTS / "latentfm_trackc_residual_operator_route_gap_gate_xverse_trackc_residual_operator_memall_resid_ep050_replay2_2k_seed42_retry1.json",
    "v2_residual_smoke": REPORTS / "latentfm_trackc_routed_distill_smoke_decision_xverse_trackc_support_context_v2_residual_ep050_replay2_2k_seed42.json",
    "v2_residual_uncapped": REPORTS / "latentfm_trackc_support_context_v2_uncapped_noharm_xverse_trackc_support_context_v2_residual_ep050_replay2_2k_seed42_20260623_decision.json",
    "archetype_multilatent_gate": REPORTS / "latentfm_soft_archetype_multilatent_state_cpu_gate_20260623.json",
    "tracka_fallback_summary": REPORTS / "LATENTFM_CROSSLATENT_TRACKA_GENE_RELIABILITY_ADAPTER_SUMMARY_20260623.md",
}

OUT_JSON = REPORTS / "latentfm_post_v2_next_modeling_gate_decision_20260623.json"
OUT_MD = REPORTS / "LATENTFM_POST_V2_NEXT_MODELING_GATE_DECISION_20260623.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def status_from_md(path: Path) -> str:
    if not path.exists():
        return "missing"
    for line in path.read_text(encoding="utf-8").splitlines()[:24]:
        if line.startswith("Status:"):
            if "`" in line:
                return line.split("`")[1]
            return line.replace("Status:", "").strip()
    return "present"


def dataset_summary(payload: dict[str, Any], dataset: str) -> dict[str, Any]:
    summary = payload.get("summary", {})
    if isinstance(summary, dict):
        return summary.get(dataset, {})
    if isinstance(summary, list):
        for row in summary:
            if isinstance(row, dict) and row.get("dataset") == dataset:
                return row
    return {}


def existing_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.suffix == ".json":
        payload = load_json(path)
        if payload.get("status") is not None:
            return str(payload["status"])
        if isinstance(payload.get("decision"), dict) and payload["decision"].get("status") is not None:
            return str(payload["decision"]["status"])
        return "present_json_no_status"
    return status_from_md(path)


def main() -> int:
    statuses = {name: existing_status(path) for name, path in INPUTS.items()}
    v2_reporting = load_json(INPUTS["v2_reporting_package"])
    v2_claim = load_json(INPUTS["v2_claim_readiness"])
    portfolio = load_json(INPUTS["post_v2_portfolio"])
    residual_cpu = load_json(INPUTS["residual_operator_cpu_gate"])
    residual_gpu = load_json(INPUTS["residual_operator_gpu_gate"])
    residual_uncapped = load_json(INPUTS["v2_residual_uncapped"])

    decisions = [
        {
            "branch": "support_context_v2_reporting",
            "decision": "continue_reporting_only",
            "reason": "v2 package is claim/figure/reporting ready and query is consumed final diagnostic evidence",
            "gpu_authorization": "none",
            "next_action": "build manuscript panels/captions from frozen figure manifest",
        },
        {
            "branch": "trackc_residual_operator_family",
            "decision": "consumed_as_v2_robustness_not_new_gpu_gate",
            "reason": (
                "residual-operator CPU gate passed earlier, but the corrected GPU retry failed route-gap closure; "
                "support-context v2 residual later passed capped/uncapped no-harm and is now recorded only as robustness with no second query"
            ),
            "gpu_authorization": "none",
            "next_action": "do not relaunch residual/operator, residual v2, endpoint, replay, memory-dose, or support-FiLM variants without a materially new CPU gate",
        },
        {
            "branch": "trackc_distinct_support_absorbability",
            "decision": "requires_new_cpu_protocol_before_gpu",
            "reason": "portfolio keeps this as backup, but no unconsumed fresh CPU gate exists after v2; mechanism must be distinct from support-FiLM/residual/context-c/endpoint/replay/memory-dose",
            "gpu_authorization": "only_after_new_cpu_gate_pass",
            "next_action": "design a support-set task adapter/gating CPU protocol that uses safe trainselect only and includes zero/shuffled controls plus wiring proof",
        },
        {
            "branch": "tracka_external_source_prior",
            "decision": "defer_until_external_feature_exists",
            "reason": "old lowcount/dataset-negative/Jiang-abstain/cross-latent families are closed; no genuinely new external/source feature is present in this decision",
            "gpu_authorization": "only_after_new_cpu_gate_pass",
            "next_action": "do not run Track A GPU from existing Jiang/cytokine evidence",
        },
        {
            "branch": "archetype_state_prior",
            "decision": "diagnostic_only",
            "reason": "hard/soft/conditional/orthogonal/multi-latent state gates failed no-harm or shuffled-control criteria",
            "gpu_authorization": "none",
            "next_action": "only a materially new continuous non-threshold CPU gate may reopen",
        },
    ]

    proposed_gate = {
        "name": "trackc_support_set_task_adapter_cpu_protocol",
        "status": "protocol_next_no_gpu_authorization",
        "hypothesis": (
            "A permutation-invariant support-set task adapter, selected by train_multi leave-one-task validation, "
            "may learn when and how to use support context without reducing to the consumed residual/FiLM/context-c families."
        ),
        "allowed_inputs": [
            "/data/cyx/1030/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json",
            "train_multi and support_val_multi rows from the safe trainselect split",
            "train_single/pert means only as anchor/background metadata",
            "closed-family support-val route-gap artifacts as baselines only",
        ],
        "forbidden_inputs": [
            "full v2 held-out query examples or query metrics",
            "canonical test_multi as selection signal",
            "post-v2 query result for route/checkpoint/feature tuning",
            "residual v2 query or any second query in the current v2 family",
        ],
        "mechanism_must_differ_from": [
            "support-FiLM shift/scale",
            "support_residual_adapter / residual operator",
            "context-c adapter",
            "endpoint/replay/memory-dose sweeps",
            "condition-only biological threshold gates",
        ],
        "minimum_gate": {
            "Wessels pp delta": ">= +0.02",
            "Wessels route-gap closure": ">= +0.05",
            "Norman pp delta": ">= -0.02",
            "support pp p_harm": "<= 0.20",
            "MMD hard harm": "none",
            "zero-support control": "must fail support gate",
            "shuffled-support control": "must fail support gate and lose Wessels closure",
            "wiring proof": "fixed-condition outputs change when support set changes",
        },
        "gpu_consequence_if_passed": "at most one capped support-only smoke after fresh resource audit and RUN_STATUS; no held-out query",
        "stop_rule": "if this protocol cannot define a mechanism distinct from consumed families or fails CPU gate, close Track C support absorbability backup for now",
    }

    failed_reasons: list[str] = []
    if statuses["v2_reporting_package"] != "support_context_v2_reporting_package_ready":
        failed_reasons.append("v2_reporting_package_not_ready")
    if statuses["v2_claim_readiness"] != "claim_ready_as_frozen_support_context_v2_diagnostic_not_formal_multi_solution":
        failed_reasons.append("v2_claim_readiness_not_ready")
    if statuses["post_v2_portfolio"] != "post_v2_portfolio_ready_reporting_plus_query_free_gates":
        failed_reasons.append("post_v2_portfolio_missing")
    if statuses["residual_operator_cpu_gate"] != "residual_operator_cpu_gate_pass_authorize_one_capped_gpu_smoke":
        failed_reasons.append("residual_cpu_gate_status_unexpected")
    if statuses["residual_operator_gpu_gate"] != "residual_route_gap_gate_fail_close_branch":
        failed_reasons.append("residual_gpu_route_gap_not_recorded_failed")
    if residual_uncapped.get("decision", {}).get("status") != "trackc_uncapped_canonical_noharm_pass_query_allowed_once":
        failed_reasons.append("v2_residual_uncapped_status_unexpected")

    status = "post_v2_no_gpu_new_cpu_protocol_required" if not failed_reasons else "post_v2_next_gate_decision_needs_review"
    payload = {
        "status": status,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "input_statuses": statuses,
        "failed_reasons": failed_reasons,
        "decisions": decisions,
        "proposed_next_gate": proposed_gate,
        "evidence_metrics": {
            "v2_query_pp": v2_reporting.get("key_metrics", {}).get("query_pp", {}).get("delta"),
            "v2_query_mmd": v2_reporting.get("key_metrics", {}).get("query_mmd", {}).get("delta"),
            "v2_unseen2_pp": v2_claim.get("metrics", {}).get("unseen2_pearson_delta"),
            "residual_cpu_wessels_delta": residual_cpu.get("decision", {}).get("wessels_delta_vs_route"),
            "residual_cpu_wessels_closure": residual_cpu.get("decision", {}).get("wessels_route_gap_closure"),
            "residual_gpu_wessels_closure": dataset_summary(residual_gpu, "Wessels").get("weighted_route_gap_closure"),
        },
        "rules": [
            "Do not use consumed held-out query for new branch selection.",
            "Do not relaunch consumed residual/FiLM/context-c/endpoint/replay/memory-dose variants.",
            "New GPU work requires a materially new query-free CPU gate and fresh resource audit.",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Post-V2 Next Modeling Gate Decision",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "## Summary",
        "",
        "No post-v2 branch currently has unconsumed GPU authorization. The support-context v2 package remains reporting/figure/provenance ready, while new modeling requires a materially new query-free CPU protocol.",
        "",
        "## Input Statuses",
        "",
        "| input | status | path |",
        "|---|---|---|",
    ]
    for name, path in INPUTS.items():
        lines.append(f"| `{name}` | `{statuses[name]}` | `{path}` |")
    lines.extend(["", "## Decisions", "", "| branch | decision | GPU | next action |", "|---|---|---|---|"])
    for d in decisions:
        lines.append(f"| `{d['branch']}` | `{d['decision']}` | `{d['gpu_authorization']}` | {d['next_action']} |")
    lines.extend(
        [
            "",
            "## Proposed Next CPU Gate",
            "",
            f"- name: `{proposed_gate['name']}`",
            f"- status: `{proposed_gate['status']}`",
            f"- hypothesis: {proposed_gate['hypothesis']}",
            "- minimum gate:",
        ]
    )
    for k, v in proposed_gate["minimum_gate"].items():
        lines.append(f"  - {k}: `{v}`")
    lines.extend(
        [
            "",
            "## Stop Rule",
            "",
            proposed_gate["stop_rule"],
            "",
            "## Boundary",
            "",
            "- This is a decision/protocol artifact only.",
            "- It does not launch or authorize GPU work.",
            "- It does not authorize held-out query evaluation.",
            "- Full v2 query and canonical multi remain forbidden for training-time selection.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 1 if failed_reasons else 0


if __name__ == "__main__":
    raise SystemExit(main())
