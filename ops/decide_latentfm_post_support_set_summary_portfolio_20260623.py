#!/usr/bin/env python3
"""Portfolio decision after support-set summary gate failure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_post_support_set_summary_portfolio_decision_20260623.json"
OUT_MD = REPORTS / "LATENTFM_POST_SUPPORT_SET_SUMMARY_PORTFOLIO_DECISION_20260623.md"

INPUTS = {
    "support_set_summary": REPORTS / "latentfm_trackc_support_set_task_summary_gate_20260623.json",
    "support_set_after_summary": REPORTS / "latentfm_trackc_support_set_task_after_summary_decision_20260623.json",
    "support_set_mmd_protocol": REPORTS / "latentfm_trackc_support_set_task_mmd_noharm_protocol_20260623.json",
    "support_context_v2_reporting": REPORTS / "latentfm_trackc_support_context_v2_reporting_package_20260623.json",
    "support_context_v2_claim": REPORTS / "latentfm_trackc_support_context_v2_claim_readiness_audit_20260623.json",
    "learned_anchor_gate": REPORTS / "latentfm_trackc_learned_anchor_gate_cpu_gate_20260623.json",
    "archetype_triage": REPORTS / "latentfm_soft_archetype_pocket_reopen_triage_20260623.json",
    "post_learned_gate": REPORTS / "latentfm_post_learned_gate_portfolio_decision_20260623.json",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def status_of(obj: dict[str, Any]) -> str | None:
    if "status" in obj:
        return str(obj["status"])
    decision = obj.get("decision")
    if isinstance(decision, dict) and decision.get("status") is not None:
        return str(decision["status"])
    return None


def build_payload() -> dict[str, Any]:
    loaded = {name: load_json(path) for name, path in INPUTS.items() if path.is_file()}
    statuses = {name: status_of(obj) for name, obj in loaded.items()}
    missing = sorted(set(INPUTS) - set(loaded))
    decisions = [
        {
            "branch": "Track C support-context v2 frozen diagnostic",
            "decision": "keep_as_current_best_reporting_candidate",
            "reason": (
                "Frozen support-context v2 reporting/claim artifacts are ready and include the "
                "one-shot held-out query diagnostic; this remains diagnostic, not blanket formal multi."
            ),
            "gpu": "none",
            "next": "reporting_package_figures_caveats_failure_analysis",
        },
        {
            "branch": "Track C support-set summary residual rule",
            "decision": "close_current_rule",
            "reason": (
                "Formal summary gate failed: no alpha passed train_multi LOO; support-val was not "
                "eligible for scoring."
            ),
            "gpu": "none",
            "next": "do_not_run_mmd_posthoc_or_query_for_this_rule",
        },
        {
            "branch": "Track C learned anchor-gate/simple deployable gate",
            "decision": "closed_by_prior_cpu_gate",
            "reason": "learned anchor-gate CPU gate failed canonical no-harm.",
            "gpu": "none",
            "next": "reopen_only_with_genuinely_new_non_scope_feature",
        },
        {
            "branch": "Archetype/state prior",
            "decision": "diagnostic_only",
            "reason": "independent review plus pocket triage closed same-feature variants as GPU candidates.",
            "gpu": "none",
            "next": "reopen_only_with_new_independent_source_or_mechanism",
        },
        {
            "branch": "Track A external/source prior",
            "decision": "defer_until_materially_new_feature_exists",
            "reason": "existing Track A lowcount/dataset-negative/Jiang/cross-latent routes are closed or near-miss with harm.",
            "gpu": "none",
            "next": "new_cpu_protocol_only_if_new_external_or_biology_source_prior_exists",
        },
    ]
    return {
        "timestamp": "2026-06-23 12:33 CST",
        "status": "post_support_set_summary_no_unconsumed_gpu_gate_reporting_or_new_cpu_source_next",
        "gpu_authorization": "none",
        "input_statuses": statuses,
        "missing_inputs": missing,
        "decisions": decisions,
        "recommended_next_action": (
            "Prioritize frozen support-context v2 reporting/failure-case consolidation, or design a "
            "materially new query-free CPU gate based on a new independent source/mechanism. Do not "
            "launch GPU from support-set summary, learned gate, archetype, or Track A fallback evidence."
        ),
        "closed_no_relaunch": [
            "support-set summary residual alpha sweep",
            "support-set MMD/no-harm posthoc for the failed summary rule",
            "condition-only learned anchor gate",
            "same-feature archetype threshold/router/ridge/abstain variants",
            "Track A lowcount/dataset-negative/Jiang-abstain seed expansion",
        ],
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Post Support-Set Summary Portfolio Decision",
        "",
        f"Status: `{payload['status']}`",
        f"GPU authorization: `{payload['gpu_authorization']}`",
        "",
        "## Input Statuses",
        "",
        "| input | status |",
        "|---|---|",
    ]
    for name, status in sorted(payload["input_statuses"].items()):
        lines.append(f"| `{name}` | `{status}` |")
    if payload["missing_inputs"]:
        lines.extend(["", "Missing inputs:"])
        lines.extend(f"- `{name}`" for name in payload["missing_inputs"])
    lines.extend(["", "## Branch Decisions", "", "| branch | decision | GPU | next |", "|---|---|---|---|"])
    for row in payload["decisions"]:
        lines.append(f"| {row['branch']} | `{row['decision']}` | `{row['gpu']}` | {row['next']} |")
    lines.extend(["", "## Recommended Next Action", "", payload["recommended_next_action"], ""])
    lines.extend(["## Closed / Do Not Relaunch", ""])
    lines.extend(f"- {item}" for item in payload["closed_no_relaunch"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
