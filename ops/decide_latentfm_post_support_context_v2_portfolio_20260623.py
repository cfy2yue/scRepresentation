#!/usr/bin/env python3
"""Portfolio decision after support-context v2 final diagnostic support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_post_support_context_v2_portfolio_decision_20260623.json"
OUT_MD = REPORTS / "LATENTFM_POST_SUPPORT_CONTEXT_V2_PORTFOLIO_DECISION_20260623.md"

INPUTS = {
    "v2_final_audit": REPORTS / "latentfm_trackc_support_context_v2_final_package_audit_20260623.json",
    "v2_synthesis": REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_FINAL_DIAGNOSTIC_SYNTHESIS_20260623.md",
    "next_candidate_review": REPORTS / "LATENTFM_HIGH_THROUGHPUT_NEXT_CANDIDATE_REVIEW_20260623_1045.md",
    "post_learned_portfolio": REPORTS / "LATENTFM_POST_LEARNED_GATE_PORTFOLIO_DECISION_20260623.md",
    "distinct_support_protocol": REPORTS / "LATENTFM_TRACKC_DISTINCT_SUPPORT_ABSORBABILITY_PROTOCOL_20260623.md",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    audit = load_json(INPUTS["v2_final_audit"])
    failed = []
    if audit.get("status") != "trackc_support_context_v2_final_package_audit_pass":
        failed.append("v2_final_package_audit_not_pass")
    missing = [name for name, path in INPUTS.items() if not path.exists()]
    failed.extend(f"missing:{name}" for name in missing)

    status = "post_v2_portfolio_ready_reporting_plus_query_free_gates" if not failed else "post_v2_portfolio_fail_missing_inputs"
    decisions = [
        {
            "branch": "support_context_v2_reporting",
            "decision": "promote_to_reporting_package",
            "reason": "final package audit passed; one-shot query supported; strict canonical no-harm passed",
            "next_action": "build manuscript figures/tables and claim-readiness prose from frozen package",
            "gpu_authorization": "none",
        },
        {
            "branch": "trackc_distinct_support_absorbability",
            "decision": "keep_as_query_free_modeling_backup",
            "reason": "future modeling may target formal/trainable support use, but cannot reuse query result; must be distinct from support-FiLM/residual/context-c/endpoint/replay/memory-dose",
            "next_action": "design CPU gate only if pursuing a new train/support-only mechanism",
            "gpu_authorization": "only_after_new_cpu_gate_pass",
        },
        {
            "branch": "tracka_external_jiang_prior",
            "decision": "keep_conditional_on_new_external_source",
            "reason": "Track A near-miss remains harmed by Jiang/cytokine rows; old lowcount/dataset-negative/Jiang-abstain families are closed",
            "next_action": "do not run unless a genuinely new external/source feature is introduced and train-only gate beats closed baselines",
            "gpu_authorization": "only_after_new_cpu_gate_pass",
        },
        {
            "branch": "archetype_state_prior",
            "decision": "diagnostic_only",
            "reason": "hard/soft/conditional/orthogonal/multi-latent state gates failed current no-harm or shuffled-control gates",
            "next_action": "do not run GPU; only materially new continuous non-threshold CPU gate may reopen",
            "gpu_authorization": "none",
        },
    ]
    payload = {
        "status": status,
        "failed_reasons": failed,
        "inputs": {name: str(path) for name, path in INPUTS.items()},
        "v2_metrics": (audit.get("metrics") or {}),
        "decisions": decisions,
        "rules": [
            "do not use v2 query result for tuning or new branch selection",
            "new GPU requires a query-free CPU gate with hypothesis, promotion gate, stop rule, launcher, RUN_STATUS, and fresh resource audit",
            "reporting package is current priority because v2 final diagnostic is supported with caveats",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM Post Support-Context V2 Portfolio Decision",
        "",
        f"Status: `{status}`",
        "",
        "## Context",
        "",
        "Support-context v2 now has a passed final package audit and a supported one-shot held-out query diagnostic.  The result is strong enough for the reporting lane, but it must not become a query-driven tuning signal.",
        "",
        "## Decisions",
        "",
        "| branch | decision | next action | GPU |",
        "|---|---|---|---|",
    ]
    for row in decisions:
        lines.append(
            f"| `{row['branch']}` | `{row['decision']}` | {row['next_action']} | `{row['gpu_authorization']}` |"
        )
    lines += [
        "",
        "## Key Metrics",
        "",
        f"- query Pearson delta: `{payload['v2_metrics'].get('query_multi_pp_delta'):+.6f}`",
        f"- query MMD delta: `{payload['v2_metrics'].get('query_multi_mmd_delta'):+.6f}`",
        f"- query unseen2 Pearson delta: `{payload['v2_metrics'].get('query_unseen2_pp_delta'):+.6f}`",
        f"- query unseen2 MMD delta: `{payload['v2_metrics'].get('query_unseen2_mmd_delta'):+.6f}`",
        "",
        "## Rules",
        "",
    ]
    for rule in payload["rules"]:
        lines.append(f"- {rule}")
    lines += [
        "",
        "## Recommended Immediate Action",
        "",
        "Continue reporting/provenance/figure preparation for support-context v2.  Do not launch another GPU job unless a new query-free CPU gate is implemented and passes.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))


if __name__ == "__main__":
    main()
