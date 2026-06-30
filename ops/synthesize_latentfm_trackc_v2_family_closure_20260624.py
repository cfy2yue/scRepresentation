#!/usr/bin/env python3
"""Synthesize Track C support-context v2 family closure.

This is a read-only decision synthesis. It reads finalized public reports and
query-free gate JSONs only. It does not read raw held-out query artifacts,
canonical multi for selection, active logs, or launch jobs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_trackc_v2_family_closure_synthesis_20260624.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_V2_FAMILY_CLOSURE_SYNTHESIS_20260624.md"


INPUTS = {
    "claim_readiness": REPORTS / "latentfm_trackc_support_context_v2_claim_readiness_audit_20260623.json",
    "post_v2_portfolio": REPORTS / "latentfm_trackc_post_v2_portfolio_20260624.json",
    "pseudo_episode": REPORTS / "latentfm_trackc_pseudo_episode_gate_20260624.json",
    "support_jackknife": REPORTS / "latentfm_trackc_support_jackknife_reliability_gate_20260624.json",
    "response_nonadditivity": REPORTS / "latentfm_trackc_response_nonadditivity_gate_20260624.json",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def nested(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def gate_row(name: str, path: Path, key_metrics: dict[str, Any]) -> dict[str, Any]:
    d = read_json(path)
    decision = d.get("decision") or {}
    return {
        "name": name,
        "path": str(path),
        "status": d.get("status") or decision.get("status"),
        "gpu_authorization": decision.get("gpu_authorization") or d.get("gpu_authorization") or "none",
        "reasons": decision.get("reasons") or d.get("decision_reasons") or [],
        "key_metrics": key_metrics,
    }


def main() -> None:
    claim = read_json(INPUTS["claim_readiness"])
    pseudo = read_json(INPUTS["pseudo_episode"])
    jack = read_json(INPUTS["support_jackknife"])
    nonadd = read_json(INPUTS["response_nonadditivity"])

    gates = [
        gate_row(
            "pseudo_episode",
            INPUTS["pseudo_episode"],
            {
                "aggregate_pp": nested(pseudo, "decision", "aggregate_pp_delta"),
                "aggregate_mmd": nested(pseudo, "decision", "aggregate_mmd_delta"),
                "zero_overlap_pp": nested(pseudo, "decision", "zero_overlap_pp_delta"),
                "norman_pp": nested(pseudo, "decision", "norman_pp_delta"),
                "wessels_pp": nested(pseudo, "decision", "wessels_pp_delta"),
                "shuffled_pp": nested(pseudo, "decision", "shuffled_pp_delta"),
            },
        ),
        gate_row(
            "support_jackknife",
            INPUTS["support_jackknife"],
            {
                "support_pp": nested(jack, "decision", "support_val_pp_delta"),
                "support_mmd": nested(jack, "decision", "support_val_mmd_delta"),
                "norman_pp": nested(jack, "decision", "norman_pp_delta"),
                "wessels_pp": nested(jack, "decision", "wessels_pp_delta"),
                "enabled_rows": nested(jack, "decision", "enabled_rows"),
                "enabled_negative_rows": nested(jack, "decision", "enabled_negative_rows"),
                "shuffled_pp": nested(jack, "decision", "shuffled_pp_delta"),
            },
        ),
        gate_row(
            "response_nonadditivity",
            INPUTS["response_nonadditivity"],
            {
                "support_pp": nested(nonadd, "decision", "support_val_pp_delta"),
                "support_mmd": nested(nonadd, "decision", "support_val_mmd_delta"),
                "norman_pp": nested(nonadd, "decision", "norman_pp_delta"),
                "wessels_pp": nested(nonadd, "decision", "wessels_pp_delta"),
                "enabled_rows": nested(nonadd, "decision", "enabled_rows"),
                "enabled_negative_rows": nested(nonadd, "decision", "enabled_negative_rows"),
                "train_full_additive": nested(nonadd, "train_interaction_coverage", "full"),
                "support_full_additive": nested(nonadd, "support_interaction_coverage", "full"),
                "shuffled_pp": nested(nonadd, "decision", "shuffled_pp_delta"),
                "inverted_pp": nested(nonadd, "decision", "inverted_pp_delta"),
            },
        ),
    ]

    current_best = {
        "route": "xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42",
        "scope": "frozen_support_context_v2_diagnostic_not_formal_multi_solution",
        "claim_status": claim.get("status"),
        "query_multi_pearson_delta": nested(claim, "metrics", "query_multi_pearson_delta"),
        "query_multi_mmd_delta": nested(claim, "metrics", "query_multi_mmd_delta"),
        "seen_pearson_delta": nested(claim, "metrics", "seen_pearson_delta"),
        "unseen1_pearson_delta": nested(claim, "metrics", "unseen1_pearson_delta"),
        "unseen2_pearson_delta": nested(claim, "metrics", "unseen2_pearson_delta"),
        "unseen2_mmd_delta": nested(claim, "metrics", "unseen2_mmd_delta"),
    }

    all_failed = all("fail" in str(g["status"]) for g in gates)
    any_gpu = any(str(g.get("gpu_authorization")).lower() not in {"", "none", "false"} for g in gates)
    status = (
        "trackc_v2_family_closed_diagnostic_only_no_gpu_no_query"
        if all_failed and not any_gpu
        else "trackc_v2_family_needs_manual_review"
    )

    payload = {
        "status": status,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "boundary": {
            "reads_raw_heldout_query": False,
            "reads_canonical_multi_for_selection": False,
            "reads_active_logs": False,
            "launches_gpu": False,
            "uses_final_query_summary_for_context_only": True,
        },
        "inputs": {k: str(v) for k, v in INPUTS.items()},
        "current_best": current_best,
        "expansion_gates": gates,
        "decision": {
            "close_support_context_v2_family_expansion": bool(status.startswith("trackc_v2_family_closed")),
            "gpu_authorized": False,
            "heldout_query_authorized": False,
            "reason": [
                "pseudo_episode_failed_zero_overlap_generalization",
                "support_jackknife_failed_tail_risk",
                "response_nonadditivity_failed_mean_signal_coverage_and_controls",
                "heldout_query_reuse_for_selection_forbidden",
            ],
            "next_allowed_action": (
                "reporting/failure-analysis for frozen v2 diagnostic, or a materially new "
                "query-blind protocol defined before any additional query/GPU work"
            ),
        },
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Track C V2 Family Closure Synthesis",
        "",
        f"Status: `{status}`",
        "GPU authorization: `none`",
        "Held-out query authorization: `none`",
        "",
        "## Boundary",
        "",
        "- Reads finalized public summaries and query-free gate JSONs only.",
        "- Does not read raw held-out query artifacts, canonical multi for selection, active logs, or launch GPU jobs.",
        "- Final query metrics are used only to state the frozen diagnostic context.",
        "",
        "## Current Best Track C Route",
        "",
        f"- route: `{current_best['route']}`",
        f"- scope: `{current_best['scope']}`",
        f"- query_multi Pearson delta: `{fmt(current_best['query_multi_pearson_delta'])}`",
        f"- query_multi MMD delta: `{fmt(current_best['query_multi_mmd_delta'])}`",
        f"- seen/unseen1 Pearson deltas: `{fmt(current_best['seen_pearson_delta'])}` / `{fmt(current_best['unseen1_pearson_delta'])}`",
        f"- unseen2 Pearson/MMD deltas: `{fmt(current_best['unseen2_pearson_delta'])}` / `{fmt(current_best['unseen2_mmd_delta'])}`",
        "",
        "## Expansion Gates",
        "",
        "| gate | status | decisive evidence |",
        "|---|---|---|",
    ]
    for gate in gates:
        metrics = gate["key_metrics"]
        if gate["name"] == "pseudo_episode":
            evidence = (
                f"agg pp {fmt(metrics['aggregate_pp'])}, MMD {fmt(metrics['aggregate_mmd'])}; "
                f"zero-overlap pp {fmt(metrics['zero_overlap_pp'])}"
            )
        elif gate["name"] == "support_jackknife":
            evidence = (
                f"support pp {fmt(metrics['support_pp'])}, MMD {fmt(metrics['support_mmd'])}; "
                f"enabled negatives {metrics['enabled_negative_rows']}/{metrics['enabled_rows']}"
            )
        else:
            evidence = (
                f"support pp {fmt(metrics['support_pp'])}, Wessels {fmt(metrics['wessels_pp'])}; "
                f"coverage train/support {metrics['train_full_additive']}/{metrics['support_full_additive']}; controls not separated"
            )
        lines.append(f"| `{gate['name']}` | `{gate['status']}` | {evidence} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Close the Track C support-context v2 family as a modeling expansion branch. "
            "Keep the frozen v2 resfilm checkpoint as a diagnostic/reporting candidate only.",
            "",
            "Do not launch another GPU smoke or held-out query from pseudo-episode, "
            "support-jackknife reliability, response-derived nonadditivity, or local variants "
            "of these gates.",
            "",
            "## Next Allowed Action",
            "",
            "- Reporting and failure analysis for the frozen v2 diagnostic route.",
            "- A materially new query-blind protocol defined before any additional GPU/query work.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(OUT_MD)
    print(OUT_JSON)
    print(status)


if __name__ == "__main__":
    main()
