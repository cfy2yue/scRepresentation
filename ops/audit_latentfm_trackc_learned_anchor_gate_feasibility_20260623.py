#!/usr/bin/env python3
"""Feasibility audit for a learned Track C anchor-gate reliability CPU gate.

This is a protocol/readiness audit, not a model evaluation.  It distinguishes
the current frozen scope-gated diagnostic from the next deployable question:
can a train/support-derived reliability gate choose when to apply the support
teacher residual without using split labels or held-out query?
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT_JSON = ROOT / "reports/latentfm_trackc_learned_anchor_gate_feasibility_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_LEARNED_ANCHOR_GATE_FEASIBILITY_20260623.md"

INPUTS = {
    "anchor_gated_cpu_gate": ROOT / "reports/latentfm_trackc_anchor_gated_support_teacher_cpu_gate_20260623.json",
    "anchor_gated_posthoc_gate": ROOT / "reports/latentfm_trackc_anchor_gated_support_teacher_blend_posthoc_gate_20260623.json",
    "route_freeze": ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_BLEND_ROUTE_FREEZE_20260623.md",
    "claim_readiness": ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_BLEND_CLAIM_READINESS_AUDIT_20260623.md",
    "next_branch_decision": ROOT / "reports/latentfm_next_query_free_branch_decision_20260623.json",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def assess() -> dict[str, Any]:
    cpu = load_json(INPUTS["anchor_gated_cpu_gate"])
    posthoc = load_json(INPUTS["anchor_gated_posthoc_gate"])
    route_text = read_text(INPUTS["route_freeze"])
    claim_text = read_text(INPUTS["claim_readiness"])
    next_decision = load_json(INPUTS["next_branch_decision"])

    present = {
        "condition_mean_cpu_gate_exists": bool(cpu),
        "full_mmd_posthoc_gate_exists": bool(posthoc),
        "route_freeze_exists": bool(route_text),
        "claim_boundary_exists": "claim_ready_as_frozen_diagnostic_not_formal_multi_solution" in claim_text,
        "next_branch_decision_exists": bool(next_decision),
    }

    existing_support = {
        "frozen_alpha": 0.75 if "0.75" in route_text else None,
        "cpu_gate_status": (cpu.get("decision") or {}).get("status"),
        "posthoc_gate_status": posthoc.get("status"),
        "current_gate_type": "scope_oracle_gate_support_query_on_canonical_off",
        "heldout_query_used_for_this_audit": False,
        "canonical_multi_selection_used": False,
    }

    gaps = [
        {
            "gap": "no_learned_trainonly_gate",
            "detail": "Current frozen blend uses evaluation scope to set gate=1 on support/query and gate=0 on canonical no-harm; it has not learned g_trainonly(condition,dataset,features).",
            "required_next": "Fit or predeclare g_trainonly using train/support-only features, before canonical no-harm and without held-out query.",
        },
        {
            "gap": "canonical_noharm_is_scope_noop",
            "detail": "Canonical no-harm is exact because residual is disabled; this is valid for a diagnostic but not proof of a deployable learned gate.",
            "required_next": "After freezing learned gate/alpha, evaluate canonical test_single and family_gene with the learned gate active and fail closed on pp/MMD harm.",
        },
        {
            "gap": "negative_controls_need_gate_level_recheck",
            "detail": "Zero/shuffled support controls exist for residual signal, but a learned gate also needs controls showing the gate does not encode split labels or dataset leakage.",
            "required_next": "Run zero-support, shuffled-support, and label-permuted-gate controls.",
        },
        {
            "gap": "mmd_required_for_promotion",
            "detail": "Mean-vector CPU gates are not enough for GPU promotion unless followed by full MMD/Pearson posthoc.",
            "required_next": "Any pass can authorize only one capped support-only GPU smoke with full support/canonical posthoc before query is considered.",
        },
    ]

    ready = all(present.values()) and existing_support["cpu_gate_status"] and existing_support["posthoc_gate_status"]
    status = "trackc_learned_anchor_gate_feasibility_protocol_ready_no_gpu" if ready else "trackc_learned_anchor_gate_feasibility_missing_inputs"

    return {
        "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "status": status,
        "gpu_authorization": "none",
        "inputs": {k: str(v) for k, v in INPUTS.items()},
        "present": present,
        "existing_support": existing_support,
        "gaps": gaps,
        "recommended_next_gate": {
            "name": "learned_anchor_gate_reliability_cpu_gate",
            "hypothesis": "A train/support-derived reliability gate can apply the support-teacher residual where true-multi support signal is reliable while preserving canonical Track A no-harm without using split labels.",
            "forbidden_inputs": [
                "held-out Track C query for selection or tuning",
                "canonical multi for selection",
                "evaluation-scope label as a gate feature",
                "post-query failure cases as gate-design labels",
            ],
            "promotion_gate": [
                "support-val Wessels pp delta >= +0.02",
                "support-val Wessels route-gap closure >= +0.05",
                "support-val Norman pp delta >= -0.02",
                "support paired pp p_harm <= 0.20",
                "canonical test_single/family_gene pp p_harm <= 0.35",
                "canonical MMD p_harm <= 0.80",
                "zero/shuffled/label-permuted gate controls fail",
            ],
            "close_rule": "If learned gate cannot beat the scope-oracle limitation without canonical harm, keep frozen blend diagnostic-only and close trainable support-residual promotion.",
        },
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Learned Anchor-Gate Feasibility Audit",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        f"Status: `{payload['status']}`",
        f"GPU authorization: `{payload['gpu_authorization']}`",
        "",
        "## Boundary",
        "",
        "This audit is query-free for selection and launches no experiment.  It separates the current frozen scope-gated diagnostic from the next deployable learned-gate question.",
        "",
        "## Existing Support",
        "",
        "| item | value |",
        "|---|---|",
    ]
    for key, value in payload["existing_support"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Readiness Checks", "", "| check | value |", "|---|---|"])
    for key, value in payload["present"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Gaps Before GPU", ""])
    for row in payload["gaps"]:
        lines.extend([f"- `{row['gap']}`: {row['detail']} Required next: {row['required_next']}"])
    nxt = payload["recommended_next_gate"]
    lines.extend(
        [
            "",
            "## Recommended Next Gate",
            "",
            f"Name: `{nxt['name']}`",
            "",
            f"Hypothesis: {nxt['hypothesis']}",
            "",
            "Forbidden inputs:",
        ]
    )
    lines.extend(f"- {item}" for item in nxt["forbidden_inputs"])
    lines.extend(["", "Promotion gate:"])
    lines.extend(f"- {item}" for item in nxt["promotion_gate"])
    lines.extend(["", f"Close rule: {nxt['close_rule']}", ""])
    return "\n".join(lines)


def main() -> int:
    payload = assess()
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
