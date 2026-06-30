#!/usr/bin/env python3
"""Decide the next LatentFM branch without using held-out query for tuning."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT_JSON = ROOT / "reports/latentfm_next_query_free_branch_decision_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_NEXT_QUERY_FREE_BRANCH_DECISION_20260623.md"


REPORTS = {
    "frozen_package": ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_BLEND_FROZEN_PACKAGE_AUDIT_20260623.md",
    "claim_readiness": ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_BLEND_CLAIM_READINESS_AUDIT_20260623.md",
    "support_film_decision": ROOT / "reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_xverse_trackc_support_film_absroute_2k_seed42_retry1.md",
    "support_film_route_gap": ROOT / "reports/LATENTFM_TRACKC_SUPPORT_FILM_ROUTE_GAP_GATE_xverse_trackc_support_film_absroute_2k_seed42_retry1.md",
    "anchor_gated_protocol": ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_SUPPORT_TEACHER_PROTOCOL_20260623.md",
    "jiang_abstain": ROOT / "reports/LATENTFM_TRACKA_JIANG_ABSTAIN_ROUTER_CPU_GATE_20260623.md",
    "archetype_orthogonal": ROOT / "reports/LATENTFM_SOFT_ARCHETYPE_ORTHOGONAL_ROUTER_CPU_GATE_20260623.md",
    "run_status_audit": ROOT / "reports/LATENTFM_RUN_STATUS_CONSISTENCY_AUDIT_20260623.md",
}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def find_status(text: str) -> str | None:
    match = re.search(r"Status:\s*`([^`]+)`", text)
    return match.group(1) if match else None


def find_value(text: str, label: str) -> str | None:
    match = re.search(re.escape(label) + r"[^+\-\d]*(?P<value>[+\-]\d+\.\d+)", text)
    return match.group("value") if match else None


def evidence() -> dict[str, Any]:
    texts = {name: read(path) for name, path in REPORTS.items()}
    return {
        "reports": {name: str(path) for name, path in REPORTS.items()},
        "status": {name: find_status(text) for name, text in texts.items()},
        "key_values": {
            "support_film_support_pp_delta": find_value(texts["support_film_decision"], "pearson_pert"),
            "support_film_wessels_closure": find_value(texts["support_film_route_gap"], "Wessels"),
            "jiang_best_crossbg_delta": find_value(texts["jiang_abstain"], "focus_margin_or_lowcount"),
        },
        "boundary_checks": {
            "run_status_recent_ok": "ok=30" in texts["run_status_audit"] or "`ok` | 30" in texts["run_status_audit"],
            "frozen_package_pass": "trackc_anchor_gated_blend_frozen_package_audit_pass" in texts["frozen_package"],
            "claim_boundary_present": "claim_ready_as_frozen_diagnostic_not_formal_multi_solution" in texts["claim_readiness"],
            "support_film_closed": "trackc_smoke_fail_canonical_harm_close_branch" in texts["support_film_decision"],
            "archetype_no_gpu": "soft_archetype_orthogonal_router_cpu_gate_fail_no_gpu" in texts["archetype_orthogonal"],
            "jiang_no_gpu": "tracka_jiang_abstain_router_cpu_gate_fail_no_gpu" in texts["jiang_abstain"],
        },
    }


def candidates() -> list[dict[str, Any]]:
    return [
        {
            "rank": 1,
            "name": "Track C learned anchor-gate reliability CPU gate",
            "status": "next_cpu_gate_recommended",
            "hypothesis": (
                "The frozen anchor-gated blend works because the support residual is useful but must be "
                "off for canonical single/family contexts.  A deployable next step must learn a "
                "train/support-derived reliability gate instead of using split/scope oracle gate=1 for "
                "support/query and gate=0 for canonical."
            ),
            "forbidden_inputs": [
                "held-out Track C query raw/decision artifacts for any selection",
                "canonical multi for selection",
                "alpha/gate/checkpoint changes from the one-shot query result",
            ],
            "cpu_gate": [
                "Use safe trainselect support artifacts plus train-only/internal features only.",
                "Freeze anchor/support-teacher checkpoints and alpha before any canonical no-harm check.",
                "Learn or predeclare g_trainonly(condition,dataset, support-coverage features).",
                "Support-val gate: Wessels pp delta >= +0.02, Wessels closure >= +0.05, Norman delta >= -0.02, pp p_harm <= 0.20, no MMD hard harm.",
                "Canonical no-harm after gate freeze: test_single and family_gene pp p_harm <= 0.35 and MMD p_harm <= 0.80.",
                "Negative controls: zero-support must no-op; shuffled-support gate/residual must fail.",
            ],
            "close_rule": (
                "If the learned train-only gate cannot preserve canonical no-harm without using split labels, "
                "close trainable Track C support-residual promotion and keep the current frozen blend as a diagnostic only."
            ),
            "not_duplicate": (
                "Distinct from support-FiLM/residual/operator because it does not alter hidden dynamics or route labels; "
                "distinct from the current frozen blend because the gate must be learned from allowed features, not assigned by evaluation scope."
            ),
            "gpu_authorization": "none_until_cpu_gate_passes",
        },
        {
            "rank": 2,
            "name": "Track A condition/source biological reliability prior CPU gate",
            "status": "secondary_cpu_gate",
            "hypothesis": (
                "Single-perturbation Track A remains near-miss; the useful signal may require source/condition biology "
                "features rather than the already-failed lowcount, dataset-negative, or simple Jiang abstain policies."
            ),
            "forbidden_inputs": [
                "canonical multi or held-out Track C query",
                "canonical Track A posthoc for router selection",
                "reusing the failed lowcount/dataset-negative/Jiang-abstain rules as a renamed GPU launch",
            ],
            "cpu_gate": [
                "Train-only/internal split only.",
                "Compare against failed lowcount and dataset-negative baselines, not just dataset_mean.",
                "cross_background_seen_gene and family proxy delta >= +0.02, p_harm <= 0.20.",
                "Dataset minimum delta >= -0.02 across Jiang_IFNG, Jiang_TNFA, Norman, and non-focus controls.",
            ],
            "close_rule": "Any material Jiang/non-Jiang dataset harm or failure to beat both failed baselines closes this Track A branch.",
            "not_duplicate": (
                "Different from the failed Jiang abstain gate only if it introduces an independently justified biological/source prior "
                "and beats both completed fallback baselines; otherwise it is closed."
            ),
            "gpu_authorization": "none_until_cpu_gate_passes",
        },
        {
            "rank": 3,
            "name": "Continuous multi-latent state prior audit",
            "status": "diagnostic_cpu_only",
            "hypothesis": (
                "Hard/soft archetype labels failed, but a continuous state prior using multi-latent agreement may still be useful "
                "as a diagnostic or reliability feature if it is seed-stable and not dataset-like."
            ),
            "forbidden_inputs": [
                "held-out query or canonical multi selection",
                "hard KMeans cluster labels as a direct adapter trigger",
                "validation target-based choice of when to enable the state prior",
            ],
            "cpu_gate": [
                "Use residualized continuous prototypes or multi-latent agreement only.",
                "Require stability, low dataset NMI/purity, focus coverage, improvement over gene/dataset baselines, and shuffled-state collapse.",
                "No GPU unless it beats the orthogonalized soft-archetype negative evidence."
            ],
            "close_rule": "If it remains dataset-like, unstable, or fails shuffled controls, keep archetype/state prior as failure analysis only.",
            "not_duplicate": "Must be continuous/multi-latent; the K16 soft-archetype router and threshold variants are already negative evidence.",
            "gpu_authorization": "none",
        },
    ]


def write_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Next Query-Free Branch Decision",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        "Status: `next_cpu_gate_required_before_gpu`",
        "",
        "## Boundary",
        "",
        "This decision does not read held-out query artifacts for selection, does not authorize a GPU run, and does not change the frozen Track C diagnostic route.",
        "",
        "Current best remains the frozen Track C anchor-gated blend diagnostic/calibrator.  It is reporting-ready, but it is not a deployable trainable formal multi solution because the current gate is evaluation-scope based.",
        "",
        "## Evidence Checks",
        "",
        "| check | value |",
        "|---|---|",
    ]
    for key, value in payload["evidence"]["boundary_checks"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Ranked Next Gates", ""])
    for cand in payload["candidates"]:
        lines.extend(
            [
                f"### {cand['rank']}. {cand['name']}",
                "",
                f"Status: `{cand['status']}`",
                "",
                f"Hypothesis: {cand['hypothesis']}",
                "",
                "Forbidden inputs:",
            ]
        )
        lines.extend([f"- {item}" for item in cand["forbidden_inputs"]])
        lines.extend(["", "CPU gate:"])
        lines.extend([f"- {item}" for item in cand["cpu_gate"]])
        lines.extend(
            [
                "",
                f"Close rule: {cand['close_rule']}",
                "",
                f"Why not duplicate: {cand['not_duplicate']}",
                "",
                f"GPU authorization: `{cand['gpu_authorization']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            "Do not launch a new GPU job from the already-consumed C1/C2/archetype/Track A fallback evidence.  The next executable research step is the ranked Track C learned anchor-gate reliability CPU gate.  Passing that gate can authorize one capped support-only GPU smoke after a fresh AGENTS.md resource audit and a new RUN_STATUS; failing it keeps the frozen blend as a diagnostic/reporting result and closes trainable support-residual promotion for now.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    payload = {
        "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "status": "next_cpu_gate_required_before_gpu",
        "evidence": evidence(),
        "candidates": candidates(),
        "decision": "run_trackc_learned_anchor_gate_reliability_cpu_gate_next_no_gpu_until_pass",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(write_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "decision": payload["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
