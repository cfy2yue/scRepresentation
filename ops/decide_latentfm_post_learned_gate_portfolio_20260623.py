#!/usr/bin/env python3
"""Portfolio decision after the learned anchor-gate CPU gate failed."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT_JSON = ROOT / "reports/latentfm_post_learned_gate_portfolio_decision_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_POST_LEARNED_GATE_PORTFOLIO_DECISION_20260623.md"

INPUTS = {
    "frozen_package": ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_BLEND_FROZEN_PACKAGE_AUDIT_20260623.md",
    "learned_anchor_gate": ROOT / "reports/latentfm_trackc_learned_anchor_gate_cpu_gate_20260623.json",
    "tracka_nearmiss": ROOT / "reports/LATENTFM_TRACKA_SCF_GENE_RELIABILITY_NEARMISS_ANALYSIS_20260623.md",
    "tracka_jiang_abstain": ROOT / "reports/latentfm_tracka_jiang_abstain_router_cpu_gate_20260623.json",
    "tracka_crosslatent_source": ROOT / "reports/LATENTFM_XVERSE_CROSSLATENT_DEPLOYABLE_SOURCE_GATE_20260622.md",
    "archetype_orthogonal": ROOT / "reports/latentfm_soft_archetype_orthogonal_router_cpu_gate_20260623.json",
    "run_status_audit": ROOT / "reports/latentfm_run_status_consistency_audit_20260623.json",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def status_from_text(text: str) -> str | None:
    match = re.search(r"Status:\s*`([^`]+)`", text)
    return match.group(1) if match else None


def evidence() -> dict[str, Any]:
    learned = load_json(INPUTS["learned_anchor_gate"])
    jiang = load_json(INPUTS["tracka_jiang_abstain"])
    archetype = load_json(INPUTS["archetype_orthogonal"])
    run_status = load_json(INPUTS["run_status_audit"])
    frozen_text = read_text(INPUTS["frozen_package"])
    nearmiss_text = read_text(INPUTS["tracka_nearmiss"])
    crosslatent_text = read_text(INPUTS["tracka_crosslatent_source"])
    return {
        "frozen_package_status": status_from_text(frozen_text),
        "learned_anchor_gate_status": (learned.get("decision") or {}).get("status"),
        "learned_anchor_gate_reasons": (learned.get("decision") or {}).get("reasons") or [],
        "tracka_nearmiss_status": status_from_text(nearmiss_text),
        "tracka_jiang_abstain_status": (jiang.get("decision") or {}).get("status"),
        "tracka_crosslatent_source_status": status_from_text(crosslatent_text),
        "archetype_status": (archetype.get("status") or (archetype.get("decision") or {}).get("status")),
        "run_status_counts": run_status.get("status_counts") or {},
        "inputs": {k: str(v) for k, v in INPUTS.items()},
    }


def branches() -> list[dict[str, Any]]:
    return [
        {
            "branch": "Track C frozen anchor-gated blend",
            "decision": "keep_as_current_best_diagnostic_reporting_package",
            "reason": "Frozen package/provenance/CI/failure-case audits pass, but claim boundary remains diagnostic/calibrator, not deployable formal multi.",
            "next": "Use for reporting, manuscript tables, and failure analysis; do not tune query.",
            "gpu": "none",
        },
        {
            "branch": "Track C learned anchor-gate from condition/train metadata",
            "decision": "close_simple_deployable_gate_family",
            "reason": "multi_condition passes support but fails canonical family_gene; stricter train-single coverage gates fail support and family_gene. Canonical family_gene includes all support-val exact Norman/Wessels conditions, making condition-only separation non-identifiable.",
            "next": "Reopen only with a genuinely new non-scope feature that can separate support-residual applicability from canonical family no-harm without query/post-query labels.",
            "gpu": "none",
        },
        {
            "branch": "Track A scFoundation/Jiang/cross-latent reliability",
            "decision": "keep_closed_until_external_or_materially_new_prior",
            "reason": "Near-miss aggregate signal is real, but Jiang_IFNG/TNFA harm persists; Jiang abstain and cross-latent deployable-source gates failed no-harm/dataset-harm rules.",
            "next": "Only an external biological/source prior or new feature family may reopen Track A; threshold tweaks or renamed lowcount/dataset-negative policies are closed.",
            "gpu": "none",
        },
        {
            "branch": "Archetype/state prior",
            "decision": "diagnostic_only",
            "reason": "Naive, residualized, soft, conditional, and orthogonalized archetype gates failed stability, baseline, or shuffled-control criteria.",
            "next": "Reopen only as continuous multi-latent state agreement with stability and shuffled controls, not hard/soft KMeans threshold variants.",
            "gpu": "none",
        },
    ]


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Post Learned-Gate Portfolio Decision",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        "Status: `no_new_gpu_authorization_after_query_free_negative_gates`",
        "",
        "## Evidence Summary",
        "",
        "| item | status |",
        "|---|---|",
    ]
    ev = payload["evidence"]
    for key in (
        "frozen_package_status",
        "learned_anchor_gate_status",
        "tracka_nearmiss_status",
        "tracka_jiang_abstain_status",
        "tracka_crosslatent_source_status",
        "archetype_status",
    ):
        lines.append(f"| `{key}` | `{ev.get(key)}` |")
    lines.extend(["", "## Branch Decisions", "", "| branch | decision | reason | next | GPU |", "|---|---|---|---|---|"])
    for row in payload["branches"]:
        lines.append(f"| {row['branch']} | `{row['decision']}` | {row['reason']} | {row['next']} | `{row['gpu']}` |")
    lines.extend(
        [
            "",
            "## Next Action",
            "",
            "Do not launch a GPU job from the consumed Track C learned-gate, Track A fallback, cross-latent, or archetype evidence. The immediate useful work is reporting/failure-case consolidation for the frozen diagnostic package, or a new CPU-only gate based on a materially new information source. A materially new source means it is not condition arity, train-single coverage, lowcount/dataset-negative fallback, Jiang thresholding, cross-latent disagreement, or hard/soft archetype thresholding.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    payload = {
        "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "status": "no_new_gpu_authorization_after_query_free_negative_gates",
        "evidence": evidence(),
        "branches": branches(),
        "decision": "report_frozen_diagnostic_or_require_materially_new_cpu_gate_before_gpu",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "decision": payload["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
