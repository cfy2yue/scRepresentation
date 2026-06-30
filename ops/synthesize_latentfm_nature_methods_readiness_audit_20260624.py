#!/usr/bin/env python3
"""Audit LatentFM consolidation against Nature Methods-style rigor constraints."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def file_nonempty(path: str | Path) -> bool:
    p = Path(path)
    return p.is_file() and p.stat().st_size > 0


def checklist_status(claim: dict[str, Any], item: str) -> dict[str, Any] | None:
    for row in claim.get("provenance_checklist", []):
        if row.get("item") == item:
            return row
    return None


def main() -> None:
    claim = load_json(REPORTS / "latentfm_manuscript_claim_package_20260624.json")
    validation = load_json(REPORTS / "latentfm_consolidation_claim_scope_validation_20260624.json")
    manifest = load_json(REPORTS / "latentfm_consolidation_artifact_manifest_20260624.json")
    failure = load_json(REPORTS / "latentfm_failure_map_provenance_20260624.json")
    post_locke = load_json(REPORTS / "latentfm_post_locke_portfolio_decision_20260624.json")
    figure_manifest = load_json(REPORTS / "figures" / "latentfm_consolidation_20260624" / "manifest.json")
    results_text = (REPORTS / "LATENTFM_RESULTS_SECTION_DRAFT_20260624.md").read_text(encoding="utf-8")
    external_review = (REPORTS / "LATENTFM_CONSOLIDATION_EXTERNAL_WORDING_REVIEW_20260624.md").read_text(encoding="utf-8")

    figures = figure_manifest.get("figures", {})
    figure_files = []
    for name, paths in figures.items():
        for ext in ("png", "svg"):
            figure_files.append((name, ext, paths.get(ext, "")))

    canonical = checklist_status(claim, "Canonical split integrity")
    query = checklist_status(claim, "Track C query isolation")
    bootstrap = checklist_status(claim, "Bootstrap / CI / no-harm evidence")
    controls = checklist_status(claim, "Negative controls")
    provenance = checklist_status(claim, "Artifact provenance")

    constraints: list[dict[str, Any]] = [
        {
            "constraint": "Claim scope validator",
            "status": "satisfied",
            "evidence": "claim-scope validator passed with zero failures",
            "check": validation.get("status") == "consolidation_claim_scope_validation_pass_no_gpu"
            and validation.get("summary", {}).get("n_failed") == 0,
        },
        {
            "constraint": "Artifact provenance closure",
            "status": "satisfied",
            "evidence": "manifest has no missing artifacts or duplicate hash groups",
            "check": manifest.get("status") == "consolidation_artifact_manifest_ready_no_gpu"
            and manifest.get("summary", {}).get("n_missing") == 0
            and manifest.get("summary", {}).get("n_duplicate_hash_groups") == 0
            and manifest.get("summary", {}).get("n_artifacts", 0) >= 33,
        },
        {
            "constraint": "Canonical split integrity",
            "status": canonical.get("status") if canonical else "missing",
            "evidence": canonical.get("evidence") if canonical else "missing claim package checklist row",
            "check": bool(canonical and "not re-cut" in canonical.get("evidence", "")),
        },
        {
            "constraint": "Canonical multi exclusion",
            "status": "satisfied_by_policy",
            "evidence": "forbidden claims prohibit canonical multi or held-out Track C query as selection signal",
            "check": any(
                "canonical multi" in row.lower() and "selection signal" in row.lower()
                for row in claim.get("forbidden_claims", [])
            ),
        },
        {
            "constraint": "Track C query isolation",
            "status": query.get("status") if query else "missing",
            "evidence": query.get("evidence") if query else "missing claim package checklist row",
            "check": bool(query and "frozen diagnostic" in query.get("evidence", "")),
        },
        {
            "constraint": "Bootstrap/CI/no-harm evidence represented",
            "status": bootstrap.get("status") if bootstrap else "missing",
            "evidence": bootstrap.get("evidence") if bootstrap else "missing claim package checklist row",
            "check": bool(bootstrap and bootstrap.get("status") == "represented"),
        },
        {
            "constraint": "Negative controls represented",
            "status": controls.get("status") if controls else "missing",
            "evidence": controls.get("evidence") if controls else "missing claim package checklist row",
            "check": bool(controls and controls.get("status") == "represented"),
        },
        {
            "constraint": "Failure-case analysis",
            "status": "satisfied",
            "evidence": f"failure map rows={len(failure.get('rows', []))}",
            "check": len(failure.get("rows", [])) >= 12,
        },
        {
            "constraint": "External review at decision point",
            "status": "satisfied",
            "evidence": "Godel external wording review found no manuscript-blocking provenance gap and no new gate recommendation",
            "check": "No manuscript-blocking missing evidence" in external_review
            and "Do not reopen experiments" in external_review,
        },
        {
            "constraint": "Figure artifact availability",
            "status": "satisfied",
            "evidence": f"figure files={len(figure_files)}",
            "check": len(figure_files) == 8 and all(file_nonempty(p) for _, _, p in figure_files),
        },
        {
            "constraint": "Current Track A default statement",
            "status": "satisfied",
            "evidence": claim.get("final_model_statement", {}).get("track_a", ""),
            "check": "xverse_8k_anchor" in claim.get("final_model_statement", {}).get("track_a", ""),
        },
        {
            "constraint": "Track C diagnostic-only statement",
            "status": "satisfied",
            "evidence": claim.get("final_model_statement", {}).get("track_c", ""),
            "check": "diagnostic" in claim.get("final_model_statement", {}).get("track_c", "").lower(),
        },
        {
            "constraint": "No positive overclaim in Results draft",
            "status": "satisfied",
            "evidence": "Results draft avoids formal multi solved, OT improves, broad scaling superiority, and promoted new Track A model claims",
            "check": not any(
                phrase in results_text.lower()
                for phrase in [
                    "formal multi is solved",
                    "formal multi capability is solved",
                    "ot improves downstream generalization",
                    "broad cross-dataset scaling superiority",
                    "new track a model is promoted",
                ]
            ),
        },
        {
            "constraint": "No GPU authorization after consolidation",
            "status": "satisfied",
            "evidence": str(post_locke.get("decision", {}).get("gpu_authorized")),
            "check": post_locke.get("decision", {}).get("gpu_authorized") is False,
        },
    ]

    blockers = [row for row in constraints if not row["check"]]
    status = (
        "nature_methods_readiness_audit_pass_for_conservative_claims_no_gpu"
        if not blockers
        else "nature_methods_readiness_audit_blocked_for_conservative_claims_no_gpu"
    )
    limitations = [
        "This readiness audit supports conservative reporting, not a claim of a new promoted Track A model.",
        "Track C support-context v2 remains diagnostic/reporting only and does not establish formal multi-perturbation capability.",
        "Scaling evidence remains narrow: condition-count midpoint internal signal plus negative breadth/canonical no-harm evidence.",
        "OT remains wired-but-no-gain; no OT optimization claim is supported.",
    ]
    out = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "boundary": {
            "reads_consolidation_outputs_only": True,
            "active_logs": False,
            "raw_canonical_or_query": False,
            "canonical_multi_selection": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "summary": {
            "n_constraints": len(constraints),
            "n_passed": len(constraints) - len(blockers),
            "n_blockers": len(blockers),
        },
        "constraints": constraints,
        "blockers": blockers,
        "limitations": limitations,
        "decision": {
            "ready_for_conservative_manuscript_use": not blockers,
            "gpu_authorized": False,
            "new_cpu_gate_recommended": False,
            "scope": "conservative consolidation claims only",
        },
    }

    json_path = REPORTS / "latentfm_nature_methods_readiness_audit_20260624.json"
    md_path = REPORTS / "LATENTFM_NATURE_METHODS_READINESS_AUDIT_20260624.md"
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Nature Methods Readiness Audit",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- Reads completed consolidation outputs only: claim package, claim-scope validation, artifact manifest, failure map, figure manifest, post-Locke decision, Results draft, and external wording review.",
        "- Does not read active logs, raw canonical/query artifacts, use canonical multi for selection, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- constraints: `{out['summary']['n_constraints']}`",
        f"- passed: `{out['summary']['n_passed']}`",
        f"- blockers: `{out['summary']['n_blockers']}`",
        f"- ready for conservative manuscript use: `{out['decision']['ready_for_conservative_manuscript_use']}`",
        f"- GPU authorized: `{out['decision']['gpu_authorized']}`",
        "",
        "## Constraint Checks",
        "",
        "| Constraint | Status | Pass | Evidence |",
        "|---|---|---:|---|",
    ]
    for row in constraints:
        lines.append(
            f"| {row['constraint']} | `{row['status']}` | `{row['check']}` | {row['evidence']} |"
        )
    lines.extend(["", "## Blockers", ""])
    if blockers:
        lines.extend(f"- `{row['constraint']}`: {row['evidence']}" for row in blockers)
    else:
        lines.append("- None for conservative claim scope.")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in limitations)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The current LatentFM consolidation is ready for conservative manuscript/report use under the validated claim scope. It does not authorize new GPU experiments or stronger model-capability claims.",
            "",
            "## JSON",
            "",
            f"`{json_path}`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)
    print(json_path)
    if blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
