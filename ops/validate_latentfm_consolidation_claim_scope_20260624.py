#!/usr/bin/env python3
"""Validate LatentFM consolidation claim scope and artifact closure."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures" / "latentfm_consolidation_20260624"
EXPECTED_ARTIFACT_COUNT = 49


REQUIRED_JSONS = [
    REPORTS / "latentfm_consolidation_artifact_manifest_20260624.json",
    REPORTS / "latentfm_manuscript_claim_package_20260624.json",
    REPORTS / "latentfm_post_locke_portfolio_decision_20260624.json",
    REPORTS / "latentfm_failure_map_provenance_20260624.json",
    REPORTS / "latentfm_figure_table_candidates_20260624.json",
    REPORTS / "latentfm_results_section_draft_20260624.json",
    REPORTS / "latentfm_nature_methods_readiness_audit_20260624.json",
    REPORTS / "latentfm_reproducibility_capsule_20260624.json",
    REPORTS / "latentfm_submission_release_index_20260624.json",
    REPORTS / "latentfm_legacy_active_run_closure_audit_20260624.json",
    REPORTS / "latentfm_training_data_normalization_closure_20260624.json",
    FIG_DIR / "manifest.json",
]

REQUIRED_REPORTS = [
    REPORTS / "LATENTFM_MANUSCRIPT_CLAIM_PACKAGE_20260624.md",
    REPORTS / "LATENTFM_RESULTS_SECTION_DRAFT_20260624.md",
    REPORTS / "LATENTFM_CONSOLIDATION_ARTIFACT_MANIFEST_20260624.md",
    REPORTS / "LATENTFM_CONSOLIDATION_FIGURES_20260624.md",
    REPORTS / "LATENTFM_CONSOLIDATION_EXTERNAL_WORDING_REVIEW_20260624.md",
    REPORTS / "LATENTFM_NATURE_METHODS_READINESS_AUDIT_20260624.md",
    REPORTS / "LATENTFM_REPRODUCIBILITY_CAPSULE_20260624.md",
    REPORTS / "LATENTFM_SUBMISSION_RELEASE_INDEX_20260624.md",
    REPORTS / "LATENTFM_LEGACY_ACTIVE_RUN_CLOSURE_AUDIT_20260624.md",
    REPORTS / "LATENTFM_TRAINING_DATA_NORMALIZATION_CLOSURE_20260624.md",
    REPORTS / "LATENTFM_TRAINING_DATA_NORMALIZATION_EXTERNAL_REVIEW_20260624.md",
]

REQUIRED_FIGURES = [
    "oracle_headroom_ladder",
    "gain_vs_tail_risk",
    "trackc_overlap_failure",
    "ot_wired_no_gain",
]

REQUIRED_ALLOWED_SUBSTRINGS = [
    "xverse_8k_anchor",
    "Forbidden-oracle headroom exists",
    "Track C support-context v2 has diagnostic",
    "OT minibatch pairing is wired",
    "Scaling evidence supports only a narrow",
]

REQUIRED_FORBIDDEN_SUBSTRINGS = [
    "Do not claim a new Track A model is promoted",
    "Do not claim formal multi perturbation capability is solved",
    "Do not claim strong unseen2 multi Pearson improvement",
    "Do not claim OT improves downstream generalization",
    "Do not claim broad cross-dataset scaling superiority",
    "Do not use canonical multi or held-out Track C query as a selection signal",
]

REQUIRED_DRAFT_SUBSTRINGS = [
    "xverse_8k_anchor",
    "diagnostic/reporting",
    "not a formal multi-perturbation capability claim",
    "wired-but-no-gain",
    "narrow train-only condition-count midpoint signal",
]

OVERCLAIM_PHRASES = [
    "formal multi is solved",
    "formal multi capability is solved",
    "formal multi perturbation capability is solved",
    "strong unseen2 multi pearson improvement",
    "ot improves downstream generalization",
    "broad cross-dataset scaling superiority",
    "new track a model is promoted",
    "promoted over xverse_8k_anchor",
    "query-tuned route",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "detail": detail,
        }
    )


def text_contains_all(text: str, needles: list[str]) -> tuple[bool, list[str]]:
    missing = [needle for needle in needles if needle not in text]
    return len(missing) == 0, missing


def main() -> None:
    checks: list[dict[str, Any]] = []
    loaded: dict[str, Any] = {}

    for path in REQUIRED_JSONS:
        try:
            loaded[rel(path)] = load_json(path)
            add_check(checks, f"parse_json:{rel(path)}", True, "parsed")
        except Exception as exc:  # noqa: BLE001
            add_check(checks, f"parse_json:{rel(path)}", False, repr(exc))

    for path in REQUIRED_REPORTS:
        add_check(
            checks,
            f"report_exists:{rel(path)}",
            path.is_file() and path.stat().st_size > 0,
            f"size_bytes={path.stat().st_size if path.exists() else None}",
        )

    artifact = loaded.get("reports/latentfm_consolidation_artifact_manifest_20260624.json", {})
    summary = artifact.get("summary", {}) if isinstance(artifact, dict) else {}
    add_check(
        checks,
        "artifact_manifest_ready",
        artifact.get("status") == "consolidation_artifact_manifest_ready_no_gpu",
        str(artifact.get("status")),
    )
    add_check(checks, "artifact_manifest_missing_zero", summary.get("n_missing") == 0, str(summary))
    add_check(
        checks,
        "artifact_manifest_duplicate_hash_zero",
        summary.get("n_duplicate_hash_groups") == 0,
        str(summary),
    )
    add_check(
        checks,
        "artifact_manifest_artifact_count",
        summary.get("n_artifacts") == EXPECTED_ARTIFACT_COUNT,
        str(summary),
    )

    claim = loaded.get("reports/latentfm_manuscript_claim_package_20260624.json", {})
    allowed = claim.get("allowed_claims", []) if isinstance(claim, dict) else []
    forbidden = claim.get("forbidden_claims", []) if isinstance(claim, dict) else []
    final_model = claim.get("final_model_statement", {}) if isinstance(claim, dict) else {}
    allowed_text = "\n".join(map(str, allowed))
    forbidden_text = "\n".join(map(str, forbidden))
    ok, missing = text_contains_all(allowed_text, REQUIRED_ALLOWED_SUBSTRINGS)
    add_check(checks, "claim_allowed_scope_contains_required_items", ok, f"missing={missing}")
    ok, missing = text_contains_all(forbidden_text, REQUIRED_FORBIDDEN_SUBSTRINGS)
    add_check(checks, "claim_forbidden_scope_contains_required_items", ok, f"missing={missing}")
    add_check(
        checks,
        "track_a_default_is_anchor",
        "xverse_8k_anchor" in str(final_model.get("track_a")),
        str(final_model.get("track_a")),
    )
    add_check(
        checks,
        "track_c_declared_diagnostic",
        "diagnostic" in str(final_model.get("track_c", "")).lower(),
        str(final_model.get("track_c")),
    )

    figure_manifest = loaded.get("reports/figures/latentfm_consolidation_20260624/manifest.json", {})
    figures = figure_manifest.get("figures", {}) if isinstance(figure_manifest, dict) else {}
    add_check(
        checks,
        "figure_manifest_ready",
        figure_manifest.get("status") == "latentfm_consolidation_figures_ready_no_gpu",
        str(figure_manifest.get("status")),
    )
    for name in REQUIRED_FIGURES:
        fig = figures.get(name, {})
        for ext in ("png", "svg"):
            p = Path(fig.get(ext, ""))
            add_check(
                checks,
                f"figure_exists:{name}.{ext}",
                p.is_file() and p.stat().st_size > 0,
                f"path={p} size_bytes={p.stat().st_size if p.exists() else None}",
            )

    claim_md = (REPORTS / "LATENTFM_MANUSCRIPT_CLAIM_PACKAGE_20260624.md").read_text(
        encoding="utf-8"
    )
    draft_md = (REPORTS / "LATENTFM_RESULTS_SECTION_DRAFT_20260624.md").read_text(
        encoding="utf-8"
    )
    ok, missing = text_contains_all(draft_md, REQUIRED_DRAFT_SUBSTRINGS)
    add_check(checks, "results_draft_contains_required_scope", ok, f"missing={missing}")

    lower_draft = draft_md.lower()
    lower_claim = claim_md.lower()
    overclaims_in_draft = [phrase for phrase in OVERCLAIM_PHRASES if phrase in lower_draft]
    # The claim package intentionally lists forbidden claims; overclaim phrases
    # are allowed there only if they appear in the Forbidden Claims section.
    claim_forbidden_start = lower_claim.find("## forbidden claims")
    claim_forbidden_ok = claim_forbidden_start >= 0
    overclaims_before_forbidden = []
    if claim_forbidden_ok:
        claim_prefix = lower_claim[:claim_forbidden_start]
        overclaims_before_forbidden = [
            phrase for phrase in OVERCLAIM_PHRASES if phrase in claim_prefix
        ]
    add_check(
        checks,
        "results_draft_has_no_overclaim_phrases",
        len(overclaims_in_draft) == 0,
        f"phrases={overclaims_in_draft}",
    )
    add_check(
        checks,
        "claim_package_overclaims_only_in_forbidden_context",
        claim_forbidden_ok and len(overclaims_before_forbidden) == 0,
        f"forbidden_section_found={claim_forbidden_ok} phrases_before={overclaims_before_forbidden}",
    )

    post_locke = loaded.get("reports/latentfm_post_locke_portfolio_decision_20260624.json", {})
    decision = post_locke.get("decision", {}) if isinstance(post_locke, dict) else {}
    add_check(
        checks,
        "post_locke_no_gpu_authorized",
        decision.get("gpu_authorized") is False,
        str(decision.get("gpu_authorized")),
    )
    add_check(
        checks,
        "post_locke_current_tracka_default",
        decision.get("current_tracka_default") == "xverse_8k_anchor",
        str(decision.get("current_tracka_default")),
    )

    failures = [check for check in checks if not check["passed"]]
    status = (
        "consolidation_claim_scope_validation_pass_no_gpu"
        if not failures
        else "consolidation_claim_scope_validation_fail_no_gpu"
    )
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
            "n_checks": len(checks),
            "n_passed": len(checks) - len(failures),
            "n_failed": len(failures),
        },
        "checks": checks,
        "failures": failures,
        "decision": {
            "claim_scope_validated": not failures,
            "track_a_default": final_model.get("track_a"),
            "track_c_scope": final_model.get("track_c"),
            "next_action": (
                "claim_package_ready_for_external_read_only_review"
                if not failures
                else "fix_failed_claim_scope_checks_before_external_review"
            ),
        },
    }

    json_path = REPORTS / "latentfm_consolidation_claim_scope_validation_20260624.json"
    md_path = REPORTS / "LATENTFM_CONSOLIDATION_CLAIM_SCOPE_VALIDATION_20260624.md"
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Consolidation Claim-Scope Validation",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- Reads completed consolidation outputs only: artifact manifest, claim package, Results draft, figure manifest, and post-Locke decision.",
        "- Does not read active logs, raw canonical/query artifacts, canonical multi for selection, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- checks: `{out['summary']['n_checks']}`",
        f"- passed: `{out['summary']['n_passed']}`",
        f"- failed: `{out['summary']['n_failed']}`",
        f"- Track A default: `{final_model.get('track_a')}`",
        f"- Track C scope: `{final_model.get('track_c')}`",
        "",
        "## Decision",
        "",
        (
            "The current consolidation claim scope is internally consistent and ready for an external read-only wording review."
            if not failures
            else "The current consolidation claim scope failed validation; fix failed checks before external review or manuscript use."
        ),
        "",
        "## Failed Checks",
        "",
    ]
    if failures:
        for check in failures:
            lines.append(f"- `{check['name']}`: {check['detail']}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Key Passed Checks",
            "",
            f"- Artifact manifest ready; missing artifacts `0`; duplicate hash groups `0`; artifact count `{EXPECTED_ARTIFACT_COUNT}`.",
            "- Claim package preserves `xverse_8k_anchor` as Track A deployable/default.",
            "- Track C support-context v2 remains diagnostic/reporting only.",
            "- Results draft contains OT wired-but-no-gain and scaling narrow-signal language.",
            "- Four consolidation figures each have non-empty PNG and SVG files.",
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
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
