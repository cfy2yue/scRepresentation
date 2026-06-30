#!/usr/bin/env python3
"""Create a submission/release index for the LatentFM consolidation package."""

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


def main() -> None:
    claim = load_json(REPORTS / "latentfm_manuscript_claim_package_20260624.json")
    readiness = load_json(REPORTS / "latentfm_nature_methods_readiness_audit_20260624.json")
    manifest = load_json(REPORTS / "latentfm_consolidation_artifact_manifest_20260624.json")
    validation = load_json(REPORTS / "latentfm_consolidation_claim_scope_validation_20260624.json")
    capsule = load_json(REPORTS / "latentfm_reproducibility_capsule_20260624.json")
    legacy_closure = load_json(REPORTS / "latentfm_legacy_active_run_closure_audit_20260624.json")
    training_closure = load_json(REPORTS / "latentfm_training_data_normalization_closure_20260624.json")
    figures = load_json(REPORTS / "figures" / "latentfm_consolidation_20260624" / "manifest.json")

    final_model = claim["final_model_statement"]
    figure_paths = claim["figure_paths"]
    recommended_reading = [
        {
            "purpose": "Start here: conservative claim boundary and navigation",
            "path": "reports/LATENTFM_SUBMISSION_RELEASE_INDEX_20260624.md",
        },
        {
            "purpose": "Results prose for conservative report/manuscript draft",
            "path": "reports/LATENTFM_RESULTS_SECTION_DRAFT_20260624.md",
        },
        {
            "purpose": "Allowed/forbidden claims and figure captions",
            "path": "reports/LATENTFM_MANUSCRIPT_CLAIM_PACKAGE_20260624.md",
        },
        {
            "purpose": "Nature Methods-style rigor checklist",
            "path": "reports/LATENTFM_NATURE_METHODS_READINESS_AUDIT_20260624.md",
        },
        {
            "purpose": "Failure-map evidence and branch closures",
            "path": "reports/LATENTFM_FAILURE_MAP_PROVENANCE_20260624.md",
        },
        {
            "purpose": "Original prompt active-run closure audit",
            "path": "reports/LATENTFM_LEGACY_ACTIVE_RUN_CLOSURE_AUDIT_20260624.md",
        },
        {
            "purpose": "Training-data/normalization closure and external review",
            "path": "reports/LATENTFM_TRAINING_DATA_NORMALIZATION_CLOSURE_20260624.md",
        },
        {
            "purpose": "External review of training-data/normalization closure",
            "path": "reports/LATENTFM_TRAINING_DATA_NORMALIZATION_EXTERNAL_REVIEW_20260624.md",
        },
        {
            "purpose": "Reproducibility commands and git-boundary notes",
            "path": "reports/LATENTFM_REPRODUCIBILITY_CAPSULE_20260624.md",
        },
        {
            "purpose": "SHA256 artifact manifest",
            "path": "reports/LATENTFM_CONSOLIDATION_ARTIFACT_MANIFEST_20260624.md",
        },
        {
            "purpose": "Claim-scope validator result",
            "path": "reports/LATENTFM_CONSOLIDATION_CLAIM_SCOPE_VALIDATION_20260624.md",
        },
    ]

    figure_table = []
    for name in ["oracle_headroom_ladder", "gain_vs_tail_risk", "trackc_overlap_failure", "ot_wired_no_gain"]:
        paths = figure_paths[name]
        figure_table.append(
            {
                "name": name,
                "png": paths["png"],
                "svg": paths["svg"],
                "caption": claim["figure_captions"][name],
            }
        )

    release_index = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "submission_release_index_ready_no_gpu",
        "boundary": {
            "reads_consolidation_outputs_only": True,
            "active_logs": False,
            "raw_canonical_or_query": False,
            "canonical_multi_selection": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "final_model_statement": final_model,
        "allowed_claims": claim["allowed_claims"],
        "forbidden_claims": claim["forbidden_claims"],
        "recommended_reading": recommended_reading,
        "figures": figure_table,
        "validation_snapshot": {
            "readiness": readiness["summary"],
            "manifest": manifest["summary"],
            "claim_scope_validation": validation["summary"],
            "figure_manifest_status": figures["status"],
            "legacy_active_run_closure": {
                "status": legacy_closure["status"],
                "n_failures": len(legacy_closure["failures"]),
            },
            "training_data_normalization_closure": {
                "status": training_closure["status"],
                "gpu_authorized": training_closure["decision"]["gpu_authorized"],
            },
        },
        "reproducibility": {
            "regeneration_order": capsule["regeneration_order"],
            "validation_commands": capsule["validation_commands"],
            "top_level_git_repo": next(
                row for row in capsule["git_boundaries"] if row["path"] == str(ROOT)
            ),
        },
        "decision": {
            "ready_for_conservative_submission_package": True,
            "gpu_authorized": False,
            "new_cpu_gate_recommended": False,
            "scope": "conservative manuscript/report package only",
        },
    }

    json_path = REPORTS / "latentfm_submission_release_index_20260624.json"
    md_path = REPORTS / "LATENTFM_SUBMISSION_RELEASE_INDEX_20260624.md"
    json_path.write_text(json.dumps(release_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Submission Release Index",
        "",
        "Status: `submission_release_index_ready_no_gpu`",
        "",
        "## Boundary",
        "",
        "- Reads completed consolidation outputs only.",
        "- Does not read active logs, raw canonical/query artifacts, use canonical multi for selection, train, infer, or use GPU.",
        "",
        "## Final Model Statement",
        "",
        f"- Track A: {final_model['track_a']}",
        f"- Track C: {final_model['track_c']}",
        f"- Track C diagnostic route: `{final_model['track_c_route']}`.",
        f"- Frozen diagnostic query_multi pp/MMD deltas: `{final_model['track_c_query_multi_pearson_delta']:+.6f}` / `{final_model['track_c_query_multi_mmd_delta']:+.6f}`.",
        f"- Unseen2 Pearson delta remains weak: `{final_model['track_c_unseen2_pearson_delta']:+.6f}`.",
        "",
        "## Recommended Reading Order",
        "",
        "| Step | Purpose | Path |",
        "|---:|---|---|",
    ]
    for idx, row in enumerate(recommended_reading, start=1):
        lines.append(f"| {idx} | {row['purpose']} | `{ROOT / row['path']}` |")

    lines.extend(["", "## Allowed Claims", ""])
    lines.extend(f"- {item}" for item in release_index["allowed_claims"])
    lines.extend(["", "## Forbidden Claims", ""])
    lines.extend(f"- {item}" for item in release_index["forbidden_claims"])
    lines.extend(["", "## Figures", "", "| Figure | PNG | SVG | Claim role |", "|---|---|---|---|"])
    for row in figure_table:
        lines.append(f"| `{row['name']}` | `{row['png']}` | `{row['svg']}` | {row['caption']} |")

    lines.extend(
        [
            "",
            "## Validation Snapshot",
            "",
            f"- Nature Methods readiness: `{readiness['summary']}`",
            f"- Artifact manifest: `{manifest['summary']}`",
            f"- Claim-scope validation: `{validation['summary']}`",
            f"- Legacy active-run closure: `status={legacy_closure['status']}; failures={len(legacy_closure['failures'])}`",
            f"- Training-data/normalization closure: `status={training_closure['status']}; gpu_authorized={training_closure['decision']['gpu_authorized']}`",
            f"- Figure manifest status: `{figures['status']}`",
            "",
            "## Reproducibility",
            "",
            "Run regeneration commands from `/data/cyx/1030/scLatent` in the order recorded in the reproducibility capsule.",
            "",
            "Top-level git status:",
            "",
            f"- `/data/cyx/1030/scLatent` is git repo: `{release_index['reproducibility']['top_level_git_repo']['is_git_repo']}`",
            f"- note: `{release_index['reproducibility']['top_level_git_repo']['toplevel_or_error']}`",
            "",
            "## Decision",
            "",
            "- Ready for conservative submission/report package: `True`.",
            "- GPU authorized: `False`.",
            "- New CPU gate recommended: `False`.",
            "- Scope: conservative manuscript/report package only.",
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


if __name__ == "__main__":
    main()
