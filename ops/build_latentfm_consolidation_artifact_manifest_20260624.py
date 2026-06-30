#!/usr/bin/env python3
"""Build a sha256 provenance manifest for LatentFM consolidation artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures" / "latentfm_consolidation_20260624"
OPS = ROOT / "ops"


ARTIFACTS: list[dict[str, str]] = [
    {
        "category": "decision",
        "purpose": "post-Locke no-gpu portfolio decision",
        "path": "reports/LATENTFM_POST_LOCKE_PORTFOLIO_DECISION_20260624.md",
    },
    {
        "category": "decision_json",
        "purpose": "machine-readable post-Locke no-gpu portfolio decision",
        "path": "reports/latentfm_post_locke_portfolio_decision_20260624.json",
    },
    {
        "category": "failure_map",
        "purpose": "paper-grade failure map and claim-scope provenance",
        "path": "reports/LATENTFM_FAILURE_MAP_PROVENANCE_20260624.md",
    },
    {
        "category": "failure_map_json",
        "purpose": "machine-readable failure map and claim-scope provenance",
        "path": "reports/latentfm_failure_map_provenance_20260624.json",
    },
    {
        "category": "failure_map_csv",
        "purpose": "tabular failure map for downstream tables",
        "path": "reports/latentfm_failure_map_provenance_20260624.csv",
    },
    {
        "category": "figure_table",
        "purpose": "figure/table candidate report",
        "path": "reports/LATENTFM_FIGURE_TABLE_CANDIDATES_20260624.md",
    },
    {
        "category": "figure_table_json",
        "purpose": "machine-readable figure/table candidate data",
        "path": "reports/latentfm_figure_table_candidates_20260624.json",
    },
    {
        "category": "figure_table_csv",
        "purpose": "oracle headroom ladder source table",
        "path": "reports/latentfm_oracle_headroom_ladder_20260624.csv",
    },
    {
        "category": "figure_table_csv",
        "purpose": "average gain versus tail-risk source table",
        "path": "reports/latentfm_gain_vs_tail_risk_20260624.csv",
    },
    {
        "category": "figure_table_csv",
        "purpose": "Track C overlap failure source table",
        "path": "reports/latentfm_trackc_overlap_failure_panel_20260624.csv",
    },
    {
        "category": "figure_table_csv",
        "purpose": "OT wired-but-no-gain source table",
        "path": "reports/latentfm_ot_wired_no_gain_panel_20260624.csv",
    },
    {
        "category": "figure_index",
        "purpose": "static figure index",
        "path": "reports/LATENTFM_CONSOLIDATION_FIGURES_20260624.md",
    },
    {
        "category": "figure_manifest",
        "purpose": "static figure path manifest",
        "path": "reports/figures/latentfm_consolidation_20260624/manifest.json",
    },
    {
        "category": "claim_package",
        "purpose": "allowed/forbidden claims, captions, and provenance checklist",
        "path": "reports/LATENTFM_MANUSCRIPT_CLAIM_PACKAGE_20260624.md",
    },
    {
        "category": "claim_package_json",
        "purpose": "machine-readable manuscript claim package",
        "path": "reports/latentfm_manuscript_claim_package_20260624.json",
    },
    {
        "category": "results_draft",
        "purpose": "conservative Results-section text draft",
        "path": "reports/LATENTFM_RESULTS_SECTION_DRAFT_20260624.md",
    },
    {
        "category": "results_draft_json",
        "purpose": "machine-readable Results-section draft metadata",
        "path": "reports/latentfm_results_section_draft_20260624.json",
    },
    {
        "category": "external_review",
        "purpose": "read-only external wording review and conservative patch decision",
        "path": "reports/LATENTFM_CONSOLIDATION_EXTERNAL_WORDING_REVIEW_20260624.md",
    },
    {
        "category": "readiness_audit",
        "purpose": "Nature Methods-style rigor/readiness audit for conservative claims",
        "path": "reports/LATENTFM_NATURE_METHODS_READINESS_AUDIT_20260624.md",
    },
    {
        "category": "readiness_audit_json",
        "purpose": "machine-readable Nature Methods-style readiness audit",
        "path": "reports/latentfm_nature_methods_readiness_audit_20260624.json",
    },
    {
        "category": "reproducibility_capsule",
        "purpose": "regeneration commands, environment, and git-boundary capsule",
        "path": "reports/LATENTFM_REPRODUCIBILITY_CAPSULE_20260624.md",
    },
    {
        "category": "reproducibility_capsule_json",
        "purpose": "machine-readable regeneration commands and git-boundary capsule",
        "path": "reports/latentfm_reproducibility_capsule_20260624.json",
    },
    {
        "category": "submission_release_index",
        "purpose": "top-level submission/report package navigation index",
        "path": "reports/LATENTFM_SUBMISSION_RELEASE_INDEX_20260624.md",
    },
    {
        "category": "submission_release_index_json",
        "purpose": "machine-readable submission/report package navigation index",
        "path": "reports/latentfm_submission_release_index_20260624.json",
    },
    {
        "category": "legacy_active_run_closure",
        "purpose": "closure audit for original Track C routed-distill wait item",
        "path": "reports/LATENTFM_LEGACY_ACTIVE_RUN_CLOSURE_AUDIT_20260624.md",
    },
    {
        "category": "legacy_active_run_closure_json",
        "purpose": "machine-readable closure audit for original Track C routed-distill wait item",
        "path": "reports/latentfm_legacy_active_run_closure_audit_20260624.json",
    },
    {
        "category": "training_data_normalization_closure",
        "purpose": "closure synthesis for training-data, normalization, weighted-loss, and OT axes",
        "path": "reports/LATENTFM_TRAINING_DATA_NORMALIZATION_CLOSURE_20260624.md",
    },
    {
        "category": "training_data_normalization_closure_json",
        "purpose": "machine-readable training-data and normalization closure synthesis",
        "path": "reports/latentfm_training_data_normalization_closure_20260624.json",
    },
    {
        "category": "external_review",
        "purpose": "read-only external review for training-data and normalization closure",
        "path": "reports/LATENTFM_TRAINING_DATA_NORMALIZATION_EXTERNAL_REVIEW_20260624.md",
    },
    {
        "category": "script",
        "purpose": "generate post-Locke portfolio synthesis",
        "path": "ops/synthesize_latentfm_post_locke_portfolio_decision_20260624.py",
    },
    {
        "category": "script",
        "purpose": "generate failure-map/provenance package",
        "path": "ops/synthesize_latentfm_failure_map_provenance_20260624.py",
    },
    {
        "category": "script",
        "purpose": "generate figure/table candidates",
        "path": "ops/synthesize_latentfm_figure_table_candidates_20260624.py",
    },
    {
        "category": "script",
        "purpose": "render consolidation figures",
        "path": "ops/render_latentfm_consolidation_figures_20260624.py",
    },
    {
        "category": "script",
        "purpose": "generate manuscript claim package",
        "path": "ops/synthesize_latentfm_manuscript_claim_package_20260624.py",
    },
    {
        "category": "script",
        "purpose": "generate Results-section draft",
        "path": "ops/synthesize_latentfm_results_section_draft_20260624.py",
    },
    {
        "category": "script",
        "purpose": "validate consolidation claim scope",
        "path": "ops/validate_latentfm_consolidation_claim_scope_20260624.py",
    },
    {
        "category": "script",
        "purpose": "generate Nature Methods-style readiness audit",
        "path": "ops/synthesize_latentfm_nature_methods_readiness_audit_20260624.py",
    },
    {
        "category": "script",
        "purpose": "generate reproducibility capsule",
        "path": "ops/synthesize_latentfm_reproducibility_capsule_20260624.py",
    },
    {
        "category": "script",
        "purpose": "generate submission/release index",
        "path": "ops/synthesize_latentfm_submission_release_index_20260624.py",
    },
    {
        "category": "script",
        "purpose": "generate legacy active-run closure audit",
        "path": "ops/synthesize_latentfm_legacy_active_run_closure_audit_20260624.py",
    },
    {
        "category": "script",
        "purpose": "generate training-data and normalization closure synthesis",
        "path": "ops/synthesize_latentfm_training_data_normalization_closure_20260624.py",
    },
]


FIGURE_NAMES = [
    "oracle_headroom_ladder",
    "gain_vs_tail_risk",
    "trackc_overlap_failure",
    "ot_wired_no_gain",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_row(entry: dict[str, str]) -> dict[str, Any]:
    path = ROOT / entry["path"]
    if not path.exists():
        return {
            **entry,
            "abs_path": str(path),
            "exists": False,
            "size_bytes": None,
            "sha256": None,
        }
    stat = path.stat()
    return {
        **entry,
        "abs_path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "sha256": sha256(path),
    }


def main() -> None:
    artifact_entries = list(ARTIFACTS)
    for name in FIGURE_NAMES:
        for ext in ["png", "svg"]:
            artifact_entries.append(
                {
                    "category": f"figure_{ext}",
                    "purpose": f"rendered consolidation figure: {name}",
                    "path": f"reports/figures/latentfm_consolidation_20260624/{name}.{ext}",
                }
            )

    rows = [artifact_row(entry) for entry in artifact_entries]
    missing = [row for row in rows if not row["exists"]]
    duplicate_hashes: dict[str, list[str]] = {}
    for row in rows:
        if row["sha256"]:
            duplicate_hashes.setdefault(row["sha256"], []).append(row["path"])
    duplicate_hashes = {k: v for k, v in duplicate_hashes.items() if len(v) > 1}

    manifest = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "consolidation_artifact_manifest_ready_no_gpu",
        "boundary": {
            "scope": "current LatentFM consolidation closure artifacts only",
            "active_logs": False,
            "raw_canonical_or_query": False,
            "canonical_multi_selection": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "summary": {
            "n_artifacts": len(rows),
            "n_missing": len(missing),
            "n_duplicate_hash_groups": len(duplicate_hashes),
        },
        "artifacts": rows,
        "missing": missing,
        "duplicate_hashes": duplicate_hashes,
    }

    json_path = REPORTS / "latentfm_consolidation_artifact_manifest_20260624.json"
    md_path = REPORTS / "LATENTFM_CONSOLIDATION_ARTIFACT_MANIFEST_20260624.md"
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Consolidation Artifact Manifest",
        "",
        "Status: `consolidation_artifact_manifest_ready_no_gpu`",
        "",
        "## Boundary",
        "",
        "- Scope: current LatentFM consolidation closure artifacts only.",
        "- No active logs, raw canonical/query artifacts, canonical multi selection, training, inference, or GPU.",
        "",
        "## Summary",
        "",
        f"- Artifacts: `{len(rows)}`",
        f"- Missing: `{len(missing)}`",
        f"- Duplicate hash groups: `{len(duplicate_hashes)}`",
        "",
        "## Artifacts",
        "",
        "| Category | Purpose | Size | SHA256 | Path |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        sha = row["sha256"][:12] + "..." if row["sha256"] else "MISSING"
        size = row["size_bytes"] if row["size_bytes"] is not None else ""
        lines.append(
            f"| `{row['category']}` | {row['purpose']} | {size} | `{sha}` | `{row['abs_path']}` |"
        )
    lines.extend(["", "## JSON", "", f"`{json_path}`", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(md_path)
    print(json_path)
    print(f"artifacts={len(rows)} missing={len(missing)} duplicate_hash_groups={len(duplicate_hashes)}")


if __name__ == "__main__":
    main()
