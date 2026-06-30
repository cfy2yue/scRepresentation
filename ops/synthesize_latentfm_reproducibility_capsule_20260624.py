#!/usr/bin/env python3
"""Create a reproducibility capsule for the LatentFM consolidation package."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"


REGENERATION_COMMANDS = [
    "python ops/synthesize_latentfm_failure_map_provenance_20260624.py",
    "python ops/synthesize_latentfm_figure_table_candidates_20260624.py",
    "python ops/render_latentfm_consolidation_figures_20260624.py",
    "python ops/synthesize_latentfm_manuscript_claim_package_20260624.py",
    "python ops/synthesize_latentfm_results_section_draft_20260624.py",
    "python ops/synthesize_latentfm_legacy_active_run_closure_audit_20260624.py",
    "python ops/synthesize_latentfm_training_data_normalization_closure_20260624.py",
    "python ops/validate_latentfm_consolidation_claim_scope_20260624.py",
    "python ops/synthesize_latentfm_submission_release_index_20260624.py",
    "python ops/synthesize_latentfm_nature_methods_readiness_audit_20260624.py",
    "python ops/build_latentfm_consolidation_artifact_manifest_20260624.py",
    "python ops/validate_latentfm_consolidation_claim_scope_20260624.py",
]

PY_COMPILE_COMMAND = (
    "python -m py_compile "
    "ops/synthesize_latentfm_failure_map_provenance_20260624.py "
    "ops/synthesize_latentfm_figure_table_candidates_20260624.py "
    "ops/render_latentfm_consolidation_figures_20260624.py "
    "ops/synthesize_latentfm_manuscript_claim_package_20260624.py "
    "ops/synthesize_latentfm_results_section_draft_20260624.py "
    "ops/synthesize_latentfm_legacy_active_run_closure_audit_20260624.py "
    "ops/synthesize_latentfm_training_data_normalization_closure_20260624.py "
    "ops/synthesize_latentfm_submission_release_index_20260624.py "
    "ops/synthesize_latentfm_nature_methods_readiness_audit_20260624.py "
    "ops/synthesize_latentfm_reproducibility_capsule_20260624.py "
    "ops/build_latentfm_consolidation_artifact_manifest_20260624.py "
    "ops/validate_latentfm_consolidation_claim_scope_20260624.py"
)

JSON_TOOL_COMMANDS = [
    "python -m json.tool reports/latentfm_consolidation_artifact_manifest_20260624.json >/tmp/manifest.json.tool",
    "python -m json.tool reports/latentfm_consolidation_claim_scope_validation_20260624.json >/tmp/validation.json.tool",
    "python -m json.tool reports/latentfm_nature_methods_readiness_audit_20260624.json >/tmp/readiness.json.tool",
    "python -m json.tool reports/latentfm_reproducibility_capsule_20260624.json >/tmp/repro_capsule.json.tool",
]

CORE_OUTPUTS = [
    "reports/LATENTFM_FAILURE_MAP_PROVENANCE_20260624.md",
    "reports/LATENTFM_FIGURE_TABLE_CANDIDATES_20260624.md",
    "reports/LATENTFM_CONSOLIDATION_FIGURES_20260624.md",
    "reports/LATENTFM_MANUSCRIPT_CLAIM_PACKAGE_20260624.md",
    "reports/LATENTFM_RESULTS_SECTION_DRAFT_20260624.md",
    "reports/LATENTFM_CONSOLIDATION_EXTERNAL_WORDING_REVIEW_20260624.md",
    "reports/LATENTFM_LEGACY_ACTIVE_RUN_CLOSURE_AUDIT_20260624.md",
    "reports/LATENTFM_TRAINING_DATA_NORMALIZATION_CLOSURE_20260624.md",
    "reports/LATENTFM_TRAINING_DATA_NORMALIZATION_EXTERNAL_REVIEW_20260624.md",
    "reports/LATENTFM_NATURE_METHODS_READINESS_AUDIT_20260624.md",
    "reports/LATENTFM_SUBMISSION_RELEASE_INDEX_20260624.md",
    "reports/LATENTFM_CONSOLIDATION_ARTIFACT_MANIFEST_20260624.md",
    "reports/LATENTFM_CONSOLIDATION_CLAIM_SCOPE_VALIDATION_20260624.md",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_text(cmd: list[str], cwd: Path = ROOT) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def git_repo_rows() -> list[dict[str, Any]]:
    rows = []
    top = run_text(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"])
    rows.append(
        {
            "path": str(ROOT),
            "is_git_repo": top["returncode"] == 0,
            "toplevel_or_error": (top["stdout"] or top["stderr"]).replace("\n", " / "),
            "commit": None,
            "dirty_short": None,
        }
    )
    candidate_repos = [ROOT / "CoupledFM", ROOT / "scFMBench"]
    candidate_repos.extend(sorted((ROOT / "scFM_third_party").glob("*")))
    for repo in candidate_repos:
        if not (repo / ".git").is_dir():
            continue
        commit = run_text(["git", "-C", str(repo), "rev-parse", "HEAD"])
        dirty = run_text(["git", "-C", str(repo), "status", "--short"])
        rows.append(
            {
                "path": str(repo),
                "is_git_repo": True,
                "toplevel_or_error": str(repo),
                "commit": commit["stdout"] if commit["returncode"] == 0 else None,
                "dirty_short": (
                    dirty["stdout"] if dirty["returncode"] == 0 else dirty["stderr"]
                ).replace("\n", "; "),
            }
        )
    return rows


def main() -> None:
    validation = load_json(REPORTS / "latentfm_consolidation_claim_scope_validation_20260624.json")
    readiness = load_json(REPORTS / "latentfm_nature_methods_readiness_audit_20260624.json")

    output_rows = []
    for rel in CORE_OUTPUTS:
        path = ROOT / rel
        output_rows.append(
            {
                "path": rel,
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
        )

    out = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "reproducibility_capsule_ready_no_gpu",
        "boundary": {
            "reads_consolidation_outputs_only": True,
            "active_logs": False,
            "raw_canonical_or_query": False,
            "canonical_multi_selection": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.replace("\n", " "),
            "platform": platform.platform(),
        },
        "git_boundaries": git_repo_rows(),
        "regeneration_order": REGENERATION_COMMANDS,
        "validation_commands": [PY_COMPILE_COMMAND, *JSON_TOOL_COMMANDS],
        "core_outputs": output_rows,
        "pre_manifest_refresh_validation_summary": validation.get("summary"),
        "pre_manifest_refresh_readiness_summary": readiness.get("summary"),
        "notes": [
            "Top-level /data/cyx/1030/scLatent is not a git repository; top-level goal/docs/reports/ops artifacts cannot be committed from that root without a separate repository decision.",
            "Regeneration commands are short CPU/file tasks and must not be treated as GPU experiments.",
            "Long-running experiment protocol from AGENTS.md remains mandatory for any future training/inference job.",
            "This capsule is generated before the final manifest/validator refresh to avoid a self-referential hash loop.",
            "Re-run the artifact manifest after any regenerated report/script, then re-run the claim-scope validator.",
        ],
    }

    json_path = REPORTS / "latentfm_reproducibility_capsule_20260624.json"
    md_path = REPORTS / "LATENTFM_REPRODUCIBILITY_CAPSULE_20260624.md"
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Reproducibility Capsule",
        "",
        "Status: `reproducibility_capsule_ready_no_gpu`",
        "",
        "## Boundary",
        "",
        "- Reads completed consolidation outputs only.",
        "- Does not read active logs, raw canonical/query artifacts, use canonical multi for selection, train, infer, or use GPU.",
        "",
        "## Environment",
        "",
        f"- Python executable: `{out['environment']['python_executable']}`",
        f"- Python version: `{out['environment']['python_version']}`",
        f"- Platform: `{out['environment']['platform']}`",
        "",
        "## Git Boundaries",
        "",
        "| Path | Git repo | Commit | Dirty status summary |",
        "|---|---:|---|---|",
    ]
    for row in out["git_boundaries"]:
        dirty = row["dirty_short"] if row["dirty_short"] else ""
        commit = row["commit"] if row["commit"] else row["toplevel_or_error"]
        lines.append(f"| `{row['path']}` | `{row['is_git_repo']}` | `{commit}` | `{dirty}` |")

    lines.extend(
        [
            "",
            "## Regeneration Order",
            "",
            "Run from `/data/cyx/1030/scLatent`:",
            "",
            "```bash",
            *REGENERATION_COMMANDS,
            "```",
            "",
            "## Validation Commands",
            "",
            "```bash",
            PY_COMPILE_COMMAND,
            *JSON_TOOL_COMMANDS,
            "```",
            "",
            "## Core Outputs",
            "",
            "| Output | Exists | Size bytes |",
            "|---|---:|---:|",
        ]
    )
    for row in output_rows:
        lines.append(f"| `{row['path']}` | `{row['exists']}` | {row['size_bytes']} |")
    lines.extend(
        [
            "",
            "## Pre-Manifest-Refresh Validation Summary",
            "",
            f"- Claim-scope validator: `{out['pre_manifest_refresh_validation_summary']}`",
            f"- Nature Methods readiness: `{out['pre_manifest_refresh_readiness_summary']}`",
            "",
            "## Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in out["notes"])
    lines.extend(["", "## JSON", "", f"`{json_path}`", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
