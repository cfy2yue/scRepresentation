#!/usr/bin/env python3
"""Validate lightweight project handoff docs without touching training jobs."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
REPORT = ROOT / "reports/HANDOFF_DOCS_VALIDATION_20260619.md"

REQUIRED = [
    ROOT / "README.md",
    ROOT / "LOCAL_4X4090_RUNBOOK.md",
    ROOT / "AGENTS.md",
    ROOT / "goal.md",
    ROOT / "docs/PROJECT_OVERVIEW.md",
    ROOT / "docs/PROJECT_REVIEW.md",
    ROOT / "docs/EXPERIMENT_INDEX.md",
    ROOT / "docs/DATA_PIPELINE.md",
    ROOT / "docs/RESULTS_SUMMARY.md",
    ROOT / "docs/MODEL_NOTES.md",
    ROOT / "docs/DECISIONS.md",
    ROOT / "docs/BUGS_AND_FIXES.md",
    ROOT / "reports/WORKSPACE_STATUS.md",
    ROOT / "reports/GOAL_REQUIREMENT_STATUS_20260619.md",
    ROOT / "reports/GOAL_COMPLETION_AUDIT_20260619_1252.md",
    ROOT / "reports/目标推进阶段报告_20260619_1309.md",
    ROOT / "reports/目标推进阶段报告_20260619_1628.md",
    ROOT / "reports/LATENTFM_RELATIONAL_RESIDUAL_AUTOMATION_CHAIN_20260619.md",
    ROOT / "reports/LATENTFM_RELATIONAL_RESIDUAL_DECISION_GATE_20260619.md",
    ROOT / "reports/LATENTFM_POST_RELATIONAL_NEXT_ACTIONS_20260619.md",
    ROOT / "reports/LATENTFM_RELATIONAL_ONE_SHOT_STATUS_20260619.md",
    ROOT / "reports/LATENTFM_CONDITION_PRIOR_TEACHER_DOSE_20260619.md",
    ROOT / "reports/LATENTFM_CONDITION_PRIOR_DOSE_NEXT_ACTIONS_20260619.md",
    ROOT / "reports/LATENTFM_CONDITION_PRIOR_DIAGNOSTIC_INTERPRETATION_20260619.md",
    ROOT / "reports/LATENTFM_CONDITION_PRIOR_CONDITION_LEVEL_COMPARISON_20260619.md",
    ROOT / "reports/LATENTFM_CONDITION_PRIOR_BIOLOGICAL_INSIGHT_20260619.md",
    ROOT / "reports/LATENTFM_CONDITION_PRIOR_INJECTION_COMPARISON_20260619.md",
    ROOT / "reports/LATENTFM_CONDITION_PRIOR_INJECTION_CONDITION_LEVEL_20260619.md",
    ROOT / "reports/LATENTFM_ADDITIVE_INTERACTION_MODULE_DESIGN_20260619.md",
    ROOT / "reports/LATENTFM_CONDITION_DELTA_DECOMPOSITION_20260619.md",
    ROOT / "reports/LATENTFM_CONDITION_PRIOR_ADDITIVE_HEAD_COMPARISON_20260619.md",
    ROOT / "reports/CONDITION_PRIOR_ADDITIVE_HEAD_READOUT_SUMMARY_20260619.md",
    ROOT / "reports/LATENTFM_CONDITION_PRIOR_ADDITIVE_HEAD_NEXT_ACTIONS_20260619.md",
    ROOT / "reports/CONDITION_PRIOR_DOSE_READOUT_SUMMARY_20260619.md",
    ROOT / "reports/OPERATIONS_HANDOFF_20260619.md",
    ROOT / "reports/MANUSCRIPT_FIGURE_ARCHITECTURE_20260619.md",
    ROOT / "ops/check_relational_decision_once.sh",
    ROOT / "ops/check_condition_prior_dose_once.sh",
    ROOT / "ops/validate_condition_prior_dose_one_shot.py",
    ROOT / "ops/validate_condition_prior_dose_watchers.py",
    ROOT / "ops/summarize_condition_prior_one_shot_readout.py",
    ROOT / "ops/summarize_condition_prior_condition_level_20260619.py",
    ROOT / "ops/summarize_latentfm_condition_prior_injection_20260619.py",
    ROOT / "ops/summarize_condition_prior_injection_condition_level_20260619.py",
    ROOT / "ops/summarize_latentfm_additive_interaction_design_20260619.py",
    ROOT / "ops/summarize_latentfm_condition_delta_decomposition_20260619.py",
    ROOT / "ops/launch_latentfm_condition_prior_additive_head_20260619.sh",
    ROOT / "ops/run_latentfm_condition_prior_additive_head_posthoc_20260619.sh",
    ROOT / "ops/run_latentfm_condition_prior_additive_head_summary_20260619.sh",
    ROOT / "ops/summarize_latentfm_condition_prior_additive_head_20260619.py",
    ROOT / "ops/summarize_condition_prior_additive_head_readout.py",
    ROOT / "ops/validate_condition_prior_additive_head_pipeline.py",
    ROOT / "ops/validate_condition_prior_additive_head_readout.py",
    ROOT / "ops/validate_condition_prior_additive_head_next_actions.py",
    ROOT / "ops/validate_additive_head_doc_sync.py",
    ROOT / "ops/validate_condition_prior_readout_summary.py",
    ROOT / "ops/validate_latentfm_condition_prior_teacher_dose_pipeline.py",
    ROOT / "ops/run_latentfm_condition_prior_teacher_injection_posthoc_20260619.sh",
    ROOT / "ops/run_latentfm_condition_prior_teacher_injection_summary_20260619.sh",
    ROOT / "ops/generate_workspace_status.py",
    ROOT / "ops/validate_workspace_status.py",
    ROOT / "runs/latentfm_scfoundation_relational_residual_20260619/RUN_STATUS.md",
    ROOT / "runs/latentfm_condition_prior_teacher_dose_summary_20260619/RUN_STATUS.md",
    ROOT / "runs/latentfm_condition_prior_teacher_injection_20260619/RUN_STATUS.md",
    ROOT / "runs/latentfm_condition_prior_teacher_injection_posthoc_20260619/RUN_STATUS.md",
    ROOT / "runs/latentfm_condition_prior_teacher_injection_summary_20260619/RUN_STATUS.md",
    ROOT / "runs/latentfm_condition_prior_additive_head_20260619/RUN_STATUS.md",
    ROOT / "runs/latentfm_condition_prior_additive_head_posthoc_20260619/RUN_STATUS.md",
    ROOT / "runs/latentfm_condition_prior_additive_head_summary_20260619/RUN_STATUS.md",
    ROOT / "runs/condition_prior_dose_one_shot_1350_20260619/RUN_STATUS.md",
]

DOCS_TO_SCAN = [
    ROOT / "README.md",
    ROOT / "LOCAL_4X4090_RUNBOOK.md",
    ROOT / "docs/PROJECT_OVERVIEW.md",
    ROOT / "docs/PROJECT_REVIEW.md",
    ROOT / "docs/EXPERIMENT_INDEX.md",
    ROOT / "docs/DATA_PIPELINE.md",
    ROOT / "docs/RESULTS_SUMMARY.md",
    ROOT / "docs/MODEL_NOTES.md",
    ROOT / "docs/DECISIONS.md",
    ROOT / "docs/BUGS_AND_FIXES.md",
    ROOT / "reports/OPERATIONS_HANDOFF_20260619.md",
    ROOT / "reports/WORKSPACE_STATUS.md",
    ROOT / "reports/GOAL_COMPLETION_AUDIT_20260619_1252.md",
]

TOKEN_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9_]+"),
    re.compile(r"GHTOKEN_1030\s*=\s*['\"]?[^'\"\s]+"),
    re.compile(r"github_token\s*=\s*['\"]?[^'\"\s]+", re.IGNORECASE),
]

ABS_PATH_RE = re.compile(r"/data/cyx/1030/scLatent/[A-Za-z0-9._/\-]+")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def path_from_match(raw: str) -> Path:
    return Path(raw.rstrip("`'\"),.;:"))


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    checked_paths: set[Path] = set()

    for path in REQUIRED:
        if not path.exists():
            errors.append(f"missing required file: {path}")

    for doc in DOCS_TO_SCAN:
        if not doc.is_file():
            continue
        text = read_text(doc)
        for pattern in TOKEN_PATTERNS:
            if pattern.search(text):
                errors.append(f"token-like pattern found in {doc}: {pattern.pattern}")
        for match in ABS_PATH_RE.finditer(text):
            path = path_from_match(match.group(0))
            checked_paths.add(path)
            # Generated future reports may be intentionally pending before the
            # scheduled decision chain. Keep those as warnings instead of hard
            # failures.
            if path.exists():
                continue
            if "LATENTFM_SCFOUNDATION_RELATIONAL_RESIDUAL_REPORT_20260619" in str(path):
                warnings.append(f"future/pending report path not present yet: {path}")
            elif "latentfm_scfoundation_relational_residual_status_20260619" in str(path):
                warnings.append(f"future/pending status JSON not present yet: {path}")
            elif "scFM_output/figures_manuscript" in str(path):
                warnings.append(f"future manuscript figure output directory not present yet: {path}")
            elif "latentfm_condition_prior_teacher_dose_20260619" in str(path):
                family = sorted(path.parent.glob(path.name + ".*"))
                if not family:
                    warnings.append(f"future/pending condition-prior dose artifact not present yet: {path}")
            elif "LATENTFM_CONDITION_PRIOR_ADDITIVE_HEAD_ONE_SHOT_STATUS" in str(path):
                warnings.append(f"future/pending additive-head one-shot report not present yet: {path}")
            elif "CoupledFM/data/raw/genepert_DE5000/metainfo.json" in str(path):
                warnings.append(f"known missing local smoke-test fixture documented in BUGS_AND_FIXES: {path}")
            else:
                errors.append(f"referenced absolute path missing in {doc}: {path}")

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "required_count": len(REQUIRED),
        "scanned_docs": [str(p) for p in DOCS_TO_SCAN if p.is_file()],
        "referenced_absolute_paths_checked": len(checked_paths),
    }

    lines = [
        "# Handoff Docs Validation 2026-06-19",
        "",
        f"Generated: {payload['generated']}",
        "",
        "This is a lightweight read-only validation. It does not inspect GPU",
        "utilization, attach to tmux, tail training logs, or launch jobs.",
        "",
        "## Status",
        "",
        f"`{payload['status']}`",
        "",
        "## Summary",
        "",
        f"- Required files checked: {payload['required_count']}",
        f"- Markdown files scanned: {len(payload['scanned_docs'])}",
        f"- Absolute `/data/cyx/1030/scLatent/...` paths checked: {payload['referenced_absolute_paths_checked']}",
        f"- Errors: {len(errors)}",
        f"- Warnings: {len(warnings)}",
        "",
        "## Errors",
        "",
    ]
    lines.extend([f"- {err}" for err in errors] or ["- none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warn}" for warn in warnings] or ["- none"])
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
