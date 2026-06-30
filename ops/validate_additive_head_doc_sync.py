#!/usr/bin/env python3
"""Validate that current-state docs point to the additive-head smoke.

This is a read-only documentation consistency check. Historical timeline files
may mention older branches, but the current-state documents should not present
relational residual, strategy probes, or the dose response as the active
LatentFM branch after the additive-head smoke launch.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")

CURRENT_DOCS = {
    "project_overview": ROOT / "docs/PROJECT_OVERVIEW.md",
    "results_summary": ROOT / "docs/RESULTS_SUMMARY.md",
    "model_notes": ROOT / "docs/MODEL_NOTES.md",
    "operations_handoff": ROOT / "reports/OPERATIONS_HANDOFF_20260619.md",
}

SUPPORTING_DOCS = {
    "experiment_index": ROOT / "docs/EXPERIMENT_INDEX.md",
    "decisions": ROOT / "docs/DECISIONS.md",
    "workspace_status": ROOT / "reports/WORKSPACE_STATUS.md",
}

REQUIRED_CURRENT = [
    "scf_prioradd005_prior010_inject_e2_4k",
    "condition-prior additive-head",
    "condition_prior_additive_delta_loss_weight=0.05",
]

REQUIRED_SUPPORTING = [
    "scf_prioradd005_prior010_inject_e2_4k",
    "LATENTFM_CONDITION_PRIOR_ADDITIVE_HEAD_NEXT_ACTIONS_20260619.md",
]

FORBIDDEN_CURRENT_ACTIVE = [
    "The active direction is a\ndefault-off scFoundation residual relational objective",
    "The active direction is now a\nshort strategy-search block",
    "Current active LatentFM direction as of 2026-06-19 12",
    "condition-prior teacher dose response\n```",
]


def read(path: Path) -> str:
    if not path.is_file():
        raise AssertionError(f"missing doc: {path}")
    return path.read_text(encoding="utf-8")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label}: missing required text: {needle}")


def main() -> int:
    for label, path in CURRENT_DOCS.items():
        text = read(path)
        for needle in REQUIRED_CURRENT:
            require(text, needle, label)
        for forbidden in FORBIDDEN_CURRENT_ACTIVE:
            if forbidden in text:
                raise AssertionError(f"{label}: stale active-state text remains: {forbidden!r}")

    for label, path in SUPPORTING_DOCS.items():
        text = read(path)
        for needle in REQUIRED_SUPPORTING:
            require(text, needle, label)

    print("additive-head doc sync validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
