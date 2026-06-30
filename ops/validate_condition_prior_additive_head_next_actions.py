#!/usr/bin/env python3
"""Validate additive-head next-action playbook content."""
from __future__ import annotations

from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
REPORT = ROOT / "reports/LATENTFM_CONDITION_PRIOR_ADDITIVE_HEAD_NEXT_ACTIONS_20260619.md"


def require(text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing required text: {needle}")


def main() -> int:
    text = REPORT.read_text(encoding="utf-8")
    for needle in (
        "Scenario A: Additive-Head Is Still Pending",
        "Scenario B: Additive-Head Becomes `repeat_candidate`",
        "Scenario C: Additive-Head Is Only `diagnostic_candidate`",
        "Scenario D: Additive-Head Is `reject_as_is`",
        "No broad scalar/additive weight sweep from partial data.",
        "Wessels unseen2",
        "combo/additive cosine",
        "train-single-only prior bank",
        "train_multi=0",
        "LATENTFM_CONDITION_PRIOR_ADDITIVE_HEAD_REPEAT_PLAN_20260619.md",
        "LATENTFM_CONDITION_PRIOR_ADDITIVE_HEAD_DIAGNOSTIC_INTERPRETATION_20260619.md",
        "LATENTFM_CONDITION_PRIOR_ADDITIVE_HEAD_NEGATIVE_RESULT_20260619.md",
        "/data/cyx/1030/scLatent/AGENTS.md",
    ):
        require(text, needle)
    print("condition-prior additive-head next-actions validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
