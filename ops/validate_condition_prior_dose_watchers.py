#!/usr/bin/env python3
"""Static validation for condition-prior dose watcher scripts.

This validator does not inspect live tmux sessions, GPUs, training logs, or
model outputs. It checks that watcher scripts keep the expected marker-based
workflow and do not contain obvious high-frequency polling patterns.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")

SCRIPTS = {
    "primary_posthoc": ROOT / "ops/run_latentfm_condition_prior_teacher_posthoc_20260619.sh",
    "sister_posthoc": ROOT / "ops/run_latentfm_condition_prior_teacher_sister_posthoc_20260619.sh",
    "dose_summary": ROOT / "ops/run_latentfm_condition_prior_teacher_dose_summary_20260619.sh",
}


FORBIDDEN_SNIPPETS = [
    "tail -f",
    "watch ",
    "nvidia-smi",
    "tmux attach",
]


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label}: missing required text: {needle}")


def main() -> int:
    for label, path in SCRIPTS.items():
        text = path.read_text(encoding="utf-8")
        require(text, "sleep 1800", label)
        require(text, "RUN_STATUS.md", label)
        require(text, "EXIT_CODE", label)
        require(text, "FINISHED", label)
        for forbidden in FORBIDDEN_SNIPPETS:
            if forbidden in text:
                raise AssertionError(f"{label}: forbidden polling/log pattern: {forbidden}")

    primary = SCRIPTS["primary_posthoc"].read_text(encoding="utf-8")
    for artifact in (
        "split_group_eval_best_ode20_mse1024_mmd1024.json",
        "condition_family_eval_best_ode20_mse1024_mmd1024.json",
        "condition_residual_full128_best.json",
        "summarize_latentfm_condition_prior_teacher_probe_20260619.py",
    ):
        require(primary, artifact, "primary_posthoc")

    sister = SCRIPTS["sister_posthoc"].read_text(encoding="utf-8")
    for label in ("scf_prior002_e2_4k", "scf_prior010_e2_4k"):
        require(sister, label, "sister_posthoc")
    for artifact in (
        "split_group_eval_best_ode20_mse1024_mmd1024.json",
        "condition_family_eval_best_ode20_mse1024_mmd1024.json",
        "condition_residual_full128_best.json",
    ):
        require(sister, artifact, "sister_posthoc")

    summary = SCRIPTS["dose_summary"].read_text(encoding="utf-8")
    for artifact in (
        "summarize_latentfm_condition_prior_teacher_dose_20260619.py",
        "plot_latentfm_condition_prior_teacher_dose_20260619.py",
        "LATENTFM_CONDITION_PRIOR_TEACHER_DOSE_20260619.md",
        "latentfm_condition_prior_teacher_dose_20260619.{pdf,svg,png}",
    ):
        require(summary, artifact, "dose_summary")

    print("condition-prior dose watcher validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
