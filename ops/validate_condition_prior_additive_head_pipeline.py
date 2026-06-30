#!/usr/bin/env python3
"""Static validation for the condition-prior additive-head smoke chain.

This validator is intentionally read-only. It does not inspect live tmux
sessions, GPUs, training logs, checkpoints, or model outputs. It checks that
the launch, posthoc, summary, and summarizer files preserve the intended
single-branch smoke design and marker-based long-task behavior.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")

RUN_SCRIPT = (
    ROOT
    / "runs/latentfm_condition_prior_additive_head_20260619/"
    "run_scf_prioradd005_prior010_inject_e2_4k.sh"
)
LAUNCH = ROOT / "ops/launch_latentfm_condition_prior_additive_head_20260619.sh"
POSTHOC = ROOT / "ops/run_latentfm_condition_prior_additive_head_posthoc_20260619.sh"
SUMMARY = ROOT / "ops/run_latentfm_condition_prior_additive_head_summary_20260619.sh"
SUMMARIZER = ROOT / "ops/summarize_latentfm_condition_prior_additive_head_20260619.py"
READOUT = ROOT / "ops/summarize_condition_prior_additive_head_readout.py"
NEXT_ACTIONS = ROOT / "reports/LATENTFM_CONDITION_PRIOR_ADDITIVE_HEAD_NEXT_ACTIONS_20260619.md"


FORBIDDEN_POLLING_SNIPPETS = [
    "tail -f",
    "watch ",
    "tmux attach",
]


def read(path: Path) -> str:
    if not path.is_file():
        raise AssertionError(f"missing required file: {path}")
    return path.read_text(encoding="utf-8")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label}: missing required text: {needle}")


def forbid(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise AssertionError(f"{label}: forbidden text: {needle}")


def validate_run_script() -> None:
    text = read(RUN_SCRIPT)
    for needle in (
        "scf_prioradd005_prior010_inject_e2_4k",
        "--total-steps 4000",
        "--init-checkpoint /data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/full_scfoundation/20260617_scfoundation_comp006_delta_w5_12k/best.pt",
        "--condition-delta-head-use-in-model",
        "--condition-prior-delta-loss-weight 0.10",
        "--condition-prior-additive-delta-loss-weight 0.05",
        "--condition-prior-additive-delta-loss-warmup-start 0",
        "--condition-prior-additive-delta-loss-warmup-end 1000",
        "--condition-prior-delta-loss-every 1",
        "--condition-prior-bank-max-cells 512",
        "--condition-prior-num-genes 2",
        "--ot-sinkhorn-iter 50",
        "OMP_NUM_THREADS=4",
    ):
        require(text, needle, "run_script")


def validate_launch() -> None:
    text = read(LAUNCH)
    for needle in (
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv",
        "select_available_gpus.py",
        "--samples 3",
        "--interval-seconds 10",
        "--need 1",
        "--max-jobs-per-gpu 3",
        "tmux new-session -d -s",
        "CUDA_VISIBLE_DEVICES='${gpu}'",
        "RUN_STATUS.md",
        "EXIT_CODE",
        "FINISHED",
        "Runtime classification: Long GPU task.",
        "downstream watchers sleep 1800 seconds",
    ):
        require(text, needle, "launch")
    for forbidden in FORBIDDEN_POLLING_SNIPPETS:
        forbid(text, forbidden, "launch")


def validate_posthoc() -> None:
    text = read(POSTHOC)
    for needle in (
        "sleep 1800",
        "waiting_for_training_exit",
        "select_available_gpus.py",
        "--samples 3",
        "--interval-seconds 10",
        "--max-jobs-per-gpu 3",
        "split_group_eval_best_ode20_mse1024_mmd1024.json",
        "condition_family_eval_best_ode20_mse1024_mmd1024.json",
        "condition_residual_full128_best.json",
        "condition_delta_decomposition_full128_best.json",
        "eval_condition_delta_decomposition",
        "RUN_STATUS.md",
        "EXIT_CODE",
        "FINISHED",
    ):
        require(text, needle, "posthoc")
    for forbidden in FORBIDDEN_POLLING_SNIPPETS:
        forbid(text, forbidden, "posthoc")


def validate_summary() -> None:
    text = read(SUMMARY)
    for needle in (
        "sleep 1800",
        "waiting_for_posthoc",
        "summarize_latentfm_condition_prior_additive_head_20260619.py",
        "summarize_condition_prior_additive_head_readout.py",
        "LATENTFM_CONDITION_PRIOR_ADDITIVE_HEAD_COMPARISON_20260619.md",
        "CONDITION_PRIOR_ADDITIVE_HEAD_READOUT_SUMMARY_20260619.md",
        "latentfm_condition_prior_additive_head_comparison_20260619.csv",
        "latentfm_condition_prior_additive_head_comparison_20260619.json",
        "RUN_STATUS.md",
        "EXIT_CODE",
        "FINISHED",
    ):
        require(text, needle, "summary")
    for forbidden in FORBIDDEN_POLLING_SNIPPETS:
        forbid(text, forbidden, "summary")


def validate_summarizer() -> None:
    text = read(SUMMARIZER)
    for needle in (
        "scf_prior010_e2_4k",
        "scf_prior010_inject_e2_4k",
        "scf_prioradd005_prior010_inject_e2_4k",
        "condition_delta_decomposition_full128_best.json",
        "decomp_wessels_unseen2_combo_additive_cosine",
        "condition_prior_additive_delta_loss_weight=0.05",
        "return 0 if status == \"complete\" else 2",
    ):
        require(text, needle, "summarizer")


def validate_readout() -> None:
    text = read(READOUT)
    for needle in (
        "latentfm_condition_prior_additive_head_comparison_20260619.json",
        "CONDITION_PRIOR_ADDITIVE_HEAD_READOUT_SUMMARY_20260619.md",
        "Additive-head 分支仍未完成",
        "repeat_candidate",
        "split-aware additive-plus-interaction",
        "return 0 if not pending else 2",
    ):
        require(text, needle, "readout")


def validate_next_actions() -> None:
    text = read(NEXT_ACTIONS)
    for needle in (
        "Scenario A: Additive-Head Is Still Pending",
        "Scenario B: Additive-Head Becomes `repeat_candidate`",
        "Scenario C: Additive-Head Is Only `diagnostic_candidate`",
        "Scenario D: Additive-Head Is `reject_as_is`",
        "No broad scalar/additive weight sweep from partial data.",
        "Wessels unseen2",
        "train_multi=0",
    ):
        require(text, needle, "next_actions")


def main() -> int:
    validate_run_script()
    validate_launch()
    validate_posthoc()
    validate_summary()
    validate_summarizer()
    validate_readout()
    validate_next_actions()
    print("condition-prior additive-head pipeline validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
