#!/usr/bin/env python3
"""CPU-only code gate for a risk-row CVaR/top-k MMD loss branch.

The gate is intentionally conservative.  It checks whether the current
LatentFM training code can implement the proposed row-tail objective as a
distinct mechanism rather than another scalar MMD gamma or dataset-filter
continuation.  It reads only source code and completed train-only internal
reports; it does not read canonical metrics, canonical multi outputs, or Track
C query artifacts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
TRAIN = ROOT / "CoupledFM/model/latent/train.py"
CONFIG = ROOT / "CoupledFM/model/latent/config.py"
DATASET = ROOT / "CoupledFM/model/latent/dataset.py"
TESTS = ROOT / "CoupledFM/model/tests/test_latent_risk_row_cvar_tail_state.py"
LAUNCHER = ROOT / "CoupledFM/model/latent/scripts/run_full_stack_latentfm.sh"
RISK_REPORT_JSON = ROOT / "reports/latentfm_risk_stratified_gate_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_risk_row_cvar_loss_code_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_RISK_ROW_CVAR_LOSS_CODE_GATE_20260624.md"


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    train = load_text(TRAIN)
    config = load_text(CONFIG)
    dataset = load_text(DATASET)
    tests = load_text(TESTS)
    launcher = load_text(LAUNCHER)
    risk = load_json(RISK_REPORT_JSON)

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, evidence: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    add(
        "source_boundary_present",
        TRAIN.is_file() and CONFIG.is_file() and DATASET.is_file() and TESTS.is_file()
        and LAUNCHER.is_file() and RISK_REPORT_JSON.is_file(),
        "Uses train/config/dataset/test/launcher source and completed risk-stratified JSON only.",
    )
    add(
        "risk_stratified_gate_failed_no_gpu",
        risk.get("status") == "risk_stratified_gate_fail_no_gpu",
        f"risk status={risk.get('status')!r}",
    )
    add(
        "train_step_has_condition_mmd",
        "mmd_raw = mmd_fn(x1_hat.float(), gt.float(), sigmas)" in train
        and "dataset_filter_matches(getattr(cfg, \"mmd_dataset_filter\"" in train,
        "Current MMD loss is computed for one yielded dataset/condition batch.",
    )
    add(
        "dataset_yields_single_condition_batches",
        "yield (" in dataset and "ds_name," in dataset and "cond," in dataset,
        "CrossDatasetFMDataset yields one `(ds_name, cond)` batch at a time.",
    )
    add(
        "default_off_cvar_config_absent",
        not re.search(r"risk_.*cvar|cvar_.*mmd|topk_.*mmd|top_k_.*mmd", config, re.I),
        "False means explicit default-off CVaR/top-k risk-row config fields are present.",
    )
    add(
        "cross_condition_tail_state_absent",
        not re.search(r"risk_.*queue|cvar_.*queue|topk_.*queue|risk_.*history|cvar_.*history", train, re.I),
        "False means cross-condition train-only tail history/state is wired in training.",
    )
    add(
        "risk_row_batch_control_present",
        "def risk_row_cvar_batch_control(" in train
        and "risk_row_cvar_batch_control(" in tests
        and "dataset_filter_matches(getattr(cfg, \"risk_row_cvar_dataset_filter\"" in train,
        "Central batch-control helper exists and is directly covered by tests.",
    )
    add(
        "risk_row_batch_control_tests_present",
        "test_batch_control_respects_dataset_filter_exclusion" in tests
        and "test_batch_control_observes_then_applies_nonzero_tail_weight" in tests,
        "Unit tests cover dataset-filter exclusion and observe-then-apply nonzero tail weight.",
    )
    add(
        "train_eval_disable_config_present",
        "train_eval_enabled: bool = True" in config,
        "Default-on train_eval_enabled config allows explicit train-only no-eval smokes.",
    )
    add(
        "epoch_eval_skip_guard_present",
        "train_eval_enabled=False; skipped epoch IID eval" in train
        and "best checkpoint selection" in train,
        "Epoch-end eval/best-selection skip log is present for train-only smokes.",
    )
    add(
        "test_dataset_not_built_when_train_eval_disabled",
        "if is_rank0 and train_eval_enabled:" in train
        and "Test  conditions (IID): skipped because train_eval_enabled=False" in train,
        "Train-only no-eval mode avoids constructing the IID test dataset.",
    )
    add(
        "final_eval_skip_guard_present",
        "Final IID/OOD evaluation skipped because train_eval_enabled=False" in train,
        "Final IID/OOD eval skip log is present for train-only smokes.",
    )
    add(
        "launcher_passes_risk_row_args",
        "RISK_ROW_CVAR_LOSS_WEIGHT" in launcher
        and "--risk-row-cvar-loss-weight" in launcher
        and "--risk-row-cvar-dataset-filter" in launcher,
        "Shared LatentFM launcher exposes and forwards risk-row CVaR config.",
    )
    add(
        "launcher_passes_train_eval_disable",
        "TRAIN_EVAL_ENABLED" in launcher
        and "--no-train-eval-enabled" in launcher,
        "Shared LatentFM launcher can disable train-time IID/OOD eval explicitly.",
    )
    add(
        "risk_row_log_counters_present",
        "risk_row_obs=" in train
        and "risk_row_apply=" in train
        and "avg_risk_row_cvar_w=" in train,
        "Training log exposes observe/apply counts and average active tail weight.",
    )

    passed = {row["name"]: row["passed"] for row in checks}
    code_gate_pass = (
        passed["source_boundary_present"]
        and passed["risk_stratified_gate_failed_no_gpu"]
        and passed["train_step_has_condition_mmd"]
        and not passed["default_off_cvar_config_absent"]
        and not passed["cross_condition_tail_state_absent"]
        and passed["risk_row_batch_control_present"]
        and passed["risk_row_batch_control_tests_present"]
        and passed["train_eval_disable_config_present"]
        and passed["epoch_eval_skip_guard_present"]
        and passed["test_dataset_not_built_when_train_eval_disabled"]
        and passed["final_eval_skip_guard_present"]
        and passed["launcher_passes_risk_row_args"]
        and passed["launcher_passes_train_eval_disable"]
        and passed["risk_row_log_counters_present"]
    )
    status = (
        "risk_row_cvar_loss_code_gate_pass_unit_validated_no_gpu"
        if code_gate_pass
        else "risk_row_cvar_loss_code_gate_fail_no_gpu"
    )

    failed = [row for row in checks if not row["passed"]]
    blocking_reasons = []
    if passed["default_off_cvar_config_absent"]:
        blocking_reasons.append("missing_default_off_cvar_topk_config")
    if passed["cross_condition_tail_state_absent"]:
        blocking_reasons.append("missing_cross_condition_tail_state")
    if not passed["train_step_has_condition_mmd"]:
        blocking_reasons.append("condition_mmd_wiring_not_detected")
    if not passed["risk_row_batch_control_present"]:
        blocking_reasons.append("missing_risk_row_batch_control_helper")
    if not passed["risk_row_batch_control_tests_present"]:
        blocking_reasons.append("missing_batch_control_tests")
    if not passed["train_eval_disable_config_present"]:
        blocking_reasons.append("missing_train_eval_enabled_config")
    if not passed["epoch_eval_skip_guard_present"]:
        blocking_reasons.append("missing_epoch_eval_skip_guard")
    if not passed["test_dataset_not_built_when_train_eval_disabled"]:
        blocking_reasons.append("test_dataset_still_built_when_train_eval_disabled")
    if not passed["final_eval_skip_guard_present"]:
        blocking_reasons.append("missing_final_eval_skip_guard")
    if not passed["launcher_passes_risk_row_args"]:
        blocking_reasons.append("launcher_does_not_forward_risk_row_args")
    if not passed["launcher_passes_train_eval_disable"]:
        blocking_reasons.append("launcher_does_not_forward_train_eval_disable")
    if not passed["risk_row_log_counters_present"]:
        blocking_reasons.append("missing_risk_row_log_counters")

    payload = {
        "status": status,
        "boundary": {
            "cpu_only": True,
            "source_code_only": True,
            "completed_train_only_internal_report_only": True,
            "canonical_metrics_read": False,
            "canonical_multi_read": False,
            "trackc_query_read": False,
        },
        "source_files": {
            "train": str(TRAIN),
            "config": str(CONFIG),
            "dataset": str(DATASET),
            "tests": str(TESTS),
            "launcher": str(LAUNCHER),
            "risk_stratified_json": str(RISK_REPORT_JSON),
        },
        "checks": checks,
        "blocking_reasons": blocking_reasons,
        "decision": {
            "gpu_authorized": False,
            "canonical_authorized": False,
            "recommendation": (
                "GPU remains unauthorized by this code gate alone. If status passes, "
                "the next step is external/code review plus a separate launcher/provenance "
                "gate for exactly one capped train-only smoke; if status fails, fix the "
                "default-off tail-state API or close the branch."
            ),
            "next_non_gpu_action": "external_review_or_launcher_provenance_gate_only_if_status_passes",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM Risk-Row CVaR Loss Code Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only source/report audit.",
        "- Reads completed train-only internal risk-stratified JSON only.",
        "- Does not read canonical metrics, canonical multi, or Track C query.",
        "",
        "## Checks",
        "",
        "| check | pass | evidence |",
        "|---|---:|---|",
    ]
    for row in checks:
        lines.append(f"| `{row['name']}` | `{row['passed']}` | {row['evidence']} |")

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- No GPU or canonical no-harm is authorized by this code gate alone.",
            "- A pass means the default-off tail-state API exists and avoids collapsing into a plain dataset-filtered scalar MMD variant.",
            "- Next action after a pass: external/code review plus a separate launcher/provenance gate for exactly one capped train-only smoke.",
            "- A fail means fix the API/unit tests or close the branch.",
            "",
            "## Output",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
