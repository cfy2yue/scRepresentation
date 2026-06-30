#!/usr/bin/env python3
"""CPU-only guards for LatentFM stablecaps gate and bootstrap helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summary = _load_module(
    "summarize_latentfm_sampling_smokes_stablecaps_20260619",
    OPS / "summarize_latentfm_sampling_smokes_stablecaps_20260619.py",
)
bootstrap = _load_module(
    "bootstrap_latentfm_condition_metrics_20260619",
    OPS / "bootstrap_latentfm_condition_metrics_20260619.py",
)
focus_audit = _load_module(
    "audit_latentfm_focus_learnability_gate_20260619",
    OPS / "audit_latentfm_focus_learnability_gate_20260619.py",
)
focus_decision = _load_module(
    "decide_latentfm_focus_next_action_20260619",
    OPS / "decide_latentfm_focus_next_action_20260619.py",
)


def _row(
    run: str,
    group: str,
    *,
    pp: float,
    mmd: float = 1.0,
    keys: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "run": run,
        "group": group,
        "n_requested": len(keys or []),
        "n_conds": len(keys or []),
        "mmd": mmd,
        "mmd_biased": None,
        "mmd_clamped": None,
        "mmd_gate": mmd,
        "mmd_gate_metric": "test_mmd",
        "dp": 0.1,
        "pc": 0.1,
        "pp": pp,
        "selected_condition_keys": keys or ["ds\tcond"],
    }


def test_gate_invalidates_selection_mismatch() -> None:
    baseline = "baseline"
    run = "candidate"
    rows = [
        _row(baseline, "test", pp=0.1, keys=["ds\tcond_a"]),
        _row(baseline, "family_gene", pp=0.1, keys=["ds\tcond_a"]),
        _row(baseline, "test_multi_unseen2", pp=-0.1, keys=["Wessels\tcond_a"]),
        _row(run, "test", pp=0.2, keys=["ds\tcond_b"]),
        _row(run, "family_gene", pp=0.2, keys=["ds\tcond_a"]),
        _row(run, "test_multi_unseen2", pp=0.1, keys=["Wessels\tcond_a"]),
    ]
    dataset_rows = [
        {
            "run": baseline,
            "group": "test_multi_unseen2",
            "dataset": "Wessels",
            "n_selected_conditions": 1,
            "mmd": 1.0,
            "dp": 0.1,
            "pc": 0.1,
            "pp": -0.1,
        },
        {
            "run": run,
            "group": "test_multi_unseen2",
            "dataset": "Wessels",
            "n_selected_conditions": 1,
            "mmd": 1.0,
            "dp": 0.1,
            "pc": 0.1,
            "pp": 0.1,
        },
    ]
    gate = summary._gate_status(rows, dataset_rows, baseline)
    assert gate[0]["triage_status"] == "invalid_selection_mismatch"
    assert gate[0]["checks"]["selected_conditions_match_baseline"]["pass"] is False


def test_mmd_gate_uses_common_available_metric() -> None:
    baseline = {"mmd": 1.0, "mmd_biased": None, "mmd_clamped": None}
    run = {"mmd": 1.2, "mmd_biased": 1.1, "mmd_clamped": 1.05}
    source_key, row_key = summary._common_mmd_gate_metric(baseline, run)
    assert source_key == "test_mmd"
    assert row_key == "mmd"

    baseline["mmd_clamped"] = 0.9
    source_key, row_key = summary._common_mmd_gate_metric(baseline, run)
    assert source_key == "test_mmd_clamped"
    assert row_key == "mmd_clamped"


def test_bootstrap_reports_coverage_and_metric_direction() -> None:
    baseline_rows = {
        ("ds1", "cond_a"): {"dataset": "ds1", "condition": "cond_a", "test_mmd": 1.0},
        ("ds1", "cond_b"): {"dataset": "ds1", "condition": "cond_b", "test_mmd": 1.2},
    }
    run_rows = {
        ("ds1", "cond_a"): {"dataset": "ds1", "condition": "cond_a", "test_mmd": 0.8},
    }
    result = bootstrap._paired_delta_ci(
        baseline_rows,
        run_rows,
        metric="test_mmd",
        seed=7,
        n_boot=16,
    )
    assert result["status"] == "ok"
    assert result["lower_is_better"] is True
    assert result["n_baseline_conditions"] == 2
    assert result["n_run_conditions"] == 1
    assert result["n_pairs"] == 1
    assert result["warning"] == "paired_condition_coverage_below_80pct"


def test_focus_audit_requires_matching_selected_conditions() -> None:
    baseline_group = {
        "n_conds": 1,
        "test_mmd": 1.0,
        "pearson_pert": -0.2,
        "pearson_ctrl": 0.1,
        "direct_pearson": 0.1,
        "selected_conditions": [{"dataset": "Wessels", "condition": "cond_a"}],
        "per_ds_p_pert": {"Wessels": -0.2},
        "per_ds_mmd": {"Wessels": 1.0},
    }
    run_group = {
        "n_conds": 1,
        "test_mmd": 1.0,
        "pearson_pert": 0.1,
        "pearson_ctrl": 0.1,
        "direct_pearson": 0.1,
        "selected_conditions": [{"dataset": "Wessels", "condition": "cond_b"}],
        "per_ds_p_pert": {"Wessels": 0.1},
        "per_ds_mmd": {"Wessels": 1.0},
    }
    payload = {
        "groups": {
            "test": baseline_group,
            "test_multi": baseline_group,
            "test_multi_seen": baseline_group,
            "test_multi_unseen1": baseline_group,
            "test_multi_unseen2": baseline_group,
        }
    }
    run_payload = {
        "groups": {
            "test": run_group,
            "test_multi": run_group,
            "test_multi_seen": run_group,
            "test_multi_unseen1": run_group,
            "test_multi_unseen2": run_group,
        }
    }

    # Test the same logic used by audit() without touching filesystem.
    mismatches = []
    for group in focus_audit.GROUPS:
        if focus_audit._selected_keys(payload, group) != focus_audit._selected_keys(run_payload, group):
            mismatches.append(group)
    assert mismatches == list(focus_audit.GROUPS)


def _focus_decision_row(status: str, *, wessels_delta: float) -> dict[str, Any]:
    return {
        "run": "scf_prior010_inject_nwg_focus_4k",
        "status": status,
        "selection_mismatches": [] if status != "invalid_selection_mismatch" else [{"group": "test"}],
        "groups": {
            "test": {"mmd_gate": {"ratio": 1.0}},
            "test_multi_unseen2": {"pearson_pert": {"delta": 0.1}},
        },
        "focus_dataset_unseen2": [
            {"dataset": "Wessels", "unseen2_pp_delta": wessels_delta},
            {"dataset": "NormanWeissman2019_filtered", "unseen2_pp_delta": 0.1},
            {"dataset": "GasperiniShendure2019_lowMOI", "unseen2_pp_delta": 0.1},
        ],
    }


def test_focus_decision_routes_wessels_rescue_to_balance() -> None:
    payload = {"baseline": "baseline", "runs": [_focus_decision_row("focus_learnability_signal", wessels_delta=0.05)]}
    decision = focus_decision.decide(payload)
    assert decision["next_action"] == "launch_stronger_all_split_balance_4k"


def test_focus_decision_rejects_selection_mismatch() -> None:
    payload = {"baseline": "baseline", "runs": [_focus_decision_row("invalid_selection_mismatch", wessels_delta=0.05)]}
    decision = focus_decision.decide(payload)
    assert decision["next_action"] == "rerun_or_reaudit_focus_posthoc_selection_mismatch"


def main() -> int:
    test_gate_invalidates_selection_mismatch()
    test_mmd_gate_uses_common_available_metric()
    test_bootstrap_reports_coverage_and_metric_direction()
    test_focus_audit_requires_matching_selected_conditions()
    test_focus_decision_routes_wessels_rescue_to_balance()
    test_focus_decision_rejects_selection_mismatch()
    print("latentfm_metric_guards_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
