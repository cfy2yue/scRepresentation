from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path("/data/cyx/1030/scLatent")
SCRIPT = ROOT / "ops/summarize_latentfm_trackc_anchor_gated_support_teacher_cpu_gate_20260623.py"


def load_module():
    spec = importlib.util.spec_from_file_location("trackc_anchor_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_row(dataset: str, condition: str, pred: list[float], gt: list[float]) -> dict:
    return {
        "dataset": dataset,
        "condition": condition,
        "pred_mean": pred,
        "gt_mean": gt,
        "pert_mean": [0.0, 0.0, 0.0, 0.0],
        "ctrl_mean": [0.0, 0.0, 0.0, 0.0],
    }


def payload(group: str, rows: list[dict]) -> dict:
    return {"groups": {group: {"condition_metrics": rows}}}


def test_support_alpha_gate_detects_anchor_preserving_residual_signal() -> None:
    mod = load_module()
    anchor_payload = payload(
        "test_multi",
        [
            make_row("Wessels", "W1+W2", [0.0, 0.0, 1.0, -1.0], [1.0, -1.0, 0.0, 0.0]),
            make_row("Wessels", "W3+W4", [0.0, 0.0, -1.0, 1.0], [1.0, -1.0, 0.0, 0.0]),
            make_row("NormanWeissman2019_filtered", "N1+N2", [1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]),
        ],
    )
    candidate_payload = payload(
        "test_multi",
        [
            make_row("Wessels", "W1+W2", [1.0, -1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0]),
            make_row("Wessels", "W3+W4", [1.0, -1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0]),
            make_row("NormanWeissman2019_filtered", "N1+N2", [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 1.0, -1.0]),
        ],
    )
    rows = mod.paired_rows(anchor_payload, candidate_payload, "test_multi")
    summary = mod.support_summary(
        rows,
        {"Wessels": 0.50, "NormanWeissman2019_filtered": 0.50},
        0.25,
    )

    assert mod.alpha_passes(summary)
    assert mod.find_dataset(summary, "Wessels")["mean_delta_pp"] >= 0.02
    assert mod.find_dataset(summary, "NormanWeissman2019_filtered")["mean_delta_pp"] >= -0.02


def test_canonical_gate_zero_is_exact_noop() -> None:
    mod = load_module()
    anchor_payload = payload(
        "test_single",
        [
            make_row("DatasetA", "G1", [1.0, -1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0]),
            make_row("DatasetB", "G2", [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 1.0, -1.0]),
        ],
    )
    candidate_payload = payload(
        "test_single",
        [
            make_row("DatasetA", "G1", [0.0, 0.0, 1.0, -1.0], [1.0, -1.0, 0.0, 0.0]),
            make_row("DatasetB", "G2", [1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]),
        ],
    )
    rows = mod.paired_rows(anchor_payload, candidate_payload, "test_single")
    summary = mod.canonical_noop_summary(rows, "test_single")

    assert summary["max_abs_delta_pp"] == 0.0
    assert summary["paired"]["delta_mean"] == 0.0
    assert summary["paired"]["p_harm"] == 0.0


def test_decision_accepts_zero_max_abs_delta_as_exact_noop() -> None:
    mod = load_module()
    selected = {
        "alpha": 0.25,
        "paired": {"p_harm": 0.0},
        "dataset_summary": [
            {
                "dataset": "Wessels",
                "mean_delta_pp": 0.03,
                "route_gap_closed_fraction": 0.10,
            },
            {
                "dataset": "NormanWeissman2019_filtered",
                "mean_delta_pp": 0.00,
                "route_gap_closed_fraction": 0.0,
            },
        ],
    }
    payload = {
        "selected_support_alpha_summary": selected,
        "selected_shuffled_summary": {
            "dataset_summary": [
                {"dataset": "Wessels", "mean_delta_pp": -0.01},
                {"dataset": "NormanWeissman2019_filtered", "mean_delta_pp": -0.01},
            ]
        },
        "canonical_noop": [
            {"group": "test_single", "paired": {"p_harm": 0.0}, "max_abs_delta_pp": 0.0},
            {"group": "family_gene", "paired": {"p_harm": 0.0}, "max_abs_delta_pp": 0.0},
        ],
    }

    decision = mod.decide(payload)

    assert decision["status"] == "trackc_anchor_gated_support_teacher_cpu_gate_pass_code_gate_next"
    assert decision["reasons"] == []


def test_main_fails_closed_when_artifacts_missing(tmp_path: Path) -> None:
    mod = load_module()
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"

    with pytest.raises(FileNotFoundError):
        mod.main_args = None
        import sys

        old_argv = sys.argv
        try:
            sys.argv = [
                str(SCRIPT),
                "--run-root",
                str(tmp_path / "missing_run"),
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ]
            mod.main()
        finally:
            sys.argv = old_argv
