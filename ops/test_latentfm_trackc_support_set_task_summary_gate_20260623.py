from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path("/data/cyx/1030/scLatent")
SCRIPT = ROOT / "ops/summarize_latentfm_trackc_support_set_task_summary_gate_20260623.py"


def load_module():
    spec = importlib.util.spec_from_file_location("support_set_task_summary_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(dataset: str, condition: str, pred: list[float], gt: list[float]) -> dict:
    return {
        "dataset": dataset,
        "condition": condition,
        "pred_mean": pred,
        "gt_mean": gt,
        "pert_mean": [0.0, 0.0, 0.0, 0.0],
        "ctrl_mean": [0.0, 0.0, 0.0, 0.0],
    }


def payload(train_rows: list[dict], support_rows: list[dict]) -> dict:
    return {
        "groups": {
            "train_multi": {"condition_metrics": train_rows},
            "support_val_multi": {"condition_metrics": support_rows},
        }
    }


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def write_route_gap(path: Path) -> None:
    write_json(
        path,
        {
            "real": {
                "dataset_breakdown": [
                    {
                        "dataset": "Wessels",
                        "support_selected_route": 0.0,
                        "candidate": 1.0,
                    },
                    {
                        "dataset": "NormanWeissman2019_filtered",
                        "support_selected_route": 0.0,
                        "candidate": 1.0,
                    },
                ]
            }
        },
    )


def test_support_set_task_summary_gate_passes_with_transferable_train_summary(tmp_path: Path) -> None:
    mod = load_module()
    run_root = tmp_path / "run"
    route_gap = tmp_path / "route_gap.json"
    write_route_gap(route_gap)
    anchor_train = [
        row("Wessels", "W_train_1", [0.0, 0.0, 1.0, -1.0], [1.0, -1.0, 0.0, 0.0]),
        row("Wessels", "W_train_2", [0.0, 0.0, 1.0, -1.0], [1.0, -1.0, 0.0, 0.0]),
        row("NormanWeissman2019_filtered", "N_train_1", [1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]),
        row("NormanWeissman2019_filtered", "N_train_2", [1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]),
    ]
    candidate_train = [
        row("Wessels", "W_train_1", [1.0, -1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0]),
        row("Wessels", "W_train_2", [1.0, -1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0]),
        row("NormanWeissman2019_filtered", "N_train_1", [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 1.0, -1.0]),
        row("NormanWeissman2019_filtered", "N_train_2", [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 1.0, -1.0]),
    ]
    anchor_support = [
        row("Wessels", "W_support", [0.0, 0.0, 1.0, -1.0], [1.0, -1.0, 0.0, 0.0]),
        row("NormanWeissman2019_filtered", "N_support", [1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]),
    ]
    candidate_support = [
        row("Wessels", "W_support", [1.0, -1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0]),
        row("NormanWeissman2019_filtered", "N_support", [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 1.0, -1.0]),
    ]
    write_json(
        run_root / "condition_means/trainselect_anchor_train_support_multi_condition_means_ode20.json",
        payload(anchor_train, anchor_support),
    )
    write_json(
        run_root / "condition_means/trainselect_candidate_train_support_multi_condition_means_ode20.json",
        payload(candidate_train, candidate_support),
    )

    out = mod.build_payload(run_root, route_gap)

    assert out["decision"]["status"] == "trackc_support_set_task_summary_gate_pass_posthoc_mmd_gate_next_no_gpu"
    assert out["selected_train_loo_summary"]["alpha"] == 0.25
    assert mod.support_gate_passes(out["support_val_summary"])
    assert not mod.support_gate_passes(out["zero_support_control"])


def test_support_set_task_summary_gate_fails_when_train_loo_has_no_signal(tmp_path: Path) -> None:
    mod = load_module()
    run_root = tmp_path / "run"
    route_gap = tmp_path / "route_gap.json"
    write_route_gap(route_gap)
    anchor_train = [
        row("Wessels", "W_train_1", [1.0, -1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0]),
        row("Wessels", "W_train_2", [1.0, -1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0]),
        row("NormanWeissman2019_filtered", "N_train_1", [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 1.0, -1.0]),
        row("NormanWeissman2019_filtered", "N_train_2", [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 1.0, -1.0]),
    ]
    candidate_train = list(anchor_train)
    anchor_support = [
        row("Wessels", "W_support", [1.0, -1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0]),
        row("NormanWeissman2019_filtered", "N_support", [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 1.0, -1.0]),
    ]
    candidate_support = list(anchor_support)
    write_json(
        run_root / "condition_means/trainselect_anchor_train_support_multi_condition_means_ode20.json",
        payload(anchor_train, anchor_support),
    )
    write_json(
        run_root / "condition_means/trainselect_candidate_train_support_multi_condition_means_ode20.json",
        payload(candidate_train, candidate_support),
    )

    out = mod.build_payload(run_root, route_gap)

    assert out["decision"]["status"] == "trackc_support_set_task_summary_gate_fail_no_gpu"
    assert "no_alpha_passed_train_multi_loo_gate" in out["decision"]["reasons"]
    assert out["support_val_summary"] is None


def test_support_set_task_summary_gate_missing_artifacts_fail_closed(tmp_path: Path) -> None:
    mod = load_module()
    with pytest.raises(FileNotFoundError):
        mod.build_payload(tmp_path / "missing", tmp_path / "route_gap.json")
