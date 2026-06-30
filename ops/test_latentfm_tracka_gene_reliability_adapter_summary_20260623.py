from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
SCRIPT = ROOT / "ops/summarize_latentfm_crosslatent_tracka_gene_reliability_adapter_block_20260623.py"


def load_module():
    spec = importlib.util.spec_from_file_location("tracka_gene_summary", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def manifest_row(tmp_path: Path, run_name: str, gate_path: Path) -> dict:
    return {
        "run_name": run_name,
        "latent": "scfoundation",
        "aggregation": "gene_shrink_k2",
        "physical_gpu": 0,
        "run_status": str(tmp_path / "runs" / run_name / "RUN_STATUS.md"),
        "gate_json": str(gate_path),
        "decision_md": str(tmp_path / "reports" / f"{run_name}.md"),
    }


def passing_gate() -> dict:
    return {
        "gate": {"status": "candidate_gate_pass", "reasons": []},
        "paired_deltas": [
            {
                "stratum": "cross_background_seen_gene",
                "metric": "pearson_pert",
                "delta_mean": 0.031,
                "p_improve": 0.95,
                "p_harm": 0.05,
                "n_matched_conditions": 10,
                "n_matched_datasets": 3,
            },
            {
                "stratum": "all_test_single",
                "metric": "pearson_pert",
                "delta_mean": 0.001,
                "p_improve": 0.70,
                "p_harm": 0.10,
                "n_matched_conditions": 10,
                "n_matched_datasets": 3,
            },
            {
                "stratum": "family_gene",
                "metric": "pearson_pert",
                "delta_mean": 0.002,
                "p_improve": 0.72,
                "p_harm": 0.12,
                "n_matched_conditions": 8,
                "n_matched_datasets": 3,
            },
        ],
    }


def test_summary_reports_pending_when_training_marker_missing(tmp_path: Path) -> None:
    mod = load_module()
    gate_path = tmp_path / "reports/gate.json"
    row = manifest_row(tmp_path, "run_pending", gate_path)

    summary = mod.summarize_run(row, tmp_path / "runs")

    assert summary["status"] == "training_running"
    assert not summary["gate_json_exists"]
    assert mod.overall_status([summary]) == "tracka_gene_reliability_adapter_block_pending"


def test_summary_promotes_only_existing_pass_gate(tmp_path: Path) -> None:
    mod = load_module()
    run_root = tmp_path / "runs"
    gate_path = tmp_path / "reports/gate.json"
    row = manifest_row(tmp_path, "run_pass", gate_path)
    run_dir = run_root / "run_pass"
    run_dir.mkdir(parents=True)
    (run_dir / "EXIT_CODE").write_text("0", encoding="utf-8")
    (run_dir / "POSTHOC_EXIT_CODE").write_text("0", encoding="utf-8")
    write_json(gate_path, passing_gate())

    summary = mod.summarize_run(row, run_root)

    assert summary["status"] == "candidate_gate_pass"
    assert summary["key_metrics"]["cross_background_seen_gene_pp"]["delta_mean"] == 0.031
    assert mod.overall_status([summary]) == "tracka_gene_reliability_adapter_has_pass_candidate_needs_seed_robustness"


def test_summary_missing_gate_after_posthoc_is_unknown_not_pass(tmp_path: Path) -> None:
    mod = load_module()
    run_root = tmp_path / "runs"
    gate_path = tmp_path / "reports/missing_gate.json"
    row = manifest_row(tmp_path, "run_missing_gate", gate_path)
    run_dir = run_root / "run_missing_gate"
    run_dir.mkdir(parents=True)
    (run_dir / "EXIT_CODE").write_text("0", encoding="utf-8")
    (run_dir / "POSTHOC_EXIT_CODE").write_text("0", encoding="utf-8")

    summary = mod.summarize_run(row, run_root)

    assert summary["status"] == "posthoc_complete_missing_gate"
    assert not summary["gate_json_exists"]
    assert mod.overall_status([summary]) == "tracka_gene_reliability_adapter_block_missing_gate_artifacts"
