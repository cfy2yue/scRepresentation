#!/usr/bin/env python3
"""Pure-Python checks for the workspace status generator.

The validation uses temporary status files and does not inspect tmux, GPUs, or
training logs.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
GENERATOR = ROOT / "ops/generate_workspace_status.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_workspace_status", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load workspace status generator: {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_status(path: Path, status: str, *, exit_code: int | None = None) -> None:
    lines = ["# Status", "", f"Status: {status}"]
    if exit_code is not None:
        lines.append(f"Exit code: {exit_code}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    module = load_generator()
    with tempfile.TemporaryDirectory(prefix="workspace_status_") as tmp_s:
        tmp = Path(tmp_s)
        launch = tmp / "launch.md"
        posthoc = tmp / "posthoc.md"

        write_status(launch, "running")
        write_status(posthoc, "running_posthoc")
        assert module.four_run_launch_summary(launch, posthoc) == "status=running"

        write_status(posthoc, "finished", exit_code=0)
        summary = module.four_run_launch_summary(launch, posthoc)
        assert summary == "superseded_by_posthoc_finished (launch file says status=running)", summary

        launch.unlink()
        summary = module.four_run_launch_summary(launch, posthoc)
        assert summary == "superseded_by_posthoc_finished (launch file says NA)", summary

        run_root = tmp / "run_with_markers"
        run_root.mkdir()
        write_status(run_root / "RUN_STATUS.md", "running")
        (run_root / "EXIT_CODE").write_text("0\n", encoding="utf-8")
        (run_root / "FINISHED").write_text("2026-06-19 13:18:01 CST\n", encoding="utf-8")
        marker_summary = module.run_marker_summary(run_root)
        assert marker_summary == "exit_code=0; finished=2026-06-19 13:18:01 CST", marker_summary

        run_root_no_marker = tmp / "run_without_markers"
        run_root_no_marker.mkdir()
        write_status(run_root_no_marker / "RUN_STATUS.md", "waiting")
        assert module.run_marker_summary(run_root_no_marker) == "status=waiting"

        pending = module.strategy_decision_state({"missing_inputs": ["expanded"], "rows": []})
        assert pending["status"] == "pending", pending
        assert "expanded" in pending["message"], pending

        no_repeat = module.strategy_decision_state(
            {
                "missing_inputs": [],
                "rows": [
                    {"run": "a", "decision": "diagnostic_candidate", "score": -1.0},
                    {"run": "b", "decision": "reject_as_is", "score": -2.0},
                ],
            }
        )
        assert no_repeat["status"] == "complete_no_repeat_candidate", no_repeat
        assert no_repeat["best"] == "a", no_repeat

        repeat = module.strategy_decision_state(
            {
                "missing_inputs": [],
                "rows": [
                    {"run": "a", "decision": "diagnostic_candidate", "score": -1.0},
                    {"run": "b", "decision": "repeat_candidate", "score": 1.0},
                ],
            }
        )
        assert repeat["status"] == "complete_with_repeat_candidate", repeat
        assert repeat["repeat_candidates"] == 1, repeat

        dose_missing = module.condition_prior_dose_state(None)
        assert dose_missing == "pending; summary JSON missing", dose_missing

        dose_pending = module.condition_prior_dose_state(
            {"status": "pending", "best": None, "rows": [{"complete": False, "decision": "pending"}]}
        )
        assert dose_pending == "pending; complete=0/1; repeat_candidates=0; best=NA", dose_pending

        dose_complete = module.condition_prior_dose_state(
            {
                "status": "complete_no_repeat_candidate",
                "best": {"run": "scf_prior002_e2_4k"},
                "rows": [
                    {"complete": True, "decision": "diagnostic_candidate"},
                    {"complete": True, "decision": "repeat_candidate"},
                    {"complete": False, "decision": "pending"},
                ],
            }
        )
        assert (
            dose_complete
            == "complete_no_repeat_candidate; complete=2/3; repeat_candidates=1; best=scf_prior002_e2_4k"
        ), dose_complete

    print("workspace status validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
