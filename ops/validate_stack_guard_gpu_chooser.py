#!/usr/bin/env python3
"""Validate the Stack guarded launcher's shared-GPU selection rules.

This is a lightweight static/unit-style check. It does not call nvidia-smi,
start tmux sessions, or launch training.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
GUARD = ROOT / "runs/latentfm_stack_composite_selection_20260618/launch_if_fullcap_pivot.py"


def load_guard():
    spec = importlib.util.spec_from_file_location("stack_guard", GUARD)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load guard: {GUARD}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_case(module, sequence, *, max_jobs=3):
    rows = [(0, 100, 0), (1, 100, 0), (2, 100, 0)]
    state = {"i": 0}
    module.SAMPLES = 3
    module.SAMPLE_SECONDS = 0
    module.gpu_rows = lambda: rows

    def fake_compute(_rows_by_gpu):
        idx = state["i"]
        state["i"] += 1
        return sequence[idx]

    module.compute_users = fake_compute
    module.time.sleep = lambda _seconds: None
    return module.choose_gpus(max_jobs=max_jobs)


def main() -> int:
    module = load_guard()

    # A foreign process seen as idle in only one sample is not stable-idle.
    chosen, reason = run_case(module, [({}, {1}, {1}), ({}, set(), set()), ({}, set(), set())])
    assert 1 not in chosen, (chosen, reason)
    assert "other_blocked=[1]" in reason, reason

    # Stable-idle foreign GPUs are eligible, but fully clean GPUs come first.
    chosen, reason = run_case(module, [({}, {1}, {1}), ({}, {1}, {1}), ({}, {1}, {1})], max_jobs=2)
    assert chosen == [0, 2], (chosen, reason)
    assert "other_stably_idle=[1]" in reason, reason

    # If clean GPUs are insufficient, a stable-idle foreign GPU can be used.
    chosen, reason = run_case(
        module,
        [({0: {"cyx"}}, {1}, {1}), ({0: {"cyx"}}, {1}, {1}), ({0: {"cyx"}}, {1}, {1})],
        max_jobs=2,
    )
    assert chosen == [2, 1], (chosen, reason)

    orig_tmux_exists = module.tmux_exists
    orig_run_root = module.RUN_ROOT
    orig_stack_out_root = module.STACK_OUT_ROOT

    class FakePath:
        def __init__(self, *, exists=False, is_file=False):
            self._exists = exists
            self._is_file = is_file

        def exists(self):
            return self._exists

        def is_file(self):
            return self._is_file

        def __truediv__(self, name):
            if str(name).endswith(".status"):
                return FakePath(is_file=name == "done.status")
            return FakePath(exists=name == "done")

    try:
        module.RUN_ROOT = FakePath()
        module.STACK_OUT_ROOT = FakePath()
        module.tmux_exists = lambda session: session == "latentfm_active"
        assert module.target_started_or_done("active") is True
        assert module.target_started_or_done("done") is True
        assert module.target_started_or_done("new") is False
    finally:
        module.tmux_exists = orig_tmux_exists
        module.RUN_ROOT = orig_run_root
        module.STACK_OUT_ROOT = orig_stack_out_root

    print("stack guard gpu chooser validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
