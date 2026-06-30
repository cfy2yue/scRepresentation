#!/usr/bin/env python3
"""Unit-style checks for ops/select_available_gpus.py.

This validation is pure Python and does not call nvidia-smi.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
HELPER = ROOT / "ops/select_available_gpus.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("select_available_gpus", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {HELPER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(module, idx, *, mem=100, util=0, users=()):
    return module.GpuRow(
        index=idx,
        uuid=f"GPU-{idx}",
        name="RTX 4090",
        memory_used_mib=mem,
        memory_total_mib=24564,
        utilization_gpu_pct=util,
        compute_pids=tuple(range(1000 + idx * 10, 1000 + idx * 10 + len(users))),
        compute_users=tuple(users),
    )


def main() -> int:
    module = load_helper()

    samples = [
        {
            0: row(module, 0, users=("cyx",)),
            1: row(module, 1),
            2: row(module, 2, users=("other",), mem=3000, util=7),
            3: row(module, 3, users=("other",), mem=7000, util=2),
        },
        {
            0: row(module, 0, users=("cyx",)),
            1: row(module, 1),
            2: row(module, 2, users=("other",), mem=3200, util=8),
            3: row(module, 3, users=("other",), mem=7000, util=2),
        },
        {
            0: row(module, 0, users=("cyx",)),
            1: row(module, 1),
            2: row(module, 2, users=("other",), mem=3500, util=9),
            3: row(module, 3, users=("other",), mem=7000, util=2),
        },
    ]
    classified = module.classify(
        samples,
        current_user="cyx",
        util_threshold_pct=10,
        memory_threshold_mib=4096,
        max_user_gpus=4,
        max_jobs_per_gpu=4,
    )
    by_idx = {gpu["index"]: gpu for gpu in classified["gpus"]}
    assert by_idx[0]["reason"] == "own_colocation_slot", by_idx[0]
    assert by_idx[0]["colocation_slots_free"] == 3, by_idx[0]
    assert by_idx[1]["reason"] == "clean", by_idx[1]
    assert by_idx[2]["reason"] == "foreign_stably_light", by_idx[2]
    assert by_idx[2]["available"] is True, by_idx[2]
    assert by_idx[3]["reason"] == "foreign_active", by_idx[3]
    assert by_idx[3]["available"] is False, by_idx[3]
    assert classified["other_blocked_gpus"] == [3], classified

    chosen = module.choose_job_gpus(classified, 4)
    assert set(chosen[:3]) == {0, 1, 2}, chosen
    assert 3 not in chosen, chosen

    all_clean = [
        {idx: row(module, idx) for idx in range(4)}
        for _ in range(3)
    ]
    classified = module.classify(
        all_clean,
        current_user="cyx",
        util_threshold_pct=10,
        memory_threshold_mib=4096,
        max_user_gpus=4,
        max_jobs_per_gpu=4,
    )
    chosen = module.choose_job_gpus(classified, 16)
    assert len(chosen) == 16, chosen
    assert {idx: chosen.count(idx) for idx in set(chosen)} == {0: 4, 1: 4, 2: 4, 3: 4}, chosen

    overloaded = [
        {
            0: row(module, 0, users=("cyx", "cyx", "cyx", "cyx")),
            1: row(module, 1, users=("other",), mem=1000, util=3),
        }
        for _ in range(3)
    ]
    classified = module.classify(
        overloaded,
        current_user="cyx",
        util_threshold_pct=10,
        memory_threshold_mib=4096,
        max_user_gpus=4,
        max_jobs_per_gpu=4,
    )
    by_idx = {gpu["index"]: gpu for gpu in classified["gpus"]}
    assert by_idx[0]["available"] is False, by_idx[0]
    assert by_idx[0]["colocation_slots_free"] == 0, by_idx[0]
    assert module.choose_job_gpus(classified, 2) == [1, 1], classified

    old_light_new_blocked = [
        {0: row(module, 0, users=("other",), mem=4500, util=9)}
        for _ in range(3)
    ]
    classified = module.classify(
        old_light_new_blocked,
        current_user="cyx",
        util_threshold_pct=10,
        memory_threshold_mib=4096,
        max_user_gpus=4,
        max_jobs_per_gpu=4,
    )
    by_idx = {gpu["index"]: gpu for gpu in classified["gpus"]}
    assert by_idx[0]["reason"] == "foreign_active", by_idx[0]
    assert by_idx[0]["available"] is False, by_idx[0]

    system = module.system_snapshot()
    for key in ("cpu_count", "load1", "load5", "load15", "load1_per_cpu", "mem_available_gib", "mem_total_gib"):
        assert key in system, system
    assert system["cpu_count"] >= 1, system

    print("gpu availability helper validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
