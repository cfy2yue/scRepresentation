#!/usr/bin/env python3
"""One-shot shared-GPU availability helper for the 8x4090 local server.

The tool is read-only: it samples ``nvidia-smi`` and ``ps`` then prints a
human-readable summary plus JSON. It does not launch jobs.
"""
from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from io import StringIO
from typing import Any


@dataclass(frozen=True)
class GpuRow:
    index: int
    uuid: str
    name: str
    memory_used_mib: int
    memory_total_mib: int
    utilization_gpu_pct: int
    compute_pids: tuple[int, ...]
    compute_users: tuple[str, ...]


def system_snapshot() -> dict[str, Any]:
    """Return a lightweight CPU/RAM snapshot for launch-time judgment."""

    meminfo: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                parts = value.strip().split()
                if parts:
                    meminfo[key] = int(parts[0])
    except OSError:
        meminfo = {}

    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = 0.0
    cpu_count = os.cpu_count() or 1
    mem_available_gib = meminfo.get("MemAvailable", 0) / 1024 / 1024
    mem_total_gib = meminfo.get("MemTotal", 0) / 1024 / 1024
    return {
        "cpu_count": cpu_count,
        "load1": round(load1, 2),
        "load5": round(load5, 2),
        "load15": round(load15, 2),
        "load1_per_cpu": round(load1 / cpu_count, 3),
        "mem_available_gib": round(mem_available_gib, 1),
        "mem_total_gib": round(mem_total_gib, 1),
    }


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def _parse_int(value: str) -> int:
    value = value.strip()
    if value in {"", "[Not Supported]", "N/A", "None"}:
        return 0
    return int(float(value))


def _csv_rows(text: str) -> list[list[str]]:
    if not text.strip():
        return []
    return [[cell.strip() for cell in row] for row in csv.reader(StringIO(text))]


def _pid_owner(pid: int) -> str:
    try:
        out = _run(["ps", "-o", "user=", "-p", str(pid)])
    except subprocess.CalledProcessError:
        return "unknown"
    owner = out.strip().splitlines()
    return owner[0].strip() if owner else "unknown"


def sample_once() -> dict[int, GpuRow]:
    gpu_text = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_rows: dict[int, dict[str, Any]] = {}
    uuid_to_index: dict[str, int] = {}
    for row in _csv_rows(gpu_text):
        if len(row) < 6:
            continue
        idx = int(row[0])
        uuid = row[1]
        uuid_to_index[uuid] = idx
        gpu_rows[idx] = {
            "index": idx,
            "uuid": uuid,
            "name": row[2],
            "memory_used_mib": _parse_int(row[3]),
            "memory_total_mib": _parse_int(row[4]),
            "utilization_gpu_pct": _parse_int(row[5]),
            "compute_pids": [],
            "compute_users": [],
        }

    try:
        app_text = _run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except subprocess.CalledProcessError as exc:
        app_text = "" if "No running processes found" in exc.output else ""

    owner_cache: dict[int, str] = {}
    for row in _csv_rows(app_text):
        if len(row) < 2:
            continue
        uuid, pid_s = row[0], row[1]
        if uuid not in uuid_to_index:
            continue
        pid = _parse_int(pid_s)
        if pid <= 0:
            continue
        owner_cache.setdefault(pid, _pid_owner(pid))
        idx = uuid_to_index[uuid]
        gpu_rows[idx]["compute_pids"].append(pid)
        gpu_rows[idx]["compute_users"].append(owner_cache[pid])

    return {
        idx: GpuRow(
            index=idx,
            uuid=str(row["uuid"]),
            name=str(row["name"]),
            memory_used_mib=int(row["memory_used_mib"]),
            memory_total_mib=int(row["memory_total_mib"]),
            utilization_gpu_pct=int(row["utilization_gpu_pct"]),
            compute_pids=tuple(int(p) for p in row["compute_pids"]),
            compute_users=tuple(str(u) for u in row["compute_users"]),
        )
        for idx, row in sorted(gpu_rows.items())
    }


def collect_samples(samples: int, interval_seconds: float) -> list[dict[int, GpuRow]]:
    out: list[dict[int, GpuRow]] = []
    for i in range(samples):
        out.append(sample_once())
        if i + 1 < samples:
            time.sleep(interval_seconds)
    return out


def classify(
    samples: list[dict[int, GpuRow]],
    *,
    current_user: str,
    util_threshold_pct: int,
    memory_threshold_mib: int,
    max_user_gpus: int,
    max_jobs_per_gpu: int,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("at least one sample is required")

    indices = sorted(set().union(*(set(sample) for sample in samples)))
    gpus: list[dict[str, Any]] = []
    blocked_by_other: list[int] = []
    active_user_gpus: list[int] = []

    for idx in indices:
        rows = [sample[idx] for sample in samples if idx in sample]
        users = sorted({u for row in rows for u in row.compute_users})
        pids = sorted({p for row in rows for p in row.compute_pids})
        own_counts = [sum(1 for u in row.compute_users if u == current_user) for row in rows]
        foreign_counts = [sum(1 for u in row.compute_users if u != current_user) for row in rows]
        max_own = max(own_counts) if own_counts else 0
        max_foreign = max(foreign_counts) if foreign_counts else 0
        max_util = max(row.utilization_gpu_pct for row in rows)
        max_memory = max(row.memory_used_mib for row in rows)
        stable_light = max_util < util_threshold_pct and max_memory < memory_threshold_mib
        has_foreign = max_foreign > 0
        has_own = max_own > 0
        foreign_blocked = has_foreign and not stable_light
        if foreign_blocked:
            blocked_by_other.append(idx)
        if has_own:
            active_user_gpus.append(idx)
        own_slot_available = max_own < max_jobs_per_gpu
        available = (not foreign_blocked) and own_slot_available
        if not pids:
            reason = "clean"
        elif foreign_blocked:
            reason = "foreign_active"
        elif has_foreign:
            reason = "foreign_stably_light"
        elif has_own:
            reason = "own_colocation_slot"
        else:
            reason = "available"
        gpus.append(
            {
                "index": idx,
                "name": rows[-1].name,
                "memory_used_mib": rows[-1].memory_used_mib,
                "memory_total_mib": rows[-1].memory_total_mib,
                "utilization_gpu_pct": rows[-1].utilization_gpu_pct,
                "max_sample_memory_used_mib": max_memory,
                "max_sample_utilization_gpu_pct": max_util,
                "compute_pids": pids,
                "compute_users": users,
                "own_process_count": max_own,
                "foreign_process_count": max_foreign,
                "stable_light": stable_light,
                "available": available,
                "reason": reason,
                "colocation_slots_free": max(0, max_jobs_per_gpu - max_own) if available else 0,
            }
        )

    other_blocked_count = len(blocked_by_other)
    total_gpus = len(indices)
    if other_blocked_count > 3:
        allowed_physical_user_gpus = max(0, min(max_user_gpus, total_gpus - other_blocked_count - 1))
    else:
        allowed_physical_user_gpus = min(max_user_gpus, total_gpus)
    active_user_count = len(set(active_user_gpus))
    new_physical_slots = max(0, allowed_physical_user_gpus - active_user_count)

    def rank_for_new_job(gpu: dict[str, Any]) -> tuple[int, int, int, int]:
        reason = str(gpu["reason"])
        # Prefer genuinely empty/stably-light GPUs when we still have physical
        # GPU budget. Colocation is allowed for low-util LatentFM jobs, but an
        # already busy own GPU should not outrank a clean idle card.
        if reason == "own_colocation_slot":
            reason_rank = 1 if bool(gpu.get("stable_light")) else 4
        else:
            reason_rank = {
                "clean": 0,
                "available": 2,
                "foreign_stably_light": 3,
                "foreign_active": 9,
            }.get(reason, 8)
        if gpu["reason"] in {"clean", "available", "foreign_stably_light"} and new_physical_slots <= 0:
            reason_rank += 5
        return (
            reason_rank,
            int(gpu["max_sample_memory_used_mib"]),
            int(gpu["max_sample_utilization_gpu_pct"]),
            int(gpu["index"]),
        )

    sorted_candidates = sorted((g for g in gpus if g["available"]), key=rank_for_new_job)
    return {
        "current_user": current_user,
        "samples": len(samples),
        "util_threshold_pct": util_threshold_pct,
        "memory_threshold_mib": memory_threshold_mib,
        "max_user_gpus": max_user_gpus,
        "max_jobs_per_gpu": max_jobs_per_gpu,
        "other_blocked_gpus": blocked_by_other,
        "active_user_gpus": sorted(set(active_user_gpus)),
        "allowed_physical_user_gpus": allowed_physical_user_gpus,
        "new_physical_slots": new_physical_slots,
        "gpus": gpus,
        "candidate_order": [int(g["index"]) for g in sorted_candidates],
    }


def choose_job_gpus(classified: dict[str, Any], need: int) -> list[int]:
    chosen: list[int] = []
    new_physical_used = 0
    active_user = set(classified["active_user_gpus"])
    new_limit = int(classified["new_physical_slots"])
    gpus_by_idx = {int(g["index"]): g for g in classified["gpus"]}

    # First spread one job across each eligible physical GPU. Low-util
    # colocation is allowed, but filling one card before using clean cards makes
    # later high-throughput exploration unnecessarily serial.
    for idx in classified["candidate_order"]:
        if len(chosen) >= need:
            return chosen
        gpu = gpus_by_idx[idx]
        if idx not in active_user and new_physical_used >= new_limit:
            continue
        slots = int(gpu["colocation_slots_free"])
        if slots <= 0:
            continue
        chosen.append(idx)
        if idx not in active_user:
            new_physical_used += 1

    for idx in classified["candidate_order"]:
        gpu = gpus_by_idx[idx]
        slots = int(gpu["colocation_slots_free"])
        already = chosen.count(idx)
        for _ in range(max(0, slots - already)):
            if len(chosen) >= need:
                return chosen
            chosen.append(idx)
    return chosen


def print_summary(classified: dict[str, Any], suggested: list[int]) -> None:
    system = classified.get("system")
    print(
        "Shared GPU snapshot: "
        f"user={classified['current_user']} samples={classified['samples']} "
        f"thresholds=util<{classified['util_threshold_pct']}%,"
        f"mem<{classified['memory_threshold_mib']}MiB"
    )
    if isinstance(system, dict):
        print(
            "System snapshot: "
            f"cpu={system.get('cpu_count')} "
            f"load1/5/15={system.get('load1')},{system.get('load5')},{system.get('load15')} "
            f"load1_per_cpu={system.get('load1_per_cpu')} "
            f"mem_available={system.get('mem_available_gib')}GiB/"
            f"{system.get('mem_total_gib')}GiB"
        )
    print(
        "Budget: "
        f"allowed_physical_user_gpus={classified['allowed_physical_user_gpus']} "
        f"active_user_gpus={classified['active_user_gpus']} "
        f"new_physical_slots={classified['new_physical_slots']} "
        f"max_jobs_per_gpu={classified['max_jobs_per_gpu']}"
    )
    for gpu in classified["gpus"]:
        users = ",".join(gpu["compute_users"]) if gpu["compute_users"] else "-"
        print(
            f"GPU {gpu['index']}: {gpu['reason']} available={gpu['available']} "
            f"util={gpu['utilization_gpu_pct']}% mem={gpu['memory_used_mib']}/"
            f"{gpu['memory_total_mib']}MiB max_sample=({gpu['max_sample_utilization_gpu_pct']}%,"
            f"{gpu['max_sample_memory_used_mib']}MiB) own={gpu['own_process_count']} "
            f"foreign={gpu['foreign_process_count']} users={users} "
            f"slots_free={gpu['colocation_slots_free']}"
        )
    print(f"Suggested single-GPU job assignment order: {suggested}")
    unique = ",".join(str(i) for i in sorted(set(suggested)))
    print(f"Suggested CUDA_VISIBLE_DEVICES for unique selected physical GPUs: {unique or '(none)'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--util-threshold-pct", type=int, default=10)
    parser.add_argument("--memory-threshold-mib", type=int, default=4096)
    parser.add_argument("--max-user-gpus", type=int, default=4)
    parser.add_argument("--max-jobs-per-gpu", type=int, default=4)
    parser.add_argument("--need", type=int, default=4, help="number of single-GPU jobs to place")
    parser.add_argument("--user", default=getpass.getuser())
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(argv)

    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.max_jobs_per_gpu < 1:
        parser.error("--max-jobs-per-gpu must be >= 1")

    samples = collect_samples(args.samples, args.interval_seconds)
    classified = classify(
        samples,
        current_user=args.user,
        util_threshold_pct=args.util_threshold_pct,
        memory_threshold_mib=args.memory_threshold_mib,
        max_user_gpus=args.max_user_gpus,
        max_jobs_per_gpu=args.max_jobs_per_gpu,
    )
    classified["system"] = system_snapshot()
    suggested = choose_job_gpus(classified, max(0, args.need))
    classified["suggested_job_gpus"] = suggested

    if not args.json_only:
        print_summary(classified, suggested)
        print("\nJSON:")
    print(json.dumps(classified, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
