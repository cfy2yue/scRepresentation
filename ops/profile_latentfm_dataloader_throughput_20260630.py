#!/usr/bin/env python3
"""CPU-only LatentFM dataloader throughput/equivalence gate.

This script profiles the existing train-only LatentFM HDF5 row loader against
a semantics-preserving condition-cache candidate. It does not train, infer,
read canonical multi/query rows, or modify model code.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path("/data/cyx/1030/scLatent")
COUPLED = ROOT / "CoupledFM"
if str(COUPLED) not in sys.path:
    sys.path.insert(0, str(COUPLED))

from model.latent.dataset import CrossDatasetFMDataset  # noqa: E402


OUT_DIR = ROOT / "reports/latentfm_dataloader_throughput_gate_20260630"


def rss_gib() -> float:
    """Return resident set size in GiB using /proc when available."""
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / (1024.0 * 1024.0)
    return 0.0


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_dataset(args: argparse.Namespace) -> CrossDatasetFMDataset:
    split = load_json(args.split_file)
    return CrossDatasetFMDataset(
        data_dir=str(args.data_dir),
        split=split,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        mode="train",
        min_cells=int(args.min_cells),
        ds_alpha=float(args.ds_alpha),
        scale_noise=0.0,
        min_selected_conditions_per_dataset=int(args.min_selected_conditions_per_dataset),
        condition_visit_power=float(args.condition_visit_power),
        condition_visit_cap=int(args.condition_visit_cap),
        use_pert_condition=False,
        biflow_dir=str(args.biflow_dir),
        latent_backbone=str(args.latent_backbone),
        perturbation_family_filter=str(args.perturbation_family_filter),
        silent=True,
    )


def build_plan(dataset: CrossDatasetFMDataset, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Mirror CrossDatasetFMDataset.__iter__ index choices for one epoch."""
    rng = np.random.RandomState(int(args.seed))
    order = dataset._build_epoch_order(rng)  # pylint: disable=protected-access
    if int(args.steps) > 0:
        order = order[: int(args.steps)]
    gt_perms: dict[tuple[str, str], np.ndarray] = {}
    gt_cursors: dict[tuple[str, str], int] = {}
    plan: list[dict[str, Any]] = []
    for ds_name, cond in order:
        key = (str(ds_name), str(cond))
        n_src_total, n_gt_total = dataset._cond_sizes[ds_name][cond]  # pylint: disable=protected-access
        if n_gt_total < dataset.batch_size:
            gt_idx = rng.choice(n_gt_total, size=dataset.batch_size, replace=True)
        else:
            if key not in gt_perms:
                gt_perms[key] = rng.permutation(n_gt_total)
                gt_cursors[key] = 0
            perm = gt_perms[key]
            cursor = gt_cursors[key]
            end = min(cursor + dataset.batch_size, n_gt_total)
            gt_idx = perm[cursor:end]
            if len(gt_idx) < dataset.batch_size:
                shortfall = dataset.batch_size - len(gt_idx)
                gt_idx = np.concatenate(
                    [gt_idx, rng.choice(n_gt_total, size=shortfall, replace=True)]
                )
            gt_cursors[key] = end if end < n_gt_total else 0
        src_idx = rng.choice(
            n_src_total,
            size=dataset.batch_size,
            replace=(n_src_total < dataset.batch_size),
        )
        plan.append(
            {
                "dataset": str(ds_name),
                "condition": str(cond),
                "src_idx": np.asarray(src_idx, dtype=np.int64),
                "gt_idx": np.asarray(gt_idx, dtype=np.int64),
                "n_src_total": int(n_src_total),
                "n_gt_total": int(n_gt_total),
            }
        )
    return plan


def tensorize(src: np.ndarray, gt: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the existing Dataset.__iter__ float32 tensor conversion."""
    return torch.from_numpy(src.astype(np.float32)), torch.from_numpy(gt.astype(np.float32))


def read_baseline(
    dataset: CrossDatasetFMDataset,
    row: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    handle = dataset.handles[row["dataset"]]
    gt = handle.read_gt_rows(row["condition"], row["gt_idx"])
    src = handle.read_src_rows(row["condition"], row["src_idx"])
    return np.asarray(src, dtype=np.float32), np.asarray(gt, dtype=np.float32)


class ConditionCache:
    """On-demand full-condition cache with a hard byte budget."""

    def __init__(self, dataset: CrossDatasetFMDataset, byte_limit: int):
        self.dataset = dataset
        self.byte_limit = max(0, int(byte_limit))
        self.items: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self.bytes = 0
        self.hits = 0
        self.misses_loaded = 0
        self.misses_budget = 0

    def _estimate_bytes(self, src: np.ndarray, gt: np.ndarray) -> int:
        return int(src.nbytes + gt.nbytes)

    def get(self, ds_name: str, cond: str) -> tuple[np.ndarray, np.ndarray] | None:
        key = (str(ds_name), str(cond))
        cached = self.items.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        handle = self.dataset.handles[str(ds_name)]
        src_full = np.asarray(handle.read_src(str(cond)), dtype=np.float32)
        gt_full = np.asarray(handle.read_gt(str(cond)), dtype=np.float32)
        need = self._estimate_bytes(src_full, gt_full)
        if self.byte_limit > 0 and self.bytes + need > self.byte_limit:
            self.misses_budget += 1
            return None
        self.items[key] = (src_full, gt_full)
        self.bytes += need
        self.misses_loaded += 1
        return self.items[key]


def profile_baseline(
    dataset: CrossDatasetFMDataset,
    plan: list[dict[str, Any]],
    warmup: int,
) -> dict[str, Any]:
    start_rss = rss_gib()
    rows: list[dict[str, Any]] = []
    measured = 0
    t0 = time.perf_counter()
    for i, row in enumerate(plan):
        step_t0 = time.perf_counter()
        src, gt = read_baseline(dataset, row)
        src_t, gt_t = tensorize(src, gt)
        elapsed = time.perf_counter() - step_t0
        if i >= warmup:
            measured += 1
            rows.append(
                {
                    "mode": "baseline_row_read",
                    "step": i,
                    "dataset": row["dataset"],
                    "condition": row["condition"],
                    "seconds": elapsed,
                    "src_shape": list(src_t.shape),
                    "gt_shape": list(gt_t.shape),
                }
            )
    elapsed_total = time.perf_counter() - t0
    measured_seconds = sum(float(r["seconds"]) for r in rows)
    return {
        "mode": "baseline_row_read",
        "steps_total": len(plan),
        "steps_measured": measured,
        "elapsed_total_s": elapsed_total,
        "measured_s": measured_seconds,
        "steps_per_s": measured / measured_seconds if measured_seconds > 0 else 0.0,
        "rss_start_gib": start_rss,
        "rss_end_gib": rss_gib(),
        "rows": rows,
    }


def profile_condition_cache(
    dataset: CrossDatasetFMDataset,
    plan: list[dict[str, Any]],
    warmup: int,
    byte_limit: int,
) -> dict[str, Any]:
    cache = ConditionCache(dataset, byte_limit=byte_limit)
    start_rss = rss_gib()
    rows: list[dict[str, Any]] = []
    measured = 0
    t0 = time.perf_counter()
    for i, row in enumerate(plan):
        step_t0 = time.perf_counter()
        cached = cache.get(row["dataset"], row["condition"])
        if cached is None:
            src, gt = read_baseline(dataset, row)
            source = "fallback_row_read"
        else:
            src_full, gt_full = cached
            src = np.asarray(src_full[row["src_idx"]], dtype=np.float32)
            gt = np.asarray(gt_full[row["gt_idx"]], dtype=np.float32)
            source = "condition_cache"
        src_t, gt_t = tensorize(src, gt)
        elapsed = time.perf_counter() - step_t0
        if i >= warmup:
            measured += 1
            rows.append(
                {
                    "mode": "condition_cache",
                    "step": i,
                    "dataset": row["dataset"],
                    "condition": row["condition"],
                    "seconds": elapsed,
                    "source": source,
                    "src_shape": list(src_t.shape),
                    "gt_shape": list(gt_t.shape),
                }
            )
    elapsed_total = time.perf_counter() - t0
    measured_seconds = sum(float(r["seconds"]) for r in rows)
    return {
        "mode": "condition_cache",
        "steps_total": len(plan),
        "steps_measured": measured,
        "elapsed_total_s": elapsed_total,
        "measured_s": measured_seconds,
        "steps_per_s": measured / measured_seconds if measured_seconds > 0 else 0.0,
        "rss_start_gib": start_rss,
        "rss_end_gib": rss_gib(),
        "cache_items": len(cache.items),
        "cache_bytes_gib": cache.bytes / (1024.0**3),
        "cache_hits": cache.hits,
        "cache_misses_loaded": cache.misses_loaded,
        "cache_misses_budget": cache.misses_budget,
        "rows": rows,
    }


def equivalence_check(
    dataset: CrossDatasetFMDataset,
    plan: list[dict[str, Any]],
    n_steps: int,
    byte_limit: int,
) -> dict[str, Any]:
    cache = ConditionCache(dataset, byte_limit=byte_limit)
    max_abs = 0.0
    checked = 0
    failures: list[dict[str, Any]] = []
    for row in plan[: max(0, int(n_steps))]:
        src_base, gt_base = read_baseline(dataset, row)
        cached = cache.get(row["dataset"], row["condition"])
        if cached is None:
            continue
        src_full, gt_full = cached
        src_cache = np.asarray(src_full[row["src_idx"]], dtype=np.float32)
        gt_cache = np.asarray(gt_full[row["gt_idx"]], dtype=np.float32)
        src_diff = float(np.max(np.abs(src_base - src_cache))) if src_base.size else 0.0
        gt_diff = float(np.max(np.abs(gt_base - gt_cache))) if gt_base.size else 0.0
        diff = max(src_diff, gt_diff)
        max_abs = max(max_abs, diff)
        checked += 1
        if diff > 0.0:
            failures.append(
                {
                    "dataset": row["dataset"],
                    "condition": row["condition"],
                    "max_abs": diff,
                    "src_max_abs": src_diff,
                    "gt_max_abs": gt_diff,
                }
            )
    return {
        "checked_steps": checked,
        "requested_steps": int(n_steps),
        "max_abs": max_abs,
        "failures": failures[:10],
        "passed": checked > 0 and max_abs == 0.0 and not failures,
        "cache_misses_budget": cache.misses_budget,
        "cache_bytes_gib": cache.bytes / (1024.0**3),
    }


def quantiles(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "mean": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    baseline = payload["profiles"]["baseline_row_read"]
    cached = payload["profiles"]["condition_cache"]
    speedup = (
        float(cached["steps_per_s"]) / float(baseline["steps_per_s"])
        if float(baseline["steps_per_s"]) > 0
        else 0.0
    )
    rss_delta = float(cached["rss_end_gib"]) - float(cached["rss_start_gib"])
    equivalence = payload["equivalence"]
    reasons: list[str] = []
    if not equivalence.get("passed"):
        reasons.append("cache_not_bitwise_equivalent_to_hdf5_row_reads")
    if speedup < 1.5:
        reasons.append("speedup_below_1p5x")
    if rss_delta > 4.0:
        reasons.append("rss_delta_above_4gib")
    if int(cached.get("cache_misses_budget", 0)) > int(cached.get("cache_misses_loaded", 0)):
        reasons.append("cache_budget_misses_exceed_loaded_conditions")
    status = "loader_cache_candidate_pass" if not reasons else "loader_cache_candidate_blocks_integration"
    return {
        "status": status,
        "speedup": speedup,
        "rss_delta_gib": rss_delta,
        "reasons": reasons,
        "next_action": (
            "wire_optional_condition_cache_behind_config_and_run_short_gpu_smoke"
            if status == "loader_cache_candidate_pass"
            else "do_not_modify_training_loader; keep current row-read path and search other bottlenecks"
        ),
    }


def write_report(payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "latentfm_dataloader_throughput_gate_20260630.json"
    csv_path = OUT_DIR / "latentfm_dataloader_throughput_step_rows_20260630.csv"
    md_path = OUT_DIR / "LATENTFM_DATALOADER_THROUGHPUT_GATE_20260630.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        fieldnames = sorted({k for row in rows for k in row})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    decision = payload["decision"]
    baseline = payload["profiles"]["baseline_row_read"]
    cached = payload["profiles"]["condition_cache"]
    lines = [
        "# LatentFM Dataloader Throughput Gate 20260630",
        "",
        "## Boundary",
        "",
        "- CPU-only engineering profile of train-split LatentFM HDF5 loading.",
        "- No training, inference, checkpoint selection, canonical multi, or Track C query access.",
        f"- Data dir: `{payload['inputs']['data_dir']}`.",
        f"- Split file: `{payload['inputs']['split_file']}`.",
        "",
        "## Decision",
        "",
        f"- status: `{decision['status']}`",
        f"- speedup: `{decision['speedup']:.3f}x`",
        f"- cache RSS delta: `{decision['rss_delta_gib']:.3f}` GiB",
        f"- reasons: `{', '.join(decision['reasons']) if decision['reasons'] else 'none'}`",
        f"- next action: `{decision['next_action']}`",
        "",
        "## Equivalence",
        "",
        f"- checked steps: `{payload['equivalence']['checked_steps']}` / `{payload['equivalence']['requested_steps']}`",
        f"- max abs difference: `{payload['equivalence']['max_abs']}`",
        f"- passed: `{payload['equivalence']['passed']}`",
        "",
        "## Throughput",
        "",
        "| mode | measured steps/s | measured seconds | RSS start GiB | RSS end GiB | cache items | cache GiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| baseline row read | "
        f"{baseline['steps_per_s']:.3f} | {baseline['measured_s']:.3f} | "
        f"{baseline['rss_start_gib']:.3f} | {baseline['rss_end_gib']:.3f} | 0 | 0 |",
        "| condition cache | "
        f"{cached['steps_per_s']:.3f} | {cached['measured_s']:.3f} | "
        f"{cached['rss_start_gib']:.3f} | {cached['rss_end_gib']:.3f} | "
        f"{cached['cache_items']} | {cached['cache_bytes_gib']:.3f} |",
        "",
        "## Step-Time Quantiles",
        "",
        "| mode | mean s | p50 s | p90 s | p95 s |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, q in payload["step_time_quantiles"].items():
        lines.append(
            f"| {mode} | {q['mean']:.6f} | {q['p50']:.6f} | {q['p90']:.6f} | {q['p95']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- JSON: `{json_path}`",
            f"- step rows: `{csv_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset/latentfm_full/xverse")
    parser.add_argument(
        "--split-file",
        type=Path,
        default=ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json",
    )
    parser.add_argument("--biflow-dir", type=Path, default=ROOT / "dataset/biFlow_data")
    parser.add_argument("--latent-backbone", default="xverse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-cells", type=int, default=32)
    parser.add_argument("--ds-alpha", type=float, default=0.7)
    parser.add_argument("--min-selected-conditions-per-dataset", type=int, default=0)
    parser.add_argument("--condition-visit-power", type=float, default=1.0)
    parser.add_argument("--condition-visit-cap", type=int, default=8)
    parser.add_argument("--perturbation-family-filter", default="all")
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--equivalence-steps", type=int, default=48)
    parser.add_argument("--cache-byte-limit-gb", type=float, default=4.0)
    args = parser.parse_args()

    dataset = build_dataset(args)
    plan = build_plan(dataset, args)
    if not plan:
        raise RuntimeError("profile plan is empty; check split/data_dir")
    byte_limit = int(float(args.cache_byte_limit_gb) * (1024.0**3))
    equivalence = equivalence_check(dataset, plan, args.equivalence_steps, byte_limit)
    baseline = profile_baseline(dataset, plan, warmup=max(0, int(args.warmup)))
    cached = profile_condition_cache(dataset, plan, warmup=max(0, int(args.warmup)), byte_limit=byte_limit)
    rows = list(baseline["rows"]) + list(cached["rows"])
    payload = {
        "inputs": {
            "data_dir": str(args.data_dir),
            "split_file": str(args.split_file),
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "min_cells": int(args.min_cells),
            "ds_alpha": float(args.ds_alpha),
            "condition_visit_cap": int(args.condition_visit_cap),
            "steps": int(args.steps),
            "warmup": int(args.warmup),
            "equivalence_steps": int(args.equivalence_steps),
            "cache_byte_limit_gb": float(args.cache_byte_limit_gb),
        },
        "environment": {
            "python": sys.executable,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "torch": torch.__version__,
        },
        "dataset_summary": {
            "train_conditions": int(dataset.total_conditions),
            "epoch_steps": int(dataset.epoch_steps),
            "datasets": {ds: len(dataset.ds_conds[ds]) for ds in dataset.ds_names},
            "profile_plan_steps": len(plan),
        },
        "leakage_boundary": {
            "train_split_only": True,
            "training_or_inference": False,
            "canonical_multi_or_trackc_query_access": False,
            "checkpoint_selection": False,
        },
        "equivalence": equivalence,
        "profiles": {
            "baseline_row_read": {k: v for k, v in baseline.items() if k != "rows"},
            "condition_cache": {k: v for k, v in cached.items() if k != "rows"},
        },
        "step_time_quantiles": {
            "baseline_row_read": quantiles([float(r["seconds"]) for r in baseline["rows"]]),
            "condition_cache": quantiles([float(r["seconds"]) for r in cached["rows"]]),
        },
    }
    payload["decision"] = decide(payload)
    write_report(payload, rows)
    dataset.close()
    print(json.dumps(payload["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
