#!/usr/bin/env python3
"""CPU gate for balanced task/no-harm scheduling in lookahead adapter steps.

This is a narrow training-set perspective audit. It compares the current
sequential train-batch pairing against a simple cross-dataset balanced
task/no-harm schedule inside the same lookahead/trust-region adapter unit
step. It does not train a checkpoint, evaluate canonical metrics, read
canonical multi, read Track C query, or use GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch

ROOT = Path("/data/cyx/1030/scLatent")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "CoupledFM"))

from model.latent.fm_ot import CondOTPath, OTPlanSampler  # noqa: E402
from model.latent.train import apply_finetune_freeze, build_model  # noqa: E402
from ops.audit_latentfm_lookahead_response_normalized_unit_gate_20260627 import mode_trial  # noqa: E402
from ops.train_latentfm_lookahead_trust_region_adapter_smoke_20260627 import (  # noqa: E402
    ANCHOR_CKPT,
    SAFE_SPLIT,
    candidate_cfg,
    cfg_from_checkpoint,
    flat_params,
    load_raw_then_ema,
    make_dataset,
    parse_step_grid,
    set_flat_params,
    to_device_batch,
    trainable_items,
)

REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "lookahead_balanced_schedule_unit_gate_20260627"
OUT_JSON = REPORTS / "latentfm_lookahead_balanced_schedule_unit_gate_20260627.json"
OUT_MD = REPORTS / "LATENTFM_LOOKAHEAD_BALANCED_SCHEDULE_UNIT_GATE_20260627.md"
ROWS_CSV = OUT_DIR / "schedule_rows.csv"
SUMMARY_CSV = OUT_DIR / "schedule_summary.csv"


def next_raw(iterator, dataset):
    try:
        raw = next(iterator)
    except StopIteration:
        iterator = iter(dataset)
        raw = next(iterator)
    return iterator, raw


def raw_dataset(raw: tuple) -> str:
    return str(raw[2])


def collect_raws(dataset, *, pool_size: int) -> list[tuple]:
    iterator = iter(dataset)
    raws: list[tuple] = []
    for _ in range(int(pool_size)):
        iterator, raw = next_raw(iterator, dataset)
        raws.append(raw)
    return raws


def build_sequential_pairs(raws: list[tuple], max_pairs: int) -> list[tuple[tuple, tuple]]:
    pairs = []
    for i in range(0, min(len(raws) - 1, int(max_pairs) * 2), 2):
        pairs.append((raws[i], raws[i + 1]))
    return pairs[: int(max_pairs)]


def build_balanced_pairs(raws: list[tuple], max_pairs: int) -> list[tuple[tuple, tuple]]:
    by_ds: dict[str, list[tuple]] = defaultdict(list)
    for raw in raws:
        by_ds[raw_dataset(raw)].append(raw)
    datasets = sorted([ds for ds, vals in by_ds.items() if vals])
    if not datasets:
        return []
    pairs = []
    offsets = defaultdict(int)
    # Prefer one task per dataset, and pair no-harm from an offset dataset to
    # make the replay constraint less likely to collapse to one background.
    for idx, task_ds in enumerate(datasets):
        if len(pairs) >= int(max_pairs):
            break
        noharm_ds = datasets[(idx + max(1, len(datasets) // 3)) % len(datasets)]
        task_vals = by_ds[task_ds]
        noharm_vals = by_ds[noharm_ds]
        task = task_vals[offsets[task_ds] % len(task_vals)]
        offsets[task_ds] += 1
        noharm = noharm_vals[offsets[noharm_ds] % len(noharm_vals)]
        offsets[noharm_ds] += 1
        pairs.append((task, noharm))
    return pairs[: int(max_pairs)]


def fmean(vals: list[float]) -> float | None:
    return float(mean(vals)) if vals else None


def fmedian(vals: list[float]) -> float | None:
    return float(median(vals)) if vals else None


def summarize(rows: list[dict[str, Any]], arms: list[str]) -> list[dict[str, Any]]:
    out = []
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        acc = [r for r in sub if r["accepted"]]
        task_datasets = sorted({r["task_dataset"] for r in sub})
        noharm_datasets = sorted({r["noharm_dataset"] for r in sub})
        per_dataset = []
        for ds in task_datasets:
            ds_rows = [r for r in sub if r["task_dataset"] == ds]
            per_dataset.append(sum(1 for r in ds_rows if r["accepted"]) / max(len(ds_rows), 1))
        out.append(
            {
                "arm": arm,
                "attempts": len(sub),
                "accepted": len(acc),
                "accepted_rate": len(acc) / max(len(sub), 1),
                "task_dataset_coverage": len(task_datasets),
                "noharm_dataset_coverage": len(noharm_datasets),
                "min_task_dataset_accept_rate": min(per_dataset) if per_dataset else 0.0,
                "mean_raw_task_delta": fmean([float(r["raw_proj_task_delta"]) for r in acc if "raw_proj_task_delta" in r]),
                "median_raw_task_delta": fmedian([float(r["raw_proj_task_delta"]) for r in acc if "raw_proj_task_delta" in r]),
                "mean_anchor_delta": fmean([float(r["proj_anchor_delta"]) for r in acc if "proj_anchor_delta" in r]),
                "mean_footprint": fmean([float(r["proj_footprint_mean_l2"]) for r in acc if "proj_footprint_mean_l2" in r]),
                "mean_material_row_frac": fmean([float(r["proj_material_row_frac"]) for r in acc if "proj_material_row_frac" in r]),
            }
        )
    return out


def decide(summary: list[dict[str, Any]], *, max_pairs: int) -> tuple[str, list[str]]:
    by_arm = {row["arm"]: row for row in summary}
    seq = by_arm.get("sequential") or {}
    bal = by_arm.get("balanced") or {}
    reasons: list[str] = []
    required_cov = min(int(max_pairs), 8)
    if int(bal.get("task_dataset_coverage") or 0) < required_cov:
        reasons.append("balanced_task_dataset_coverage_lt_required")
    if int(bal.get("noharm_dataset_coverage") or 0) < max(4, required_cov // 2):
        reasons.append("balanced_noharm_dataset_coverage_lt_required")
    if float(bal.get("accepted_rate") or 0.0) < max(0.75, float(seq.get("accepted_rate") or 0.0) - 0.10):
        reasons.append("balanced_accepted_rate_too_low")
    if float(bal.get("min_task_dataset_accept_rate") or 0.0) < 0.50:
        reasons.append("balanced_dataset_specific_accept_collapse")
    bal_delta = bal.get("mean_raw_task_delta")
    seq_delta = seq.get("mean_raw_task_delta")
    if bal_delta is None or (seq_delta is not None and float(bal_delta) > float(seq_delta) + 1e-10):
        reasons.append("balanced_task_delta_not_better_than_sequential")
    if float(bal.get("mean_anchor_delta") or 0.0) > 1e-6:
        reasons.append("balanced_anchor_delta_above_threshold")
    if float(bal.get("mean_footprint") or 0.0) < max(float(seq.get("mean_footprint") or 0.0) * 0.95, 1e-7):
        reasons.append("balanced_footprint_below_sequential")
    if reasons:
        return "lookahead_balanced_schedule_unit_fail_no_gpu", reasons
    return "lookahead_balanced_schedule_unit_pass_gpu_smoke_candidate", []


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-pairs", type=int, default=8)
    ap.add_argument("--pool-size", type=int, default=96)
    ap.add_argument("--step-grid", default="1,3,10,30,100,300")
    ap.add_argument("--anchor-threshold", type=float, default=1e-6)
    ap.add_argument("--min-task-delta", type=float, default=1e-10)
    ap.add_argument("--min-footprint", type=float, default=1e-7)
    ap.add_argument("--num-threads", type=int, default=8)
    args = ap.parse_args()

    if str(args.device).startswith("cuda"):
        raise ValueError("This audit is CPU-only by default; do not use CUDA for this gate.")
    torch.set_num_threads(max(1, int(args.num_threads)))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    anchor_cfg = cfg_from_checkpoint(ANCHOR_CKPT, batch_size=int(args.batch_size), seed=int(args.seed))
    cfg = candidate_cfg(anchor_cfg)
    anchor = build_model(anchor_cfg, device)
    candidate = build_model(cfg, device)
    anchor_load = load_raw_then_ema(ANCHOR_CKPT, anchor, anchor_cfg, device)
    cand_load = load_raw_then_ema(ANCHOR_CKPT, candidate, cfg, device)
    for p in anchor.parameters():
        p.requires_grad = False
    anchor.eval()
    candidate.eval()
    apply_finetune_freeze(candidate, cfg)
    trainable = trainable_items(candidate)
    trainable_names = [name for name, _ in trainable]
    allowed_trainable = all(
        name.startswith("condition_delta_head.") or name.startswith("condition_delta_to_c.")
        for name in trainable_names
    )
    allowed_missing = all(
        key == "condition_delta_prior_gene_allowlist"
        or key.startswith("condition_delta_head.")
        or key.startswith("condition_delta_to_c.")
        for key in cand_load["missing"]
    )
    if not (allowed_trainable and allowed_missing and not cand_load["unexpected"] and not cand_load["skipped_shape_mismatch"]):
        raise RuntimeError("provenance/scope failure")

    dataset = make_dataset(cfg, seed=int(args.seed), batch_size=int(args.batch_size))
    sampler = OTPlanSampler(method="exact", num_threads=min(4, int(args.num_threads)))
    path = CondOTPath()
    steps = parse_step_grid(args.step_grid)
    p0 = flat_params(trainable).clone()
    rows: list[dict[str, Any]] = []
    try:
        raws = collect_raws(dataset, pool_size=int(args.pool_size))
        arms = {
            "sequential": build_sequential_pairs(raws, int(args.max_pairs)),
            "balanced": build_balanced_pairs(raws, int(args.max_pairs)),
        }
        for arm, pairs in arms.items():
            for pair_idx, (task_raw, noharm_raw) in enumerate(pairs):
                set_flat_params(trainable, p0)
                task_batch = to_device_batch(task_raw, sampler, device, idx=2 * pair_idx)
                noharm_batch = to_device_batch(noharm_raw, sampler, device, idx=2 * pair_idx + 1)
                trial = mode_trial(
                    model=candidate,
                    anchor=anchor,
                    trainable=trainable,
                    task_batch=task_batch,
                    noharm_batch=noharm_batch,
                    path=path,
                    steps=steps,
                    anchor_threshold=float(args.anchor_threshold),
                    min_task_delta=float(args.min_task_delta),
                    min_footprint=float(args.min_footprint),
                    normalizer=None,
                    norm_dataset=None,
                )
                rows.append(
                    {
                        "arm": arm,
                        "pair_idx": pair_idx,
                        "task_dataset": task_batch["dataset"],
                        "task_condition": task_batch["condition"],
                        "noharm_dataset": noharm_batch["dataset"],
                        "noharm_condition": noharm_batch["condition"],
                        **trial,
                    }
                )
    finally:
        set_flat_params(trainable, p0)
        dataset.close()

    arms = ["sequential", "balanced"]
    summary = summarize(rows, arms)
    status, reasons = decide(summary, max_pairs=int(args.max_pairs))
    row_fields = sorted({key for row in rows for key in row})
    summary_fields = [
        "arm",
        "attempts",
        "accepted",
        "accepted_rate",
        "task_dataset_coverage",
        "noharm_dataset_coverage",
        "min_task_dataset_accept_rate",
        "mean_raw_task_delta",
        "median_raw_task_delta",
        "mean_anchor_delta",
        "mean_footprint",
        "mean_material_row_frac",
    ]
    write_csv(ROWS_CSV, rows, row_fields)
    write_csv(SUMMARY_CSV, summary, summary_fields)

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": status,
        "gpu_authorized": status.endswith("gpu_smoke_candidate"),
        "reasons": reasons,
        "boundary": {
            "cpu_only": True,
            "device": str(device),
            "safe_split": str(SAFE_SPLIT),
            "anchor_checkpoint": str(ANCHOR_CKPT),
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "trains_checkpoint": False,
            "uses_canonical_metrics": False,
        },
        "args": vars(args),
        "provenance": {
            "anchor_ema_applied": bool(anchor_load.get("ema_applied")),
            "candidate_ema_applied": bool(cand_load.get("ema_applied")),
            "trainable_names": trainable_names,
        },
        "summary": summary,
        "rows_csv": str(ROWS_CSV),
        "summary_csv": str(SUMMARY_CSV),
        "next_action": (
            "prepare one bounded balanced-schedule lookahead GPU smoke only after current canonical no-harm branch frees/authorizes capacity"
            if status.endswith("gpu_smoke_candidate")
            else "do not launch a balanced-schedule GPU smoke from this evidence"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM Lookahead Balanced Schedule Unit Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        f"GPU authorized: `{str(payload['gpu_authorized']).lower()}`",
        "",
        "## Boundary",
        "",
        "- CPU-only adapter-gradient audit on safe train-only batches.",
        "- Compares sequential vs cross-dataset balanced task/no-harm scheduling.",
        "- No checkpoint training, no canonical metrics, no canonical multi, no Track C query, and no GPU.",
        "",
        "## Summary",
        "",
        "| arm | attempts | accepted | rate | task ds | noharm ds | min ds accept | mean task delta | mean anchor delta | mean footprint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['arm']}` | {row['attempts']} | {row['accepted']} | {fmt(row['accepted_rate'])} | "
            f"{row['task_dataset_coverage']} | {row['noharm_dataset_coverage']} | "
            f"{fmt(row['min_task_dataset_accept_rate'])} | {fmt(row['mean_raw_task_delta'])} | "
            f"{fmt(row['mean_anchor_delta'])} | {fmt(row['mean_footprint'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            f"- next action: {payload['next_action']}",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- rows: `{ROWS_CSV}`",
            f"- summary: `{SUMMARY_CSV}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(OUT_MD)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
