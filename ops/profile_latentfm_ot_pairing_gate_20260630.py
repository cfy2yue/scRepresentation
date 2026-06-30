#!/usr/bin/env python3
"""Short LatentFM minibatch OT pairing profile.

This script uses train-only LatentFM batches to ask whether the default
minibatch Sinkhorn pairing materially changes pair geometry versus random
pairs, and whether assignment-style pairing is a plausible ablation candidate.
It does not train, infer, select checkpoints, or read held-out query data.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(COUPLED) not in sys.path:
    sys.path.insert(0, str(COUPLED))

from model.utils.data.ot_pairer import sinkhorn_pair  # noqa: E402
from ops.profile_latentfm_dataloader_throughput_20260630 import (  # noqa: E402
    build_dataset,
    build_plan,
    read_baseline,
)


OUT_DIR = ROOT / "reports/latentfm_ot_pairing_gate_20260630"


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def pair_cost(src: torch.Tensor, gt: torch.Tensor) -> float:
    return float((src.float() - gt.float()).pow(2).sum(dim=1).mean().detach().cpu().item())


def summarize(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def profile(args: argparse.Namespace) -> dict[str, Any]:
    dataset = build_dataset(args)
    plan = build_plan(dataset, args)
    if not plan:
        raise RuntimeError("empty OT profile plan")
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + 1701)

    rows: list[dict[str, Any]] = []
    by_mode: dict[str, dict[str, list[float]]] = {
        "random_identity": {"pair_cost": [], "pair_seconds": []},
        "torch_sinkhorn_multinomial": {"pair_cost": [], "pair_seconds": []},
        "torch_sinkhorn_assignment": {"pair_cost": [], "pair_seconds": []},
    }
    load_seconds: list[float] = []

    for step, row in enumerate(plan):
        t0 = time.perf_counter()
        src_np, gt_np = read_baseline(dataset, row)
        load_seconds.append(time.perf_counter() - t0)
        src = torch.as_tensor(src_np, dtype=torch.float32, device=device)
        gt = torch.as_tensor(gt_np, dtype=torch.float32, device=device)
        sync_if_needed(device)

        mode_t0 = time.perf_counter()
        random_cost = pair_cost(src, gt)
        sync_if_needed(device)
        random_seconds = time.perf_counter() - mode_t0
        by_mode["random_identity"]["pair_cost"].append(random_cost)
        by_mode["random_identity"]["pair_seconds"].append(random_seconds)
        rows.append(
            {
                "step": step,
                "dataset": row["dataset"],
                "condition": row["condition"],
                "mode": "random_identity",
                "pair_cost": random_cost,
                "pair_seconds": random_seconds,
                "cost_reduction_vs_random": 0.0,
            }
        )

        for mode, use_assignment in (
            ("torch_sinkhorn_multinomial", False),
            ("torch_sinkhorn_assignment", True),
        ):
            mode_t0 = time.perf_counter()
            i, j = sinkhorn_pair(
                src,
                gt,
                n_samples=int(src.shape[0]),
                reg=float(args.ot_sinkhorn_reg),
                n_iter=int(args.ot_sinkhorn_iter),
                normalize_cost=True,
                generator=gen,
                cost_fn=str(args.cost_fn),
                use_assignment=bool(use_assignment),
            )
            paired_cost = pair_cost(src[i], gt[j])
            sync_if_needed(device)
            elapsed = time.perf_counter() - mode_t0
            reduction = 1.0 - paired_cost / random_cost if random_cost > 0 else 0.0
            by_mode[mode]["pair_cost"].append(paired_cost)
            by_mode[mode]["pair_seconds"].append(elapsed)
            rows.append(
                {
                    "step": step,
                    "dataset": row["dataset"],
                    "condition": row["condition"],
                    "mode": mode,
                    "pair_cost": paired_cost,
                    "pair_seconds": elapsed,
                    "cost_reduction_vs_random": reduction,
                }
            )
        del src, gt
        if device.type == "cuda" and (step + 1) % 16 == 0:
            torch.cuda.empty_cache()

    dataset.close()
    mode_summary: dict[str, Any] = {}
    random_mean_cost = float(np.mean(by_mode["random_identity"]["pair_cost"]))
    for mode, vals in by_mode.items():
        mean_cost = float(np.mean(vals["pair_cost"])) if vals["pair_cost"] else 0.0
        mode_summary[mode] = {
            "pair_cost": summarize(vals["pair_cost"]),
            "pair_seconds": summarize(vals["pair_seconds"]),
            "cost_reduction_vs_random_mean": (
                1.0 - mean_cost / random_mean_cost if random_mean_cost > 0 else 0.0
            ),
        }
    default_red = float(mode_summary["torch_sinkhorn_multinomial"]["cost_reduction_vs_random_mean"])
    assign_red = float(mode_summary["torch_sinkhorn_assignment"]["cost_reduction_vs_random_mean"])
    assign_extra = assign_red - default_red
    default_p50 = float(mode_summary["torch_sinkhorn_multinomial"]["pair_seconds"]["p50"])
    assign_p50 = float(mode_summary["torch_sinkhorn_assignment"]["pair_seconds"]["p50"])
    reasons: list[str] = []
    if default_red < 0.20:
        reasons.append("default_sinkhorn_cost_reduction_below_20pct")
    if assign_extra >= 0.08 and assign_p50 <= max(0.25, 1.75 * default_p50):
        status = "ot_pairing_assignment_candidate_cpu_pass_no_model_claim"
        next_action = "consider_bounded_assignment_vs_multinomial_internal_smoke_after_external_audit"
    elif default_red >= 0.20:
        status = "ot_pairing_default_effective_no_gpu_training_change"
        next_action = "keep_default_sinkhorn_multinomial; do_not_launch_pairing_ablation_from_profile_alone"
    else:
        status = "ot_pairing_profile_blocks_pairing_ablation"
        next_action = "do_not_launch_pairing_ablation; inspect another bottleneck_or_scientific_axis"
    decision = {
        "status": status,
        "default_cost_reduction_vs_random": default_red,
        "assignment_cost_reduction_vs_random": assign_red,
        "assignment_extra_reduction": assign_extra,
        "default_pair_seconds_p50": default_p50,
        "assignment_pair_seconds_p50": assign_p50,
        "reasons": reasons,
        "next_action": next_action,
    }
    return {
        "inputs": {
            "data_dir": str(args.data_dir),
            "split_file": str(args.split_file),
            "seed": int(args.seed),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "device": str(device),
            "ot_sinkhorn_reg": float(args.ot_sinkhorn_reg),
            "ot_sinkhorn_iter": int(args.ot_sinkhorn_iter),
            "cost_fn": str(args.cost_fn),
        },
        "environment": {
            "python": sys.executable,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        },
        "dataset_summary": {
            "train_conditions": int(dataset.total_conditions),
            "epoch_steps": int(dataset.epoch_steps),
            "profile_plan_steps": len(plan),
        },
        "leakage_boundary": {
            "train_split_only": True,
            "training_or_inference": False,
            "canonical_multi_or_trackc_query_access": False,
            "checkpoint_selection": False,
        },
        "load_seconds": summarize(load_seconds),
        "mode_summary": mode_summary,
        "decision": decision,
        "rows": rows,
    }


def write_report(payload: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "latentfm_ot_pairing_gate_20260630.json"
    csv_path = OUT_DIR / "latentfm_ot_pairing_step_rows_20260630.csv"
    md_path = OUT_DIR / "LATENTFM_OT_PAIRING_GATE_20260630.md"
    rows = payload.get("rows", [])
    json_payload = {k: v for k, v in payload.items() if k != "rows"}
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        fieldnames = sorted({k for row in rows for k in row})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    decision = payload["decision"]
    lines = [
        "# LatentFM OT Pairing Gate 20260630",
        "",
        "## Boundary",
        "",
        "- Short train-only engineering/mechanism profile.",
        "- No training, inference, checkpoint selection, canonical multi, or Track C query access.",
        f"- Split file: `{payload['inputs']['split_file']}`.",
        f"- Device: `{payload['inputs']['device']}` with CUDA_VISIBLE_DEVICES=`{payload['environment']['cuda_visible_devices']}`.",
        "",
        "## Decision",
        "",
        f"- status: `{decision['status']}`",
        f"- default Sinkhorn reduction vs random: `{decision['default_cost_reduction_vs_random']:.4f}`",
        f"- assignment reduction vs random: `{decision['assignment_cost_reduction_vs_random']:.4f}`",
        f"- assignment extra reduction: `{decision['assignment_extra_reduction']:.4f}`",
        f"- default p50 pair seconds: `{decision['default_pair_seconds_p50']:.6f}`",
        f"- assignment p50 pair seconds: `{decision['assignment_pair_seconds_p50']:.6f}`",
        f"- next action: `{decision['next_action']}`",
        "",
        "## Mode Summary",
        "",
        "| mode | cost mean | cost p50 | pair p50 s | pair p90 s | reduction vs random |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, summary in payload["mode_summary"].items():
        lines.append(
            f"| {mode} | {summary['pair_cost']['mean']:.6f} | "
            f"{summary['pair_cost']['p50']:.6f} | "
            f"{summary['pair_seconds']['p50']:.6f} | "
            f"{summary['pair_seconds']['p90']:.6f} | "
            f"{summary['cost_reduction_vs_random_mean']:.4f} |"
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
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ot-sinkhorn-reg", type=float, default=0.05)
    parser.add_argument("--ot-sinkhorn-iter", type=int, default=50)
    parser.add_argument("--cost-fn", default="l2")
    args = parser.parse_args()
    payload = profile(args)
    write_report(payload)
    print(json.dumps(payload["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
