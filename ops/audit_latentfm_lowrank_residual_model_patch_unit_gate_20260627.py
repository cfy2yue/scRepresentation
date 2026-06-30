#!/usr/bin/env python3
"""CPU provenance/unit gate for the in-model low-rank residual adapter patch."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
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
from ops.audit_latentfm_lookahead_balanced_schedule_unit_gate_20260627 import (  # noqa: E402
    build_balanced_pairs,
    collect_raws,
)
from ops.audit_latentfm_lookahead_response_normalized_unit_gate_20260627 import mode_trial  # noqa: E402
from ops.train_latentfm_lookahead_trust_region_adapter_smoke_20260627 import (  # noqa: E402
    ANCHOR_CKPT,
    SAFE_SPLIT,
    cfg_from_checkpoint,
    flat_params,
    load_raw_then_ema,
    make_dataset,
    next_batch,
    parse_step_grid,
    set_flat_params,
    to_device_batch,
    trainable_items,
)

REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "lowrank_residual_model_patch_unit_gate_20260627"
OUT_JSON = REPORTS / "latentfm_lowrank_residual_model_patch_unit_gate_20260627.json"
OUT_MD = REPORTS / "LATENTFM_LOWRANK_RESIDUAL_MODEL_PATCH_UNIT_GATE_20260627.md"
ROWS_CSV = OUT_DIR / "model_patch_rows.csv"
SUMMARY_CSV = OUT_DIR / "model_patch_summary.csv"


def candidate_cfg(anchor_cfg, *, rank: int):
    cfg = deepcopy(anchor_cfg)
    cfg.condition_lowrank_residual_use_in_model = True
    cfg.condition_lowrank_residual_rank = int(rank)
    cfg.finetune_trainable_scope = "condition_lowrank_residual_adapter"
    # Keep the old condition-delta head disabled: this gate is specifically
    # for the new residual path and should not mix adapter families.
    cfg.condition_delta_head_use_in_model = False
    cfg.condition_delta_head_loss_weight = 0.0
    cfg.additive_condition_delta_loss_weight = 0.0
    cfg.condition_prior_additive_delta_loss_weight = 0.0
    cfg.trackc_routed_distill_loss_weight = 0.0
    cfg.trackc_routed_endpoint_loss_weight = 0.0
    return cfg


def allowed_missing(keys: list[str]) -> bool:
    return all(
        key == "condition_delta_prior_gene_allowlist"
        or key.startswith("condition_lowrank_residual_down.")
        or key.startswith("condition_lowrank_residual_up.")
        for key in keys
    )


def shared_state_mismatches(anchor: torch.nn.Module, candidate: torch.nn.Module) -> list[str]:
    a_state = anchor.state_dict()
    c_state = candidate.state_dict()
    mismatches = []
    for key, aval in a_state.items():
        if key.startswith("condition_lowrank_residual_"):
            continue
        cval = c_state.get(key)
        if cval is None:
            mismatches.append(f"{key}:missing_in_candidate")
            continue
        if aval.shape != cval.shape:
            mismatches.append(f"{key}:shape_mismatch")
            continue
        if torch.is_floating_point(aval):
            if not torch.equal(aval.cpu(), cval.cpu()):
                mismatches.append(f"{key}:value_mismatch")
        elif not torch.equal(aval.cpu(), cval.cpu()):
            mismatches.append(f"{key}:value_mismatch")
    return mismatches


def fmean(vals: list[float]) -> float | None:
    return float(mean(vals)) if vals else None


def fmedian(vals: list[float]) -> float | None:
    return float(median(vals)) if vals else None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-pairs", type=int, default=8)
    ap.add_argument("--pool-size", type=int, default=128)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--step-grid", default="1,3,10,30,100,300")
    ap.add_argument("--anchor-threshold", type=float, default=1e-6)
    ap.add_argument("--min-task-delta", type=float, default=1e-10)
    ap.add_argument("--min-footprint", type=float, default=5e-6)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument(
        "--tag",
        default="",
        help="Optional output tag to avoid overwriting the canonical rank32 gate report.",
    )
    args = ap.parse_args()
    global OUT_DIR, OUT_JSON, OUT_MD, ROWS_CSV, SUMMARY_CSV
    if str(args.tag).strip():
        safe_tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(args.tag).strip())
        OUT_DIR = REPORTS / f"lowrank_residual_model_patch_unit_gate_20260627_{safe_tag}"
        OUT_JSON = REPORTS / f"latentfm_lowrank_residual_model_patch_unit_gate_20260627_{safe_tag}.json"
        OUT_MD = REPORTS / f"LATENTFM_LOWRANK_RESIDUAL_MODEL_PATCH_UNIT_GATE_20260627_{safe_tag}.md"
        ROWS_CSV = OUT_DIR / "model_patch_rows.csv"
        SUMMARY_CSV = OUT_DIR / "model_patch_summary.csv"
    if str(args.device).startswith("cuda"):
        raise ValueError("CPU-only gate: do not use CUDA")
    torch.set_num_threads(max(1, int(args.num_threads)))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    anchor_cfg = cfg_from_checkpoint(ANCHOR_CKPT, batch_size=int(args.batch_size), seed=int(args.seed))
    cand_cfg = candidate_cfg(anchor_cfg, rank=int(args.rank))
    anchor = build_model(anchor_cfg, device)
    candidate = build_model(cand_cfg, device)
    anchor_load = load_raw_then_ema(ANCHOR_CKPT, anchor, anchor_cfg, device)
    cand_load = load_raw_then_ema(ANCHOR_CKPT, candidate, cand_cfg, device)
    for p in anchor.parameters():
        p.requires_grad = False
    anchor.eval()
    candidate.eval()
    apply_finetune_freeze(candidate, cand_cfg)
    trainable = trainable_items(candidate)
    trainable_names = [name for name, _ in trainable]
    expected_trainable = all(
        name.startswith("condition_lowrank_residual_down.")
        or name.startswith("condition_lowrank_residual_up.")
        for name in trainable_names
    )
    shared_mismatches = shared_state_mismatches(anchor, candidate)
    provenance_reasons = []
    if not allowed_missing(cand_load["missing"]):
        provenance_reasons.append("missing_keys_outside_lowrank_allowlist")
    if cand_load["unexpected"]:
        provenance_reasons.append("unexpected_keys_present")
    if cand_load["skipped_shape_mismatch"]:
        provenance_reasons.append("shape_mismatch_skips_present")
    if not expected_trainable:
        provenance_reasons.append("trainable_scope_not_lowrank_only")
    if shared_mismatches:
        provenance_reasons.append("shared_state_mismatch")

    dataset = make_dataset(cand_cfg, seed=int(args.seed), batch_size=int(args.batch_size))
    iterator = iter(dataset)
    sampler = OTPlanSampler(method="exact", num_threads=min(4, int(args.num_threads)))
    path = CondOTPath()
    steps = parse_step_grid(args.step_grid)
    p0 = flat_params(trainable).clone()
    noop_drifts: list[float] = []
    rows: list[dict[str, Any]] = []
    try:
        for i in range(2):
            iterator, batch = next_batch(iterator, dataset, sampler, device, idx=i)
            with torch.no_grad():
                src, gt, t, pb = batch["src"], batch["gt"], batch["t"], batch["pb"]
                ps = path.sample(src, gt, t)
                av = torch.nan_to_num(
                    __import__("model.latent.train", fromlist=["_model_latent_velocity"])._model_latent_velocity(
                        anchor, ps.x_t, ps.t, src, pb
                    )
                )
                cv = torch.nan_to_num(
                    __import__("model.latent.train", fromlist=["_model_latent_velocity"])._model_latent_velocity(
                        candidate, ps.x_t, ps.t, src, pb
                    )
                )
                noop_drifts.append(float((cv - av).abs().max().item()))
        raws = collect_raws(dataset, pool_size=int(args.pool_size))
        pairs = build_balanced_pairs(raws, int(args.max_pairs))
        for pair_idx, (task_raw, noharm_raw) in enumerate(pairs):
            set_flat_params(trainable, p0)
            task_batch = to_device_batch(task_raw, sampler, device, idx=2 * pair_idx + 2)
            noharm_batch = to_device_batch(noharm_raw, sampler, device, idx=2 * pair_idx + 3)
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

    acc = [r for r in rows if r["accepted"]]
    summary = {
        "attempts": len(rows),
        "accepted": len(acc),
        "accepted_rate": len(acc) / max(len(rows), 1),
        "task_dataset_coverage": len({r["task_dataset"] for r in rows}),
        "noharm_dataset_coverage": len({r["noharm_dataset"] for r in rows}),
        "mean_task_delta": fmean([float(r["raw_proj_task_delta"]) for r in acc if "raw_proj_task_delta" in r]),
        "median_task_delta": fmedian([float(r["raw_proj_task_delta"]) for r in acc if "raw_proj_task_delta" in r]),
        "mean_anchor_delta": fmean([float(r["proj_anchor_delta"]) for r in acc if "proj_anchor_delta" in r]),
        "mean_footprint": fmean([float(r["proj_footprint_mean_l2"]) for r in acc if "proj_footprint_mean_l2" in r]),
        "mean_material_row_frac": fmean([float(r["proj_material_row_frac"]) for r in acc if "proj_material_row_frac" in r]),
        "mean_task_grad_norm": fmean([float(r["task_grad_norm"]) for r in rows]),
    }
    gate_reasons = list(provenance_reasons)
    if max(noop_drifts or [999.0]) > 1e-7:
        gate_reasons.append("initial_noop_drift_above_1e-7")
    if summary["accepted_rate"] < 0.75:
        gate_reasons.append("accepted_rate_below_0p75")
    if (summary["mean_footprint"] or 0.0) < 5e-6:
        gate_reasons.append("mean_footprint_below_5e-6")
    if (summary["mean_anchor_delta"] or 0.0) > 1e-6:
        gate_reasons.append("mean_anchor_delta_above_1e-6")
    if summary["task_dataset_coverage"] < min(int(args.max_pairs), 8):
        gate_reasons.append("task_dataset_coverage_too_low")
    if summary["noharm_dataset_coverage"] < min(int(args.max_pairs), 8):
        gate_reasons.append("noharm_dataset_coverage_too_low")
    status = (
        "lowrank_residual_model_patch_unit_pass_gpu_launcher_candidate"
        if not gate_reasons
        else "lowrank_residual_model_patch_unit_fail_no_gpu"
    )

    row_fields = sorted({key for row in rows for key in row})
    write_csv(ROWS_CSV, rows, row_fields)
    write_csv(SUMMARY_CSV, [summary], list(summary.keys()))
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": status,
        "gpu_authorized": False,
        "gpu_launcher_candidate": not gate_reasons,
        "reasons": gate_reasons,
        "boundary": {
            "cpu_only": True,
            "safe_split": str(SAFE_SPLIT),
            "anchor_checkpoint": str(ANCHOR_CKPT),
            "core_model_patch_enabled": True,
            "trains_checkpoint": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
        },
        "args": vars(args),
        "anchor_load": anchor_load,
        "candidate_load": cand_load,
        "trainable_names": trainable_names,
        "shared_mismatches": shared_mismatches[:20],
        "max_noop_drift": max(noop_drifts or [999.0]),
        "summary": summary,
        "rows_csv": str(ROWS_CSV),
        "summary_csv": str(SUMMARY_CSV),
        "next_action": (
            "prepare a bounded GPU smoke launcher with RUN_STATUS and canonical no-harm gates"
            if not gate_reasons
            else "do not launch GPU from this model patch"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# LatentFM Low-Rank Residual Model Patch Unit Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized now: `false`",
        "",
        "## Boundary",
        "",
        "- CPU-only provenance/unit gate for the in-model default-off low-rank residual adapter.",
        "- No checkpoint training, no canonical metrics, no canonical multi, no Track C query, and no GPU.",
        "",
        "## Provenance",
        "",
        f"- candidate missing keys: `{cand_load['missing']}`",
        f"- unexpected keys: `{cand_load['unexpected']}`",
        f"- trainable names: `{trainable_names}`",
        f"- shared state mismatches shown: `{shared_mismatches[:5]}`",
        f"- max no-op drift: `{payload['max_noop_drift']}`",
        "",
        "## Summary",
        "",
        "| attempts | accepted | rate | task ds | noharm ds | mean task delta | mean anchor delta | mean footprint | mean material rows |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['attempts']} | {summary['accepted']} | {summary['accepted_rate']:.6g} | "
        f"{summary['task_dataset_coverage']} | {summary['noharm_dataset_coverage']} | "
        f"{summary['mean_task_delta']} | {summary['mean_anchor_delta']} | "
        f"{summary['mean_footprint']} | {summary['mean_material_row_frac']} |",
        "",
        "## Decision",
        "",
        f"- reasons: `{gate_reasons}`",
        f"- next action: {payload['next_action']}",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- rows: `{ROWS_CSV}`",
        f"- summary: `{SUMMARY_CSV}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(OUT_MD)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
