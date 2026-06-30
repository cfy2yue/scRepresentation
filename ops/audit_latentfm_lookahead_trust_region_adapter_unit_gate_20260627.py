#!/usr/bin/env python3
"""Lookahead/trust-region adapter unit gate.

CPU/report-only follow-up to the no-harm PCGrad unit gate. Vanilla PCGrad is
first-order blind at an exact default-off adapter because anchor replay has zero
gradient at initialization. This gate tests a distinct lookahead/trust-region
idea on frozen internal means:

1. take the task gradient at the no-op adapter;
2. probe the anchor/no-harm gradient after a virtual unprojected step;
3. project the original task gradient against that lookahead anchor gradient;
4. accept only if every seed/group slice has task improvement, bounded anchor
   loss, and nonzero condition-specific footprint.

No training, inference, checkpoint selection, canonical multi selection,
Track C query, or GPU is used. Passing only authorizes external audit and a
real train-batch unit gate, not a GPU run.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path("/data/cyx/1030/scLatent")))

from ops.audit_latentfm_noharm_pcgrad_adapter_unit_gate_20260627 import (  # noqa: E402
    ROOT,
    STEP_SIZES,
    build_batch,
    build_rows,
    evaluate_vector,
    flat_params,
    grad_vector,
    losses,
    pcgrad,
    set_flat_params,
)


REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "lookahead_trust_region_adapter_unit_gate_20260627"
OUT_ROWS = OUT_DIR / "lookahead_trust_region_adapter_step_rows.csv"
OUT_SUMMARY = OUT_DIR / "lookahead_trust_region_adapter_summary.csv"
OUT_JSON = REPORTS / "latentfm_lookahead_trust_region_adapter_unit_gate_20260627.json"
OUT_MD = REPORTS / "LATENTFM_LOOKAHEAD_TRUST_REGION_ADAPTER_UNIT_GATE_20260627.md"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def evaluate_slice(seed: str, group: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model, batch, meta = build_batch(rows)
    p0 = flat_params(model).clone()
    base_metrics = evaluate_vector(model, batch, p0)
    task_loss0, anchor_loss0, footprint0 = losses(model, batch)
    task_grad = grad_vector(task_loss0, model)
    task_grad_norm = float(torch.linalg.norm(task_grad).item())
    _, anchor_loss0b, _ = losses(model, batch)
    anchor_grad0 = grad_vector(anchor_loss0b, model)
    anchor_grad0_norm = float(torch.linalg.norm(anchor_grad0).item())

    step_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for step in STEP_SIZES:
        unproj_vec = p0 - float(step) * task_grad
        unproj_metrics = evaluate_vector(model, batch, unproj_vec)

        # Probe anchor gradient after the unprojected virtual step.
        _, anchor_probe_loss, _ = losses(model, batch)
        anchor_grad_probe = grad_vector(anchor_probe_loss, model)
        set_flat_params(model, p0)
        proj_grad, proj_stats = pcgrad(task_grad, anchor_grad_probe)
        proj_vec = p0 - float(step) * proj_grad
        proj_metrics = evaluate_vector(model, batch, proj_vec)
        set_flat_params(model, p0)

        unproj_task_delta = unproj_metrics["task_loss"] - base_metrics["task_loss"]
        proj_task_delta = proj_metrics["task_loss"] - base_metrics["task_loss"]
        task_retention = (
            abs(proj_task_delta) / max(abs(unproj_task_delta), 1e-12)
            if proj_task_delta < 0 and unproj_task_delta < 0
            else 0.0
        )
        row = {
            "seed": seed,
            "group": group,
            "step": step,
            "n_rows": meta["n_rows"],
            "n_genes": meta["n_genes"],
            "task_rows": meta["task_rows"],
            "task_grad_norm": task_grad_norm,
            "anchor_grad0_norm": anchor_grad0_norm,
            "unproj_task_delta": unproj_task_delta,
            "unproj_anchor_delta": unproj_metrics["anchor_loss"] - base_metrics["anchor_loss"],
            "unproj_footprint_mean_l2": unproj_metrics["footprint_mean_l2"],
            "proj_task_delta": proj_task_delta,
            "proj_anchor_delta": proj_metrics["anchor_loss"] - base_metrics["anchor_loss"],
            "proj_footprint_mean_l2": proj_metrics["footprint_mean_l2"],
            "proj_material_row_frac": proj_metrics["material_row_frac"],
            "proj_condition_specific_unique_frac": proj_metrics["condition_specific_unique_frac"],
            "task_retention_vs_unprojected": task_retention,
            "probe_anchor_grad_norm": proj_stats["anchor_norm"],
            "probe_dot_before": proj_stats["dot_before"],
            "probe_dot_after": proj_stats["dot_after"],
            "probe_projection_coeff": proj_stats["projection_coeff"],
            "projection_reduced_anchor_delta": proj_metrics["anchor_loss"] <= unproj_metrics["anchor_loss"] + 1e-12,
        }
        step_rows.append(row)
        candidate_ok = (
            row["proj_task_delta"] <= -1e-7
            and row["proj_anchor_delta"] <= 1e-6
            and row["proj_footprint_mean_l2"] >= 1e-4
            and row["proj_material_row_frac"] >= 0.15
            and row["proj_condition_specific_unique_frac"] >= 0.15
            and row["task_retention_vs_unprojected"] >= 0.20
            and row["projection_reduced_anchor_delta"]
        )
        if candidate_ok and (
            best is None
            or (row["proj_anchor_delta"], -row["proj_task_delta"]) < (best["proj_anchor_delta"], -best["proj_task_delta"])
        ):
            best = row

    reasons: list[str] = []
    if anchor_grad0_norm > 1e-10:
        reasons.append("unexpected_nonzero_initial_anchor_gradient")
    if task_grad_norm <= 1e-8:
        reasons.append("task_gradient_not_live")
    if best is None:
        reasons.append("no_step_passed_slice_gate")
    summary = {
        "seed": seed,
        "group": group,
        "n_rows": meta["n_rows"],
        "n_genes": meta["n_genes"],
        "task_rows": meta["task_rows"],
        "task_grad_norm": task_grad_norm,
        "anchor_grad0_norm": anchor_grad0_norm,
        "slice_pass": best is not None and task_grad_norm > 1e-8 and anchor_grad0_norm <= 1e-10,
        "fail_reasons": ";".join(reasons),
        "best_step": best["step"] if best else None,
        "best_proj_task_delta": best["proj_task_delta"] if best else None,
        "best_proj_anchor_delta": best["proj_anchor_delta"] if best else None,
        "best_proj_footprint_mean_l2": best["proj_footprint_mean_l2"] if best else None,
        "best_task_retention_vs_unprojected": best["task_retention_vs_unprojected"] if best else None,
        "best_condition_specific_unique_frac": best["proj_condition_specific_unique_frac"] if best else None,
    }
    return step_rows, summary


def main() -> None:
    rows = build_rows()
    all_step_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for seed in sorted({row["seed"] for row in rows}):
        for group in sorted({row["group"] for row in rows if row["seed"] == seed}):
            sub = [row for row in rows if row["seed"] == seed and row["group"] == group]
            step_rows, summary = evaluate_slice(seed, group, sub)
            all_step_rows.extend(step_rows)
            summaries.append(summary)

    reasons: list[str] = []
    if len(summaries) < 4:
        reasons.append("missing_seed_group_slices")
    failed = [row for row in summaries if not row["slice_pass"]]
    if failed:
        reasons.append("one_or_more_seed_group_slices_failed")
    worst_task_delta = min(float(row["best_proj_task_delta"] or 0.0) for row in summaries)
    worst_anchor_delta = max(float(row["best_proj_anchor_delta"] or 999.0) for row in summaries)
    min_footprint = min(float(row["best_proj_footprint_mean_l2"] or 0.0) for row in summaries)
    min_retention = min(float(row["best_task_retention_vs_unprojected"] or 0.0) for row in summaries)
    min_unique = min(float(row["best_condition_specific_unique_frac"] or 0.0) for row in summaries)
    if worst_anchor_delta > 1e-6:
        reasons.append("worst_anchor_delta_above_1e-6")
    if min_footprint < 1e-4:
        reasons.append("min_footprint_below_1e-4")
    if min_retention < 0.20:
        reasons.append("min_task_retention_below_0p20")
    if min_unique < 0.15:
        reasons.append("min_condition_specific_unique_frac_below_0p15")

    status = (
        "lookahead_trust_region_adapter_unit_gate_pass_external_audit_only_no_gpu"
        if not reasons
        else "lookahead_trust_region_adapter_unit_gate_fail_no_gpu"
    )

    write_csv(
        OUT_ROWS,
        all_step_rows,
        [
            "seed",
            "group",
            "step",
            "n_rows",
            "n_genes",
            "task_rows",
            "task_grad_norm",
            "anchor_grad0_norm",
            "unproj_task_delta",
            "unproj_anchor_delta",
            "unproj_footprint_mean_l2",
            "proj_task_delta",
            "proj_anchor_delta",
            "proj_footprint_mean_l2",
            "proj_material_row_frac",
            "proj_condition_specific_unique_frac",
            "task_retention_vs_unprojected",
            "probe_anchor_grad_norm",
            "probe_dot_before",
            "probe_dot_after",
            "probe_projection_coeff",
            "projection_reduced_anchor_delta",
        ],
    )
    write_csv(
        OUT_SUMMARY,
        summaries,
        [
            "seed",
            "group",
            "n_rows",
            "n_genes",
            "task_rows",
            "task_grad_norm",
            "anchor_grad0_norm",
            "slice_pass",
            "fail_reasons",
            "best_step",
            "best_proj_task_delta",
            "best_proj_anchor_delta",
            "best_proj_footprint_mean_l2",
            "best_task_retention_vs_unprojected",
            "best_condition_specific_unique_frac",
        ],
    )
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": status,
        "gpu_authorized": False,
        "reasons": reasons,
        "summary": summaries,
        "aggregate": {
            "worst_task_delta": worst_task_delta,
            "worst_anchor_delta": worst_anchor_delta,
            "min_footprint": min_footprint,
            "min_task_retention": min_retention,
            "min_condition_specific_unique_frac": min_unique,
        },
        "outputs": {
            "rows": str(OUT_ROWS),
            "summary": str(OUT_SUMMARY),
            "report": str(OUT_MD),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM Lookahead Trust-Region Adapter Unit Gate",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M %Z')}`",
        "",
        f"Status: `{status}`",
        "",
        "## Scope",
        "",
        "CPU/report-only frozen-means unit gate. This is a distinct method from "
        "vanilla PCGrad. It does not train, infer, select checkpoints, read "
        "canonical multi for selection, read Track C query, or use GPU.",
        "",
        "## Aggregate Gate",
        "",
        f"- worst anchor delta: `{worst_anchor_delta:.6g}`",
        f"- min footprint mean L2: `{min_footprint:.6g}`",
        f"- min task retention vs unprojected: `{min_retention:.6g}`",
        f"- min condition-specific unique fraction: `{min_unique:.6g}`",
        f"- reasons: `{reasons}`",
        "",
        "## Slice Summary",
        "",
        "| seed | group | pass | step | task delta | anchor delta | footprint | retention | unique frac |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['seed']}` | `{row['group']}` | `{row['slice_pass']}` | "
            f"`{row['best_step']}` | `{row['best_proj_task_delta']}` | "
            f"`{row['best_proj_anchor_delta']}` | `{row['best_proj_footprint_mean_l2']}` | "
            f"`{row['best_task_retention_vs_unprojected']}` | "
            f"`{row['best_condition_specific_unique_frac']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Even if this gate passes, it only authorizes external audit and a real "
            "train-batch/checkpoint-provenance unit gate. It is not a GPU launcher "
            "and not a model-improvement claim.",
            "",
            "## Outputs",
            "",
            f"- Rows: `{OUT_ROWS}`",
            f"- Summary: `{OUT_SUMMARY}`",
            f"- JSON: `{OUT_JSON}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "reasons": reasons, "aggregate": payload["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
