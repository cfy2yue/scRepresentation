#!/usr/bin/env python3
"""Direction-preserving residual norm trust-region preflight.

CPU/report-only. This follow-up tests whether the clipping branch failed
because coordinate-wise clipping distorted residual direction. It scales the
whole predicted perturbation residual vector only when its norm exceeds a
dataset-specific baseline radius:

    candidate_effect = effect * min(1, tau * radius / ||effect||)

It uses frozen train/internal condition means only. It does not train, infer,
select checkpoints, read canonical multi for selection, read Track C query, or
use GPU.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

sys.path.insert(0, str(Path("/data/cyx/1030/scLatent")))

from ops.audit_latentfm_control_radius_residual_clip_preflight_20260627 import (  # noqa: E402
    EPS,
    ROOT,
    TAUS,
    dataset_bootstrap_ci_low,
    load_conditions,
    pearson_np,
)


REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "direction_preserving_residual_norm_trust_region_preflight_20260627"
OUT_ROWS = OUT_DIR / "direction_preserving_residual_norm_trust_region_rows.csv"
OUT_SUMMARY = OUT_DIR / "direction_preserving_residual_norm_trust_region_summary.csv"
OUT_JSON = REPORTS / "latentfm_direction_preserving_residual_norm_trust_region_preflight_20260627.json"
OUT_MD = REPORTS / "LATENTFM_DIRECTION_PRESERVING_RESIDUAL_NORM_TRUST_REGION_PREFLIGHT_20260627.md"

NORM_TAUS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def compute_dataset_radii(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], float]:
    """Use pert-mean dispersion as a query-blind dataset radius."""
    by_key: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    global_pert: list[np.ndarray] = []
    for row in rows:
        by_key[(row["seed"], row["group"], row["dataset"])].append(row["pert"])
        global_pert.append(row["pert"])
    global_stack = np.stack(global_pert, axis=0)
    global_center = global_stack.mean(axis=0)
    global_dists = np.linalg.norm(global_stack - global_center, axis=1)
    global_radius = float(np.median(global_dists[global_dists > 0])) if np.any(global_dists > 0) else 1.0
    radii: dict[tuple[str, str, str], float] = {}
    for key, vals in by_key.items():
        if len(vals) >= 3:
            stack = np.stack(vals, axis=0)
            center = stack.mean(axis=0)
            dists = np.linalg.norm(stack - center, axis=1)
            radius = float(np.median(dists[dists > 0])) if np.any(dists > 0) else global_radius
        else:
            radius = global_radius
        radii[key] = max(radius, global_radius * 0.05, EPS)
    return radii


def summarize(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for seed in sorted({row["seed"] for row in candidate_rows}):
        for group in sorted({row["group"] for row in candidate_rows if row["seed"] == seed}):
            for tau in NORM_TAUS:
                sub = [row for row in candidate_rows if row["seed"] == seed and row["group"] == group and row["tau"] == tau]
                if not sub:
                    continue
                hard = [row for row in sub if row["hard_tail"]]
                per_dataset: dict[str, float] = {}
                for dataset in sorted({row["dataset"] for row in sub}):
                    vals = [float(row["delta_pp"]) for row in sub if row["dataset"] == dataset]
                    if vals:
                        per_dataset[dataset] = mean(vals)
                mmd_hi = [float(row["delta_pp"]) for row in sub if float(row.get("mmd_original") or 0.0) >= 0.01]
                mmd_lo = [float(row["delta_pp"]) for row in sub if float(row.get("mmd_original") or 0.0) < 0.01]
                summaries.append(
                    {
                        "seed": seed,
                        "group": group,
                        "tau": tau,
                        "n": len(sub),
                        "datasets": len(per_dataset),
                        "changed_condition_frac": sum(float(row["scale_factor"]) < 1.0 - 1e-7 for row in sub) / len(sub),
                        "scale_factor_mean": mean(float(row["scale_factor"]) for row in sub),
                        "mean_delta_pp": mean(float(row["delta_pp"]) for row in sub),
                        "hard_tail_delta_pp": mean(float(row["delta_pp"]) for row in hard) if hard else None,
                        "endpoint_mse_delta_mean": mean(float(row["endpoint_mse_delta"]) for row in sub),
                        "dataset_min_delta_pp": min(per_dataset.values()) if per_dataset else None,
                        "dataset_bootstrap_ci_low": dataset_bootstrap_ci_low(sub, "delta_pp"),
                        "mmd_high_minus_low_delta_pp": (mean(mmd_hi) - mean(mmd_lo)) if mmd_hi and mmd_lo else None,
                    }
                )
    return summaries


def choose(summaries: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str, list[str]]:
    by_tau: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_tau[float(row["tau"])].append(row)
    candidates: list[dict[str, Any]] = []
    for tau, rows in sorted(by_tau.items()):
        if len(rows) < 4:
            continue
        candidates.append(
            {
                "tau": tau,
                "slices": rows,
                "worst_mean_delta_pp": min(float(row["mean_delta_pp"]) for row in rows),
                "worst_hard_tail_delta_pp": min(float(row["hard_tail_delta_pp"] or -999.0) for row in rows),
                "worst_dataset_min_delta_pp": min(float(row["dataset_min_delta_pp"] or -999.0) for row in rows),
                "worst_ci_low": min(float(row["dataset_bootstrap_ci_low"] or -999.0) for row in rows),
                "max_endpoint_mse_delta": max(float(row["endpoint_mse_delta_mean"]) for row in rows),
                "min_changed_condition_frac": min(float(row["changed_condition_frac"]) for row in rows),
                "max_mmd_high_minus_low_delta_pp": max(float(row["mmd_high_minus_low_delta_pp"] or 0.0) for row in rows),
            }
        )
    if not candidates:
        return None, "direction_preserving_residual_norm_trust_region_preflight_fail_no_gpu", ["no_complete_candidate"]
    best = max(
        candidates,
        key=lambda row: (
            row["worst_dataset_min_delta_pp"],
            row["worst_mean_delta_pp"],
            row["worst_hard_tail_delta_pp"],
            -row["max_endpoint_mse_delta"],
        ),
    )
    reasons = []
    if best["min_changed_condition_frac"] < 0.05:
        reasons.append("changed_condition_frac_below_0p05")
    if best["worst_mean_delta_pp"] < -0.002:
        reasons.append("worst_internal_mean_delta_pp_below_minus_0p002")
    if best["worst_hard_tail_delta_pp"] < 0.03:
        reasons.append("worst_hard_tail_delta_pp_below_0p03")
    if best["worst_dataset_min_delta_pp"] < -0.02:
        reasons.append("dataset_min_delta_pp_below_minus_0p02")
    if best["worst_ci_low"] < -0.005:
        reasons.append("dataset_bootstrap_ci_low_below_minus_0p005")
    if best["max_endpoint_mse_delta"] > 0:
        reasons.append("endpoint_mse_surrogate_harm")
    if best["max_mmd_high_minus_low_delta_pp"] < -0.001:
        reasons.append("mmd_high_rows_delta_pp_worse_than_low_by_gt_0p001")
    status = (
        "direction_preserving_residual_norm_trust_region_preflight_pass_external_audit_only_no_gpu"
        if not reasons
        else "direction_preserving_residual_norm_trust_region_preflight_fail_no_gpu"
    )
    return best, status, reasons


def main() -> None:
    rows = load_conditions()
    radii = compute_dataset_radii(rows)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        radius = radii[(row["seed"], row["group"], row["dataset"])]
        effect = row["effect"]
        gt_effect = row["gt_effect"]
        effect_norm = float(np.linalg.norm(effect))
        for tau in NORM_TAUS:
            limit = tau * radius
            scale_factor = min(1.0, limit / (effect_norm + EPS))
            candidate_effect = effect * scale_factor
            candidate_pred = row["pert"] + candidate_effect
            candidate_pp = pearson_np(candidate_effect, gt_effect)
            if candidate_pp is None:
                continue
            out_rows.append(
                {
                    "seed": row["seed"],
                    "group": row["group"],
                    "dataset": row["dataset"],
                    "condition": row["condition"],
                    "tau": tau,
                    "radius": radius,
                    "effect_norm": effect_norm,
                    "scale_factor": scale_factor,
                    "base_pp": row["base_pp"],
                    "candidate_pp": candidate_pp,
                    "delta_pp": candidate_pp - float(row["base_pp"]),
                    "base_endpoint_mse": row["base_endpoint_mse"],
                    "candidate_endpoint_mse": float(np.mean((candidate_pred - row["gt"]) ** 2)),
                    "endpoint_mse_delta": float(np.mean((candidate_pred - row["gt"]) ** 2)) - float(row["base_endpoint_mse"]),
                    "mmd_original": row["mmd_original"],
                    "hard_tail": row["hard_tail"],
                }
            )
    summaries = summarize(out_rows)
    best, status, reasons = choose(summaries)

    write_csv(
        OUT_ROWS,
        out_rows,
        [
            "seed",
            "group",
            "dataset",
            "condition",
            "tau",
            "radius",
            "effect_norm",
            "scale_factor",
            "base_pp",
            "candidate_pp",
            "delta_pp",
            "base_endpoint_mse",
            "candidate_endpoint_mse",
            "endpoint_mse_delta",
            "mmd_original",
            "hard_tail",
        ],
    )
    write_csv(
        OUT_SUMMARY,
        summaries,
        [
            "seed",
            "group",
            "tau",
            "n",
            "datasets",
            "changed_condition_frac",
            "scale_factor_mean",
            "mean_delta_pp",
            "hard_tail_delta_pp",
            "endpoint_mse_delta_mean",
            "dataset_min_delta_pp",
            "dataset_bootstrap_ci_low",
            "mmd_high_minus_low_delta_pp",
        ],
    )
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_report_only": True,
            "train_or_infer": False,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
            "true_distribution_mmd_recomputed": False,
        },
        "best_candidate": best,
        "reasons": reasons,
        "n_condition_rows": len(out_rows),
        "outputs": {"rows": str(OUT_ROWS), "summary": str(OUT_SUMMARY), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    rows_for_table = sorted(
        summaries,
        key=lambda row: (
            abs(float(row["tau"]) - float(best["tau"])) if best else 999.0,
            row["seed"],
            row["group"],
        ),
    )[:24]
    lines = []
    for row in rows_for_table:
        lines.append(
            "| {seed} | {group} | {tau:g} | {changed:.3f} | {scale:.3f} | {dpp:+.6f} | {tail} | {dsmin} | {ci} | {mse:+.6e} |".format(
                seed=row["seed"],
                group=row["group"],
                tau=float(row["tau"]),
                changed=float(row["changed_condition_frac"]),
                scale=float(row["scale_factor_mean"]),
                dpp=float(row["mean_delta_pp"]),
                tail=f"{row['hard_tail_delta_pp']:+.6f}" if isinstance(row.get("hard_tail_delta_pp"), float) else "NA",
                dsmin=f"{row['dataset_min_delta_pp']:+.6f}" if isinstance(row.get("dataset_min_delta_pp"), float) else "NA",
                ci=f"{row['dataset_bootstrap_ci_low']:+.6f}" if isinstance(row.get("dataset_bootstrap_ci_low"), float) else "NA",
                mse=float(row["endpoint_mse_delta_mean"]),
            )
        )
    best_text = "None"
    if best:
        best_text = (
            f"tau `{best['tau']}`; worst mean delta `{best['worst_mean_delta_pp']:+.6f}`; "
            f"worst hard-tail delta `{best['worst_hard_tail_delta_pp']:+.6f}`; "
            f"worst dataset min `{best['worst_dataset_min_delta_pp']:+.6f}`; "
            f"worst CI low `{best['worst_ci_low']:+.6f}`"
        )
    md = f"""# LatentFM Direction-Preserving Residual Norm Trust-Region Preflight 2026-06-27

Timestamp: `{payload['timestamp']}`

Status: `{status}`

GPU authorized: `False`

## Boundary

- CPU/report-only posthoc simulation on frozen train/internal condition means.
- No training, inference, checkpoint selection, canonical multi selection,
  Track C query, or GPU.
- This preflight cannot authorize GPU by itself because distribution-level
  MMD/no-harm is not recomputed.

## Hypothesis

Coordinate clipping rescued recurrent hard tails but harmed no-harm metrics by
distorting residual direction. A vector-norm trust region preserves direction
while reducing residual overshoot.

## Best Candidate

{best_text}

Reasons: `{reasons}`

## Nearby Slice Summary

| seed | group | tau | changed condition frac | mean scale | mean delta pp | hard-tail delta pp | dataset min delta | dataset bootstrap CI low | endpoint-MSE delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

## Decision

GPU remains unauthorized. If this gate fails, clipping-derived posthoc
interventions should be closed unless a real distribution-level implementation
produces new no-harm evidence. If it passes, it still requires external audit
and real internal/canonical MMD/no-harm before GPU.

## Outputs

- JSON: `{OUT_JSON}`
- rows: `{OUT_ROWS}`
- summary: `{OUT_SUMMARY}`
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
