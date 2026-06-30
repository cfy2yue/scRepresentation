#!/usr/bin/env python3
"""Selective high-pressure control-radius clipping preflight.

This is a follow-up to the broad control-radius clip preflight. Broad clipping
improved recurrent hard tails but harmed mean/dataset internal pp because it
changed almost every condition. This gate only activates clipping on conditions
whose standardized predicted residual pressure is in the top within-dataset
quantile.

CPU/report-only; no training, inference, checkpoint selection, canonical multi
selection, Track C query, or GPU.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

sys.path.insert(0, str(Path("/data/cyx/1030/scLatent")))

from ops.audit_latentfm_control_radius_residual_clip_preflight_20260627 import (
    ROOT,
    TAUS,
    compute_scales,
    dataset_bootstrap_ci_low,
    load_conditions,
    pearson_np,
)


REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "selective_control_radius_clip_preflight_20260627"
OUT_ROWS = OUT_DIR / "selective_control_radius_clip_condition_rows.csv"
OUT_SUMMARY = OUT_DIR / "selective_control_radius_clip_summary.csv"
OUT_JSON = REPORTS / "latentfm_selective_control_radius_clip_preflight_20260627.json"
OUT_MD = REPORTS / "LATENTFM_SELECTIVE_CONTROL_RADIUS_CLIP_PREFLIGHT_20260627.md"
ACTIVATION_QS = [0.50, 0.70, 0.80, 0.90, 0.95]
EPS = 1e-8


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def pressure_scores(rows: list[dict[str, Any]], scales: dict[tuple[str, str, str], np.ndarray]) -> None:
    for row in rows:
        scale = scales[(row["seed"], row["group"], row["dataset"])]
        standardized = np.abs(row["effect"]) / (scale + EPS)
        row["pressure_l2"] = float(np.linalg.norm(standardized) / math.sqrt(standardized.size))
        row["pressure_p95"] = float(np.percentile(standardized, 95))


def summarize(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for seed in sorted({row["seed"] for row in candidate_rows}):
        for group in sorted({row["group"] for row in candidate_rows if row["seed"] == seed}):
            for tau in TAUS:
                for q in ACTIVATION_QS:
                    sub = [row for row in candidate_rows if row["seed"] == seed and row["group"] == group and row["tau"] == tau and row["activation_q"] == q]
                    if not sub:
                        continue
                    hard = [row for row in sub if row["hard_tail"]]
                    per_dataset: dict[str, float] = {}
                    for dataset in sorted({row["dataset"] for row in sub}):
                        vals = [float(row["delta_pp"]) for row in sub if row["dataset"] == dataset]
                        if vals:
                            per_dataset[dataset] = mean(vals)
                    summaries.append(
                        {
                            "seed": seed,
                            "group": group,
                            "tau": tau,
                            "activation_q": q,
                            "n": len(sub),
                            "datasets": len(per_dataset),
                            "activated_condition_frac": sum(bool(row["activated"]) for row in sub) / len(sub),
                            "changed_condition_frac": sum(row["changed_coord_frac"] > 0 for row in sub) / len(sub),
                            "changed_coord_frac_mean": mean(float(row["changed_coord_frac"]) for row in sub),
                            "mean_delta_pp": mean(float(row["delta_pp"]) for row in sub),
                            "hard_tail_delta_pp": mean(float(row["delta_pp"]) for row in hard) if hard else None,
                            "endpoint_mse_delta_mean": mean(float(row["endpoint_mse_delta"]) for row in sub),
                            "dataset_min_delta_pp": min(per_dataset.values()) if per_dataset else None,
                            "dataset_bootstrap_ci_low": dataset_bootstrap_ci_low(sub, "delta_pp"),
                        }
                    )
    return summaries


def choose(summaries: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str, list[str]]:
    by_key: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_key[(float(row["tau"]), float(row["activation_q"]))].append(row)
    candidates: list[dict[str, Any]] = []
    for (tau, q), rows in sorted(by_key.items()):
        if len(rows) < 4:
            continue
        candidates.append(
            {
                "tau": tau,
                "activation_q": q,
                "slices": rows,
                "worst_mean_delta_pp": min(float(row["mean_delta_pp"]) for row in rows),
                "worst_hard_tail_delta_pp": min(float(row["hard_tail_delta_pp"] or -999.0) for row in rows),
                "worst_dataset_min_delta_pp": min(float(row["dataset_min_delta_pp"] or -999.0) for row in rows),
                "worst_ci_low": min(float(row["dataset_bootstrap_ci_low"] or -999.0) for row in rows),
                "max_endpoint_mse_delta": max(float(row["endpoint_mse_delta_mean"]) for row in rows),
                "min_activated_condition_frac": min(float(row["activated_condition_frac"]) for row in rows),
                "max_activated_condition_frac": max(float(row["activated_condition_frac"]) for row in rows),
            }
        )
    if not candidates:
        return None, "selective_control_radius_clip_preflight_fail_no_gpu", ["no_complete_candidate"]
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
    if best["min_activated_condition_frac"] < 0.03:
        reasons.append("activated_condition_frac_below_0p03")
    if best["max_activated_condition_frac"] > 0.35:
        reasons.append("activated_condition_frac_above_0p35")
    if best["worst_mean_delta_pp"] < -0.005:
        reasons.append("worst_internal_mean_delta_pp_below_minus_0p005")
    if best["worst_hard_tail_delta_pp"] < 0.02:
        reasons.append("worst_hard_tail_delta_pp_below_0p02")
    if best["worst_dataset_min_delta_pp"] < -0.02:
        reasons.append("dataset_min_delta_pp_below_minus_0p02")
    if best["worst_ci_low"] < -0.005:
        reasons.append("dataset_bootstrap_ci_low_below_minus_0p005")
    if best["max_endpoint_mse_delta"] > 0:
        reasons.append("endpoint_mse_surrogate_harm")
    status = "selective_control_radius_clip_preflight_pass_external_audit_only_no_gpu" if not reasons else "selective_control_radius_clip_preflight_fail_no_gpu"
    return best, status, reasons


def main() -> None:
    rows = load_conditions()
    scales = compute_scales(rows)
    pressure_scores(rows, scales)
    pressure_by_slice_dataset: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        pressure_by_slice_dataset[(row["seed"], row["group"], row["dataset"])].append(float(row["pressure_l2"]))
    thresholds: dict[tuple[str, str, str, float], float] = {}
    for key, vals in pressure_by_slice_dataset.items():
        for q in ACTIVATION_QS:
            thresholds[(*key, q)] = float(np.quantile(vals, q))

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        scale = scales[(row["seed"], row["group"], row["dataset"])]
        effect = row["effect"]
        gt_effect = row["gt_effect"]
        for tau in TAUS:
            limit = tau * scale
            clipped_effect = np.clip(effect, -limit, limit)
            for q in ACTIVATION_QS:
                threshold = thresholds[(row["seed"], row["group"], row["dataset"], q)]
                activated = float(row["pressure_l2"]) >= threshold
                candidate_effect = clipped_effect if activated else effect
                candidate_pred = row["pert"] + candidate_effect
                candidate_pp = pearson_np(candidate_effect, gt_effect)
                if candidate_pp is None:
                    continue
                changed = np.abs(candidate_effect - effect) > EPS
                out_rows.append(
                    {
                        "seed": row["seed"],
                        "group": row["group"],
                        "dataset": row["dataset"],
                        "condition": row["condition"],
                        "tau": tau,
                        "activation_q": q,
                        "pressure_l2": row["pressure_l2"],
                        "activated": activated,
                        "base_pp": row["base_pp"],
                        "candidate_pp": candidate_pp,
                        "delta_pp": candidate_pp - float(row["base_pp"]),
                        "endpoint_mse_delta": float(np.mean((candidate_pred - row["gt"]) ** 2)) - float(row["base_endpoint_mse"]),
                        "changed_coord_frac": float(np.mean(changed)),
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
            "activation_q",
            "pressure_l2",
            "activated",
            "base_pp",
            "candidate_pp",
            "delta_pp",
            "endpoint_mse_delta",
            "changed_coord_frac",
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
            "activation_q",
            "n",
            "datasets",
            "activated_condition_frac",
            "changed_condition_frac",
            "changed_coord_frac_mean",
            "mean_delta_pp",
            "hard_tail_delta_pp",
            "endpoint_mse_delta_mean",
            "dataset_min_delta_pp",
            "dataset_bootstrap_ci_low",
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

    top = sorted(
        summaries,
        key=lambda row: (
            abs(float(row["tau"]) - float(best["tau"])) if best else 999.0,
            abs(float(row["activation_q"]) - float(best["activation_q"])) if best else 999.0,
            row["seed"],
            row["group"],
        ),
    )[:20]
    lines = []
    for row in top:
        lines.append(
            "| {seed} | {group} | {tau:g} | {q:.2f} | {act:.3f} | {dpp:+.6f} | {tail} | {dsmin} | {ci} | {mse:+.6e} |".format(
                seed=row["seed"],
                group=row["group"],
                tau=float(row["tau"]),
                q=float(row["activation_q"]),
                act=float(row["activated_condition_frac"]),
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
            f"tau `{best['tau']}`, activation q `{best['activation_q']}`; "
            f"worst mean delta `{best['worst_mean_delta_pp']:+.6f}`; "
            f"worst hard-tail delta `{best['worst_hard_tail_delta_pp']:+.6f}`; "
            f"worst dataset min `{best['worst_dataset_min_delta_pp']:+.6f}`; "
            f"worst CI low `{best['worst_ci_low']:+.6f}`"
        )
    md = f"""# LatentFM Selective Control-Radius Clip Preflight 2026-06-27

Timestamp: `{payload['timestamp']}`

Status: `{status}`

GPU authorized: `False`

## Boundary

- CPU/report-only selective posthoc simulation on frozen train/internal
  condition means.
- No training, inference, checkpoint selection, canonical multi selection,
  Track C query, or GPU.
- This preflight cannot authorize GPU by itself because distribution-level
  MMD/no-harm is not recomputed.

## Hypothesis

Broad clipping improved recurrent hard tails but harmed average internal pp.
Selective clipping may preserve hard-tail gains by only clipping high-pressure
predicted residual conditions.

## Best Candidate

{best_text}

Reasons: `{reasons}`

## Nearby Slice Summary

| seed | group | tau | activation q | activated frac | mean delta pp | hard-tail delta pp | dataset min delta | dataset bootstrap CI low | endpoint-MSE delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

## Decision

GPU remains unauthorized. If this fails, the control-radius clipping mechanism
is useful as failure anatomy only. If it passes, it still requires external
audit and real distribution-level internal/canonical no-harm before GPU.

## Outputs

- JSON: `{OUT_JSON}`
- rows: `{OUT_ROWS}`
- summary: `{OUT_SUMMARY}`
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
