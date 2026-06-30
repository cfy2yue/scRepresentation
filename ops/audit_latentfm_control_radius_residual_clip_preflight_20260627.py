#!/usr/bin/env python3
"""Control-radius residual clipping preflight for LatentFM.

CPU/report-only. This simulates a default-off endpoint postprocessor on frozen
train/internal condition means:

    clipped_pred = pert_mean + clip(pred_mean - pert_mean, +/- tau * scale)

where scale is the per-dataset coordinate std of pert/control baseline means.
The goal is only to test whether a nonzero control-radius mechanism has any
internal no-harm promise before considering implementation or GPU. It does not
train, infer, select checkpoints, read canonical multi for Track A selection,
read Track C query, or use GPU.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
SEED_FILES = {
    "seed42": REPORTS / "latentfm_xverse_8k_seed_ensemble_internal_means_20260627/seed42_internal_split_group_means_evalseed42.json",
    "seed43": REPORTS / "latentfm_xverse_8k_seed_ensemble_internal_means_20260627/seed43_internal_split_group_means_evalseed42.json",
}
TAIL_ROWS = REPORTS / "train_internal_recurrent_tail_analogue_gate_20260627/train_internal_recurrent_tail_analogue_rows.csv"
OUT_DIR = REPORTS / "control_radius_residual_clip_preflight_20260627"
OUT_ROWS = OUT_DIR / "control_radius_residual_clip_condition_rows.csv"
OUT_SUMMARY = OUT_DIR / "control_radius_residual_clip_summary.csv"
OUT_JSON = REPORTS / "latentfm_control_radius_residual_clip_preflight_20260627.json"
OUT_MD = REPORTS / "LATENTFM_CONTROL_RADIUS_RESIDUAL_CLIP_PREFLIGHT_20260627.md"

TAUS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
RNG_SEED = 20260627
EPS = 1e-8


def norm(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def pearson_np(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3 or x.shape != y.shape:
        return None
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - float(x.mean())
    y = y - float(y.mean())
    denom = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    if denom <= 0:
        return None
    return float(np.dot(x, y) / denom)


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rank(xs), rank(ys)) if len(xs) >= 3 else None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_tail_labels() -> dict[tuple[str, str, str], bool]:
    labels: dict[tuple[str, str, str], bool] = {}
    for row in read_csv(TAIL_ROWS):
        labels[(norm(row.get("group")), norm(row.get("dataset")), norm(row.get("condition")))] = (
            norm(row.get("internal_recurrent_hard_tail")).lower() == "true"
        )
    return labels


def load_conditions() -> list[dict[str, Any]]:
    tail_labels = load_tail_labels()
    rows: list[dict[str, Any]] = []
    for seed, path in SEED_FILES.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        for group, gdata in data.get("groups", {}).items():
            if not group.startswith("internal_val_"):
                continue
            for rec in gdata.get("condition_metrics", []):
                dataset = norm(rec.get("dataset"))
                condition = norm(rec.get("condition"))
                pred = np.asarray(rec.get("pred_mean"), dtype=np.float32)
                gt = np.asarray(rec.get("gt_mean"), dtype=np.float32)
                pert = np.asarray(rec.get("pert_mean"), dtype=np.float32)
                if not dataset or not condition or pred.size == 0 or pred.shape != gt.shape or pred.shape != pert.shape:
                    continue
                effect = pred - pert
                gt_effect = gt - pert
                base_pp = pearson_np(effect, gt_effect)
                if base_pp is None:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "group": group,
                        "dataset": dataset,
                        "condition": condition,
                        "pred": pred,
                        "gt": gt,
                        "pert": pert,
                        "effect": effect,
                        "gt_effect": gt_effect,
                        "base_pp": base_pp,
                        "base_endpoint_mse": float(np.mean((pred - gt) ** 2)),
                        "base_effect_l2": float(np.linalg.norm(effect)),
                        "mmd_original": rec.get("test_mmd_clamped"),
                        "hard_tail": bool(tail_labels.get((group, dataset, condition), False)),
                    }
                )
    return rows


def compute_scales(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], np.ndarray]:
    by_key: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    all_pert: list[np.ndarray] = []
    for row in rows:
        by_key[(row["seed"], row["group"], row["dataset"])].append(row["pert"])
        all_pert.append(row["pert"])
    global_scale = np.std(np.stack(all_pert, axis=0), axis=0).astype(np.float32)
    global_floor = float(np.median(global_scale[global_scale > 0])) * 0.05 if np.any(global_scale > 0) else 1e-4
    global_scale = np.maximum(global_scale, global_floor)
    scales: dict[tuple[str, str, str], np.ndarray] = {}
    for key, vals in by_key.items():
        if len(vals) >= 3:
            scale = np.std(np.stack(vals, axis=0), axis=0).astype(np.float32)
            floor = float(np.median(scale[scale > 0])) * 0.05 if np.any(scale > 0) else global_floor
            scale = np.maximum(scale, floor)
        else:
            scale = global_scale
        scales[key] = scale
    return scales


def dataset_bootstrap_ci_low(rows: list[dict[str, Any]], key: str, *, n_boot: int = 1000) -> float | None:
    datasets = sorted({row["dataset"] for row in rows})
    if len(datasets) < 3:
        return None
    by_dataset = {dataset: [row for row in rows if row["dataset"] == dataset] for dataset in datasets}
    rng = random.Random(RNG_SEED)
    vals: list[float] = []
    for _ in range(n_boot):
        sample = []
        for dataset in [rng.choice(datasets) for _ in datasets]:
            pool = by_dataset[dataset]
            sample.extend(pool[rng.randrange(len(pool))] for _ in range(len(pool)))
        vals.append(mean(float(row[key]) for row in sample))
    vals.sort()
    return vals[int(0.025 * (len(vals) - 1))] if vals else None


def summarize(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for seed in sorted({row["seed"] for row in candidate_rows}):
        for group in sorted({row["group"] for row in candidate_rows if row["seed"] == seed}):
            for tau in TAUS:
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
                        "changed_condition_frac": sum(row["changed_coord_frac"] > 0 for row in sub) / len(sub),
                        "changed_coord_frac_mean": mean(float(row["changed_coord_frac"]) for row in sub),
                        "mean_delta_pp": mean(float(row["delta_pp"]) for row in sub),
                        "hard_tail_delta_pp": mean(float(row["delta_pp"]) for row in hard) if hard else None,
                        "endpoint_mse_delta_mean": mean(float(row["endpoint_mse_delta"]) for row in sub),
                        "dataset_min_delta_pp": min(per_dataset.values()) if per_dataset else None,
                        "dataset_bootstrap_ci_low": dataset_bootstrap_ci_low(sub, "delta_pp"),
                        "mmd_high_minus_low_delta_pp": (mean(mmd_hi) - mean(mmd_lo)) if mmd_hi and mmd_lo else None,
                    }
                )
    return summaries


def choose_and_decide(summaries: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str, list[str]]:
    # A candidate must satisfy both internal groups/seeds. Rank by the worst
    # mean delta across the four seed/group slices.
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
        return None, "control_radius_residual_clip_preflight_fail_no_gpu", ["no_complete_tau_candidate"]
    best = max(candidates, key=lambda row: (row["worst_mean_delta_pp"], row["worst_hard_tail_delta_pp"], -row["max_endpoint_mse_delta"]))
    reasons = []
    if best["min_changed_condition_frac"] < 0.10:
        reasons.append("nonzero_footprint_condition_frac_below_0p10")
    if best["worst_mean_delta_pp"] < 0.02:
        reasons.append("worst_internal_mean_delta_pp_below_0p02")
    if best["worst_hard_tail_delta_pp"] < 0.02:
        reasons.append("worst_hard_tail_delta_pp_below_0p02")
    if best["worst_dataset_min_delta_pp"] < -0.01:
        reasons.append("dataset_min_delta_pp_below_minus_0p01")
    if best["worst_ci_low"] <= 0:
        reasons.append("dataset_bootstrap_ci_low_not_positive")
    if best["max_endpoint_mse_delta"] > 0:
        reasons.append("endpoint_mse_surrogate_harm")
    if best["max_mmd_high_minus_low_delta_pp"] < -0.001:
        reasons.append("mmd_high_rows_delta_pp_worse_than_low_by_gt_0p001")
    # Even a pass would require real distribution-level no-harm before GPU.
    status = "control_radius_residual_clip_preflight_pass_external_audit_only_no_gpu" if not reasons else "control_radius_residual_clip_preflight_fail_no_gpu"
    return best, status, reasons


def main() -> None:
    rows = load_conditions()
    scales = compute_scales(rows)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        scale = scales[(row["seed"], row["group"], row["dataset"])]
        effect = row["effect"]
        gt_effect = row["gt_effect"]
        for tau in TAUS:
            limit = tau * scale
            clipped_effect = np.clip(effect, -limit, limit)
            clipped_pred = row["pert"] + clipped_effect
            clipped_pp = pearson_np(clipped_effect, gt_effect)
            if clipped_pp is None:
                continue
            changed = np.abs(clipped_effect - effect) > EPS
            out_rows.append(
                {
                    "seed": row["seed"],
                    "group": row["group"],
                    "dataset": row["dataset"],
                    "condition": row["condition"],
                    "tau": tau,
                    "base_pp": row["base_pp"],
                    "clipped_pp": clipped_pp,
                    "delta_pp": clipped_pp - float(row["base_pp"]),
                    "base_endpoint_mse": row["base_endpoint_mse"],
                    "clipped_endpoint_mse": float(np.mean((clipped_pred - row["gt"]) ** 2)),
                    "endpoint_mse_delta": float(np.mean((clipped_pred - row["gt"]) ** 2)) - float(row["base_endpoint_mse"]),
                    "changed_coord_frac": float(np.mean(changed)),
                    "effect_l2_ratio": float(np.linalg.norm(clipped_effect) / (np.linalg.norm(effect) + EPS)),
                    "mmd_original": row["mmd_original"],
                    "hard_tail": row["hard_tail"],
                }
            )
    summaries = summarize(out_rows)
    best, status, reasons = choose_and_decide(summaries)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUT_ROWS,
        out_rows,
        [
            "seed",
            "group",
            "dataset",
            "condition",
            "tau",
            "base_pp",
            "clipped_pp",
            "delta_pp",
            "base_endpoint_mse",
            "clipped_endpoint_mse",
            "endpoint_mse_delta",
            "changed_coord_frac",
            "effect_l2_ratio",
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
            "changed_coord_frac_mean",
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
        "inputs": {seed: str(path) for seed, path in SEED_FILES.items()},
        "n_condition_tau_rows": len(out_rows),
        "best_tau_candidate": best,
        "reasons": reasons,
        "outputs": {"rows": str(OUT_ROWS), "summary": str(OUT_SUMMARY), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    top = sorted(summaries, key=lambda row: (row["tau"], row["seed"], row["group"]))
    table_lines = []
    for row in top:
        table_lines.append(
            "| {seed} | {group} | {tau:g} | {changed:.3f} | {dpp:+.6f} | {tail} | {dsmin} | {ci} | {mse:+.6e} |".format(
                seed=row["seed"],
                group=row["group"],
                tau=float(row["tau"]),
                changed=float(row["changed_condition_frac"]),
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
            f"tau `{best['tau']}`; worst mean delta pp `{best['worst_mean_delta_pp']:+.6f}`; "
            f"worst hard-tail delta `{best['worst_hard_tail_delta_pp']:+.6f}`; "
            f"worst dataset min `{best['worst_dataset_min_delta_pp']:+.6f}`; "
            f"max endpoint-MSE delta `{best['max_endpoint_mse_delta']:+.6e}`"
        )
    md = f"""# LatentFM Control-Radius Residual Clip Preflight 2026-06-27

Timestamp: `{payload['timestamp']}`

Status: `{status}`

GPU authorized: `False`

## Boundary

- CPU/report-only posthoc simulation on frozen train/internal condition means.
- No training, inference, checkpoint selection, canonical multi selection,
  Track C query, or GPU.
- This preflight cannot authorize GPU by itself because it does not recompute
  distribution-level MMD/no-harm; it only tests nonzero footprint and mean-level
  internal no-harm promise.

## Hypothesis

Coordinate-level clipping of predicted perturbation residuals by a
dataset-specific control/baseline radius may reduce overshoot on recurrent
tails while preserving internal cross-background/family proxy performance.

## Best Candidate

{best_text}

Reasons: `{reasons}`

## Slice Summary

| seed | group | tau | changed condition frac | mean delta pp | hard-tail delta pp | dataset min delta | dataset bootstrap CI low | endpoint-MSE delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_lines)}

## Decision

GPU remains unauthorized. A future branch would require external audit plus a
real implementation that recomputes distribution-level internal and canonical
no-harm metrics. Current status is `{status}`.

## Outputs

- JSON: `{OUT_JSON}`
- rows: `{OUT_ROWS}`
- summary: `{OUT_SUMMARY}`
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
