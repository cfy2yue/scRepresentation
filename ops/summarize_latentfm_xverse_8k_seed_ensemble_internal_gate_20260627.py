#!/usr/bin/env python3
"""Summarize the internal seed42/seed43 frozen-anchor ensemble gate.

Inputs must be condition-mean eval artifacts produced on the train-only
cross-background validation split with a locked eval seed. This script compares
seed42, seed43, and simple mean prediction ensembles. It is CPU-only and cannot
authorize canonical claims by itself.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
IN_DIR = ROOT / "reports/latentfm_xverse_8k_seed_ensemble_internal_means_20260627"
SEED42 = IN_DIR / "seed42_internal_split_group_means_evalseed42.json"
SEED43 = IN_DIR / "seed43_internal_split_group_means_evalseed42.json"
OUT_JSON = ROOT / "reports/latentfm_xverse_8k_seed_ensemble_internal_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_8K_SEED_ENSEMBLE_INTERNAL_GATE_20260627.md"


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / den)


def load_rows(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for group, group_obj in obj.get("groups", {}).items():
        for row in group_obj.get("condition_metrics", []):
            key = (str(group), str(row.get("dataset")), str(row.get("condition")))
            if "pred_mean" not in row or "pert_mean" not in row:
                raise ValueError(f"condition means missing for {path}: {key}")
            rows[key] = row
    return rows


def bootstrap(vals: list[float], seed: int, n_boot: int = 5000) -> dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "p_gt0": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_gt0": float(np.mean(means > 0)),
    }


def pearson_pert(pred: np.ndarray, gt: np.ndarray, pert: np.ndarray) -> float:
    return pearson(np.asarray(pred, dtype=float) - pert, np.asarray(gt, dtype=float) - pert)


def summarize_group(rows42: dict[tuple[str, str, str], dict[str, Any]], rows43: dict[tuple[str, str, str], dict[str, Any]], group: str) -> dict[str, Any]:
    keys = sorted(k for k in rows42 if k[0] == group and k in rows43)
    per_rows = []
    for key in keys:
        r42 = rows42[key]
        r43 = rows43[key]
        pred42 = np.asarray(r42["pred_mean"], dtype=float)
        pred43 = np.asarray(r43["pred_mean"], dtype=float)
        pert = np.asarray(r42["pert_mean"], dtype=float)
        gt = np.asarray(r42["gt_mean"], dtype=float)
        pp42 = pearson_pert(pred42, gt, pert)
        pp43 = pearson_pert(pred43, gt, pert)
        ppens = pearson_pert(0.5 * (pred42 + pred43), gt, pert)
        base = max(pp42, pp43)
        per_rows.append(
            {
                "dataset": key[1],
                "condition": key[2],
                "pp_seed42": pp42,
                "pp_seed43": pp43,
                "pp_ensemble": ppens,
                "delta_vs_seed42": ppens - pp42,
                "delta_vs_best_seed": ppens - base,
            }
        )
    by_ds = defaultdict(list)
    for row in per_rows:
        by_ds[row["dataset"]].append(row)
    ds_delta42 = [float(np.mean([r["delta_vs_seed42"] for r in part])) for part in by_ds.values()]
    ds_delta_best = [float(np.mean([r["delta_vs_best_seed"] for r in part])) for part in by_ds.values()]
    return {
        "n_conditions": len(per_rows),
        "n_datasets": len(by_ds),
        "mean_pp_seed42": float(np.mean([r["pp_seed42"] for r in per_rows])) if per_rows else 0.0,
        "mean_pp_seed43": float(np.mean([r["pp_seed43"] for r in per_rows])) if per_rows else 0.0,
        "mean_pp_ensemble": float(np.mean([r["pp_ensemble"] for r in per_rows])) if per_rows else 0.0,
        "mean_delta_vs_seed42": float(np.mean(ds_delta42)) if ds_delta42 else 0.0,
        "mean_delta_vs_best_seed": float(np.mean(ds_delta_best)) if ds_delta_best else 0.0,
        "dataset_min_delta_vs_seed42": float(min(ds_delta42)) if ds_delta42 else 0.0,
        "dataset_min_delta_vs_best_seed": float(min(ds_delta_best)) if ds_delta_best else 0.0,
        "bootstrap_delta_vs_seed42": bootstrap(ds_delta42, seed=20260627),
        "bootstrap_delta_vs_best_seed": bootstrap(ds_delta_best, seed=20260628),
        "per_dataset": {
            ds: {
                "n": len(part),
                "delta_vs_seed42": float(np.mean([r["delta_vs_seed42"] for r in part])),
                "delta_vs_best_seed": float(np.mean([r["delta_vs_best_seed"] for r in part])),
            }
            for ds, part in sorted(by_ds.items())
        },
    }


def main() -> None:
    if not SEED42.is_file() or not SEED43.is_file():
        missing = [str(p) for p in (SEED42, SEED43) if not p.is_file()]
        raise SystemExit("missing required condition-mean artifacts: " + ", ".join(missing))
    rows42 = load_rows(SEED42)
    rows43 = load_rows(SEED43)
    groups = sorted({k[0] for k in rows42} & {k[0] for k in rows43})
    summaries = {group: summarize_group(rows42, rows43, group) for group in groups}

    reasons: list[str] = []
    cross = summaries.get("internal_val_cross_background_seen_gene_proxy", {})
    family = summaries.get("internal_val_family_gene_proxy", {})
    for label, obj in (("cross", cross), ("family", family)):
        if not obj:
            reasons.append(f"{label}_group_missing")
            continue
        if obj["mean_delta_vs_seed42"] < 0.005:
            reasons.append(f"{label}_delta_vs_seed42_lt_0p005")
        if obj["bootstrap_delta_vs_seed42"]["ci_low"] <= 0:
            reasons.append(f"{label}_delta_vs_seed42_ci_low_not_above_0")
        if obj["dataset_min_delta_vs_seed42"] < -0.02:
            reasons.append(f"{label}_dataset_min_delta_vs_seed42_below_minus_0p02")
        if obj["mean_delta_vs_best_seed"] < -0.001:
            reasons.append(f"{label}_ensemble_worse_than_best_seed_by_more_than_0p001")
    reasons.append("real_internal_mmd_and_canonical_noharm_not_run_no_gpu")

    status = "xverse_8k_seed_ensemble_internal_gate_fail_no_gpu"
    if not any(r != "real_internal_mmd_and_canonical_noharm_not_run_no_gpu" for r in reasons):
        status = "xverse_8k_seed_ensemble_internal_gate_pass_needs_real_eval_no_gpu"

    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_summary_only": True,
            "inputs_trainonly_internal_split": True,
            "pearson_pert_metric": "corr(pred_mean - pert_mean, gt_mean - pert_mean)",
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
        },
        "inputs": {"seed42": str(SEED42), "seed43": str(SEED43)},
        "groups": summaries,
        "decision_reasons": reasons,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# xverse 8k Seed Ensemble Internal Gate",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU-only summary of frozen seed42/seed43 condition means from train-only internal validation groups. `pearson_pert` uses the evaluator definition `corr(pred_mean - pert_mean, gt_mean - pert_mean)`. This does not use canonical multi for selection or Track C query.",
        "",
        "## Metrics",
        "",
        "| group | n | datasets | seed42 pp | seed43 pp | ensemble pp | delta vs seed42 | delta vs best seed | dataset min vs seed42 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, obj in summaries.items():
        lines.append(
            f"| `{group}` | {obj['n_conditions']} | {obj['n_datasets']} | "
            f"{obj['mean_pp_seed42']:+.6f} | {obj['mean_pp_seed43']:+.6f} | {obj['mean_pp_ensemble']:+.6f} | "
            f"{obj['mean_delta_vs_seed42']:+.6f} | {obj['mean_delta_vs_best_seed']:+.6f} | "
            f"{obj['dataset_min_delta_vs_seed42']:+.6f} |"
        )
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend(f"- `{reason}`" for reason in reasons)
    lines.extend(["", "## Outputs", "", f"- JSON: `{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
