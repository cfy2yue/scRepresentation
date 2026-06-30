#!/usr/bin/env python3
"""CPU-only target-observability residual localization v3 gate.

This is a diagnostic gate for the queued target-observability follow-up. It
uses completed train-only/internal rows only. It does not train, infer, read
checkpoints, read canonical multi, read Track C query outputs, or use GPU.
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
ACTION_ROWS = REPORTS / "latentfm_background_target_actionability_rows_20260625.csv"
S0_TSV = REPORTS / "latentfm_scaling_s0_provenance_freeze_20260625.tsv"
OUT_JSON = REPORTS / "latentfm_target_observability_residual_v3_gate_20260626.json"
OUT_MD = REPORTS / "LATENTFM_TARGET_OBSERVABILITY_RESIDUAL_V3_GATE_20260626.md"
OUT_CSV = REPORTS / "latentfm_target_observability_residual_v3_rows_20260626.csv"
SEED = 20260626


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return text


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def load_s0() -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    with S0_TSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, dialect="excel-tab"):
            ds = norm(row.get("dataset"))
            cond = norm(row.get("condition"))
            if ds and cond:
                out[(ds, cond)] = row
    return out


def load_rows() -> list[dict[str, Any]]:
    s0 = load_s0()
    rows: list[dict[str, Any]] = []
    with ACTION_ROWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ds = norm(row.get("dataset"))
            cond = norm(row.get("condition"))
            if not ds or not cond:
                continue
            meta = s0.get((ds, cond), {})
            rows.append(
                {
                    "dataset": ds,
                    "condition": cond,
                    "pp_delta": fnum(row.get("pp_delta")),
                    "mmd_delta": fnum(row.get("mmd_delta")),
                    "actionability_score": fnum(row.get("actionability_score")),
                    "target_expr_mean": fnum(row.get("target_expr_mean")),
                    "target_expr_nonzero_fraction": fnum(row.get("target_expr_nonzero_fraction")),
                    "prior_go": fnum(row.get("prior_go")),
                    "prior_reactome": fnum(row.get("prior_reactome")),
                    "prior_corum": fnum(row.get("prior_corum")),
                    "prior_omnipath": fnum(row.get("prior_omnipath")),
                    "n_cells": fnum(meta.get("n_cells"), default=float("nan")),
                    "n_conditions_trainonly": fnum(meta.get("n_conditions_trainonly"), default=float("nan")),
                    "cell_background_source": norm(meta.get("cell_background_source")) or "unknown",
                    "perturbation_type": norm(meta.get("perturbation_type")) or "unknown",
                    "source_label": norm(meta.get("source_label")) or "unknown",
                }
            )
    return rows


def add_residuals(rows: list[dict[str, Any]]) -> None:
    datasets = sorted({r["dataset"] for r in rows})
    ds_index = {ds: i for i, ds in enumerate(datasets)}
    # Intercept, all but one dataset fixed effects, log cell count, and
    # train-condition count. Missing numeric covariates are median-imputed.
    cell_vals = [r["n_cells"] for r in rows if math.isfinite(float(r["n_cells"]))]
    cond_vals = [r["n_conditions_trainonly"] for r in rows if math.isfinite(float(r["n_conditions_trainonly"]))]
    cell_med = float(np.median(cell_vals)) if cell_vals else 1.0
    cond_med = float(np.median(cond_vals)) if cond_vals else 1.0
    x_rows = []
    y = []
    for row in rows:
        x = [1.0]
        # Drop the last dataset to avoid singular design.
        for ds in datasets[:-1]:
            x.append(1.0 if row["dataset"] == ds else 0.0)
        n_cells = row["n_cells"] if math.isfinite(float(row["n_cells"])) else cell_med
        n_cond = row["n_conditions_trainonly"] if math.isfinite(float(row["n_conditions_trainonly"])) else cond_med
        x.extend([math.log1p(max(0.0, float(n_cells))), math.log1p(max(0.0, float(n_cond)))])
        x_rows.append(x)
        y.append(float(row["pp_delta"]))
    x_mat = np.asarray(x_rows, dtype=float)
    y_vec = np.asarray(y, dtype=float)
    coef, *_ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
    pred = x_mat @ coef
    for row, resid, fitted in zip(rows, y_vec - pred, pred):
        row["pp_residual_dataset_count"] = float(resid)
        row["pp_fitted_dataset_count"] = float(fitted)


def within_dataset_top(rows: list[dict[str, Any]], score_key: str, q: float = 0.75) -> list[dict[str, Any]]:
    out = []
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ds[row["dataset"]].append(row)
    for ds_rows in by_ds.values():
        if len(ds_rows) < 4:
            continue
        vals = sorted(float(r[score_key]) for r in ds_rows)
        threshold = vals[int(q * (len(vals) - 1))]
        out.extend(r for r in ds_rows if float(r[score_key]) >= threshold)
    return out


def bootstrap_dataset_mean(rows: list[dict[str, Any]], value_key: str, n_boot: int = 2000) -> tuple[float, float]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_ds[row["dataset"]].append(float(row[value_key]))
    ds_means = [mean(vals) for vals in by_ds.values()]
    if not ds_means:
        return float("nan"), float("nan")
    rng = random.Random(SEED)
    boots = sorted(mean(rng.choice(ds_means) for _ in ds_means) for _ in range(n_boot))
    return float(boots[int(0.025 * (n_boot - 1))]), float(boots[int(0.975 * (n_boot - 1))])


def summarize(policy: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "policy": policy,
            "status": "fail_no_rows",
            "n": 0,
            "datasets": 0,
            "reasons": ["no_rows"],
            "gpu_authorized": False,
        }
    residuals = [float(r["pp_residual_dataset_count"]) for r in rows]
    raw_pp = [float(r["pp_delta"]) for r in rows]
    mmd = [float(r["mmd_delta"]) for r in rows]
    by_ds_resid: dict[str, list[float]] = defaultdict(list)
    by_ds_raw: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_ds_resid[row["dataset"]].append(float(row["pp_residual_dataset_count"]))
        by_ds_raw[row["dataset"]].append(float(row["pp_delta"]))
    ci_low, ci_high = bootstrap_dataset_mean(rows, "pp_residual_dataset_count")
    dataset_min_resid = min(mean(vals) for vals in by_ds_resid.values())
    dataset_min_raw = min(mean(vals) for vals in by_ds_raw.values())
    hard_harm_frac = sum(v < -0.020 for v in raw_pp) / len(raw_pp)
    reasons: list[str] = []
    if len(rows) < 25:
        reasons.append("too_few_conditions")
    if len(by_ds_resid) < 5:
        reasons.append("too_few_datasets")
    if mean(residuals) <= 0.010:
        reasons.append("residual_pp_mean_below_0p010")
    if ci_low <= 0.0:
        reasons.append("dataset_bootstrap_residual_ci_lower_not_positive")
    if dataset_min_resid < -0.020:
        reasons.append("dataset_residual_tail_below_minus_0p020")
    if dataset_min_raw < -0.020:
        reasons.append("dataset_raw_tail_below_minus_0p020")
    if hard_harm_frac > 0.20:
        reasons.append("raw_hard_harm_frac_above_0p20")
    if max(mmd) > 0.001:
        reasons.append("mmd_max_above_0p001")
    return {
        "policy": policy,
        "status": "pass_needs_external_review_no_gpu" if not reasons else "fail_no_gpu",
        "gpu_authorized": False,
        "n": len(rows),
        "datasets": len(by_ds_resid),
        "residual_pp_mean": float(mean(residuals)),
        "raw_pp_mean": float(mean(raw_pp)),
        "mmd_mean": float(mean(mmd)),
        "mmd_max": float(max(mmd)),
        "dataset_min_residual_pp": float(dataset_min_resid),
        "dataset_min_raw_pp": float(dataset_min_raw),
        "hard_harm_frac": float(hard_harm_frac),
        "dataset_bootstrap_residual_ci95": [ci_low, ci_high],
        "reasons": reasons,
    }


def within_dataset_shuffle(rows: list[dict[str, Any]], score_key: str, selected_count: int) -> dict[str, Any]:
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_by_ds: dict[str, int] = defaultdict(int)
    for row in rows:
        by_ds[row["dataset"]].append(row)
    for row in within_dataset_top(rows, score_key):
        selected_by_ds[row["dataset"]] += 1
    observed = within_dataset_top(rows, score_key)
    observed_mean = mean(float(r["pp_residual_dataset_count"]) for r in observed) if observed else float("nan")
    rng = random.Random(SEED + len(score_key))
    controls = []
    for _ in range(2000):
        picked = []
        for ds, ds_rows in by_ds.items():
            k = selected_by_ds.get(ds, 0)
            if k:
                picked.extend(rng.sample(ds_rows, k=min(k, len(ds_rows))))
        controls.append(mean(float(r["pp_residual_dataset_count"]) for r in picked) if picked else 0.0)
    controls.sort()
    p_ge = (1 + sum(v >= observed_mean for v in controls)) / (1 + len(controls))
    return {
        "score_key": score_key,
        "selected_count": selected_count,
        "observed_residual_mean": float(observed_mean),
        "shuffle_mean": float(mean(controls)),
        "shuffle_p95": float(controls[int(0.95 * (len(controls) - 1))]),
        "p_ge_observed": float(p_ge),
    }


def write_rows(rows: list[dict[str, Any]], policy_labels: dict[tuple[str, str], list[str]]) -> None:
    fields = [
        "dataset",
        "condition",
        "pp_delta",
        "mmd_delta",
        "pp_residual_dataset_count",
        "pp_fitted_dataset_count",
        "actionability_score",
        "target_expr_mean",
        "target_expr_nonzero_fraction",
        "n_cells",
        "n_conditions_trainonly",
        "policies",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            key = (row["dataset"], row["condition"])
            writer.writerow({field: row.get(field, "") for field in fields[:-1]} | {"policies": ";".join(policy_labels.get(key, []))})


def main() -> int:
    rows = load_rows()
    add_residuals(rows)
    policy_sets = {
        "within_dataset_top_expr_mean": within_dataset_top(rows, "target_expr_mean"),
        "within_dataset_top_nonzero_fraction": within_dataset_top(rows, "target_expr_nonzero_fraction"),
        "within_dataset_top_actionability": within_dataset_top(rows, "actionability_score"),
        "nonzero_target_expr": [r for r in rows if float(r["target_expr_nonzero_fraction"]) > 0.0],
    }
    policy_labels: dict[tuple[str, str], list[str]] = defaultdict(list)
    for name, subset in policy_sets.items():
        for row in subset:
            policy_labels[(row["dataset"], row["condition"])].append(name)
    summaries = [summarize(name, subset) for name, subset in policy_sets.items()]
    controls = [
        within_dataset_shuffle(rows, "target_expr_mean", len(policy_sets["within_dataset_top_expr_mean"])),
        within_dataset_shuffle(rows, "target_expr_nonzero_fraction", len(policy_sets["within_dataset_top_nonzero_fraction"])),
        within_dataset_shuffle(rows, "actionability_score", len(policy_sets["within_dataset_top_actionability"])),
    ]
    best = max(summaries, key=lambda x: (x.get("status", "").startswith("pass"), x.get("residual_pp_mean", -999)))
    control_reasons = []
    for ctrl in controls:
        if ctrl["p_ge_observed"] > 0.01:
            control_reasons.append(f"{ctrl['score_key']}_within_dataset_shuffle_p_gt_0p01")
    pass_candidates = [s for s in summaries if s["status"].startswith("pass")]
    status = "target_observability_residual_v3_pass_review_only_no_gpu" if pass_candidates and not control_reasons else "target_observability_residual_v3_fail_no_gpu"
    decision_reasons = [] if status.startswith("target_observability_residual_v3_pass") else [
        "no_residual_target_policy_passed_tail_mmd_and_shuffle_gate",
        *control_reasons,
    ]
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M CST"),
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "reads_completed_train_only_internal_rows": True,
            "reads_s0_metadata": True,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "trains_or_infers": False,
            "uses_gpu": False,
        },
        "row_count": len(rows),
        "policy_summaries": summaries,
        "within_dataset_shuffle_controls": controls,
        "best_policy": best,
        "decision_reasons": decision_reasons,
        "next_action": "no GPU; target observability remains hint-only/failure-localization evidence" if decision_reasons else "external review before any bounded target-observability route",
        "outputs": {"rows": str(OUT_CSV)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_rows(rows, policy_labels)

    lines = [
        "# LatentFM Target Observability Residual V3 Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only residual localization over completed train-only/internal target-actionability rows.",
        "- Residualizes pp delta with dataset fixed effects plus log cell count and train-condition count.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "",
        "## Policy Summary",
        "",
        "| policy | status | rows | datasets | residual pp | raw pp | dataset min residual | dataset min raw | MMD max | hard harm | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| `{policy}` | `{status}` | {n} | {datasets} | {resid:+.6f} | {raw:+.6f} | {min_resid:+.6f} | {min_raw:+.6f} | {mmd:+.6f} | {harm:.3f} | `{reasons}` |".format(
                policy=row["policy"],
                status=row["status"],
                n=row.get("n", 0),
                datasets=row.get("datasets", 0),
                resid=float(row.get("residual_pp_mean", 0.0)),
                raw=float(row.get("raw_pp_mean", 0.0)),
                min_resid=float(row.get("dataset_min_residual_pp", 0.0)),
                min_raw=float(row.get("dataset_min_raw_pp", 0.0)),
                mmd=float(row.get("mmd_max", 0.0)),
                harm=float(row.get("hard_harm_frac", 0.0)),
                reasons=row.get("reasons", []),
            )
        )
    lines += [
        "",
        "## Within-Dataset Shuffle Controls",
        "",
        "| score | selected | observed residual | shuffle mean | shuffle p95 | p_ge_observed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for ctrl in controls:
        lines.append(
            f"| `{ctrl['score_key']}` | {ctrl['selected_count']} | {ctrl['observed_residual_mean']:+.6f} | {ctrl['shuffle_mean']:+.6f} | {ctrl['shuffle_p95']:+.6f} | {ctrl['p_ge_observed']:.6f} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- pass candidates: `{[s['policy'] for s in pass_candidates]}`",
        f"- reasons: `{decision_reasons}`",
        f"- next action: `{payload['next_action']}`",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- rows: `{OUT_CSV}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": status, "gpu_authorized": False, "out_md": str(OUT_MD)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
