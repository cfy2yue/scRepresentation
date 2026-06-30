#!/usr/bin/env python3
"""Deduped/residualized strict-v2 gate for Replogle bulk artifacts.

This follows the external audit recommendations after the first Replogle gate:
aggregate duplicate obs-index rows to one value per dataset/condition/raw_column,
dedupe eval rows, test only pooled K562_gwps+RPE1 raw-column artifacts, and
residualize against pooled QC controls before any promotion discussion.

CPU/report-only. No training, inference, checkpoint selection, canonical multi
selection, Track C query, or GPU.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
ARTIFACT_CSV = ROOT / "reports/replogle_bulk_artifacts_20260627/replogle_bulk_condition_artifacts.csv"
OUT_DIR = ROOT / "reports/replogle_bulk_artifact_strict_v2_20260627"
OUT_COND = OUT_DIR / "replogle_bulk_deduped_condition_artifacts.csv"
OUT_SUMMARY = OUT_DIR / "replogle_bulk_strict_v2_summary.csv"
OUT_JSON = ROOT / "reports/latentfm_replogle_bulk_artifact_strict_v2_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_REPLOGLE_BULK_ARTIFACT_STRICT_V2_20260627.md"

ANCHOR_ROOT = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    / "xverse_comp006_endpoint5_8k_seed42_fulleval"
)
REPLICATE_ROOT = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_seed_replicate_20260621/"
    / "xverse_comp006_endpoint5_8k_seed43_fulleval"
)
INPUTS = {
    "seed42_split": ANCHOR_ROOT
    / "posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
    "seed43_split": REPLICATE_ROOT
    / "posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
}

DATASETS = {"ReplogleWeissman2022_K562_gwps", "Replogle_RPE1essential"}
SOURCE_BY_DATASET = {"ReplogleWeissman2022_K562_gwps": "K562_gwps", "Replogle_RPE1essential": "RPE1"}
PRE_REGISTERED_CANDIDATES = ["std_leverage_score", "cnv_score_z", "TE_ratio"]
QC_CONTROLS = ["UMI_count_unfiltered", "num_cells_filtered", "mitopercent", "z_gemgroup_UMI"]


def fnum(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:12], 16) % (2**32)


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    mx, my = mean(x), mean(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0 or vy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(rankdata(x), rankdata(y))


def residualize(y: list[float], xs: list[list[float]]) -> list[float]:
    """Residualize y against intercept + covariates via small ridge OLS."""
    n = len(y)
    if n == 0:
        return []
    cols = [[1.0] * n] + xs
    p = len(cols)
    xtx = [[0.0 for _ in range(p)] for _ in range(p)]
    xty = [0.0 for _ in range(p)]
    for i in range(n):
        row = [col[i] for col in cols]
        for a in range(p):
            xty[a] += row[a] * y[i]
            for b in range(p):
                xtx[a][b] += row[a] * row[b]
    for i in range(p):
        xtx[i][i] += 1e-8
    # Gaussian elimination.
    aug = [xtx[i] + [xty[i]] for i in range(p)]
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            continue
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        aug[col] = [v / div for v in aug[col]]
        for r in range(p):
            if r == col:
                continue
            factor = aug[r][col]
            aug[r] = [a - factor * b for a, b in zip(aug[r], aug[col])]
    beta = [aug[i][-1] for i in range(p)]
    residuals = []
    for i in range(n):
        pred = sum(beta[j] * cols[j][i] for j in range(p))
        residuals.append(y[i] - pred)
    return residuals


def load_deduped_artifacts() -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, Any]]:
    vals: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    with ARTIFACT_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dataset = row["dataset"]
            if dataset not in DATASETS or row["source_label"] != SOURCE_BY_DATASET[dataset]:
                continue
            value = fnum(row.get("artifact_value"))
            if value is None:
                continue
            vals[(dataset, row["condition"], row["raw_column"])].append(value)
    by_condition: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    dup_counts = []
    for (dataset, condition, raw), items in vals.items():
        by_condition[(dataset, condition)][raw] = median(items)
        dup_counts.append(len(items))
    meta = {
        "deduped_conditions": len(by_condition),
        "max_duplicate_per_dataset_condition_raw": max(dup_counts) if dup_counts else 0,
        "duplicate_keys_gt1": sum(1 for x in dup_counts if x > 1),
    }
    return by_condition, meta


def load_eval_rows() -> list[dict[str, Any]]:
    out: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for key, path in INPUTS.items():
        seed, source = key.split("_", 1)
        payload = json.loads(path.read_text(encoding="utf-8"))
        group = "test_single"
        for row in payload.get("groups", {}).get(group, {}).get("condition_metrics") or []:
            dataset = str(row.get("dataset", ""))
            condition = str(row.get("condition", ""))
            if dataset not in DATASETS:
                continue
            pp = fnum(row.get("pearson_pert"))
            mmd = fnum(row.get("test_mmd_clamped"))
            if pp is None or mmd is None:
                continue
            out[(seed, group, dataset, condition)] = {
                "seed": seed,
                "group": group,
                "dataset": dataset,
                "condition": condition,
                "pearson_pert": pp,
                "test_mmd_clamped": mmd,
            }
    return list(out.values())


def build_rows(eval_rows: list[dict[str, Any]], artifacts: dict[tuple[str, str], dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    needed = set(PRE_REGISTERED_CANDIDATES + QC_CONTROLS)
    for erow in eval_rows:
        vals = artifacts.get((erow["dataset"], erow["condition"]), {})
        if not needed.issubset(vals):
            continue
        rows.append({**erow, **{k: vals[k] for k in needed}})
    return rows


def perm_p(rows: list[dict[str, Any]], raw: str, direction: float, observed: float, seed_label: str, n_perm: int = 10000) -> float | None:
    if len(rows) < 20:
        return None
    rng = random.Random(stable_seed(seed_label))
    pp = [float(r["pearson_pert"]) for r in rows]
    idx_by_ds: dict[str, list[int]] = defaultdict(list)
    vals_by_ds: dict[str, list[float]] = defaultdict(list)
    for i, row in enumerate(rows):
        idx_by_ds[row["dataset"]].append(i)
        vals_by_ds[row["dataset"]].append(float(row[raw]))
    ge = 0
    for _ in range(n_perm):
        shuffled = [0.0] * len(rows)
        for ds, vals in vals_by_ds.items():
            items = list(vals)
            rng.shuffle(items)
            for idx, value in zip(idx_by_ds[ds], items):
                shuffled[idx] = value
        rho = spearman(shuffled, pp)
        if rho is not None and direction * rho >= observed - 1e-12:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def summarize_raw(rows: list[dict[str, Any]], raw: str) -> dict[str, Any]:
    # Direction is fixed from source semantics when possible; otherwise use
    # positive direction but mark test-selected risk in the global decision.
    base = [r for r in rows if r["seed"] == "seed42"]
    rho0 = spearman([float(r[raw]) for r in base], [float(r["pearson_pert"]) for r in base]) or 0.0
    direction = 1.0 if rho0 >= 0 else -1.0
    per_seed = {}
    signed = []
    mmd_abs = []
    residual_signed = []
    residual_ds_signed = []
    ds_signed_all = []
    pvals = []
    for seed in ("seed42", "seed43"):
        sub = [r for r in rows if r["seed"] == seed]
        vals = [float(r[raw]) for r in sub]
        pp = [float(r["pearson_pert"]) for r in sub]
        mmd = [float(r["test_mmd_clamped"]) for r in sub]
        rho = spearman(vals, pp)
        srho = None if rho is None else direction * rho
        pval = perm_p(sub, raw, direction, srho or -999.0, f"{raw}:{seed}") if srho is not None else None
        ds_signed = []
        for ds in sorted(DATASETS):
            dsub = [r for r in sub if r["dataset"] == ds]
            drho = spearman([float(r[raw]) for r in dsub], [float(r["pearson_pert"]) for r in dsub])
            if drho is not None:
                ds_signed.append(direction * drho)
        covariates = []
        for qc in QC_CONTROLS:
            if qc != raw:
                covariates.append(rankdata([float(r[qc]) for r in sub]))
        # Dataset binary covariate.
        covariates.append([1.0 if r["dataset"] == "Replogle_RPE1essential" else 0.0 for r in sub])
        art_resid = residualize(rankdata(vals), covariates)
        rho_resid = spearman(art_resid, pp)
        srho_resid = None if rho_resid is None else direction * rho_resid
        ds_resid = []
        for ds in sorted(DATASETS):
            dsub = [r for r in sub if r["dataset"] == ds]
            if len(dsub) < 3:
                continue
            dvals = [float(r[raw]) for r in dsub]
            dpp = [float(r["pearson_pert"]) for r in dsub]
            dcovs = [rankdata([float(r[qc]) for r in dsub]) for qc in QC_CONTROLS if qc != raw]
            dres = residualize(rankdata(dvals), dcovs)
            drho = spearman(dres, dpp)
            if drho is not None:
                ds_resid.append(direction * drho)
        rho_mmd = spearman(vals, mmd)
        per_seed[seed] = {
            "n": len(sub),
            "datasets": len({r["dataset"] for r in sub}),
            "signed_rho": srho,
            "perm_p": pval,
            "dataset_min_signed_rho": min(ds_signed) if ds_signed else None,
            "rho_mmd": rho_mmd,
            "residual_signed_rho": srho_resid,
            "residual_dataset_min_signed_rho": min(ds_resid) if ds_resid else None,
        }
        if srho is not None:
            signed.append(srho)
        if pval is not None:
            pvals.append(pval)
        ds_signed_all.extend(ds_signed)
        if rho_mmd is not None:
            mmd_abs.append(abs(rho_mmd))
        if srho_resid is not None:
            residual_signed.append(srho_resid)
        residual_ds_signed.extend(ds_resid)
    return {
        "raw_column": raw,
        "direction": direction,
        "min_signed_rho": min(signed) if signed else None,
        "max_perm_p": max(pvals) if pvals else None,
        "min_dataset_signed_rho": min(ds_signed_all) if ds_signed_all else None,
        "max_abs_rho_mmd": max(mmd_abs) if mmd_abs else None,
        "min_residual_signed_rho": min(residual_signed) if residual_signed else None,
        "min_residual_dataset_signed_rho": min(residual_ds_signed) if residual_ds_signed else None,
        "seed42": per_seed.get("seed42", {}),
        "seed43": per_seed.get("seed43", {}),
    }


def decide(summary: dict[str, Any], best_qc: float | None) -> tuple[str, list[str]]:
    reasons = ["test_metric_selected_artifact_diagnostic_only"]
    if (summary.get("min_signed_rho") or -999.0) < 0.35:
        reasons.append("signed_rho_below_0p35")
    if (summary.get("min_dataset_signed_rho") or -999.0) < 0.25:
        reasons.append("dataset_signed_rho_below_0p25")
    if summary.get("max_perm_p") is None or summary["max_perm_p"] > 0.01:
        reasons.append("perm_p_above_0p01")
    signed = summary.get("min_signed_rho") or 0.0
    if summary.get("max_abs_rho_mmd") is None or summary["max_abs_rho_mmd"] > min(0.20, 0.5 * signed):
        reasons.append("mmd_qc_noharm_fail")
    if (summary.get("min_residual_signed_rho") or -999.0) < 0.25:
        reasons.append("residual_signed_rho_below_0p25")
    if (summary.get("min_residual_dataset_signed_rho") or -999.0) < 0.15:
        reasons.append("residual_dataset_signed_rho_below_0p15")
    if best_qc is not None and signed < best_qc + 0.15:
        reasons.append("qc_margin_below_0p15")
    return "diagnostic_only_no_gpu" if reasons else "strict_v2_pass_needs_review_no_gpu", reasons


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):+.6f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts, dedup_meta = load_deduped_artifacts()
    eval_rows = load_eval_rows()
    rows = build_rows(eval_rows, artifacts)
    with OUT_COND.open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "group", "dataset", "condition", "pearson_pert", "test_mmd_clamped"] + PRE_REGISTERED_CANDIDATES + QC_CONTROLS
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in rows])
    raw_summaries = [summarize_raw(rows, raw) for raw in PRE_REGISTERED_CANDIDATES + QC_CONTROLS]
    qc_best = max((s["min_signed_rho"] for s in raw_summaries if s["raw_column"] in QC_CONTROLS and s["min_signed_rho"] is not None), default=None)
    for summary in raw_summaries:
        status, reasons = decide(summary, qc_best if summary["raw_column"] in PRE_REGISTERED_CANDIDATES else None)
        if summary["raw_column"] in QC_CONTROLS:
            status = "qc_control_diagnostic"
            reasons = ["qc_control_not_promotable"]
        summary["status"] = status
        summary["reasons"] = ";".join(reasons)
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "raw_column",
            "status",
            "reasons",
            "min_signed_rho",
            "max_perm_p",
            "min_dataset_signed_rho",
            "max_abs_rho_mmd",
            "min_residual_signed_rho",
            "min_residual_dataset_signed_rho",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in raw_summaries:
            writer.writerow({k: row.get(k, "") for k in fields})
    pass_candidates = [s["raw_column"] for s in raw_summaries if s["status"] == "strict_v2_pass_needs_review_no_gpu"]
    status = "replogle_bulk_artifact_strict_v2_fail_no_gpu" if not pass_candidates else "replogle_bulk_artifact_strict_v2_pass_needs_review_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "dedup_meta": dedup_meta,
        "eval_unique_rows": len(eval_rows),
        "joined_unique_rows": len(rows),
        "best_qc_min_signed_rho": qc_best,
        "pass_candidates": pass_candidates,
        "summaries": raw_summaries,
        "outputs": {"condition_csv": str(OUT_COND), "summary_csv": str(OUT_SUMMARY), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Replogle Bulk Artifact Strict V2 2026-06-27",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only follow-up to external audit.",
        "- Dedupes source artifacts to one median value per dataset/condition/raw column.",
        "- Uses unique seed42/43 split `test_single` rows only; no canonical multi, Track C query, training, or inference.",
        "- Because candidates were discovered on frozen test metrics, any result remains diagnostic unless a new train-only/discovery-confirm route is built.",
        "",
        "## Deduplication",
        "",
        f"- deduped conditions: `{dedup_meta['deduped_conditions']}`",
        f"- duplicate raw keys >1: `{dedup_meta['duplicate_keys_gt1']}`",
        f"- max duplicate per dataset/condition/raw: `{dedup_meta['max_duplicate_per_dataset_condition_raw']}`",
        f"- joined unique rows: `{len(rows)}`",
        f"- best QC min signed rho: `{fmt(qc_best)}`",
        "",
        "## Summary",
        "",
        "| raw column | status | min signed rho | perm p | dataset min | max abs rho MMD | residual rho | residual dataset min | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in raw_summaries:
        lines.append(
            f"| `{row['raw_column']}` | `{row['status']}` | {fmt(row.get('min_signed_rho'))} | "
            f"{fmt(row.get('max_perm_p'))} | {fmt(row.get('min_dataset_signed_rho'))} | "
            f"{fmt(row.get('max_abs_rho_mmd'))} | {fmt(row.get('min_residual_signed_rho'))} | "
            f"{fmt(row.get('min_residual_dataset_signed_rho'))} | `{row['reasons']}` |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "Fail/no-GPU unless a future route uses train-only or discovery/confirm selection and passes residualized MMD/QC no-harm. Current Replogle artifacts are retained as diagnostic/failure-mechanism evidence only.",
        "",
        f"- condition rows: `{OUT_COND}`",
        f"- summary: `{OUT_SUMMARY}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "pass_candidates": pass_candidates, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
