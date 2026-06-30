#!/usr/bin/env python3
"""Train-only internal-val feasibility gate for Replogle bulk difficulty signals.

This is a CPU/report-only diagnostic after the strict-v2 Replogle held-out gate
failed. It asks whether pre-registered Replogle author bulk signals still align
with xverse anchor error on the train-only internal validation split, without
using canonical held-out test rows for candidate promotion.

No training, inference, checkpoint selection, canonical multi selection, Track C
query, or GPU.
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
INTERNAL_JSON = ROOT / "reports/latentfm_xverse_tracka_anchor_internal_val_error_map_20260622.json"
OUT_DIR = ROOT / "reports/replogle_trainonly_internal_difficulty_gate_20260627"
OUT_JOINED = OUT_DIR / "replogle_trainonly_internal_difficulty_joined_rows.csv"
OUT_SUMMARY = OUT_DIR / "replogle_trainonly_internal_difficulty_summary.csv"
OUT_JSON = ROOT / "reports/latentfm_replogle_trainonly_internal_difficulty_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_REPLOGLE_TRAINONLY_INTERNAL_DIFFICULTY_GATE_20260627.md"

DATASETS = {"ReplogleWeissman2022_K562_gwps", "Replogle_RPE1essential"}
SOURCE_BY_DATASET = {"ReplogleWeissman2022_K562_gwps": "K562_gwps", "Replogle_RPE1essential": "RPE1"}
GROUP_DISCOVERY = "internal_val_cross_background_seen_gene_proxy"
GROUP_CONFIRM = "internal_val_family_gene_proxy"
PRE_REGISTERED_CANDIDATES = ["cnv_score_z", "TE_ratio", "std_leverage_score"]
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
    return [y[i] - sum(beta[j] * cols[j][i] for j in range(p)) for i in range(n)]


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
    return by_condition, {
        "deduped_conditions": len(by_condition),
        "duplicate_keys_gt1": sum(1 for x in dup_counts if x > 1),
        "max_duplicate_per_dataset_condition_raw": max(dup_counts) if dup_counts else 0,
    }


def load_internal_rows() -> list[dict[str, Any]]:
    payload = json.loads(INTERNAL_JSON.read_text(encoding="utf-8"))
    rows = []
    for row in payload.get("condition_rows") or []:
        group = str(row.get("group", ""))
        dataset = str(row.get("dataset", ""))
        if group not in {GROUP_DISCOVERY, GROUP_CONFIRM} or dataset not in DATASETS:
            continue
        pp = fnum(row.get("anchor_pearson_pert"))
        mmd = fnum(row.get("anchor_mmd_clamped"))
        if pp is None or mmd is None:
            continue
        rows.append(
            {
                "group": group,
                "dataset": dataset,
                "condition": str(row.get("condition", "")),
                "gene": str(row.get("gene", "")),
                "anchor_pearson_pert": pp,
                "anchor_mmd_clamped": mmd,
                "gene_train_count": row.get("gene_train_count"),
                "gene_count_bucket": row.get("gene_count_bucket"),
            }
        )
    return rows


def build_joined(rows: list[dict[str, Any]], artifacts: dict[tuple[str, str], dict[str, float]]) -> list[dict[str, Any]]:
    needed = set(PRE_REGISTERED_CANDIDATES + QC_CONTROLS)
    joined = []
    for row in rows:
        vals = artifacts.get((row["dataset"], row["condition"]), {})
        if needed.issubset(vals):
            joined.append({**row, **{k: vals[k] for k in needed}})
    return joined


def perm_p(rows: list[dict[str, Any]], raw: str, direction: float, observed: float, label: str, n_perm: int = 10000) -> float | None:
    if len(rows) < 20:
        return None
    rng = random.Random(stable_seed(label))
    pp = [float(r["anchor_pearson_pert"]) for r in rows]
    idx_by_ds: dict[str, list[int]] = defaultdict(list)
    vals_by_ds: dict[str, list[float]] = defaultdict(list)
    for i, row in enumerate(rows):
        idx_by_ds[row["dataset"]].append(i)
        vals_by_ds[row["dataset"]].append(float(row[raw]))
    ge = 0
    for _ in range(n_perm):
        shuffled = [0.0] * len(rows)
        for ds, values in vals_by_ds.items():
            items = list(values)
            rng.shuffle(items)
            for idx, value in zip(idx_by_ds[ds], items):
                shuffled[idx] = value
        rho = spearman(shuffled, pp)
        if rho is not None and direction * rho >= observed - 1e-12:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def metric_block(rows: list[dict[str, Any]], raw: str, direction: float, label: str) -> dict[str, Any]:
    vals = [float(r[raw]) for r in rows]
    pp = [float(r["anchor_pearson_pert"]) for r in rows]
    mmd = [float(r["anchor_mmd_clamped"]) for r in rows]
    rho = spearman(vals, pp)
    signed = None if rho is None else direction * rho
    pval = perm_p(rows, raw, direction, signed or -999.0, label) if signed is not None else None
    ds_signed = {}
    for ds in sorted(DATASETS):
        dsub = [r for r in rows if r["dataset"] == ds]
        drho = spearman([float(r[raw]) for r in dsub], [float(r["anchor_pearson_pert"]) for r in dsub])
        ds_signed[ds] = None if drho is None else direction * drho
    covariates = [rankdata([float(r[qc]) for r in rows]) for qc in QC_CONTROLS if qc != raw]
    covariates.append([1.0 if r["dataset"] == "Replogle_RPE1essential" else 0.0 for r in rows])
    resid = residualize(rankdata(vals), covariates)
    rrho = spearman(resid, pp)
    rds = {}
    for ds in sorted(DATASETS):
        idx = [i for i, row in enumerate(rows) if row["dataset"] == ds]
        if len(idx) >= 3:
            drho = spearman([resid[i] for i in idx], [pp[i] for i in idx])
            rds[ds] = None if drho is None else direction * drho
    mrho = spearman(vals, mmd)
    return {
        "n": len(rows),
        "signed_rho": signed,
        "shuffle_p": pval,
        "dataset_signed_rho": ds_signed,
        "dataset_min_signed_rho": min([v for v in ds_signed.values() if v is not None], default=None),
        "abs_rho_mmd": None if mrho is None else abs(mrho),
        "residual_signed_rho": None if rrho is None else direction * rrho,
        "residual_dataset_signed_rho": rds,
        "residual_dataset_min_signed_rho": min([v for v in rds.values() if v is not None], default=None),
    }


def summarize_candidate(rows: list[dict[str, Any]], raw: str) -> dict[str, Any]:
    disc = [r for r in rows if r["group"] == GROUP_DISCOVERY]
    conf = [r for r in rows if r["group"] == GROUP_CONFIRM]
    rho0 = spearman([float(r[raw]) for r in disc], [float(r["anchor_pearson_pert"]) for r in disc]) or 0.0
    direction = 1.0 if rho0 >= 0 else -1.0
    dblock = metric_block(disc, raw, direction, f"{raw}:discovery")
    cblock = metric_block(conf, raw, direction, f"{raw}:confirm")
    reasons = []
    if dblock["n"] < 40 or cblock["n"] < 40:
        reasons.append("too_few_joined_rows")
    if (dblock["signed_rho"] or -999.0) < 0.35 or (cblock["signed_rho"] or -999.0) < 0.35:
        reasons.append("signed_rho_below_0p35")
    if (dblock["dataset_min_signed_rho"] or -999.0) < 0.20 or (cblock["dataset_min_signed_rho"] or -999.0) < 0.20:
        reasons.append("dataset_min_below_0p20")
    if (dblock["shuffle_p"] or 1.0) > 0.05 or (cblock["shuffle_p"] or 1.0) > 0.05:
        reasons.append("shuffle_p_above_0p05")
    mmd_limit = min(0.20, 0.5 * min(dblock["signed_rho"] or 0.0, cblock["signed_rho"] or 0.0))
    if (dblock["abs_rho_mmd"] or 999.0) > mmd_limit or (cblock["abs_rho_mmd"] or 999.0) > mmd_limit:
        reasons.append("mmd_correlation_too_large")
    if (dblock["residual_signed_rho"] or -999.0) < 0.25 or (cblock["residual_signed_rho"] or -999.0) < 0.25:
        reasons.append("qc_dataset_residual_rho_below_0p25")
    if (dblock["residual_dataset_min_signed_rho"] or -999.0) < 0.10 or (cblock["residual_dataset_min_signed_rho"] or -999.0) < 0.10:
        reasons.append("residual_dataset_min_below_0p10")
    status = "trainonly_internal_feasibility_signal_no_gpu" if not reasons else "trainonly_internal_feasibility_fail_no_gpu"
    return {
        "raw_column": raw,
        "direction": direction,
        "status": status,
        "reasons": ";".join(reasons) if reasons else "internal_val_signal_only_not_gpu_authorizing",
        "discovery": dblock,
        "confirm": cblock,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def flat_summary(row: dict[str, Any]) -> dict[str, Any]:
    d, c = row["discovery"], row["confirm"]
    return {
        "raw_column": row["raw_column"],
        "status": row["status"],
        "direction": row["direction"],
        "discovery_n": d["n"],
        "confirm_n": c["n"],
        "discovery_signed_rho": d["signed_rho"],
        "confirm_signed_rho": c["signed_rho"],
        "discovery_dataset_min": d["dataset_min_signed_rho"],
        "confirm_dataset_min": c["dataset_min_signed_rho"],
        "discovery_shuffle_p": d["shuffle_p"],
        "confirm_shuffle_p": c["shuffle_p"],
        "discovery_abs_rho_mmd": d["abs_rho_mmd"],
        "confirm_abs_rho_mmd": c["abs_rho_mmd"],
        "discovery_residual_rho": d["residual_signed_rho"],
        "confirm_residual_rho": c["residual_signed_rho"],
        "discovery_residual_dataset_min": d["residual_dataset_min_signed_rho"],
        "confirm_residual_dataset_min": c["residual_dataset_min_signed_rho"],
        "reasons": row["reasons"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts, artifact_meta = load_deduped_artifacts()
    internal_rows = load_internal_rows()
    joined = build_joined(internal_rows, artifacts)
    summaries = [summarize_candidate(joined, raw) for raw in PRE_REGISTERED_CANDIDATES]
    summaries.sort(key=lambda r: (0 if r["status"].endswith("signal_no_gpu") else 1, -(r["confirm"]["signed_rho"] or -999.0)))
    signals = [r["raw_column"] for r in summaries if r["status"] == "trainonly_internal_feasibility_signal_no_gpu"]
    status = "replogle_trainonly_internal_feasibility_signal_no_gpu" if signals else "replogle_trainonly_internal_feasibility_fail_no_gpu"

    with OUT_JOINED.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "group",
            "dataset",
            "condition",
            "gene",
            "anchor_pearson_pert",
            "anchor_mmd_clamped",
            "gene_train_count",
            "gene_count_bucket",
        ] + PRE_REGISTERED_CANDIDATES + QC_CONTROLS
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in joined])

    flat = [flat_summary(row) for row in summaries]
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)

    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "mode": "trainonly_internal_val_report_only",
            "discovery_group": GROUP_DISCOVERY,
            "confirm_group": GROUP_CONFIRM,
            "canonical_multi_used": False,
            "trackc_query_used": False,
            "training_or_inference_run": False,
        },
        "artifact_meta": artifact_meta,
        "internal_rows": len(internal_rows),
        "joined_rows": len(joined),
        "signals": signals,
        "summaries": summaries,
        "outputs": {"joined_csv": str(OUT_JOINED), "summary_csv": str(OUT_SUMMARY), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Replogle Train-Only Internal Difficulty Gate 2026-06-27",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only feasibility gate over train-only internal validation rows.",
        f"- Discovery group: `{GROUP_DISCOVERY}`; confirmation group: `{GROUP_CONFIRM}`.",
        "- Uses pre-registered Replogle bulk raw columns only: `cnv_score_z`, `TE_ratio`, `std_leverage_score`.",
        "- Dedupes author bulk rows by median to one value per dataset/condition/raw column.",
        "- No canonical multi, no Track C query, no training, no inference, and no checkpoint selection.",
        "- Internal-val groups share many conditions and are not independent manuscript evidence; a signal here would only justify further design/audit.",
        "",
        "## Summary",
        "",
        f"- internal rows: `{len(internal_rows)}`",
        f"- joined rows: `{len(joined)}`",
        f"- deduped artifact conditions: `{artifact_meta['deduped_conditions']}`",
        f"- signals: `{signals}`",
        "",
        "## Candidate Results",
        "",
    ]
    for row in flat:
        lines.extend(
            [
                f"### {row['raw_column']}",
                "",
                f"- status: `{row['status']}`",
                f"- discovery signed rho / dataset min / shuffle p / MMD: `{fmt(row['discovery_signed_rho'])}` / `{fmt(row['discovery_dataset_min'])}` / `{fmt(row['discovery_shuffle_p'])}` / `{fmt(row['discovery_abs_rho_mmd'])}`",
                f"- confirm signed rho / dataset min / shuffle p / MMD: `{fmt(row['confirm_signed_rho'])}` / `{fmt(row['confirm_dataset_min'])}` / `{fmt(row['confirm_shuffle_p'])}` / `{fmt(row['confirm_abs_rho_mmd'])}`",
                f"- residual discovery / confirm rho: `{fmt(row['discovery_residual_rho'])}` / `{fmt(row['confirm_residual_rho'])}`",
                f"- reasons: `{row['reasons']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            "This gate does not authorize GPU. If it fails, keep Replogle bulk as a diagnostic/failure-mechanism branch only. If it signals, require external review plus a train-only mechanism that never feeds held-out response-derived artifacts to eval conditions.",
            "",
            "## Outputs",
            "",
            f"- joined rows: `{OUT_JOINED}`",
            f"- summary: `{OUT_SUMMARY}`",
            f"- json: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "signals": signals, "report": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
