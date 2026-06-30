#!/usr/bin/env python3
"""Replogle residualized source/background LODO v2 gate.

CPU/report-only. This gate revisits the Replogle author bulk artifacts after
the earlier raw artifact gate found strong but MMD/QC-confounded associations.
It uses only train/internal anchor rows, residualizes candidate artifact scores
against local QC controls plus dataset/background means, and requires robust
within-dataset shuffle, LODO, MMD, and QC-control checks before any later GPU
design.

It does not train, infer, select checkpoints, read canonical multi for
selection, read Track C query, or use GPU.
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
ARTIFACTS = REPORTS / "replogle_bulk_artifacts_20260627/replogle_bulk_condition_artifacts.csv"
OUTCOMES = REPORTS / "replogle_trainonly_internal_difficulty_gate_20260627/replogle_trainonly_internal_difficulty_joined_rows.csv"
OUT_DIR = REPORTS / "replogle_bulk_residualized_source_lodo_v2_gate_20260627"
OUT_ROWS = OUT_DIR / "replogle_bulk_residualized_source_lodo_v2_rows.csv"
OUT_SUMMARY = OUT_DIR / "replogle_bulk_residualized_source_lodo_v2_summary.csv"
OUT_JSON = REPORTS / "latentfm_replogle_bulk_residualized_source_lodo_v2_gate_20260627.json"
OUT_MD = REPORTS / "LATENTFM_REPLOGLE_BULK_RESIDUALIZED_SOURCE_LODO_V2_GATE_20260627.md"

SEED = 20260627
QC_RAW_COLUMNS = {
    "UMI_count_unfiltered",
    "num_cells_filtered",
    "num_cells_unfiltered",
    "mitopercent",
    "z_gemgroup_UMI",
}
CANDIDATE_RAW_COLUMNS = {
    "cnv_score_z",
    "TE_ratio",
    "std_leverage_score",
    "mean_leverage_score",
    "control_expr",
    "fold_expr",
    "pct_expr",
    "anderson_darling_counts",
    "energy_test_p_value",
    "mann_whitney_counts",
}


def norm(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def fnum(value: Any) -> float | None:
    text = norm(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


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
    if len(xs) < 3:
        return None
    return pearson(rank(xs), rank(ys))


def zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mu = mean(values)
    var = sum((v - mu) ** 2 for v in values) / max(1, len(values) - 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    return [0.0 if sd == 0 else (v - mu) / sd for v in values]


def finite_pairs(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and math.isfinite(x) and math.isfinite(y):
            xs.append(float(x))
            ys.append(float(y))
    return xs, ys


def shuffle_p_abs(rows: list[dict[str, Any]], feature: str, target: str, *, n_perm: int = 1000) -> float | None:
    pairs = []
    for row in rows:
        x = row.get(feature)
        y = row.get(target)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and math.isfinite(x) and math.isfinite(y):
            pairs.append((float(x), float(y), norm(row.get("dataset"))))
    if len(pairs) < 10:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    actual = spearman(xs, ys)
    if actual is None:
        return None
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for idx, (_, _, dataset) in enumerate(pairs):
        by_dataset[dataset].append(idx)
    rng = random.Random(SEED)
    hits = 0
    total = 0
    for _ in range(n_perm):
        shuffled = xs[:]
        for idxs in by_dataset.values():
            vals = [shuffled[i] for i in idxs]
            rng.shuffle(vals)
            for i, val in zip(idxs, vals):
                shuffled[i] = val
        rho = spearman(shuffled, ys)
        if rho is None:
            continue
        total += 1
        if abs(rho) >= abs(actual):
            hits += 1
    return (hits + 1) / (total + 1) if total else None


def lodo_min(rows: list[dict[str, Any]], feature: str, target: str) -> float | None:
    values: list[float] = []
    datasets = sorted({norm(row.get("dataset")) for row in rows if norm(row.get("dataset"))})
    for dataset in datasets:
        sub = [row for row in rows if norm(row.get("dataset")) != dataset]
        xs, ys = finite_pairs(sub, feature, target)
        rho = spearman(xs, ys)
        if rho is not None:
            values.append(rho)
    return min(values) if values else None


def load_artifact_matrix() -> dict[tuple[str, str], dict[str, Any]]:
    matrix: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for row in read_csv(ARTIFACTS):
        dataset = norm(row.get("dataset"))
        condition = norm(row.get("condition"))
        raw_col = norm(row.get("raw_column"))
        val = fnum(row.get("artifact_value"))
        if not dataset or not condition or not raw_col or val is None:
            continue
        rec = matrix[(dataset, condition)]
        rec["dataset"] = dataset
        rec["condition"] = condition
        rec["cell_background"] = norm(row.get("cell_background"))
        rec[raw_col] = val
    return matrix


def build_rows() -> list[dict[str, Any]]:
    matrix = load_artifact_matrix()
    rows: list[dict[str, Any]] = []
    for out in read_csv(OUTCOMES):
        key = (norm(out.get("dataset")), norm(out.get("condition")))
        art = matrix.get(key)
        if not art:
            continue
        pp = fnum(out.get("anchor_pearson_pert"))
        if pp is None:
            continue
        rec: dict[str, Any] = {
            "group": norm(out.get("group")),
            "dataset": key[0],
            "condition": key[1],
            "gene": norm(out.get("gene")),
            "cell_background": art.get("cell_background", ""),
            "bad_pp": -pp,
            "anchor_mmd_clamped": fnum(out.get("anchor_mmd_clamped")),
            "gene_train_count": fnum(out.get("gene_train_count")),
        }
        for col in sorted(CANDIDATE_RAW_COLUMNS | QC_RAW_COLUMNS):
            rec[col] = art.get(col)
        rows.append(rec)
    return rows


def residualize(values: list[float], design_cols: list[list[float]]) -> list[float]:
    if len(values) < 5:
        return values[:]
    X_cols = [[1.0] * len(values)] + design_cols
    X = np.array(list(zip(*X_cols)), dtype=float)
    y = np.array(values, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return list(y - X @ beta)


def prepare_feature(rows: list[dict[str, Any]], artifact: str) -> list[dict[str, Any]]:
    sub = [dict(row) for row in rows if fnum(row.get(artifact)) is not None]
    if len(sub) < 10:
        return []

    raw_vals = [float(fnum(row.get(artifact)) or 0.0) for row in sub]
    # Normalize within source dataset/background enough to prevent simple
    # background shifts from masquerading as target-specific response effects.
    by_group: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(sub):
        by_group[(norm(row.get("dataset")), norm(row.get("cell_background")))].append(idx)
    raw_z = [0.0] * len(sub)
    for idxs in by_group.values():
        vals = [raw_vals[i] for i in idxs]
        zs = zscore(vals)
        for i, z in zip(idxs, zs):
            raw_z[i] = z

    qc_cols = []
    for col in sorted(QC_RAW_COLUMNS | {"gene_train_count"}):
        vals: list[float] = []
        present = 0
        for row in sub:
            val = fnum(row.get(col))
            if val is not None:
                present += 1
            vals.append(math.log1p(max(0.0, float(val or 0.0))) if col != "mitopercent" else float(val or 0.0))
        if present >= max(10, len(sub) // 2):
            qc_cols.append(zscore(vals))
    resid = residualize(raw_z, qc_cols)
    for row, raw, res in zip(sub, raw_z, resid):
        row["artifact"] = artifact
        row["artifact_z"] = raw
        row["artifact_resid"] = float(res)
    xs, ys = finite_pairs(sub, "artifact_resid", "bad_pp")
    rho = spearman(xs, ys)
    direction = 1.0 if rho is None or rho >= 0 else -1.0
    for row in sub:
        row["artifact_resid_oriented"] = direction * float(row["artifact_resid"])
    return sub


def summarize_feature(rows: list[dict[str, Any]], artifact: str) -> dict[str, Any]:
    sub = prepare_feature(rows, artifact)
    if not sub:
        return {"artifact": artifact, "n": 0, "status": "fail_no_gpu", "reasons": "insufficient_rows"}
    xs, ys = finite_pairs(sub, "artifact_resid_oriented", "bad_pp")
    rho = spearman(xs, ys)
    _, mmd = finite_pairs(sub, "artifact_resid_oriented", "anchor_mmd_clamped")
    xs_mmd, ys_mmd = finite_pairs(sub, "artifact_resid_oriented", "anchor_mmd_clamped")
    mmd_rho = spearman(xs_mmd, ys_mmd)
    qc_rhos = []
    for qc in sorted(QC_RAW_COLUMNS | {"gene_train_count"}):
        xs_qc, ys_qc = finite_pairs(sub, qc, "bad_pp")
        rho_qc = spearman(xs_qc, ys_qc)
        if rho_qc is not None:
            qc_rhos.append(abs(rho_qc))
    max_qc_abs_rho = max(qc_rhos) if qc_rhos else None
    sp = shuffle_p_abs(sub, "artifact_resid_oriented", "bad_pp")
    ds_min = lodo_min(sub, "artifact_resid_oriented", "bad_pp")
    high_low_mmd = None
    if mmd:
        ordered = sorted(sub, key=lambda row: float(row["artifact_resid_oriented"]))
        k = max(5, len(ordered) // 4)
        low = [fnum(row.get("anchor_mmd_clamped")) for row in ordered[:k]]
        high = [fnum(row.get("anchor_mmd_clamped")) for row in ordered[-k:]]
        low = [float(v) for v in low if v is not None]
        high = [float(v) for v in high if v is not None]
        if low and high:
            high_low_mmd = mean(high) - mean(low)

    reasons = []
    if len(sub) < 80:
        reasons.append("deduped_rows_below_80")
    if rho is None or rho < 0.25:
        reasons.append("residual_signed_rho_below_0p25")
    if ds_min is None or ds_min < 0.15:
        reasons.append("lodo_min_below_0p15")
    if sp is None or sp > 0.01:
        reasons.append("within_dataset_shuffle_p_gt_0p01")
    if mmd_rho is not None and abs(mmd_rho) >= 0.15:
        reasons.append("abs_mmd_rho_ge_0p15")
    if high_low_mmd is not None and high_low_mmd > 0.001:
        reasons.append("high_low_mmd_gt_0p001")
    if max_qc_abs_rho is not None and rho is not None and max_qc_abs_rho >= abs(rho):
        reasons.append("qc_control_signal_not_weaker")
    return {
        "artifact": artifact,
        "n": len(sub),
        "datasets": len({row["dataset"] for row in sub}),
        "conditions": len({(row["dataset"], row["condition"]) for row in sub}),
        "signed_rho_bad_pp": rho,
        "shuffle_p_abs": sp,
        "lodo_min_signed_rho": ds_min,
        "mmd_abs_rho": None if mmd_rho is None else abs(mmd_rho),
        "high_low_mmd": high_low_mmd,
        "max_qc_abs_rho": max_qc_abs_rho,
        "status": "pass_needs_external_audit_no_gpu" if not reasons else "fail_no_gpu",
        "reasons": ";".join(reasons),
        "_rows": sub,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    summaries = [summarize_feature(rows, artifact) for artifact in sorted(CANDIDATE_RAW_COLUMNS)]
    summaries = sorted(
        summaries,
        key=lambda rec: (
            rec.get("status") != "pass_needs_external_audit_no_gpu",
            -(rec.get("signed_rho_bad_pp") or -999),
            rec.get("artifact", ""),
        ),
    )
    all_feature_rows: list[dict[str, Any]] = []
    for summary in summaries:
        all_feature_rows.extend(summary.pop("_rows", []))
    passed = [rec for rec in summaries if rec.get("status") == "pass_needs_external_audit_no_gpu"]
    status = "replogle_bulk_residualized_source_lodo_v2_pass_needs_external_audit_no_gpu" if passed else "replogle_bulk_residualized_source_lodo_v2_fail_no_gpu"

    write_csv(
        OUT_ROWS,
        all_feature_rows,
        [
            "artifact",
            "group",
            "dataset",
            "condition",
            "gene",
            "cell_background",
            "bad_pp",
            "anchor_mmd_clamped",
            "gene_train_count",
            "artifact_z",
            "artifact_resid",
            "artifact_resid_oriented",
        ],
    )
    write_csv(
        OUT_SUMMARY,
        summaries,
        [
            "artifact",
            "n",
            "datasets",
            "conditions",
            "signed_rho_bad_pp",
            "shuffle_p_abs",
            "lodo_min_signed_rho",
            "mmd_abs_rho",
            "high_low_mmd",
            "max_qc_abs_rho",
            "status",
            "reasons",
        ],
    )
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": status,
        "gpu_authorized": False,
        "inputs": [str(ARTIFACTS), str(OUTCOMES)],
        "n_join_rows": len(rows),
        "n_feature_rows": len(all_feature_rows),
        "passed_artifacts": [rec["artifact"] for rec in passed],
        "top_artifacts": summaries[:8],
        "outputs": {"rows": str(OUT_ROWS), "summary": str(OUT_SUMMARY), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    top_lines = []
    for rec in summaries[:10]:
        top_lines.append(
            "| {artifact} | {n} | {signed_rho_bad_pp} | {shuffle_p_abs} | {lodo_min_signed_rho} | {mmd_abs_rho} | {high_low_mmd} | {max_qc_abs_rho} | `{status}` | {reasons} |".format(
                artifact=rec.get("artifact", ""),
                n=rec.get("n", ""),
                signed_rho_bad_pp=f"{rec.get('signed_rho_bad_pp'):+.6f}" if isinstance(rec.get("signed_rho_bad_pp"), float) else "NA",
                shuffle_p_abs=f"{rec.get('shuffle_p_abs'):.6f}" if isinstance(rec.get("shuffle_p_abs"), float) else "NA",
                lodo_min_signed_rho=f"{rec.get('lodo_min_signed_rho'):+.6f}" if isinstance(rec.get("lodo_min_signed_rho"), float) else "NA",
                mmd_abs_rho=f"{rec.get('mmd_abs_rho'):+.6f}" if isinstance(rec.get("mmd_abs_rho"), float) else "NA",
                high_low_mmd=f"{rec.get('high_low_mmd'):+.6f}" if isinstance(rec.get("high_low_mmd"), float) else "NA",
                max_qc_abs_rho=f"{rec.get('max_qc_abs_rho'):+.6f}" if isinstance(rec.get("max_qc_abs_rho"), float) else "NA",
                status=rec.get("status", ""),
                reasons=rec.get("reasons", ""),
            )
        )
    md = f"""# LatentFM Replogle Bulk Residualized Source LODO V2 Gate 2026-06-27

Timestamp: `{payload['timestamp']}`

Status: `{status}`

GPU authorized: `False`

## Boundary

- CPU/report-only residualized Replogle bulk source/background LODO gate.
- Inputs are train/internal anchor outcomes and Replogle author bulk artifacts.
- No training, inference, checkpoint selection, canonical multi selection,
  Track C query, or GPU.

## Coverage

- joined train/internal rows: `{len(rows)}`
- residualized feature rows: `{len(all_feature_rows)}`
- passed artifacts: `{[rec['artifact'] for rec in passed]}`

## Top Artifacts

| artifact | n | rho vs bad pp | shuffle p(abs) | LODO min | abs MMD rho | high-low MMD | max QC abs rho | status | reasons |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
{chr(10).join(top_lines)}

## Decision

This gate can only authorize an external audit, not direct GPU. GPU remains
unauthorized because status is `{status}` and all candidates must satisfy the
predeclared residual, LODO, shuffle, MMD, and QC-control checks.

## Outputs

- JSON: `{OUT_JSON}`
- rows: `{OUT_ROWS}`
- summary: `{OUT_SUMMARY}`
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
