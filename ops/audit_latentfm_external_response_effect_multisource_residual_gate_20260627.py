#!/usr/bin/env python3
"""External response/effect multi-source residual gate v2.

CPU/report-only. This gate reconciles Replogle train/internal residualized bulk
effects with Frangieh ORCS and DepMap dependency diagnostic artifacts. Only the
train/internal scope can authorize future work; diagnostic/test-only evidence is
reported separately and cannot authorize GPU.
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


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
REPLOGLE = REPORTS / "replogle_bulk_residualized_source_lodo_v2_gate_20260627/replogle_bulk_residualized_source_lodo_v2_rows.csv"
FRANGIEH = REPORTS / "frangieh_orcs_response_preview_gate_20260627/frangieh_orcs_response_preview_joined_rows.csv"
DEPMAP = REPORTS / "depmap_24q4_dependency_gate_20260627/depmap_24q4_dependency_gate_joined_rows.csv"
OUT_DIR = REPORTS / "external_response_effect_multisource_residual_gate_20260627"
OUT_ROWS = OUT_DIR / "external_response_effect_multisource_residual_rows.csv"
OUT_SUMMARY = OUT_DIR / "external_response_effect_multisource_residual_summary.csv"
OUT_JSON = REPORTS / "latentfm_external_response_effect_multisource_residual_gate_20260627.json"
OUT_MD = REPORTS / "LATENTFM_EXTERNAL_RESPONSE_EFFECT_MULTISOURCE_RESIDUAL_GATE_20260627.md"
SEED = 20260627


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
    return pearson(rank(xs), rank(ys)) if len(xs) >= 3 else None


def zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mu = mean(values)
    var = sum((v - mu) ** 2 for v in values) / max(1, len(values) - 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    return [0.0 if sd == 0 else (v - mu) / sd for v in values]


def add_row(
    rows: list[dict[str, Any]],
    *,
    source_family: str,
    scope: str,
    dataset: str,
    condition: str,
    artifact: str,
    role: str,
    artifact_value: float | None,
    bad_pp: float | None,
    mmd: float | None,
) -> None:
    if not dataset or not condition or not artifact or artifact_value is None or bad_pp is None:
        return
    rows.append(
        {
            "source_family": source_family,
            "evidence_scope": scope,
            "dataset": dataset,
            "condition": condition,
            "artifact": artifact,
            "artifact_role": role or "response_candidate",
            "artifact_value": float(artifact_value),
            "bad_pp": float(bad_pp),
            "mmd": mmd,
        }
    )


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(REPLOGLE):
        add_row(
            rows,
            source_family="replogle_bulk_traininternal_resid",
            scope="train_internal",
            dataset=norm(row.get("dataset")),
            condition=norm(row.get("condition")),
            artifact="replogle_" + norm(row.get("artifact")),
            role="response_candidate",
            artifact_value=fnum(row.get("artifact_resid_oriented")),
            bad_pp=fnum(row.get("bad_pp")),
            mmd=fnum(row.get("anchor_mmd_clamped")),
        )
    for row in read_csv(FRANGIEH):
        role = norm(row.get("artifact_role")) or "response_candidate"
        add_row(
            rows,
            source_family="frangieh_orcs_diagnostic",
            scope="diagnostic_test_metric_only",
            dataset=norm(row.get("dataset")),
            condition=norm(row.get("condition")),
            artifact=norm(row.get("artifact")),
            role=role,
            artifact_value=fnum(row.get("artifact_value")),
            bad_pp=None if fnum(row.get("pearson_pert")) is None else -float(fnum(row.get("pearson_pert")) or 0.0),
            mmd=fnum(row.get("test_mmd_clamped")),
        )
    for row in read_csv(DEPMAP):
        add_row(
            rows,
            source_family="depmap_dependency_diagnostic",
            scope="diagnostic_test_metric_only",
            dataset=norm(row.get("dataset")),
            condition=norm(row.get("condition")),
            artifact=norm(row.get("artifact")),
            role="response_candidate",
            artifact_value=fnum(row.get("artifact_value")),
            bad_pp=None if fnum(row.get("pearson_pert")) is None else -float(fnum(row.get("pearson_pert")) or 0.0),
            mmd=fnum(row.get("test_mmd_clamped")),
        )
    return rows


def orient_rows(rows: list[dict[str, Any]]) -> None:
    by_artifact: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_artifact[(row["evidence_scope"], row["source_family"], row["artifact"])].append(row)
    for subset in by_artifact.values():
        vals = [float(row["artifact_value"]) for row in subset]
        zs = zscore(vals)
        for row, z in zip(subset, zs):
            row["artifact_z"] = z
        xs = [float(row["artifact_z"]) for row in subset]
        ys = [float(row["bad_pp"]) for row in subset]
        rho = spearman(xs, ys)
        direction = 1.0 if rho is None or rho >= 0 else -1.0
        for row in subset:
            row["artifact_oriented"] = direction * float(row["artifact_z"])


def finite(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and math.isfinite(x) and math.isfinite(y):
            xs.append(float(x))
            ys.append(float(y))
    return xs, ys


def shuffle_p(rows: list[dict[str, Any]], x_key: str, y_key: str, *, n_perm: int = 1000) -> float | None:
    pairs = [(float(row[x_key]), float(row[y_key]), row["dataset"]) for row in rows if isinstance(row.get(x_key), (int, float)) and isinstance(row.get(y_key), (int, float))]
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
    hits = total = 0
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


def heldout_min(rows: list[dict[str, Any]], key: str, x_key: str, y_key: str) -> float | None:
    vals = []
    for value in sorted({row[key] for row in rows if row.get(key)}):
        sub = [row for row in rows if row.get(key) != value]
        xs, ys = finite(sub, x_key, y_key)
        rho = spearman(xs, ys)
        if rho is not None:
            vals.append(rho)
    return min(vals) if vals else None


def summarize_scope(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    sub = [row for row in rows if row["evidence_scope"] == scope and row.get("artifact_role") == "response_candidate"]
    xs, ys = finite(sub, "artifact_oriented", "bad_pp")
    rho = spearman(xs, ys)
    mmd_x, mmd_y = finite(sub, "artifact_oriented", "mmd")
    mmd_rho = spearman(mmd_x, mmd_y)
    source_min = heldout_min(sub, "source_family", "artifact_oriented", "bad_pp")
    dataset_min = heldout_min(sub, "dataset", "artifact_oriented", "bad_pp")
    sp = shuffle_p(sub, "artifact_oriented", "bad_pp")
    high_low_mmd = None
    if mmd_y:
        ordered = sorted([row for row in sub if isinstance(row.get("mmd"), (int, float))], key=lambda row: float(row["artifact_oriented"]))
        k = max(5, len(ordered) // 4) if ordered else 0
        if k:
            low = [float(row["mmd"]) for row in ordered[:k]]
            high = [float(row["mmd"]) for row in ordered[-k:]]
            high_low_mmd = mean(high) - mean(low)
    control_sub = [row for row in rows if row["evidence_scope"] == scope and row.get("artifact_role") != "response_candidate"]
    cx, cy = finite(control_sub, "artifact_oriented", "bad_pp")
    control_rho = spearman(cx, cy)
    reasons = []
    if scope == "train_internal":
        if len({row["source_family"] for row in sub}) < 3:
            reasons.append("source_families_below_3")
        if len({(row["dataset"], row["condition"]) for row in sub}) < 100:
            reasons.append("conditions_below_100")
        if rho is None or rho < 0.25:
            reasons.append("residual_signed_rho_below_0p25")
        if source_min is None or source_min < 0.15:
            reasons.append("source_family_min_below_0p15")
        if dataset_min is None or dataset_min < 0.10:
            reasons.append("dataset_min_below_0p10")
        if sp is None or sp > 0.01:
            reasons.append("within_dataset_shuffle_p_gt_0p01")
        if mmd_rho is not None and abs(mmd_rho) >= 0.15:
            reasons.append("mmd_abs_rho_ge_0p15")
        if high_low_mmd is not None and high_low_mmd > 0.001:
            reasons.append("high_low_mmd_gt_0p001")
        if control_rho is not None and rho is not None and abs(control_rho) >= abs(rho):
            reasons.append("control_signal_not_weaker")
    else:
        reasons.append("diagnostic_test_metric_only_not_gpu_eligible")
    return {
        "scope": scope,
        "rows": len(sub),
        "conditions": len({(row["dataset"], row["condition"]) for row in sub}),
        "datasets": len({row["dataset"] for row in sub}),
        "source_families": len({row["source_family"] for row in sub}),
        "signed_rho": rho,
        "shuffle_p_abs": sp,
        "dataset_min": dataset_min,
        "source_family_min": source_min,
        "abs_mmd_rho": None if mmd_rho is None else abs(mmd_rho),
        "high_low_mmd": high_low_mmd,
        "control_abs_rho": None if control_rho is None else abs(control_rho),
        "status": "pass_needs_external_audit_no_gpu" if not reasons else "fail_no_gpu",
        "reasons": ";".join(reasons),
    }


def main() -> None:
    rows = load_rows()
    orient_rows(rows)
    summaries = [summarize_scope(rows, scope) for scope in ["train_internal", "diagnostic_test_metric_only"]]
    passed_train = [row for row in summaries if row["scope"] == "train_internal" and row["status"] == "pass_needs_external_audit_no_gpu"]
    status = "external_response_effect_multisource_residual_pass_needs_external_audit_no_gpu" if passed_train else "external_response_effect_multisource_residual_fail_no_gpu"
    write_csv(
        OUT_ROWS,
        rows,
        [
            "source_family",
            "evidence_scope",
            "dataset",
            "condition",
            "artifact",
            "artifact_role",
            "artifact_value",
            "artifact_z",
            "artifact_oriented",
            "bad_pp",
            "mmd",
        ],
    )
    write_csv(
        OUT_SUMMARY,
        summaries,
        [
            "scope",
            "rows",
            "conditions",
            "datasets",
            "source_families",
            "signed_rho",
            "shuffle_p_abs",
            "dataset_min",
            "source_family_min",
            "abs_mmd_rho",
            "high_low_mmd",
            "control_abs_rho",
            "status",
            "reasons",
        ],
    )
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": status,
        "gpu_authorized": False,
        "inputs": [str(REPLOGLE), str(FRANGIEH), str(DEPMAP)],
        "summaries": summaries,
        "outputs": {"rows": str(OUT_ROWS), "summary": str(OUT_SUMMARY), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = []
    for row in summaries:
        lines.append(
            "| {scope} | {rows} | {conditions} | {datasets} | {sources} | {rho} | {sp} | {dsmin} | {srcmin} | {mmd} | {hlmmd} | `{status}` | {reasons} |".format(
                scope=row["scope"],
                rows=row["rows"],
                conditions=row["conditions"],
                datasets=row["datasets"],
                sources=row["source_families"],
                rho=f"{row['signed_rho']:+.6f}" if isinstance(row["signed_rho"], float) else "NA",
                sp=f"{row['shuffle_p_abs']:.6f}" if isinstance(row["shuffle_p_abs"], float) else "NA",
                dsmin=f"{row['dataset_min']:+.6f}" if isinstance(row["dataset_min"], float) else "NA",
                srcmin=f"{row['source_family_min']:+.6f}" if isinstance(row["source_family_min"], float) else "NA",
                mmd=f"{row['abs_mmd_rho']:+.6f}" if isinstance(row["abs_mmd_rho"], float) else "NA",
                hlmmd=f"{row['high_low_mmd']:+.6f}" if isinstance(row["high_low_mmd"], float) else "NA",
                status=row["status"],
                reasons=row["reasons"],
            )
        )
    md = f"""# LatentFM External Response/Effect Multi-Source Residual Gate 2026-06-27

Timestamp: `{payload['timestamp']}`

Status: `{status}`

GPU authorized: `False`

## Boundary

- CPU/report-only multi-source residual gate over Replogle, Frangieh ORCS, and
  DepMap dependency artifacts.
- Only train/internal evidence can authorize future GPU design.
- Frangieh and DepMap rows are diagnostic/test-metric-only under current
  artifacts and are not GPU-eligible.
- No training, inference, checkpoint selection, canonical multi selection,
  Track C query, or GPU.

## Scope Summary

| scope | rows | conditions | datasets | source families | signed rho | shuffle p | dataset min | source min | abs MMD rho | high-low MMD | status | reasons |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
{chr(10).join(lines)}

## Decision

GPU remains unauthorized. The train/internal scope does not have three
independent source families and remains MMD-confounded; the multi-source-looking
diagnostic pool cannot be used for training or checkpoint selection.

## Outputs

- JSON: `{OUT_JSON}`
- rows: `{OUT_ROWS}`
- summary: `{OUT_SUMMARY}`
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
