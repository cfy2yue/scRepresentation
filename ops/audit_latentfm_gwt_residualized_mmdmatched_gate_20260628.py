#!/usr/bin/env python3
"""Residualized/MMD-matched closure gate for GWT reliability artifacts.

Short CPU/report-only task. Reads the already materialized GWT preflight rows
and tests whether any artifact retains train/internal pp signal after
within-dataset centering and MMD residualization. It does not train, infer, read
checkpoints, use canonical multi, read Track C query, or use GPU.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
IN_ROWS = REPORTS / "latentfm_gwt_condition_reliability_artifact_preflight_20260627_rows.csv"
OUT_DIR = REPORTS / "gwt_residualized_mmdmatched_gate_20260628"
OUT_MD = REPORTS / "LATENTFM_GWT_RESIDUALIZED_MMDMATCHED_GATE_20260628.md"
OUT_JSON = REPORTS / "latentfm_gwt_residualized_mmdmatched_gate_20260628.json"


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def finite(value: Any) -> float | None:
    text = norm(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except (TypeError, ValueError):
        return str(value)


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def bootstrap_mean(values: list[float], *, seed: int, n_boot: int = 2000) -> dict[str, Any]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return {"mean": None, "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    n = len(vals)
    draws = [sum(rng.choice(vals) for _ in range(n)) / n for _ in range(n_boot)]
    return {"mean": mean(vals), "ci_low": quantile(draws, 0.025), "ci_high": quantile(draws, 0.975)}


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
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return pearson(rankdata(xs), rankdata(ys))


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with IN_ROWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if norm(row.get("outcome_overlap")).lower() != "true":
                continue
            art = norm(row.get("artifact"))
            ds = norm(row.get("dataset"))
            cond = norm(row.get("condition"))
            x = finite(row.get("artifact_value"))
            pp = finite(row.get("pp_proxy_mean"))
            mmd = finite(row.get("mmd_proxy_max"))
            if not art or not ds or not cond or x is None or pp is None:
                continue
            rows.append({"artifact": art, "dataset": ds, "condition": cond, "x": x, "pp": pp, "mmd": mmd})
    return rows


def center_by_dataset(rows: list[dict[str, Any]], key: str) -> list[float]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = row.get(key)
        if val is not None:
            by_ds[str(row["dataset"])].append(float(val))
    means = {ds: mean(vals) for ds, vals in by_ds.items() if vals}
    return [float(row[key]) - means[str(row["dataset"])] for row in rows]


def residualize(y: list[float], covar: list[float]) -> list[float]:
    if len(y) != len(covar) or len(y) < 3:
        return y
    mx = mean(covar)
    my = mean(y)
    denom = sum((x - mx) ** 2 for x in covar)
    if denom <= 0:
        return [v - my for v in y]
    beta = sum((x - mx) * (v - my) for x, v in zip(covar, y)) / denom
    alpha = my - beta * mx
    return [v - (alpha + beta * x) for x, v in zip(covar, y)]


def high_low_stats(rows: list[dict[str, Any]], x_resid: list[float], pp_resid: list[float], mmd_resid: list[float]) -> dict[str, Any]:
    enriched = [dict(row, x_resid=x, pp_resid=pp, mmd_resid=mmd) for row, x, pp, mmd in zip(rows, x_resid, pp_resid, mmd_resid)]
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_ds[str(row["dataset"])].append(row)
    ds_rows = []
    for ds, ds_rows_in in sorted(by_ds.items()):
        if len(ds_rows_in) < 4:
            continue
        ordered = sorted(ds_rows_in, key=lambda r: float(r["x_resid"]))
        k = max(1, len(ordered) // 4)
        lo = ordered[:k]
        hi = ordered[-k:]
        pp_delta = mean([float(r["pp_resid"]) for r in hi]) - mean([float(r["pp_resid"]) for r in lo])
        mmd_delta = mean([float(r["mmd_resid"]) for r in hi]) - mean([float(r["mmd_resid"]) for r in lo])
        ds_rows.append({"dataset": ds, "n": len(ordered), "k": k, "pp_high_minus_low": pp_delta, "mmd_high_minus_low": mmd_delta})
    pp_ds = [float(r["pp_high_minus_low"]) for r in ds_rows]
    mmd_ds = [float(r["mmd_high_minus_low"]) for r in ds_rows]
    boot = bootstrap_mean(pp_ds, seed=20260628) if pp_ds else {"mean": None, "ci_low": None, "ci_high": None}
    return {
        "dataset_rows": ds_rows,
        "dataset_count_for_highlow": len(ds_rows),
        "pp_high_minus_low_mean": boot["mean"],
        "pp_high_minus_low_ci_low": boot["ci_low"],
        "pp_high_minus_low_ci_high": boot["ci_high"],
        "pp_high_minus_low_dataset_min": min(pp_ds) if pp_ds else None,
        "mmd_high_minus_low_max": max(mmd_ds) if mmd_ds else None,
    }


def shuffle_p(rows: list[dict[str, Any]], x_resid: list[float], pp_resid: list[float], observed: float | None, *, seed: int = 20260628, n_perm: int = 1000) -> float | None:
    if observed is None:
        return None
    rng = random.Random(seed)
    by_ds_idx: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_ds_idx[str(row["dataset"])].append(i)
    more = 0
    total = 0
    for _ in range(n_perm):
        perm_x = list(x_resid)
        for idxs in by_ds_idx.values():
            vals = [perm_x[i] for i in idxs]
            rng.shuffle(vals)
            for i, val in zip(idxs, vals):
                perm_x[i] = val
        stat = high_low_stats(rows, perm_x, pp_resid, [0.0] * len(pp_resid))["pp_high_minus_low_mean"]
        if stat is None:
            continue
        total += 1
        if abs(float(stat)) >= abs(float(observed)):
            more += 1
    if total == 0:
        return None
    return (more + 1.0) / (total + 1.0)


def summarize_artifact(artifact: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x_centered = center_by_dataset(rows, "x")
    pp_centered = center_by_dataset(rows, "pp")
    mmd_centered = center_by_dataset([r for r in rows if r.get("mmd") is not None], "mmd")
    # Preserve row order; rows without MMD get zero centered covariate.
    mmd_by_key = {
        (r["dataset"], r["condition"]): v
        for r, v in zip([r for r in rows if r.get("mmd") is not None], mmd_centered)
    }
    mmd_aligned = [float(mmd_by_key.get((r["dataset"], r["condition"]), 0.0)) for r in rows]
    x_resid = residualize(x_centered, mmd_aligned)
    pp_resid = residualize(pp_centered, mmd_aligned)
    rho = spearman(x_resid, pp_resid)
    hls = high_low_stats(rows, x_resid, pp_resid, mmd_aligned)
    p_shuffle = shuffle_p(rows, x_resid, pp_resid, hls["pp_high_minus_low_mean"], seed=20260628 + abs(hash(artifact)) % 10000)
    datasets = sorted({str(r["dataset"]) for r in rows})
    reasons = []
    if len(datasets) < 3:
        reasons.append("dataset_count_below_3")
    if len(rows) < 50:
        reasons.append("overlap_rows_below_50")
    if rho is None or abs(rho) < 0.25:
        reasons.append("abs_residual_spearman_below_0p25")
    if hls["pp_high_minus_low_mean"] is None or hls["pp_high_minus_low_mean"] < 0.020:
        reasons.append("residualized_pp_highlow_mean_below_0p020")
    if hls["pp_high_minus_low_ci_low"] is None or hls["pp_high_minus_low_ci_low"] <= 0:
        reasons.append("residualized_pp_highlow_ci_low_not_above_0")
    if hls["pp_high_minus_low_dataset_min"] is None or hls["pp_high_minus_low_dataset_min"] < -0.020:
        reasons.append("residualized_dataset_tail_below_minus_0p020")
    if hls["mmd_high_minus_low_max"] is not None and hls["mmd_high_minus_low_max"] > 0.001:
        reasons.append("mmd_highlow_max_above_0p001")
    if p_shuffle is None or p_shuffle > 0.01:
        reasons.append("within_dataset_shuffle_p_above_0p01")
    status = "pass_external_review_only_no_gpu" if not reasons else "fail_no_gpu"
    summary = {
        "artifact": artifact,
        "status": status,
        "n": len(rows),
        "datasets": len(datasets),
        "residual_spearman": rho,
        "pp_high_minus_low_mean": hls["pp_high_minus_low_mean"],
        "pp_high_minus_low_ci_low": hls["pp_high_minus_low_ci_low"],
        "pp_high_minus_low_ci_high": hls["pp_high_minus_low_ci_high"],
        "pp_high_minus_low_dataset_min": hls["pp_high_minus_low_dataset_min"],
        "mmd_high_minus_low_max": hls["mmd_high_minus_low_max"],
        "within_dataset_shuffle_p": p_shuffle,
        "reasons": reasons,
    }
    dataset_rows = [dict(row, artifact=artifact) for row in hls["dataset_rows"]]
    return summary, dataset_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    rows = load_rows()
    by_art: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_art[str(row["artifact"])].append(row)
    summaries = []
    dataset_rows_all = []
    for art, art_rows in sorted(by_art.items()):
        summary, dataset_rows = summarize_artifact(art, art_rows)
        summaries.append(summary)
        dataset_rows_all.extend(dataset_rows)
    pass_candidates = [s["artifact"] for s in summaries if str(s["status"]).startswith("pass")]
    status = "gwt_residualized_mmdmatched_gate_pass_external_review_only_no_gpu" if pass_candidates else "gwt_residualized_mmdmatched_gate_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_report_only": True,
            "input_rows": str(IN_ROWS),
            "training_or_inference": False,
            "checkpoints": False,
            "canonical_multi_selection": False,
            "trackc_query": False,
            "gpu": False,
        },
        "pass_candidates": pass_candidates,
        "summaries": summaries,
        "outputs": {
            "markdown": str(OUT_MD),
            "json": str(OUT_JSON),
            "summary_csv": str(OUT_DIR / "summary.csv"),
            "dataset_highlow_csv": str(OUT_DIR / "dataset_highlow.csv"),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(
        OUT_DIR / "summary.csv",
        summaries,
        [
            "artifact",
            "status",
            "n",
            "datasets",
            "residual_spearman",
            "pp_high_minus_low_mean",
            "pp_high_minus_low_ci_low",
            "pp_high_minus_low_ci_high",
            "pp_high_minus_low_dataset_min",
            "mmd_high_minus_low_max",
            "within_dataset_shuffle_p",
            "reasons",
        ],
    )
    write_csv(
        OUT_DIR / "dataset_highlow.csv",
        dataset_rows_all,
        ["artifact", "dataset", "n", "k", "pp_high_minus_low", "mmd_high_minus_low"],
    )

    lines = [
        "# LatentFM GWT Residualized MMD-Matched Gate",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only over already materialized GWT preflight rows.",
        "- Within-dataset centered artifact/pp values; pp is additionally residualized against MMD proxy.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C query, or GPU.",
        "",
        "## Summary",
        "",
        "| artifact | status | n | datasets | residual rho | pp high-low | pp CI95 | dataset min | MMD high-low max | shuffle p | reasons |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for s in summaries:
        lines.append(
            "| `{artifact}` | `{status}` | {n} | {datasets} | {rho} | {pp} | [{lo}, {hi}] | {dsmin} | {mmd} | {p} | `{reasons}` |".format(
                artifact=s["artifact"],
                status=s["status"],
                n=s["n"],
                datasets=s["datasets"],
                rho=fmt(s["residual_spearman"]),
                pp=fmt(s["pp_high_minus_low_mean"]),
                lo=fmt(s["pp_high_minus_low_ci_low"]),
                hi=fmt(s["pp_high_minus_low_ci_high"]),
                dsmin=fmt(s["pp_high_minus_low_dataset_min"]),
                mmd=fmt(s["mmd_high_minus_low_max"]),
                p=fmt(s["within_dataset_shuffle_p"]),
                reasons=s["reasons"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- pass candidates: `{pass_candidates}`",
            "- A pass would only authorize external review and a predeclared design, not GPU training.",
            "- If all candidates fail, close GWT reliability as an unsafe external stabilizer for the current route.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- summary: `{OUT_DIR / 'summary.csv'}`",
            f"- dataset high-low rows: `{OUT_DIR / 'dataset_highlow.csv'}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "pass_candidates": pass_candidates, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
