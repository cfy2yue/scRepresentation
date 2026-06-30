#!/usr/bin/env python3
"""CPU residual/MMD gate for the DepMap dependency signal.

This is a stricter follow-up to the DepMap association gate. It treats the
matched held-out single-gene rows as one evidence set per seed, normalizes the
dependency score within dataset, and tests whether the dependency-vs-anchor
failure signal survives MMD control and leave-one-dataset checks.

No training, inference, checkpoint selection, canonical multi selection, Track C
query, or GPU.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
JOINED = ROOT / "reports/depmap_24q4_dependency_gate_20260627/depmap_24q4_dependency_gate_joined_rows.csv"
OUT_DIR = ROOT / "reports/depmap_24q4_dependency_residual_mmd_gate_20260627"
OUT_ROWS = OUT_DIR / "depmap_dependency_residual_rows.csv"
OUT_SUMMARY = OUT_DIR / "depmap_dependency_residual_summary.csv"
OUT_JSON = ROOT / "reports/latentfm_depmap_dependency_residual_mmd_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_DEPMAP_DEPENDENCY_RESIDUAL_MMD_GATE_20260627.md"


def fnum(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


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


def residualize(y: list[float], x: list[float]) -> list[float]:
    """Return y residual after ordinary least squares on intercept + x."""
    if len(y) != len(x) or not y:
        return []
    mx, my = mean(x), mean(y)
    vx = sum((v - mx) ** 2 for v in x)
    if vx <= 0:
        return [v - my for v in y]
    beta = sum((a - mx) * (b - my) for a, b in zip(x, y)) / vx
    alpha = my - beta * mx
    return [yy - (alpha + beta * xx) for yy, xx in zip(y, x)]


def load_unique_rows() -> list[dict[str, Any]]:
    # Curie audit: group rows are duplicated evidence. Use only unique test_single rows.
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    with JOINED.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("group") != "test_single":
                continue
            key = (row["seed"], row["dataset"], row["condition"])
            dep = fnum(row.get("artifact_value"))
            pp = fnum(row.get("pearson_pert"))
            mmd = fnum(row.get("test_mmd_clamped"))
            if dep is None or pp is None or mmd is None:
                continue
            out[key] = {
                "seed": row["seed"],
                "dataset": row["dataset"],
                "condition": row["condition"],
                "split": row.get("split", ""),
                "dependency_score": dep,
                "pearson_pert": pp,
                "test_mmd_clamped": mmd,
                "failure_flag": pp < 0.05 or mmd > 0.05,
            }
    rows = list(out.values())
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    for ds_rows in by_dataset.values():
        vals = [float(r["dependency_score"]) for r in ds_rows]
        mu = mean(vals)
        sd = pstdev(vals) if len(vals) > 1 else 0.0
        if sd <= 0:
            sd = 1.0
        sorted_vals = sorted(vals)
        q1 = sorted_vals[int((len(sorted_vals) - 1) / 3)]
        q2 = sorted_vals[int(2 * (len(sorted_vals) - 1) / 3)]
        for row in ds_rows:
            z = (float(row["dependency_score"]) - mu) / sd
            row["dependency_z_within_dataset"] = z
            if float(row["dependency_score"]) <= q1:
                row["dependency_tercile_within_dataset"] = "low"
            elif float(row["dependency_score"]) <= q2:
                row["dependency_tercile_within_dataset"] = "mid"
            else:
                row["dependency_tercile_within_dataset"] = "high"
    return rows


def summarize_seed(rows: list[dict[str, Any]], seed: str) -> dict[str, Any]:
    srows = [r for r in rows if r["seed"] == seed]
    dep = [float(r["dependency_z_within_dataset"]) for r in srows]
    pp = [float(r["pearson_pert"]) for r in srows]
    mmd = [float(r["test_mmd_clamped"]) for r in srows]
    pp_resid = residualize(pp, mmd)
    rho_pp = spearman(dep, pp)
    rho_mmd = spearman(dep, mmd)
    rho_resid = spearman(dep, pp_resid)
    # Higher dependency should mean lower pp, so signed signal is -rho.
    signed_rho_pp = None if rho_pp is None else -rho_pp
    signed_rho_resid = None if rho_resid is None else -rho_resid
    lodo_signed = []
    lodo_details = []
    for ds in sorted({r["dataset"] for r in srows}):
        sub = [r for r in srows if r["dataset"] != ds]
        sub_dep = [float(r["dependency_z_within_dataset"]) for r in sub]
        sub_pp = [float(r["pearson_pert"]) for r in sub]
        sub_mmd = [float(r["test_mmd_clamped"]) for r in sub]
        sub_resid = residualize(sub_pp, sub_mmd)
        rho = spearman(sub_dep, sub_resid)
        if rho is not None:
            signed = -rho
            lodo_signed.append(signed)
            lodo_details.append(f"{ds}:{signed:+.4f}")
    high = [r for r in srows if r["dependency_tercile_within_dataset"] == "high"]
    lowmid = [r for r in srows if r["dependency_tercile_within_dataset"] != "high"]
    high_pp = [float(r["pearson_pert"]) for r in high]
    lowmid_pp = [float(r["pearson_pert"]) for r in lowmid]
    high_mmd = [float(r["test_mmd_clamped"]) for r in high]
    lowmid_mmd = [float(r["test_mmd_clamped"]) for r in lowmid]
    return {
        "seed": seed,
        "n": len(srows),
        "datasets": len({r["dataset"] for r in srows}),
        "rho_dependency_z_vs_pp": rho_pp,
        "signed_rho_dependency_z_vs_pp": signed_rho_pp,
        "rho_dependency_z_vs_mmd": rho_mmd,
        "rho_dependency_z_vs_pp_residual_mmd": rho_resid,
        "signed_rho_dependency_z_vs_pp_residual_mmd": signed_rho_resid,
        "lodo_min_signed_residual_rho": min(lodo_signed) if lodo_signed else None,
        "lodo_signed_residual_rhos": ";".join(lodo_details),
        "high_tercile_n": len(high),
        "high_tercile_pp_mean": mean(high_pp) if high_pp else None,
        "lowmid_pp_mean": mean(lowmid_pp) if lowmid_pp else None,
        "high_minus_lowmid_pp": (mean(high_pp) - mean(lowmid_pp)) if high_pp and lowmid_pp else None,
        "high_tercile_mmd_mean": mean(high_mmd) if high_mmd else None,
        "lowmid_mmd_mean": mean(lowmid_mmd) if lowmid_mmd else None,
        "high_minus_lowmid_mmd": (mean(high_mmd) - mean(lowmid_mmd)) if high_mmd and lowmid_mmd else None,
    }


def decide(summaries: list[dict[str, Any]]) -> tuple[str, list[str]]:
    reasons = []
    for row in summaries:
        seed = row["seed"]
        if int(row["n"]) < 100:
            reasons.append(f"{seed}_n_below_100")
        if int(row["datasets"]) < 5:
            reasons.append(f"{seed}_datasets_below_5")
        if (row["signed_rho_dependency_z_vs_pp_residual_mmd"] or 0.0) < 0.15:
            reasons.append(f"{seed}_mmd_residual_signed_rho_below_0p15")
        if (row["lodo_min_signed_residual_rho"] or 0.0) < 0.05:
            reasons.append(f"{seed}_lodo_min_signed_residual_rho_below_0p05")
        if (row["high_minus_lowmid_mmd"] or 0.0) > 0.01 and abs(row["rho_dependency_z_vs_mmd"] or 0.0) >= abs(row["rho_dependency_z_vs_pp_residual_mmd"] or 0.0):
            reasons.append(f"{seed}_dependency_signal_mmd_confounded")
    status = "depmap_dependency_residual_mmd_gate_pass_gpu_design_allowed" if not reasons else "depmap_dependency_residual_mmd_gate_fail_no_gpu"
    return status, reasons


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
    rows = load_unique_rows()
    summaries = [summarize_seed(rows, seed) for seed in sorted({r["seed"] for r in rows})]
    status, reasons = decide(summaries)

    row_fields = [
        "seed",
        "dataset",
        "condition",
        "split",
        "dependency_score",
        "dependency_z_within_dataset",
        "dependency_tercile_within_dataset",
        "pearson_pert",
        "test_mmd_clamped",
        "failure_flag",
    ]
    with OUT_ROWS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in row_fields})

    summary_fields = [
        "seed",
        "n",
        "datasets",
        "rho_dependency_z_vs_pp",
        "signed_rho_dependency_z_vs_pp",
        "rho_dependency_z_vs_mmd",
        "rho_dependency_z_vs_pp_residual_mmd",
        "signed_rho_dependency_z_vs_pp_residual_mmd",
        "lodo_min_signed_residual_rho",
        "lodo_signed_residual_rhos",
        "high_tercile_n",
        "high_tercile_pp_mean",
        "lowmid_pp_mean",
        "high_minus_lowmid_pp",
        "high_tercile_mmd_mean",
        "lowmid_mmd_mean",
        "high_minus_lowmid_mmd",
    ]
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({k: row.get(k, "") for k in summary_fields})

    payload = {
        "status": status,
        "gpu_authorized": False,
        "reasons": reasons,
        "input": str(JOINED),
        "unique_rows": len(rows),
        "summaries": summaries,
        "outputs": {
            "rows": str(OUT_ROWS),
            "summary": str(OUT_SUMMARY),
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
        },
        "decision": "Pass allows GPU design work only, not immediate promotion or final claims.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM DepMap Dependency Residual/MMD Gate 2026-06-27",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only follow-up gate requested by external audit.",
        "- Uses unique `test_single` matched rows only; `family_gene/test_all` duplicates are not treated as independent evidence.",
        "- Dependency is normalized within dataset before association tests.",
        "- Does not train, infer, select checkpoints, read canonical multi for selection, read Track C query, or use GPU.",
        "",
        "## Summary",
        "",
        f"- unique rows: `{len(rows)}`",
        f"- reasons: `{', '.join(reasons) or 'none'}`",
        "",
        "| seed | n | datasets | signed rho dep~pp | rho dep~MMD | signed rho dep~pp residual(MMD) | LODO min signed residual rho | high-lowmid pp | high-lowmid MMD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['seed']}` | {row['n']} | {row['datasets']} | "
            f"{fmt(row['signed_rho_dependency_z_vs_pp'])} | {fmt(row['rho_dependency_z_vs_mmd'])} | "
            f"{fmt(row['signed_rho_dependency_z_vs_pp_residual_mmd'])} | {fmt(row['lodo_min_signed_residual_rho'])} | "
            f"{fmt(row['high_minus_lowmid_pp'])} | {fmt(row['high_minus_lowmid_mmd'])} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "A pass only authorizes dependency-aware GPU design with a predeclared candidate-vs-anchor no-harm gate. It does not authorize default-model replacement or manuscript-level claims.",
        "",
        f"- rows CSV: `{OUT_ROWS}`",
        f"- summary CSV: `{OUT_SUMMARY}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "reasons": reasons, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
