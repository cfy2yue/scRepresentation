#!/usr/bin/env python3
"""Single-source diagnostic preview gate for Frangieh ORCS response artifacts.

Joins author MAGeCK response/fitness artifacts to frozen xverse_8k Frangieh
test_single metrics. This is diagnostic only because it is one dataset/source
and uses held-out test metrics for association. No training, inference,
canonical multi selection, Track C query, or GPU.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
ARTIFACT_CSV = ROOT / "reports/frangieh_orcs_response_artifacts_20260627/frangieh_orcs_response_artifacts.csv"
OUT_DIR = ROOT / "reports/frangieh_orcs_response_preview_gate_20260627"
OUT_JOINED = OUT_DIR / "frangieh_orcs_response_preview_joined_rows.csv"
OUT_SUMMARY = OUT_DIR / "frangieh_orcs_response_preview_summary.csv"
OUT_JSON = ROOT / "reports/latentfm_frangieh_orcs_response_preview_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_FRANGIEH_ORCS_RESPONSE_PREVIEW_GATE_20260627.md"

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
    "seed42": ANCHOR_ROOT
    / "posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
    "seed43": REPLICATE_ROOT
    / "posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
}


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


def load_eval_rows() -> list[dict[str, Any]]:
    rows = []
    for seed, path in INPUTS.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("groups", {}).get("test_single", {}).get("condition_metrics") or []:
            if row.get("dataset") != "Frangieh":
                continue
            pp = fnum(row.get("pearson_pert"))
            mmd = fnum(row.get("test_mmd_clamped"))
            if pp is None or mmd is None:
                continue
            rows.append({"seed": seed, "dataset": "Frangieh", "condition": row["condition"], "pearson_pert": pp, "test_mmd_clamped": mmd})
    return rows


def load_artifacts() -> list[dict[str, Any]]:
    rows = []
    with ARTIFACT_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "test_single":
                continue
            value = fnum(row.get("artifact_value"))
            if value is None:
                continue
            rows.append(
                {
                    "condition": row["condition"],
                    "artifact": row["artifact"],
                    "artifact_value": value,
                    "artifact_role": row["artifact_role"],
                    "response_context": row["response_context"],
                    "raw_column": row["raw_column"],
                }
            )
    return rows


def join_rows(eval_rows: list[dict[str, Any]], artifact_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_gene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in artifact_rows:
        by_gene[row["condition"]].append(row)
    joined = []
    for erow in eval_rows:
        for arow in by_gene.get(erow["condition"], []):
            joined.append({**erow, **arow})
    return joined


def perm_p(rows: list[dict[str, Any]], direction: float, observed: float, label: str, n_perm: int = 5000) -> float | None:
    if len(rows) < 20:
        return None
    rng = random.Random(stable_seed(label))
    vals = [float(r["artifact_value"]) for r in rows]
    pp = [float(r["pearson_pert"]) for r in rows]
    ge = 0
    for _ in range(n_perm):
        sv = list(vals)
        rng.shuffle(sv)
        rho = spearman(sv, pp)
        if rho is not None and direction * rho >= observed - 1e-12:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def summarize_artifact(rows: list[dict[str, Any]], artifact: str) -> dict[str, Any]:
    arows = [r for r in rows if r["artifact"] == artifact]
    role = sorted({r["artifact_role"] for r in arows})[0]
    context = sorted({r["response_context"] for r in arows})[0]
    raw = sorted({r["raw_column"] for r in arows})[0]
    seed42 = [r for r in arows if r["seed"] == "seed42"]
    rho0 = spearman([float(r["artifact_value"]) for r in seed42], [float(r["pearson_pert"]) for r in seed42]) or 0.0
    direction = 1.0 if rho0 >= 0 else -1.0
    signed = []
    mmd_abs = []
    pvals = []
    for seed in ("seed42", "seed43"):
        sub = [r for r in arows if r["seed"] == seed]
        rho = spearman([float(r["artifact_value"]) for r in sub], [float(r["pearson_pert"]) for r in sub])
        srho = None if rho is None else direction * rho
        if srho is not None:
            signed.append(srho)
            pvals.append(perm_p(sub, direction, srho, f"{artifact}:{seed}") or 1.0)
        mrho = spearman([float(r["artifact_value"]) for r in sub], [float(r["test_mmd_clamped"]) for r in sub])
        if mrho is not None:
            mmd_abs.append(abs(mrho))
    reasons = ["single_source_test_metric_preview_only"]
    if not signed or min(signed) < 0.35:
        reasons.append("min_signed_rho_below_0p35")
    if not pvals or max(pvals) > 0.05:
        reasons.append("shuffle_p_above_0p05")
    if mmd_abs and signed and max(mmd_abs) >= min(signed):
        reasons.append("mmd_correlation_not_weaker_than_pp_signal")
    status = "single_source_preview_signal_no_gpu" if role == "response_candidate" and len(reasons) == 1 else "diagnostic_only_no_gpu"
    return {
        "artifact": artifact,
        "artifact_role": role,
        "response_context": context,
        "raw_column": raw,
        "n_per_seed": len(seed42),
        "direction": direction,
        "min_signed_rho": min(signed) if signed else None,
        "mean_signed_rho": mean(signed) if signed else None,
        "max_shuffle_p": max(pvals) if pvals else None,
        "max_abs_rho_mmd": max(mmd_abs) if mmd_abs else None,
        "status": status,
        "reasons": ";".join(reasons),
    }


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
    eval_rows = load_eval_rows()
    artifact_rows = load_artifacts()
    joined = join_rows(eval_rows, artifact_rows)
    artifacts = sorted({r["artifact"] for r in joined})
    summaries = [summarize_artifact(joined, artifact) for artifact in artifacts]
    summaries.sort(key=lambda r: (0 if r["artifact_role"] == "response_candidate" else 1, -(r["min_signed_rho"] or -999.0)))
    signals = [r["artifact"] for r in summaries if r["status"] == "single_source_preview_signal_no_gpu"]
    status = "frangieh_orcs_response_preview_signal_diagnostic_no_gpu" if signals else "frangieh_orcs_response_preview_fail_no_gpu"
    with OUT_JOINED.open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "dataset", "condition", "pearson_pert", "test_mmd_clamped", "artifact", "artifact_value", "artifact_role", "response_context", "raw_column"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in joined])
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        fields = list(summaries[0].keys()) if summaries else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    payload = {
        "status": status,
        "gpu_authorized": False,
        "eval_rows": len(eval_rows),
        "joined_rows": len(joined),
        "artifacts_tested": len(summaries),
        "single_source_signals": signals,
        "top_summary": summaries[:10],
        "outputs": {"joined_csv": str(OUT_JOINED), "summary_csv": str(OUT_SUMMARY), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Frangieh ORCS Response Preview Gate 2026-06-27",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only single-source diagnostic preview over Frangieh ORCS author response artifacts.",
        "- Uses frozen seed42/43 xverse test_single Frangieh metrics only; no canonical multi, Track C query, training, or inference.",
        "- Single-source and test-metric association means no GPU authorization even if a signal appears.",
        "",
        "## Summary",
        "",
        f"- eval rows: `{len(eval_rows)}`",
        f"- joined rows: `{len(joined)}`",
        f"- artifacts tested: `{len(summaries)}`",
        f"- single-source signals: `{signals}`",
        "",
        "| artifact | role | context | min signed rho | max shuffle p | max abs rho MMD | status | reasons |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in summaries[:16]:
        lines.append(
            f"| `{row['artifact']}` | `{row['artifact_role']}` | `{row['response_context']}` | "
            f"{fmt(row.get('min_signed_rho'))} | {fmt(row.get('max_shuffle_p'))} | {fmt(row.get('max_abs_rho_mmd'))} | "
            f"`{row['status']}` | `{row['reasons']}` |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "This branch is diagnostic/no-GPU. A formal route would need a second source or source-control plus train-only/discovery-confirm selection.",
        "",
        f"- joined rows: `{OUT_JOINED}`",
        f"- summary: `{OUT_SUMMARY}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "signals": signals[:8], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
