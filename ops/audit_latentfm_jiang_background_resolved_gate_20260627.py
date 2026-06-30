#!/usr/bin/env python3
"""Gate Jiang author-DE artifacts against background-resolved anchor outcomes.

CPU/report-only. Requires frozen seed42 and seed43 background-resolved posthoc
JSONs from `model.latent.eval_background_groups`. It does not train, infer,
select checkpoints, read canonical multi for selection, read Track C query, or
use GPU.
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
SEED42_JSON = ROOT / "reports/jiang_background_resolved_anchor_seed42_20260627.json"
SEED43_JSON = ROOT / "reports/jiang_background_resolved_anchor_seed43_20260627.json"
ARTIFACT_CSV = ROOT / "reports/jiang_author_de_artifacts_20260627/jiang_author_de_background_artifacts.csv"
OUT_DIR = ROOT / "reports/jiang_background_resolved_gate_20260627"
OUT_ROWS = OUT_DIR / "jiang_background_resolved_gate_joined_rows.csv"
OUT_SUMMARY = OUT_DIR / "jiang_background_resolved_gate_summary.csv"
OUT_JSON = ROOT / "reports/latentfm_jiang_background_resolved_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_JIANG_BACKGROUND_RESOLVED_GATE_20260627.md"

ARTIFACT_METRICS = [
    "mean_abs_log2fc",
    "mean_signed_log2fc",
    "mean_abs_beta",
    "mean_signed_beta",
    "mean_abs_lfc_neglog10p",
    "sig_frac_p05_abs005",
    "valid_gene_count",
]

BONFERRONI_ALPHA = 0.05 / len(ARTIFACT_METRICS)


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16) % (2**32)


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
    mx = mean(x)
    my = mean(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0 or vy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(rankdata(x), rankdata(y))


def load_artifacts() -> dict[tuple[str, str, str], dict[str, float]]:
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    with ARTIFACT_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["condition"], row["cell_background"])
            vals = {}
            for metric in ARTIFACT_METRICS:
                val = fnum(row.get(metric))
                if val is not None:
                    vals[metric] = val
            if vals:
                out[key] = vals
    return out


def load_eval(path: Path, seed: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in payload.get("condition_background_metrics") or []:
        if not str(row.get("dataset", "")).startswith("Jiang_"):
            continue
        rows.append({"seed": seed, **row})
    return rows


def permutation_p(rows: list[dict[str, Any]], *, metric: str, target: str, seed: int, n_perm: int = 1000) -> float | None:
    x = [float(r[metric]) for r in rows]
    y = [float(r[target]) for r in rows]
    obs = spearman(x, y)
    if obs is None:
        return None
    rng = random.Random(seed)
    blocks: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        blocks[(str(row["dataset"]), str(row["cell_background"]))].append(i)
    ge = 1
    for _ in range(n_perm):
        shuf = x[:]
        for idxs in blocks.values():
            vals = [shuf[i] for i in idxs]
            rng.shuffle(vals)
            for i, val in zip(idxs, vals):
                shuf[i] = val
        rho = spearman(shuf, y)
        if rho is not None and abs(rho) >= abs(obs):
            ge += 1
    return ge / (n_perm + 1)


def join_rows(artifacts: dict[tuple[str, str, str], dict[str, float]], eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for erow in eval_rows:
        key = (str(erow["dataset"]), str(erow["condition"]), str(erow["cell_background"]))
        vals = artifacts.get(key)
        if not vals:
            continue
        for metric, val in vals.items():
            pp = fnum(erow.get("pearson_pert"))
            direct = fnum(erow.get("direct_pearson"))
            ctrl = fnum(erow.get("pearson_ctrl"))
            mmd = fnum(erow.get("test_mmd_clamped"))
            if pp is None or mmd is None:
                continue
            out.append(
                {
                    "seed": erow["seed"],
                    "dataset": erow["dataset"],
                    "condition": erow["condition"],
                    "cell_background": erow["cell_background"],
                    "artifact_metric": metric,
                    "artifact_value": val,
                    "direct_pearson": direct,
                    "pearson_ctrl": ctrl,
                    "pearson_pert": pp,
                    "test_mmd_clamped": mmd,
                    "n_src_bg": erow.get("n_src_bg"),
                    "n_gt_bg": erow.get("n_gt_bg"),
                    "n_src_eval": erow.get("n_src_eval"),
                    "n_gt_eval": erow.get("n_gt_eval"),
                    "source_pool_mode": erow.get("source_pool_mode"),
                }
            )
    return out


def summarize(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_key[(row["artifact_metric"], row["seed"])].append(row)
    for (metric, seed), rows in sorted(by_key.items()):
        x = [float(r["artifact_value"]) for r in rows]
        pp = [float(r["pearson_pert"]) for r in rows]
        direct_rows = [r for r in rows if fnum(r.get("direct_pearson")) is not None]
        ctrl_rows = [r for r in rows if fnum(r.get("pearson_ctrl")) is not None]
        mmd = [float(r["test_mmd_clamped"]) for r in rows]
        rho_pp = spearman(x, pp)
        rho_direct = (
            spearman([float(r["artifact_value"]) for r in direct_rows], [float(r["direct_pearson"]) for r in direct_rows])
            if direct_rows
            else None
        )
        rho_ctrl = (
            spearman([float(r["artifact_value"]) for r in ctrl_rows], [float(r["pearson_ctrl"]) for r in ctrl_rows])
            if ctrl_rows
            else None
        )
        rho_mmd = spearman(x, mmd)
        # Direction is fixed in advance: larger absolute author-DE response
        # should associate with higher frozen-anchor perturbation agreement.
        # Do not let each seed choose its own sign.
        sign = 1.0
        datasets = sorted({r["dataset"] for r in rows})
        bgs = sorted({r["cell_background"] for r in rows})
        conditions = sorted({r["condition"] for r in rows})
        lodo = []
        for ds in datasets:
            sub = [r for r in rows if r["dataset"] != ds]
            val = spearman([float(r["artifact_value"]) for r in sub], [float(r["pearson_pert"]) for r in sub])
            if val is not None:
                lodo.append(sign * val)
        lobo = []
        for bg in bgs:
            sub = [r for r in rows if r["cell_background"] != bg]
            val = spearman([float(r["artifact_value"]) for r in sub], [float(r["pearson_pert"]) for r in sub])
            if val is not None:
                lobo.append(sign * val)
        loco = []
        for condition in conditions:
            sub = [r for r in rows if r["condition"] != condition]
            val = spearman([float(r["artifact_value"]) for r in sub], [float(r["pearson_pert"]) for r in sub])
            if val is not None:
                loco.append(sign * val)
        out.append(
            {
                "artifact_metric": metric,
                "seed": seed,
                "n": len(rows),
                "datasets": len(datasets),
                "backgrounds": len(bgs),
                "signed_rho_pp": None if rho_pp is None else sign * rho_pp,
                "raw_rho_pp": rho_pp,
                "rho_direct": rho_direct,
                "rho_ctrl": rho_ctrl,
                "rho_mmd": rho_mmd,
                "shuffle_p_abs": permutation_p(rows, metric="artifact_value", target="pearson_pert", seed=stable_seed(metric, seed)),
                "lodo_min_signed_rho": min(lodo) if lodo else None,
                "lobo_min_signed_rho": min(lobo) if lobo else None,
                "loco_min_signed_rho": min(loco) if loco else None,
            }
        )
    return out


def decide(summary: list[dict[str, Any]]) -> tuple[str, list[str], list[dict[str, Any]]]:
    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary:
        by_metric[row["artifact_metric"]].append(row)
    candidates = []
    for metric, rows in by_metric.items():
        seeds = {r["seed"] for r in rows}
        if not {"seed42", "seed43"}.issubset(seeds):
            continue
        usable = [r for r in rows if r["n"] >= 300 and r["datasets"] >= 5 and r["backgrounds"] >= 6]
        if len(usable) < 2:
            continue
        min_signed = min(float(r["signed_rho_pp"]) for r in usable if r["signed_rho_pp"] is not None)
        min_raw = min(float(r["raw_rho_pp"]) for r in usable if r["raw_rho_pp"] is not None)
        max_p = max(float(r["shuffle_p_abs"]) for r in usable if r["shuffle_p_abs"] is not None)
        min_lodo = min(float(r["lodo_min_signed_rho"]) for r in usable if r["lodo_min_signed_rho"] is not None)
        min_lobo = min(float(r["lobo_min_signed_rho"]) for r in usable if r["lobo_min_signed_rho"] is not None)
        min_loco = min(float(r["loco_min_signed_rho"]) for r in usable if r["loco_min_signed_rho"] is not None)
        max_abs_mmd = max(abs(float(r["rho_mmd"])) for r in usable if r["rho_mmd"] is not None)
        min_direct = min(float(r["rho_direct"]) for r in usable if r["rho_direct"] is not None)
        min_ctrl = min(float(r["rho_ctrl"]) for r in usable if r["rho_ctrl"] is not None)
        passed = (
            min_signed >= 0.15
            and min_raw >= 0.15
            and max_p <= BONFERRONI_ALPHA
            and min_lodo >= 0.05
            and min_lobo >= 0.05
            and min_loco >= 0.0
            and max_abs_mmd <= 0.20
        )
        candidates.append(
            {
                "artifact_metric": metric,
                "min_signed_rho_pp": min_signed,
                "min_raw_rho_pp": min_raw,
                "max_shuffle_p_abs": max_p,
                "min_lodo_signed_rho": min_lodo,
                "min_lobo_signed_rho": min_lobo,
                "min_loco_signed_rho": min_loco,
                "max_abs_rho_mmd": max_abs_mmd,
                "min_rho_direct": min_direct,
                "min_rho_ctrl": min_ctrl,
                "pass_background_resolved_gate": passed,
            }
        )
    candidates.sort(
        key=lambda r: (
            bool(r["pass_background_resolved_gate"]),
            float(r["min_signed_rho_pp"]),
            -float(r["max_shuffle_p_abs"]),
            -float(r["max_abs_rho_mmd"]),
        ),
        reverse=True,
    )
    passing = [c for c in candidates if c["pass_background_resolved_gate"]]
    reasons = []
    if not passing:
        reasons.append("no_background_artifact_passed_cross_seed_shuffle_lodo_lobo_mmd_gate")
    status = "jiang_background_resolved_gate_pass_needs_external_audit_no_gpu" if passing else "jiang_background_resolved_gate_fail_no_gpu"
    return status, reasons, candidates[:10]


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):+.6f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    missing = [str(p) for p in (SEED42_JSON, SEED43_JSON, ARTIFACT_CSV) if not p.exists()]
    if missing:
        payload = {
            "status": "jiang_background_resolved_gate_waiting_for_inputs",
            "gpu_authorized": False,
            "missing_inputs": missing,
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joined = join_rows(load_artifacts(), load_eval(SEED42_JSON, "seed42") + load_eval(SEED43_JSON, "seed43"))
    summary = summarize(joined)
    status, reasons, top = decide(summary)
    fields = [
        "seed",
        "dataset",
        "condition",
        "cell_background",
        "artifact_metric",
        "artifact_value",
        "pearson_pert",
            "test_mmd_clamped",
            "direct_pearson",
            "pearson_ctrl",
            "n_src_bg",
            "n_gt_bg",
            "n_src_eval",
            "n_gt_eval",
            "source_pool_mode",
        ]
    with OUT_ROWS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in fields} for row in joined)
    sfields = [
        "artifact_metric",
        "seed",
        "n",
        "datasets",
        "backgrounds",
        "signed_rho_pp",
        "raw_rho_pp",
        "rho_mmd",
        "rho_direct",
        "rho_ctrl",
        "shuffle_p_abs",
        "lodo_min_signed_rho",
        "lobo_min_signed_rho",
        "loco_min_signed_rho",
    ]
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sfields)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in sfields} for row in summary)
    payload = {
        "status": status,
        "gpu_authorized": False,
        "reasons": reasons,
        "joined_rows": len(joined),
        "summary_rows": len(summary),
        "top_candidates": top,
        "outputs": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "joined_rows": str(OUT_ROWS),
            "summary_csv": str(OUT_SUMMARY),
        },
        "inputs": {"seed42": str(SEED42_JSON), "seed43": str(SEED43_JSON), "artifacts": str(ARTIFACT_CSV)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Jiang Background-Resolved Gate 2026-06-27",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only gate over frozen seed42/43 background-resolved anchor posthoc JSONs.",
        "- Joins only materialized Jiang author-DE background artifacts by `(dataset, condition, cell_background)`.",
        "- Does not train, infer, select checkpoints, read canonical multi for selection, read Track C query, or use GPU.",
        "",
        "## Summary",
        "",
        f"- joined rows: `{len(joined)}`",
        f"- summary rows: `{len(summary)}`",
        f"- shuffle multiplicity threshold: `{BONFERRONI_ALPHA:.6f}` (0.05 / {len(ARTIFACT_METRICS)} metrics)",
        f"- reasons: `{'; '.join(reasons) if reasons else 'none'}`",
        "",
        "| artifact | pass | min raw rho pp | max shuffle p | min LODO | min LOBO | min LOCO | max abs rho MMD | min direct rho | min ctrl rho |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top:
        lines.append(
            f"| `{row['artifact_metric']}` | `{row['pass_background_resolved_gate']}` | "
            f"{fmt(row['min_raw_rho_pp'])} | {fmt(row['max_shuffle_p_abs'])} | "
            f"{fmt(row['min_lodo_signed_rho'])} | {fmt(row['min_lobo_signed_rho'])} | "
            f"{fmt(row['min_loco_signed_rho'])} | {fmt(row['max_abs_rho_mmd'])} | "
            f"{fmt(row['min_rho_direct'])} | {fmt(row['min_rho_ctrl'])} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "A pass only authorizes external audit and a predeclared adapter/no-harm design.",
        "It does not authorize model replacement or manuscript claims.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- joined rows: `{OUT_ROWS}`",
        f"- summary CSV: `{OUT_SUMMARY}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "joined_rows": len(joined), "top": top[:3]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
