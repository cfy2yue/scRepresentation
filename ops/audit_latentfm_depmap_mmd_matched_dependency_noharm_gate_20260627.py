#!/usr/bin/env python3
"""MMD-matched DepMap dependency no-harm gate.

CPU/report-only follow-up to the DepMap residual/MMD gate. This tests whether
the dependency-vs-anchor-difficulty signal survives matching within seed,
dataset, and MMD quantile bins. It does not train, infer, select checkpoints,
read canonical multi for selection, read Track C query, or use GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_ROWS = ROOT / "reports/depmap_24q4_dependency_residual_mmd_gate_20260627/depmap_dependency_residual_rows.csv"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_depmap_mmd_matched_dependency_noharm_gate_20260627.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_DEPMAP_MMD_MATCHED_DEPENDENCY_NOHARM_GATE_20260627.md"
DEFAULT_OUT_DIR = ROOT / "reports/depmap_mmd_matched_dependency_noharm_gate_20260627"


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


def quantile_bins(values: list[float], n_bins: int) -> list[int]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    bins = [0] * len(values)
    for rank, idx in enumerate(order):
        bins[idx] = min(n_bins - 1, int(rank * n_bins / len(values)))
    return bins


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dep = fnum(row.get("dependency_z_within_dataset"))
            pp = fnum(row.get("pearson_pert"))
            mmd = fnum(row.get("test_mmd_clamped"))
            if dep is None or pp is None or mmd is None:
                continue
            rows.append(
                {
                    "seed": row["seed"],
                    "dataset": row["dataset"],
                    "condition": row["condition"],
                    "dependency_z_within_dataset": dep,
                    "pearson_pert": pp,
                    "test_mmd_clamped": mmd,
                }
            )
    return rows


def assign_mmd_bins(rows: list[dict[str, Any]], n_bins: int) -> None:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["seed"], row["dataset"])].append(row)
    for key_rows in by_key.values():
        bins = quantile_bins([float(r["test_mmd_clamped"]) for r in key_rows], n_bins)
        for row, bin_id in zip(key_rows, bins):
            row["mmd_bin"] = bin_id


def matched_rows(rows: list[dict[str, Any]], *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_block: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_block[(row["seed"], row["dataset"], int(row["mmd_bin"]))].append(row)
    selected: list[dict[str, Any]] = []
    dropped_blocks = 0
    for block_rows in by_block.values():
        highs = [r for r in block_rows if float(r["dependency_z_within_dataset"]) > 0]
        lows = [r for r in block_rows if float(r["dependency_z_within_dataset"]) <= 0]
        n = min(len(highs), len(lows))
        if n < 1:
            dropped_blocks += 1
            continue
        rng.shuffle(highs)
        rng.shuffle(lows)
        selected.extend(highs[:n])
        selected.extend(lows[:n])
    return selected, {"matched_rows": len(selected), "dropped_blocks": dropped_blocks, "blocks": len(by_block)}


def summarize(rows: list[dict[str, Any]], *, n_boot: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row)
    rng = random.Random(seed)
    summaries = []
    for seed_name, srows in sorted(by_seed.items()):
        dep = [float(r["dependency_z_within_dataset"]) for r in srows]
        pp = [float(r["pearson_pert"]) for r in srows]
        mmd = [float(r["test_mmd_clamped"]) for r in srows]
        rho_pp = spearman(dep, pp)
        rho_mmd = spearman(dep, mmd)
        signed_rho = None if rho_pp is None else -rho_pp
        lodo = []
        for dataset in sorted({r["dataset"] for r in srows}):
            sub = [r for r in srows if r["dataset"] != dataset]
            rho = spearman(
                [float(r["dependency_z_within_dataset"]) for r in sub],
                [float(r["pearson_pert"]) for r in sub],
            )
            if rho is not None:
                lodo.append(-rho)
        boot = []
        for _ in range(n_boot):
            sample = [srows[rng.randrange(len(srows))] for _ in range(len(srows))]
            rho = spearman(
                [float(r["dependency_z_within_dataset"]) for r in sample],
                [float(r["pearson_pert"]) for r in sample],
            )
            if rho is not None:
                boot.append(-rho)
        boot.sort()
        ci_low = boot[int(0.025 * (len(boot) - 1))] if boot else None
        ci_high = boot[int(0.975 * (len(boot) - 1))] if boot else None
        high = [r for r in srows if float(r["dependency_z_within_dataset"]) > 0]
        low = [r for r in srows if float(r["dependency_z_within_dataset"]) <= 0]
        summaries.append(
            {
                "seed": seed_name,
                "n": len(srows),
                "datasets": len({r["dataset"] for r in srows}),
                "blocks": len({(r["dataset"], r["mmd_bin"]) for r in srows}),
                "signed_rho_dependency_pp": signed_rho,
                "signed_rho_ci_low": ci_low,
                "signed_rho_ci_high": ci_high,
                "rho_dependency_mmd": rho_mmd,
                "lodo_min_signed_rho": min(lodo) if lodo else None,
                "high_minus_low_pp": mean([float(r["pearson_pert"]) for r in high]) - mean([float(r["pearson_pert"]) for r in low]) if high and low else None,
                "high_minus_low_mmd": mean([float(r["test_mmd_clamped"]) for r in high]) - mean([float(r["test_mmd_clamped"]) for r in low]) if high and low else None,
            }
        )
    aggregate = {
        "matched_rows": len(rows),
        "matched_seeds": sorted(by_seed),
        "matched_datasets": sorted({r["dataset"] for r in rows}),
    }
    return summaries, aggregate


def permutation_p(rows: list[dict[str, Any]], *, n_perm: int, seed: int) -> dict[str, float | None]:
    rng = random.Random(seed)
    out = {}
    for seed_name in sorted({r["seed"] for r in rows}):
        srows = [r for r in rows if r["seed"] == seed_name]
        obs = spearman([float(r["dependency_z_within_dataset"]) for r in srows], [float(r["pearson_pert"]) for r in srows])
        if obs is None:
            out[seed_name] = None
            continue
        obs_signed = -obs
        ge = 1
        blocks: dict[tuple[str, int], list[int]] = defaultdict(list)
        for idx, row in enumerate(srows):
            blocks[(row["dataset"], int(row["mmd_bin"]))].append(idx)
        dep = [float(r["dependency_z_within_dataset"]) for r in srows]
        pp = [float(r["pearson_pert"]) for r in srows]
        for _ in range(n_perm):
            shuf = dep[:]
            for idxs in blocks.values():
                vals = [shuf[i] for i in idxs]
                rng.shuffle(vals)
                for i, val in zip(idxs, vals):
                    shuf[i] = val
            rho = spearman(shuf, pp)
            if rho is not None and -rho >= obs_signed:
                ge += 1
        out[seed_name] = ge / (n_perm + 1)
    return out


def decide(summaries: list[dict[str, Any]], pvals: dict[str, float | None]) -> tuple[str, list[str]]:
    reasons = []
    if {r["seed"] for r in summaries} != {"seed42", "seed43"}:
        reasons.append("missing_seed42_or_seed43")
    for row in summaries:
        seed = row["seed"]
        if int(row["n"]) < 80:
            reasons.append(f"{seed}_matched_n_below_80")
        if int(row["datasets"]) < 5:
            reasons.append(f"{seed}_datasets_below_5")
        if (row["signed_rho_dependency_pp"] or 0.0) < 0.15:
            reasons.append(f"{seed}_signed_rho_below_0p15")
        if (row["signed_rho_ci_low"] or 0.0) <= 0.0:
            reasons.append(f"{seed}_bootstrap_ci_low_not_positive")
        if (pvals.get(seed) or 1.0) > 0.05:
            reasons.append(f"{seed}_within_mmdbin_shuffle_p_gt_0p05")
        if (row["lodo_min_signed_rho"] or 0.0) < 0.05:
            reasons.append(f"{seed}_lodo_min_below_0p05")
        if abs(row["rho_dependency_mmd"] or 0.0) > 0.10:
            reasons.append(f"{seed}_dependency_mmd_corr_gt_0p10_after_matching")
        if abs(row["high_minus_low_mmd"] or 0.0) > 0.001:
            reasons.append(f"{seed}_high_low_mmd_delta_gt_0p001")
    status = "depmap_mmd_matched_dependency_noharm_gate_pass_gpu_design_allowed_no_gpu" if not reasons else "depmap_mmd_matched_dependency_noharm_gate_fail_no_gpu"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--mmd-bins", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.rows)
    assign_mmd_bins(rows, args.mmd_bins)
    matched, match_info = matched_rows(rows, seed=args.seed)
    summaries, aggregate = summarize(matched, n_boot=args.n_boot, seed=args.seed)
    pvals = permutation_p(matched, n_perm=args.n_perm, seed=args.seed)
    status, reasons = decide(summaries, pvals)

    matched_csv = args.out_dir / "depmap_mmd_matched_dependency_rows.csv"
    summary_csv = args.out_dir / "depmap_mmd_matched_dependency_summary.csv"
    row_fields = ["seed", "dataset", "condition", "dependency_z_within_dataset", "pearson_pert", "test_mmd_clamped", "mmd_bin"]
    with matched_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields)
        writer.writeheader()
        for row in matched:
            writer.writerow({k: row.get(k, "") for k in row_fields})
    summary_fields = [
        "seed",
        "n",
        "datasets",
        "blocks",
        "signed_rho_dependency_pp",
        "signed_rho_ci_low",
        "signed_rho_ci_high",
        "within_mmdbin_shuffle_p",
        "rho_dependency_mmd",
        "lodo_min_signed_rho",
        "high_minus_low_pp",
        "high_minus_low_mmd",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in summaries:
            full = dict(row)
            full["within_mmdbin_shuffle_p"] = pvals.get(row["seed"])
            writer.writerow({k: full.get(k, "") for k in summary_fields})

    payload = {
        "status": status,
        "gpu_authorized": False,
        "reasons": reasons,
        "input": str(args.rows),
        "match_info": match_info,
        "aggregate": aggregate,
        "summaries": [{**row, "within_mmdbin_shuffle_p": pvals.get(row["seed"])} for row in summaries],
        "outputs": {"json": str(args.out_json), "markdown": str(args.out_md), "matched_rows": str(matched_csv), "summary": str(summary_csv)},
    }
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM DepMap MMD-Matched Dependency No-Harm Gate 2026-06-27",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only gate over existing DepMap residual/MMD rows.",
        "- Matches rows within seed, dataset, and MMD quantile bin before testing dependency association.",
        "- Does not train, infer, select checkpoints, read canonical multi for selection, read Track C query, or use GPU.",
        "",
        "## Matching",
        "",
        f"- input rows: `{len(rows)}`",
        f"- matched rows: `{match_info['matched_rows']}`",
        f"- blocks: `{match_info['blocks']}`",
        f"- dropped one-sided blocks: `{match_info['dropped_blocks']}`",
        f"- reasons: `{', '.join(reasons) or 'none'}`",
        "",
        "| seed | n | datasets | signed rho dep~pp | CI low | shuffle p | rho dep~MMD | LODO min | high-low pp | high-low MMD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['seed']}` | {row['n']} | {row['datasets']} | "
            f"{fmt(row['signed_rho_dependency_pp'])} | {fmt(row['signed_rho_ci_low'])} | "
            f"{fmt(pvals.get(row['seed']))} | {fmt(row['rho_dependency_mmd'])} | "
            f"{fmt(row['lodo_min_signed_rho'])} | {fmt(row['high_minus_low_pp'])} | "
            f"{fmt(row['high_minus_low_mmd'])} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "A pass would only authorize external audit and dependency-aware GPU design with a predeclared candidate-vs-anchor no-harm gate.",
        "A fail closes the current DepMap dependency sidecar as a GPU-enabling source.",
        "",
        f"- JSON: `{args.out_json}`",
        f"- matched rows: `{matched_csv}`",
        f"- summary CSV: `{summary_csv}`",
        "",
    ]
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "reasons": reasons, "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
