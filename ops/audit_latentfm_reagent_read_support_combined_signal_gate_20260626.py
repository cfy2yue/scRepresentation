#!/usr/bin/env python3
"""Combined reagent/read-support signal gate for external artifacts.

Short CPU task. Joins the combined external artifact manifest to completed
train-only/internal row metrics and tests whether artifact values explain
condition-level outcomes across datasets. It does not train, infer, read
checkpoints, read canonical multi, read Track C query, or use GPU.
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
CONFIG = ROOT / "configs/latentfm_reagent_read_support_combined_manifest_20260626.json"
OUT_JSON = ROOT / "reports/latentfm_reagent_read_support_combined_signal_gate_20260626.json"
OUT_MD = ROOT / "reports/LATENTFM_REAGENT_READ_SUPPORT_COMBINED_SIGNAL_GATE_20260626.md"
OUT_CSV = ROOT / "reports/latentfm_reagent_read_support_combined_signal_gate_rows_20260626.csv"
SEED = 20260626
N_PERM = 2000

OUTCOME_FILES = [
    ROOT / "reports/latentfm_condition_exposure_row_bootstrap_rows_20260625.csv",
    ROOT / "reports/latentfm_qc_support_reliability_rows_20260625.csv",
    ROOT / "reports/latentfm_response_program_projection_rows_20260625.csv",
    ROOT / "reports/latentfm_lodo_domain_conflict_rows_20260625.csv",
    ROOT / "reports/latentfm_background_target_actionability_rows_20260625.csv",
    ROOT / "reports/latentfm_truecell_riskrow_complementarity_rows_20260625.csv",
]


def norm(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def to_float(value: Any) -> float | None:
    text = norm(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
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
    if len(xs) < 3:
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def load_outcomes() -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for path in OUTCOME_FILES:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"dataset", "condition"}.issubset(set(reader.fieldnames or [])):
                continue
            for row in reader:
                ds = norm(row.get("dataset"))
                cond = norm(row.get("condition"))
                if not ds or not cond:
                    continue
                rec = rows.setdefault((ds, cond), {"dataset": ds, "condition": cond, "pp_values": [], "mmd_values": []})
                for key in ("cross_pp_diff", "pp_delta", "pp_mean", "truecell_pp_delta_mean"):
                    val = to_float(row.get(key))
                    if val is not None:
                        rec["pp_values"].append(val)
                for key in ("cross_mmd_diff", "mmd_delta", "mmd_mean", "truecell_mmd_delta_mean"):
                    val = to_float(row.get(key))
                    if val is not None:
                        rec["mmd_values"].append(val)
    out = {}
    for key, rec in rows.items():
        if rec["pp_values"]:
            out[key] = {
                "pp_proxy_mean": mean(rec["pp_values"]),
                "mmd_proxy_max": max(rec["mmd_values"]) if rec["mmd_values"] else None,
            }
    return out


def read_source_rows(path: Path, artifact: str, outcomes: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ds = norm(row.get("dataset"))
            cond = norm(row.get("condition"))
            val = to_float(row.get("artifact_value"))
            outcome = outcomes.get((ds, cond))
            if not ds or not cond or val is None or outcome is None:
                continue
            rows.append(
                {
                    "artifact": artifact,
                    "dataset": ds,
                    "condition": cond,
                    "artifact_value": val,
                    "pp_proxy_mean": outcome["pp_proxy_mean"],
                    "mmd_proxy_max": outcome["mmd_proxy_max"],
                    "source_file": str(path),
                }
            )
    return rows


def add_within_dataset_z(rows: list[dict[str, Any]]) -> None:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    for ds_rows in by_dataset.values():
        vals = [float(row["artifact_value"]) for row in ds_rows]
        mu = mean(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals) if vals else 0.0
        sd = math.sqrt(var)
        for row in ds_rows:
            row["artifact_z_within_dataset"] = 0.0 if sd <= 0 else (float(row["artifact_value"]) - mu) / sd


def high_low_dataset_effect(rows: list[dict[str, Any]]) -> tuple[float | None, dict[str, float]]:
    effects = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        ds_rows = sorted([row for row in rows if row["dataset"] == dataset], key=lambda r: r["artifact_value"])
        if len(ds_rows) < 6:
            continue
        k = max(1, len(ds_rows) // 3)
        low = ds_rows[:k]
        high = ds_rows[-k:]
        effects[dataset] = mean([row["pp_proxy_mean"] for row in high]) - mean([row["pp_proxy_mean"] for row in low])
    if not effects:
        return None, effects
    return mean(effects.values()), effects


def permute_high_low(rows: list[dict[str, Any]], observed: float | None, artifact: str) -> float | None:
    if observed is None:
        return None
    rng = random.Random(SEED + sum(ord(c) for c in artifact))
    datasets = sorted({row["dataset"] for row in rows})
    ge = 0
    n = 0
    for _ in range(N_PERM):
        permuted = []
        for dataset in datasets:
            ds_rows = [dict(row) for row in rows if row["dataset"] == dataset]
            pp_vals = [row["pp_proxy_mean"] for row in ds_rows]
            rng.shuffle(pp_vals)
            for row, pp in zip(ds_rows, pp_vals):
                row["pp_proxy_mean"] = pp
            permuted.extend(ds_rows)
        effect, _ = high_low_dataset_effect(permuted)
        if effect is None:
            continue
        n += 1
        if effect >= observed:
            ge += 1
    return None if n == 0 else (ge + 1) / (n + 1)


def summarize_artifact(spec: dict[str, Any], outcomes: dict[tuple[str, str], dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifact = spec["artifact"]
    source_files = [ROOT / p if not Path(p).is_absolute() else Path(p) for p in spec.get("source_files", [])]
    missing = [str(path) for path in source_files if not path.is_file()]
    rows: list[dict[str, Any]] = []
    for path in source_files:
        rows.extend(read_source_rows(path, artifact, outcomes))
    add_within_dataset_z(rows)
    datasets = sorted({row["dataset"] for row in rows})
    xs = [float(row.get("artifact_z_within_dataset", 0.0)) for row in rows]
    ys = [float(row["pp_proxy_mean"]) for row in rows]
    spearman_z = pearson(rank(xs), rank(ys)) if len(rows) >= 3 else None
    pearson_z = pearson(xs, ys)
    high_low_mean, high_low_by_dataset = high_low_dataset_effect(rows)
    p_pos = permute_high_low(rows, high_low_mean, artifact)
    mmd_max = max((row["mmd_proxy_max"] or -999.0) for row in rows) if rows else None
    dataset_min = min(high_low_by_dataset.values()) if high_low_by_dataset else None
    reasons = []
    if missing:
        reasons.append(f"missing_source_files:{len(missing)}")
    if len(datasets) < int(spec.get("minimum_datasets", 2)):
        reasons.append("dataset_count_below_minimum")
    if len(rows) < int(spec.get("minimum_overlap_rows", 20)):
        reasons.append("overlap_rows_below_minimum")
    if len([ds for ds in datasets if len({round(r['artifact_value'], 8) for r in rows if r['dataset'] == ds}) >= 2]) < int(
        spec.get("minimum_varying_datasets", 2)
    ):
        reasons.append("varying_dataset_count_below_minimum")
    if spearman_z is None or spearman_z < 0.30:
        reasons.append("within_dataset_spearman_lt_0p30")
    if high_low_mean is None or high_low_mean <= 0.020:
        reasons.append("high_minus_low_pp_lte_0p020")
    if p_pos is None or p_pos > 0.10:
        reasons.append("dataset_permutation_p_pos_gt_0p10")
    if dataset_min is None or dataset_min < -0.020:
        reasons.append("dataset_high_low_tail_below_minus_0p020")
    if mmd_max is not None and mmd_max > 0.001:
        reasons.append("mmd_proxy_max_above_0p001")
    status = "pass_needs_external_review_no_gpu" if not reasons else "fail_no_gpu"
    summary = {
        "artifact": artifact,
        "status": status,
        "gpu_authorized": False,
        "datasets": len(datasets),
        "dataset_names": datasets,
        "overlap_rows": len(rows),
        "within_dataset_spearman_z_vs_pp": spearman_z,
        "within_dataset_pearson_z_vs_pp": pearson_z,
        "dataset_high_minus_low_pp_mean": high_low_mean,
        "dataset_high_minus_low_pp_min": dataset_min,
        "dataset_permutation_p_pos": p_pos,
        "mmd_proxy_max": mmd_max,
        "missing_source_files": missing,
        "reasons": reasons,
        "high_low_by_dataset": high_low_by_dataset,
    }
    return summary, rows


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):+.6f}"


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    outcomes = load_outcomes()
    summaries = []
    all_rows = []
    for spec in config.get("artifacts", []):
        summary, rows = summarize_artifact(spec, outcomes)
        summaries.append(summary)
        all_rows.extend(rows)

    pass_candidates = [row["artifact"] for row in summaries if row["status"] == "pass_needs_external_review_no_gpu"]
    status = "reagent_read_support_combined_signal_pass_no_gpu" if pass_candidates else "reagent_read_support_combined_signal_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": "combined external artifact signal gate; no checkpoints/canonical multi/Track C query/training/inference/GPU",
        "config": str(CONFIG),
        "pass_candidates": pass_candidates,
        "summaries": summaries,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "artifact",
            "dataset",
            "condition",
            "artifact_value",
            "artifact_z_within_dataset",
            "pp_proxy_mean",
            "mmd_proxy_max",
            "source_file",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    lines = [
        "# LatentFM Reagent Read-Support Combined Signal Gate",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M CST')}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only combined signal gate over extracted external reagent/read/guide-support artifacts.",
        "- Uses completed train-only/internal row metrics as outcome proxies.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "",
        "## Summary",
        "",
        "| artifact | status | datasets | rows | z-spearman | high-low pp mean | high-low pp min | p_pos | MMD max | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['artifact']}` | `{row['status']}` | {row['datasets']} | {row['overlap_rows']} | "
            f"{fmt(row['within_dataset_spearman_z_vs_pp'])} | {fmt(row['dataset_high_minus_low_pp_mean'])} | "
            f"{fmt(row['dataset_high_minus_low_pp_min'])} | {fmt(row['dataset_permutation_p_pos'])} | "
            f"{fmt(row['mmd_proxy_max'])} | {', '.join(row['reasons']) or 'none'} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- pass candidates: `{pass_candidates}`",
        "- A pass here still does not launch GPU automatically; it authorizes external review and a bounded smoke proposal only.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- rows: `{OUT_CSV}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
