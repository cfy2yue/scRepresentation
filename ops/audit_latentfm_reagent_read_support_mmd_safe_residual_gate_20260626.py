#!/usr/bin/env python3
"""MMD-safe residual signal gate for reagent/read-support artifacts.

CPU-only gate over already generated train-only/internal proxy rows. It asks
whether the positive reagent/read-support signal survives after retaining only
rows with MMD proxy <= 0.001.

Does not train, infer, read checkpoints, read canonical multi, read Track C
query, or use GPU.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
ROWS_CSV = REPORTS / "latentfm_reagent_read_support_combined_signal_gate_rows_20260626.csv"
OUT_JSON = REPORTS / "latentfm_reagent_read_support_mmd_safe_residual_gate_20260626.json"
OUT_MD = REPORTS / "LATENTFM_REAGENT_READ_SUPPORT_MMD_SAFE_RESIDUAL_GATE_20260626.md"
OUT_ROWS = REPORTS / "latentfm_reagent_read_support_mmd_safe_residual_gate_rows_20260626.csv"

MMD_SAFE_MAX = 0.001
MIN_TOTAL_ROWS = 20
MIN_DATASETS = 2
MIN_ROWS_PER_DATASET = 5
MIN_DATASET_HIGH_LOW = 0.02
MIN_POOLED_SPEARMAN = 0.20
MAX_PERMUTATION_P_POS = 0.01
BOOTSTRAPS = 1000
SEED = 20260626


def to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs)
    deny = sum((y - my) ** 2 for y in ys)
    if denx <= 0 or deny <= 0:
        return None
    return num / math.sqrt(denx * deny)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rank(xs), rank(ys))


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def load_rows() -> list[dict[str, Any]]:
    with ROWS_CSV.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed = dict(row)
            for key in ["artifact_value", "artifact_z_within_dataset", "pp_proxy_mean", "mmd_proxy_max"]:
                parsed[key] = to_float(parsed.get(key))
            rows.append(parsed)
        return rows


def high_low(rows: list[dict[str, Any]]) -> float | None:
    paired = [
        (row["artifact_z_within_dataset"], row["pp_proxy_mean"])
        for row in rows
        if row.get("artifact_z_within_dataset") is not None and row.get("pp_proxy_mean") is not None
    ]
    if len(paired) < 2:
        return None
    med = median([x[0] for x in paired])
    high = [x[1] for x in paired if x[0] > med]
    low = [x[1] for x in paired if x[0] <= med]
    if not high or not low:
        return None
    return mean(high) - mean(low)


def bootstrap_lower(rows: list[dict[str, Any]], rng: random.Random) -> float | None:
    if len(rows) < 3:
        return None
    values = []
    for _ in range(BOOTSTRAPS):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        val = high_low(sample)
        if val is not None:
            values.append(val)
    return quantile(values, 0.025)


def permutation_p_pos(rows: list[dict[str, Any]], rng: random.Random) -> float | None:
    paired = [
        (row["artifact_z_within_dataset"], row["pp_proxy_mean"])
        for row in rows
        if row.get("artifact_z_within_dataset") is not None and row.get("pp_proxy_mean") is not None
    ]
    if len(paired) < 3:
        return None
    observed = high_low(rows)
    if observed is None:
        return None
    z_vals = [x[0] for x in paired]
    pp_vals = [x[1] for x in paired]
    ge_count = 0
    valid = 0
    for _ in range(BOOTSTRAPS):
        shuffled = pp_vals[:]
        rng.shuffle(shuffled)
        pseudo = [
            {"artifact_z_within_dataset": z, "pp_proxy_mean": pp}
            for z, pp in zip(z_vals, shuffled)
        ]
        val = high_low(pseudo)
        if val is None:
            continue
        valid += 1
        if val >= observed:
            ge_count += 1
    if valid == 0:
        return None
    return (ge_count + 1) / (valid + 1)


def summarize_artifact(artifact: str, rows: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    safe_rows = [
        row for row in rows
        if row.get("mmd_proxy_max") is not None and row["mmd_proxy_max"] <= MMD_SAFE_MAX
    ]
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in safe_rows:
        by_dataset[str(row.get("dataset"))].append(row)
    dataset_summaries = []
    varying_datasets = 0
    for dataset, ds_rows in sorted(by_dataset.items()):
        z_values = [row["artifact_z_within_dataset"] for row in ds_rows if row.get("artifact_z_within_dataset") is not None]
        pp_values = [row["pp_proxy_mean"] for row in ds_rows if row.get("pp_proxy_mean") is not None]
        ds_high_low = high_low(ds_rows)
        has_variation = len(set(z_values)) > 1 and len(ds_rows) >= MIN_ROWS_PER_DATASET
        if has_variation:
            varying_datasets += 1
        dataset_summaries.append(
            {
                "dataset": dataset,
                "safe_rows": len(ds_rows),
                "has_variation": has_variation,
                "pp_high_minus_low": ds_high_low,
                "spearman": spearman(z_values, pp_values),
                "mmd_max": max(row["mmd_proxy_max"] for row in ds_rows if row.get("mmd_proxy_max") is not None),
            }
        )
    z_all = [row["artifact_z_within_dataset"] for row in safe_rows if row.get("artifact_z_within_dataset") is not None]
    pp_all = [row["pp_proxy_mean"] for row in safe_rows if row.get("pp_proxy_mean") is not None]
    pooled_spearman = spearman(z_all, pp_all)
    pooled_high_low = high_low(safe_rows)
    boot_lower = bootstrap_lower(safe_rows, rng)
    p_pos = permutation_p_pos(safe_rows, rng)

    reasons = []
    if len(safe_rows) < MIN_TOTAL_ROWS:
        reasons.append("safe_overlap_rows_below_20")
    if varying_datasets < MIN_DATASETS:
        reasons.append("varying_safe_dataset_count_below_2")
    weak_datasets = [
        row["dataset"]
        for row in dataset_summaries
        if row["has_variation"] and (row.get("pp_high_minus_low") is None or row["pp_high_minus_low"] < MIN_DATASET_HIGH_LOW)
    ]
    if weak_datasets:
        reasons.append("safe_dataset_high_low_below_0p02")
    if pooled_spearman is None or pooled_spearman < MIN_POOLED_SPEARMAN:
        reasons.append("pooled_safe_spearman_lt_0p20")
    if boot_lower is None or boot_lower <= 0:
        reasons.append("bootstrap_lower_lte_0")
    if p_pos is None or p_pos > MAX_PERMUTATION_P_POS:
        reasons.append("permutation_p_pos_gt_0p01")
    status = "pass_review_only_no_gpu" if not reasons else "fail_no_gpu"
    return {
        "artifact": artifact,
        "status": status,
        "safe_rows": len(safe_rows),
        "safe_datasets": len(by_dataset),
        "varying_safe_datasets": varying_datasets,
        "pooled_safe_spearman": pooled_spearman,
        "pooled_safe_high_low": pooled_high_low,
        "bootstrap_lower": boot_lower,
        "permutation_p_pos": p_pos,
        "dataset_summaries": dataset_summaries,
        "reasons": reasons,
        "gpu_authorized": False,
    }


def main() -> int:
    rows = load_rows()
    by_artifact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_artifact[str(row.get("artifact"))].append(row)
    rng = random.Random(SEED)
    artifact_rows = [summarize_artifact(artifact, group_rows, rng) for artifact, group_rows in sorted(by_artifact.items())]
    pass_candidates = [row["artifact"] for row in artifact_rows if row["status"] == "pass_review_only_no_gpu"]
    payload = {
        "status": "reagent_read_support_mmd_safe_residual_gate_" + ("review_only_no_gpu" if pass_candidates else "fail_no_gpu"),
        "boundary": {
            "cpu_only": True,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "uses_gpu": False,
        },
        "thresholds": {
            "mmd_safe_max": MMD_SAFE_MAX,
            "min_total_rows": MIN_TOTAL_ROWS,
            "min_datasets": MIN_DATASETS,
            "min_rows_per_dataset": MIN_ROWS_PER_DATASET,
            "min_dataset_high_low": MIN_DATASET_HIGH_LOW,
            "min_pooled_spearman": MIN_POOLED_SPEARMAN,
            "max_permutation_p_pos": MAX_PERMUTATION_P_POS,
            "bootstraps": BOOTSTRAPS,
            "seed": SEED,
        },
        "inputs": {"rows_csv": str(ROWS_CSV)},
        "artifact_rows": artifact_rows,
        "pass_candidates": pass_candidates,
        "gpu_authorized": False,
        "decision": (
            "pass candidates require external review and source-block confound gate before any GPU"
            if pass_candidates
            else "MMD-safe residual signal does not pass; do not launch GPU from reagent/read-support route"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with OUT_ROWS.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "artifact",
            "dataset",
            "safe_rows",
            "has_variation",
            "pp_high_minus_low",
            "spearman",
            "mmd_max",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in artifact_rows:
            for ds_row in row["dataset_summaries"]:
                writer.writerow({"artifact": row["artifact"], **ds_row})

    lines = [
        "# LatentFM Reagent Read-Support MMD-Safe Residual Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only gate over rows with `mmd_proxy_max <= 0.001`.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "",
        "## Artifact Summary",
        "",
        "| artifact | status | safe rows | varying datasets | pooled Spearman | pooled high-low | boot lower | p_pos | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in artifact_rows:
        lines.append(
            "| `{artifact}` | `{status}` | {rows} | {datasets} | {sp} | {hl} | {boot} | {p} | `{reasons}` |".format(
                artifact=row["artifact"],
                status=row["status"],
                rows=row["safe_rows"],
                datasets=row["varying_safe_datasets"],
                sp=fmt(row.get("pooled_safe_spearman")),
                hl=fmt(row.get("pooled_safe_high_low")),
                boot=fmt(row.get("bootstrap_lower")),
                p=fmt(row.get("permutation_p_pos")),
                reasons=",".join(row.get("reasons", [])),
            )
        )
    lines.extend(
        [
            "",
            "## Dataset Summary",
            "",
            "| artifact | dataset | safe rows | variation | Spearman | high-low | MMD max |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in artifact_rows:
        for ds_row in row["dataset_summaries"]:
            lines.append(
                "| `{artifact}` | `{dataset}` | {rows} | `{variation}` | {sp} | {hl} | {mmd} |".format(
                    artifact=row["artifact"],
                    dataset=ds_row["dataset"],
                    rows=ds_row["safe_rows"],
                    variation=ds_row["has_variation"],
                    sp=fmt(ds_row.get("spearman")),
                    hl=fmt(ds_row.get("pp_high_minus_low")),
                    mmd=fmt(ds_row.get("mmd_max")),
                )
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- pass candidates: `{pass_candidates}`",
            f"- {payload['decision']}.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- rows: `{OUT_ROWS}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
