#!/usr/bin/env python3
"""Source-block and LODO confound gate for reagent/read-support artifacts.

CPU-only gate over MMD-safe combined signal rows. It checks whether the retained
read/guide-support signal survives source-file block bootstrap, leave-one-
dataset-out folds, duplicate condition collapse, and within-dataset shuffles.

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
SAFE_GATE_JSON = REPORTS / "latentfm_reagent_read_support_mmd_safe_residual_gate_20260626.json"
OUT_JSON = REPORTS / "latentfm_reagent_read_support_source_block_lodo_gate_20260626.json"
OUT_MD = REPORTS / "LATENTFM_REAGENT_READ_SUPPORT_SOURCE_BLOCK_LODO_GATE_20260626.md"
OUT_ROWS = REPORTS / "latentfm_reagent_read_support_source_block_lodo_gate_rows_20260626.csv"

MMD_SAFE_MAX = 0.001
BOOTSTRAPS = 1000
SEED = 20260626
MIN_SIGNAL = 0.02
MAX_SHUFFLE_P = 0.05


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


def spearman(rows: list[dict[str, Any]]) -> float | None:
    paired = [
        (row["artifact_z_within_dataset"], row["pp_proxy_mean"])
        for row in rows
        if row.get("artifact_z_within_dataset") is not None and row.get("pp_proxy_mean") is not None
    ]
    if len(paired) < 3:
        return None
    return pearson(rank([x[0] for x in paired]), rank([x[1] for x in paired]))


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


def collapse_condition(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("dataset")), str(row.get("condition")))].append(row)
    collapsed = []
    for (dataset, condition), group_rows in grouped.items():
        z_vals = [r["artifact_z_within_dataset"] for r in group_rows if r.get("artifact_z_within_dataset") is not None]
        pp_vals = [r["pp_proxy_mean"] for r in group_rows if r.get("pp_proxy_mean") is not None]
        mmd_vals = [r["mmd_proxy_max"] for r in group_rows if r.get("mmd_proxy_max") is not None]
        if not z_vals or not pp_vals:
            continue
        collapsed.append(
            {
                "dataset": dataset,
                "condition": condition,
                "artifact_z_within_dataset": mean(z_vals),
                "pp_proxy_mean": mean(pp_vals),
                "mmd_proxy_max": max(mmd_vals) if mmd_vals else None,
                "source_file": "condition_collapsed",
            }
        )
    return collapsed


def block_bootstrap_lower(rows: list[dict[str, Any]], rng: random.Random) -> float | None:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source_file"))].append(row)
    blocks = list(by_source)
    if len(blocks) < 2:
        return None
    values = []
    for _ in range(BOOTSTRAPS):
        sampled = []
        for _ in blocks:
            sampled.extend(by_source[rng.choice(blocks)])
        val = high_low(sampled)
        if val is not None:
            values.append(val)
    return quantile(values, 0.025)


def within_dataset_shuffle_p(rows: list[dict[str, Any]], observed: float | None, rng: random.Random) -> float | None:
    if observed is None:
        return None
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row.get("dataset"))].append(row)
    valid = 0
    ge_count = 0
    for _ in range(BOOTSTRAPS):
        shuffled_rows = []
        for dataset, ds_rows in by_dataset.items():
            pp_vals = [r["pp_proxy_mean"] for r in ds_rows]
            rng.shuffle(pp_vals)
            for row, pp in zip(ds_rows, pp_vals):
                new_row = dict(row)
                new_row["pp_proxy_mean"] = pp
                shuffled_rows.append(new_row)
        val = high_low(shuffled_rows)
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
    datasets = sorted({str(row.get("dataset")) for row in safe_rows})
    sources = sorted({str(row.get("source_file")) for row in safe_rows})
    observed_hl = high_low(safe_rows)
    observed_sp = spearman(safe_rows)
    source_boot_lower = block_bootstrap_lower(safe_rows, rng)
    shuffle_p = within_dataset_shuffle_p(safe_rows, observed_hl, rng)
    lodo_rows = []
    for dataset in datasets:
        subset = [row for row in safe_rows if row.get("dataset") != dataset]
        lodo_rows.append(
            {
                "left_out_dataset": dataset,
                "rows": len(subset),
                "high_low": high_low(subset),
                "spearman": spearman(subset),
            }
        )
    collapsed = collapse_condition(safe_rows)
    collapsed_hl = high_low(collapsed)
    collapsed_sp = spearman(collapsed)
    reasons = []
    if observed_hl is None or observed_hl < MIN_SIGNAL:
        reasons.append("observed_high_low_below_0p02")
    if source_boot_lower is None or source_boot_lower <= 0:
        reasons.append("source_block_bootstrap_lower_lte_0")
    if shuffle_p is None or shuffle_p > MAX_SHUFFLE_P:
        reasons.append("within_dataset_shuffle_p_gt_0p05")
    if any(row.get("high_low") is None or row["high_low"] <= 0 for row in lodo_rows):
        reasons.append("lodo_fold_nonpositive")
    if collapsed_hl is None or collapsed_hl <= 0:
        reasons.append("condition_collapsed_signal_nonpositive")
    if len(datasets) < 2:
        reasons.append("dataset_count_below_2")
    if any(row.get("mmd_proxy_max") is None or row["mmd_proxy_max"] > MMD_SAFE_MAX for row in safe_rows):
        reasons.append("mmd_safe_filter_violation")
    status = "pass_external_review_only_no_gpu" if not reasons else "fail_no_gpu"
    return {
        "artifact": artifact,
        "status": status,
        "safe_rows": len(safe_rows),
        "datasets": datasets,
        "source_files": sources,
        "observed_high_low": observed_hl,
        "observed_spearman": observed_sp,
        "source_block_bootstrap_lower": source_boot_lower,
        "within_dataset_shuffle_p": shuffle_p,
        "lodo_rows": lodo_rows,
        "condition_collapsed_rows": len(collapsed),
        "condition_collapsed_high_low": collapsed_hl,
        "condition_collapsed_spearman": collapsed_sp,
        "reasons": reasons,
        "gpu_authorized": False,
    }


def main() -> int:
    safe_gate = json.loads(SAFE_GATE_JSON.read_text(encoding="utf-8"))
    candidates = set(safe_gate.get("pass_candidates", []))
    rows = load_rows()
    by_artifact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        artifact = str(row.get("artifact"))
        if artifact in candidates:
            by_artifact[artifact].append(row)
    rng = random.Random(SEED)
    artifact_rows = [summarize_artifact(artifact, group_rows, rng) for artifact, group_rows in sorted(by_artifact.items())]
    pass_candidates = [row["artifact"] for row in artifact_rows if row["status"] == "pass_external_review_only_no_gpu"]
    payload = {
        "status": "reagent_read_support_source_block_lodo_gate_" + ("external_review_only_no_gpu" if pass_candidates else "fail_no_gpu"),
        "boundary": {
            "cpu_only": True,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "uses_gpu": False,
        },
        "thresholds": {
            "mmd_safe_max": MMD_SAFE_MAX,
            "bootstraps": BOOTSTRAPS,
            "seed": SEED,
            "min_signal": MIN_SIGNAL,
            "max_shuffle_p": MAX_SHUFFLE_P,
        },
        "inputs": {"rows_csv": str(ROWS_CSV), "safe_gate_json": str(SAFE_GATE_JSON)},
        "artifact_rows": artifact_rows,
        "pass_candidates": pass_candidates,
        "gpu_authorized": False,
        "decision": (
            "pass candidates require independent external review before drafting any bounded GPU smoke"
            if pass_candidates
            else "source-block/LODO confound checks failed; close or keep as mechanism evidence only"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with OUT_ROWS.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["artifact", "left_out_dataset", "rows", "high_low", "spearman"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in artifact_rows:
            for lodo in row["lodo_rows"]:
                writer.writerow({"artifact": row["artifact"], **lodo})

    lines = [
        "# LatentFM Reagent Read-Support Source-Block LODO Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only confound gate over MMD-safe rows from the residual signal gate.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "",
        "## Artifact Summary",
        "",
        "| artifact | status | rows | datasets | high-low | Spearman | source boot lower | shuffle p | collapsed high-low | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in artifact_rows:
        lines.append(
            "| `{artifact}` | `{status}` | {rows} | {datasets} | {hl} | {sp} | {boot} | {p} | {collapsed} | `{reasons}` |".format(
                artifact=row["artifact"],
                status=row["status"],
                rows=row["safe_rows"],
                datasets=len(row["datasets"]),
                hl=fmt(row.get("observed_high_low")),
                sp=fmt(row.get("observed_spearman")),
                boot=fmt(row.get("source_block_bootstrap_lower")),
                p=fmt(row.get("within_dataset_shuffle_p")),
                collapsed=fmt(row.get("condition_collapsed_high_low")),
                reasons=",".join(row.get("reasons", [])),
            )
        )
    lines.extend(
        [
            "",
            "## LODO Summary",
            "",
            "| artifact | left-out dataset | remaining rows | high-low | Spearman |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in artifact_rows:
        for lodo in row["lodo_rows"]:
            lines.append(
                "| `{artifact}` | `{dataset}` | {rows} | {hl} | {sp} |".format(
                    artifact=row["artifact"],
                    dataset=lodo["left_out_dataset"],
                    rows=lodo["rows"],
                    hl=fmt(lodo.get("high_low")),
                    sp=fmt(lodo.get("spearman")),
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
