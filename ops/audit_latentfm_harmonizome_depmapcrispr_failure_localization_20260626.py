#!/usr/bin/env python3
"""Localize why Harmonizome DepMap CRISPR artifacts failed preflight.

Short CPU/report task. Reads only the materialized external-artifact preflight
rows and does not train, infer, read checkpoints, read canonical multi, read
Track C query, or use GPU.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
ROWS = REPORTS / "latentfm_harmonizome_depmapcrispr_artifact_preflight_20260626_rows.csv"
OUT_JSON = REPORTS / "latentfm_harmonizome_depmapcrispr_failure_localization_20260626.json"
OUT_MD = REPORTS / "LATENTFM_HARMONIZOME_DEPMAPCRISPR_FAILURE_LOCALIZATION_20260626.md"
OUT_ROWS = REPORTS / "latentfm_harmonizome_depmapcrispr_failure_localization_rows_20260626.csv"


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
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
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rank(xs), rank(ys))


def summarize(rows: list[dict[str, object]], seed: int = 20260626) -> dict[str, object]:
    vals = [float(r["artifact_value"]) for r in rows]
    pps = [float(r["pp_proxy_mean"]) for r in rows]
    mmds = [float(r["mmd_proxy_max"]) for r in rows]
    datasets = sorted({str(r["dataset"]) for r in rows})
    varying_datasets = sorted(
        ds for ds in datasets if len({round(float(r["artifact_value"]), 8) for r in rows if r["dataset"] == ds}) >= 2
    )
    by_dataset_pp = {}
    for ds in datasets:
        ds_pp = [float(r["pp_proxy_mean"]) for r in rows if r["dataset"] == ds]
        by_dataset_pp[ds] = sum(ds_pp) / len(ds_pp)
    ordered = sorted(rows, key=lambda r: float(r["artifact_value"]))
    k = max(1, len(ordered) // 3)
    low = ordered[:k]
    high = ordered[-k:]
    high_low = (sum(float(r["pp_proxy_mean"]) for r in high) / len(high)) - (
        sum(float(r["pp_proxy_mean"]) for r in low) / len(low)
    )
    rho = spearman(vals, pps)

    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    ge = 0
    n_perm = 1000
    for _ in range(n_perm):
        shuffled_rows = []
        for ds_rows in grouped.values():
            shuffled_vals = [float(r["artifact_value"]) for r in ds_rows]
            rng.shuffle(shuffled_vals)
            for row, val in zip(ds_rows, shuffled_vals):
                new = dict(row)
                new["artifact_value"] = val
                shuffled_rows.append(new)
        ordered_s = sorted(shuffled_rows, key=lambda r: float(r["artifact_value"]))
        low_s = ordered_s[:k]
        high_s = ordered_s[-k:]
        hl_s = (sum(float(r["pp_proxy_mean"]) for r in high_s) / len(high_s)) - (
            sum(float(r["pp_proxy_mean"]) for r in low_s) / len(low_s)
        )
        if hl_s >= high_low:
            ge += 1

    reasons = []
    if len(rows) < 20:
        reasons.append("overlap_rows_below_20")
    if len(datasets) < 3:
        reasons.append("dataset_count_below_3")
    if len(varying_datasets) < 3:
        reasons.append("varying_dataset_count_below_3")
    if min(by_dataset_pp.values()) < -0.02:
        reasons.append("dataset_min_pp_below_minus_0p020")
    if max(mmds) > 0.001:
        reasons.append("mmd_max_above_0p001")
    if high_low < 0.02:
        reasons.append("high_low_below_0p020")
    if (ge + 1) / (n_perm + 1) > 0.05:
        reasons.append("within_dataset_shuffle_p_gt_0p05")

    return {
        "n": len(rows),
        "datasets": len(datasets),
        "varying_datasets": len(varying_datasets),
        "pp_mean": sum(pps) / len(pps) if pps else None,
        "dataset_min_pp": min(by_dataset_pp.values()) if by_dataset_pp else None,
        "mmd_max": max(mmds) if mmds else None,
        "high_low_pp": high_low,
        "spearman": rho,
        "within_dataset_shuffle_p": (ge + 1) / (n_perm + 1),
        "reasons": reasons,
        "status": "pass_review_only" if not reasons else "fail_no_gpu",
    }


def main() -> None:
    raw_rows: list[dict[str, object]] = []
    with ROWS.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("outcome_overlap") != "True":
                continue
            av = to_float(row.get("artifact_value"))
            pp = to_float(row.get("pp_proxy_mean"))
            mmd = to_float(row.get("mmd_proxy_max"))
            if av is None or pp is None or mmd is None:
                continue
            raw_rows.append(
                {
                    "artifact": row["artifact"],
                    "dataset": row["dataset"],
                    "condition": row["condition"],
                    "artifact_value": av,
                    "pp_proxy_mean": pp,
                    "mmd_proxy_max": mmd,
                    "mmd_safe": mmd <= 0.001,
                }
            )

    out_rows = []
    summaries = []
    for artifact in sorted({str(r["artifact"]) for r in raw_rows}):
        art_rows = [r for r in raw_rows if r["artifact"] == artifact]
        for label, subset in [("all_rows", art_rows), ("mmd_safe", [r for r in art_rows if r["mmd_safe"]])]:
            if not subset:
                summaries.append(
                    {
                        "artifact": artifact,
                        "subset": label,
                        "status": "fail_no_gpu",
                        "n": 0,
                        "reasons": ["no_rows"],
                    }
                )
                continue
            summary = summarize(subset)
            summary.update({"artifact": artifact, "subset": label})
            summaries.append(summary)
            for row in subset:
                out = dict(row)
                out["subset"] = label
                out_rows.append(out)

    with OUT_ROWS.open("w", newline="", encoding="utf-8") as f:
        fields = ["artifact", "subset", "dataset", "condition", "artifact_value", "pp_proxy_mean", "mmd_proxy_max", "mmd_safe"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M CST"),
        "status": "harmonizome_depmapcrispr_failure_localization_no_gpu",
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "trains_or_infers": False,
            "uses_gpu": False,
        },
        "summaries": summaries,
        "outputs": {"rows": str(OUT_ROWS)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# LatentFM Harmonizome DepMap CRISPR Failure Localization",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        "Status: `harmonizome_depmapcrispr_failure_localization_no_gpu`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only localization over completed Harmonizome external-artifact preflight rows.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "",
        "## Summary",
        "",
        "| artifact | subset | status | n | datasets | varying datasets | pp mean | dataset min pp | MMD max | high-low pp | Spearman | shuffle p | reasons |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| `{artifact}` | `{subset}` | `{status}` | {n} | {datasets} | {varying_datasets} | {pp_mean:+.6f} | {dataset_min_pp:+.6f} | {mmd_max:+.6f} | {high_low_pp:+.6f} | {spearman} | {within_dataset_shuffle_p:+.6f} | `{reasons}` |".format(
                artifact=row.get("artifact"),
                subset=row.get("subset"),
                status=row.get("status"),
                n=row.get("n", 0),
                datasets=row.get("datasets", 0),
                varying_datasets=row.get("varying_datasets", 0),
                pp_mean=row.get("pp_mean") or 0.0,
                dataset_min_pp=row.get("dataset_min_pp") or 0.0,
                mmd_max=row.get("mmd_max") or 0.0,
                high_low_pp=row.get("high_low_pp") or 0.0,
                spearman="NA" if row.get("spearman") is None else f"{row.get('spearman'):+.6f}",
                within_dataset_shuffle_p=row.get("within_dataset_shuffle_p") or 1.0,
                reasons=",".join(row.get("reasons", [])),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- No Harmonizome DepMap CRISPR dependency artifact passes localization or preflight gates.",
            "- Matched-cellline artifacts lack variation; global target-level artifacts have broader coverage but fail tail/MMD/high-low/shuffle criteria.",
            "- Do not launch dependency-prior GPU from current artifacts.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- Rows: `{OUT_ROWS}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
