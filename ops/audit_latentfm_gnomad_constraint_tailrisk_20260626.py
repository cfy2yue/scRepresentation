#!/usr/bin/env python3
"""Audit gnomAD constraint artifacts as tail-risk/scaling signals.

Short CPU/report task. Reads only materialized gnomAD external-artifact
preflight rows and completed train-only/internal outcome proxies.

It does not train, infer, read checkpoints, read canonical multi, read Track C
query, or use GPU.
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
ROWS = REPORTS / "latentfm_gnomad_constraint_artifact_preflight_20260626_rows.csv"
OUT_JSON = REPORTS / "latentfm_gnomad_constraint_tailrisk_20260626.json"
OUT_MD = REPORTS / "LATENTFM_GNOMAD_CONSTRAINT_TAILRISK_20260626.md"
OUT_ROWS = REPORTS / "latentfm_gnomad_constraint_tailrisk_rows_20260626.csv"

HIGH_MEANS_MORE_CONSTRAINED = {
    "gnomad_lof_constraint_score_neglog10_loeuf": True,
    "gnomad_pli": True,
    "gnomad_mis_z": True,
    "gnomad_oe_lof_upper": False,
}


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


def high_low_delta(rows: list[dict[str, object]]) -> float:
    ordered = sorted(rows, key=lambda r: float(r["risk_value"]))
    k = max(1, len(ordered) // 3)
    low = ordered[:k]
    high = ordered[-k:]
    return (sum(float(r["pp_proxy_mean"]) for r in high) / len(high)) - (
        sum(float(r["pp_proxy_mean"]) for r in low) / len(low)
    )


def summarize(rows: list[dict[str, object]], seed: int = 20260626) -> dict[str, object]:
    risk_values = [float(r["risk_value"]) for r in rows]
    pps = [float(r["pp_proxy_mean"]) for r in rows]
    mmds = [float(r["mmd_proxy_max"]) for r in rows]
    datasets = sorted({str(r["dataset"]) for r in rows})
    varying_datasets = sorted(
        ds for ds in datasets if len({round(float(r["risk_value"]), 8) for r in rows if r["dataset"] == ds}) >= 2
    )
    by_dataset_pp = {}
    for ds in datasets:
        ds_pp = [float(r["pp_proxy_mean"]) for r in rows if r["dataset"] == ds]
        by_dataset_pp[ds] = sum(ds_pp) / len(ds_pp)

    delta = high_low_delta(rows)
    rho = spearman(risk_values, pps)
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)

    n_perm = 2000
    ge_abs = 0
    le = 0
    ge = 0
    for _ in range(n_perm):
        shuffled_rows = []
        for ds_rows in grouped.values():
            shuffled_values = [float(r["risk_value"]) for r in ds_rows]
            rng.shuffle(shuffled_values)
            for row, value in zip(ds_rows, shuffled_values):
                new = dict(row)
                new["risk_value"] = value
                shuffled_rows.append(new)
        perm_delta = high_low_delta(shuffled_rows)
        if abs(perm_delta) >= abs(delta):
            ge_abs += 1
        if perm_delta <= delta:
            le += 1
        if perm_delta >= delta:
            ge += 1

    p_abs = (ge_abs + 1) / (n_perm + 1)
    p_less = (le + 1) / (n_perm + 1)
    p_greater = (ge + 1) / (n_perm + 1)
    reasons = []
    if len(rows) < 20:
        reasons.append("overlap_rows_below_20")
    if len(datasets) < 3:
        reasons.append("dataset_count_below_3")
    if len(varying_datasets) < 3:
        reasons.append("varying_dataset_count_below_3")
    if max(mmds) > 0.001:
        reasons.append("mmd_max_above_0p001")
    if abs(delta) < 0.020:
        reasons.append("abs_high_low_below_0p020")
    if p_abs > 0.05:
        reasons.append("within_dataset_shuffle_abs_p_gt_0p05")

    interpretation = "no_signal"
    if abs(delta) >= 0.020 and p_abs <= 0.05:
        interpretation = "high_constraint_harder" if delta < 0 else "high_constraint_easier"
    return {
        "n": len(rows),
        "datasets": len(datasets),
        "varying_datasets": len(varying_datasets),
        "pp_mean": sum(pps) / len(pps) if pps else None,
        "dataset_min_pp": min(by_dataset_pp.values()) if by_dataset_pp else None,
        "mmd_max": max(mmds) if mmds else None,
        "high_minus_low_pp": delta,
        "spearman_risk_pp": rho,
        "within_dataset_shuffle_abs_p": p_abs,
        "within_dataset_shuffle_less_p": p_less,
        "within_dataset_shuffle_greater_p": p_greater,
        "interpretation": interpretation,
        "reasons": reasons,
        "status": "pass_review_only_no_gpu" if not reasons else "fail_no_gpu",
    }


def main() -> None:
    raw_rows: list[dict[str, object]] = []
    with ROWS.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("outcome_overlap") != "True":
                continue
            artifact = row.get("artifact", "")
            av = to_float(row.get("artifact_value"))
            pp = to_float(row.get("pp_proxy_mean"))
            mmd = to_float(row.get("mmd_proxy_max"))
            if not artifact or av is None or pp is None or mmd is None:
                continue
            risk_value = av if HIGH_MEANS_MORE_CONSTRAINED.get(artifact, True) else -av
            raw_rows.append(
                {
                    "artifact": artifact,
                    "dataset": row["dataset"],
                    "condition": row["condition"],
                    "artifact_value": av,
                    "risk_value": risk_value,
                    "pp_proxy_mean": pp,
                    "mmd_proxy_max": mmd,
                    "mmd_safe": mmd <= 0.001,
                }
            )

    summaries = []
    out_rows = []
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
        fields = [
            "artifact",
            "subset",
            "dataset",
            "condition",
            "artifact_value",
            "risk_value",
            "pp_proxy_mean",
            "mmd_proxy_max",
            "mmd_safe",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    any_review_pass = any(row.get("status") == "pass_review_only_no_gpu" for row in summaries)
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M CST"),
        "status": "gnomad_constraint_tailrisk_review_pass_no_gpu" if any_review_pass else "gnomad_constraint_tailrisk_fail_no_gpu",
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
        "# LatentFM gnomAD Constraint Tail-Risk Audit",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only tail-risk localization over completed gnomAD external-artifact preflight rows.",
        "- Risk value is oriented so higher means more constrained for all four artifacts.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "",
        "## Summary",
        "",
        "| artifact | subset | status | n | datasets | varying datasets | pp mean | dataset min pp | MMD max | high-low pp | Spearman | abs shuffle p | interpretation | reasons |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summaries:
        lines.append(
            "| `{artifact}` | `{subset}` | `{status}` | {n} | {datasets} | {varying_datasets} | {pp_mean:+.6f} | {dataset_min_pp:+.6f} | {mmd_max:+.6f} | {high_minus_low_pp:+.6f} | {spearman} | {within_dataset_shuffle_abs_p:+.6f} | `{interpretation}` | `{reasons}` |".format(
                artifact=row.get("artifact"),
                subset=row.get("subset"),
                status=row.get("status"),
                n=row.get("n", 0),
                datasets=row.get("datasets", 0),
                varying_datasets=row.get("varying_datasets", 0),
                pp_mean=row.get("pp_mean") or 0.0,
                dataset_min_pp=row.get("dataset_min_pp") or 0.0,
                mmd_max=row.get("mmd_max") or 0.0,
                high_minus_low_pp=row.get("high_minus_low_pp") or 0.0,
                spearman="NA" if row.get("spearman_risk_pp") is None else f"{row.get('spearman_risk_pp'):+.6f}",
                within_dataset_shuffle_abs_p=row.get("within_dataset_shuffle_abs_p") or 1.0,
                interpretation=row.get("interpretation", "NA"),
                reasons=",".join(row.get("reasons", [])),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- gnomAD constraint is useful only if it shows a stable within-dataset tail-risk signal after MMD-safe filtering.",
            "- This report does not authorize GPU by itself; a GPU branch would still require a documented sampler/loss hypothesis plus no-leakage RUN_STATUS.",
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
