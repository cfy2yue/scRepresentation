#!/usr/bin/env python3
"""Audit seed-recurrent Track A exact-evaluator tails.

CPU/report-only. This consumes the exact simple-single/cross-background rows
from ``audit_latentfm_tracka_simple_single_unseen_exact_20260627.py`` and asks
which failures recur across seed42 and seed43. It does not train, infer, read
canonical multi, read Track C query, or authorize GPU.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
IN_ROWS = ROOT / "reports/tracka_simple_single_unseen_exact_20260627/condition_rows.csv"
OUT_DIR = ROOT / "reports/tracka_recurrent_tail_gate_20260627"
OUT_ROWS = OUT_DIR / "recurrent_tail_rows.csv"
OUT_SUMMARY = OUT_DIR / "recurrent_tail_dataset_summary.csv"
OUT_JSON = ROOT / "reports/latentfm_tracka_recurrent_tail_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_RECURRENT_TAIL_GATE_20260627.md"

GROUPS = ("simple_single_unseen", "cross_background_seen_gene_exact")
HARD_TAIL_PP = -0.50
NEGATIVE_TAIL_PP = 0.0
MMD_RISK = 0.05


def fnum(v: Any) -> float | None:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except Exception:
        return None


def fmt(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):+.6f}"
    except Exception:
        return str(v)


def mean(vals: list[float]) -> float | None:
    return float(np.mean(vals)) if vals else None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with IN_ROWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["group"] in GROUPS:
                rows.append(row)

    by_key: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["group"], row["dataset"], row["condition"])
        by_key[key][row["seed"]] = row

    recurrent_rows: list[dict[str, Any]] = []
    for (group, dataset, condition), seed_rows in sorted(by_key.items()):
        if "seed42" not in seed_rows or "seed43" not in seed_rows:
            continue
        r42 = seed_rows["seed42"]
        r43 = seed_rows["seed43"]
        pp42 = fnum(r42.get("pearson_pert"))
        pp43 = fnum(r43.get("pearson_pert"))
        mmd42 = fnum(r42.get("test_mmd_clamped"))
        mmd43 = fnum(r43.get("test_mmd_clamped"))
        if pp42 is None or pp43 is None or mmd42 is None or mmd43 is None:
            continue
        mean_pp = (pp42 + pp43) / 2.0
        min_pp = min(pp42, pp43)
        max_mmd = max(mmd42, mmd43)
        hard = pp42 <= HARD_TAIL_PP and pp43 <= HARD_TAIL_PP
        negative = pp42 < NEGATIVE_TAIL_PP and pp43 < NEGATIVE_TAIL_PP
        mmd_risk = max_mmd >= MMD_RISK
        recurrent_rows.append(
            {
                "group": group,
                "dataset": dataset,
                "condition": condition,
                "gene": r42["gene"],
                "perturbation_type": r42["perturbation_type"],
                "pp_seed42": pp42,
                "pp_seed43": pp43,
                "pp_mean": mean_pp,
                "pp_min": min_pp,
                "pp_abs_seed_delta": abs(pp43 - pp42),
                "mmd_seed42": mmd42,
                "mmd_seed43": mmd43,
                "mmd_max": max_mmd,
                "recurrent_hard_tail": hard,
                "recurrent_negative_tail": negative,
                "mmd_risk": mmd_risk,
                "tail_priority_score": -mean_pp + max_mmd,
            }
        )

    recurrent_rows.sort(key=lambda r: (not r["recurrent_hard_tail"], -float(r["tail_priority_score"])))
    fields = [
        "group",
        "dataset",
        "condition",
        "gene",
        "perturbation_type",
        "pp_seed42",
        "pp_seed43",
        "pp_mean",
        "pp_min",
        "pp_abs_seed_delta",
        "mmd_seed42",
        "mmd_seed43",
        "mmd_max",
        "recurrent_hard_tail",
        "recurrent_negative_tail",
        "mmd_risk",
        "tail_priority_score",
    ]
    with OUT_ROWS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(recurrent_rows)

    summaries: dict[str, dict[str, Any]] = {}
    dataset_rows: list[dict[str, Any]] = []
    for group in GROUPS:
        grows = [r for r in recurrent_rows if r["group"] == group]
        hard_rows = [r for r in grows if r["recurrent_hard_tail"]]
        neg_rows = [r for r in grows if r["recurrent_negative_tail"]]
        mmd_rows = [r for r in grows if r["mmd_risk"]]
        by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in grows:
            by_ds[row["dataset"]].append(row)
        summaries[group] = {
            "n_seed_paired_conditions": len(grows),
            "n_datasets": len(by_ds),
            "n_recurrent_negative_tail": len(neg_rows),
            "n_recurrent_hard_tail_pp_le_minus_0p5": len(hard_rows),
            "n_mmd_risk_ge_0p05": len(mmd_rows),
            "mean_pp": mean([float(r["pp_mean"]) for r in grows]),
            "min_pp": min([float(r["pp_min"]) for r in grows]) if grows else None,
            "max_mmd": max([float(r["mmd_max"]) for r in grows]) if grows else None,
            "top_hard_tail_conditions": [
                {
                    "dataset": r["dataset"],
                    "condition": r["condition"],
                    "gene": r["gene"],
                    "pp_mean": r["pp_mean"],
                    "mmd_max": r["mmd_max"],
                }
                for r in hard_rows[:10]
            ],
        }
        for ds, ds_rows in sorted(by_ds.items()):
            dataset_rows.append(
                {
                    "group": group,
                    "dataset": ds,
                    "n_conditions": len(ds_rows),
                    "n_recurrent_negative_tail": sum(1 for r in ds_rows if r["recurrent_negative_tail"]),
                    "n_recurrent_hard_tail": sum(1 for r in ds_rows if r["recurrent_hard_tail"]),
                    "n_mmd_risk": sum(1 for r in ds_rows if r["mmd_risk"]),
                    "mean_pp": mean([float(r["pp_mean"]) for r in ds_rows]),
                    "min_pp": min(float(r["pp_min"]) for r in ds_rows),
                    "max_mmd": max(float(r["mmd_max"]) for r in ds_rows),
                }
            )

    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "dataset",
                "n_conditions",
                "n_recurrent_negative_tail",
                "n_recurrent_hard_tail",
                "n_mmd_risk",
                "mean_pp",
                "min_pp",
                "max_mmd",
            ],
        )
        writer.writeheader()
        writer.writerows(dataset_rows)

    gpu_authorized = False
    status = "tracka_recurrent_tail_gate_ready_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": gpu_authorized,
        "boundary": {
            "cpu_report_only": True,
            "no_training": True,
            "no_inference": True,
            "no_checkpoint_selection": True,
            "canonical_multi_selection_weight": 0,
            "trackc_query_read": False,
        },
        "thresholds": {
            "recurrent_hard_tail_pp_each_seed_lte": HARD_TAIL_PP,
            "recurrent_negative_tail_pp_each_seed_lt": NEGATIVE_TAIL_PP,
            "mmd_risk_max_seed_gte": MMD_RISK,
        },
        "inputs": {"exact_condition_rows": str(IN_ROWS)},
        "summaries": summaries,
        "outputs": {"rows": str(OUT_ROWS), "dataset_summary": str(OUT_SUMMARY), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Track A Recurrent Tail Gate",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only over exact Track A evaluator condition rows.",
        "- Uses seed42/seed43 recurrence only; no train/infer/checkpoint selection.",
        "- Does not read canonical multi for selection or Track C query.",
        "",
        "## Gate Summary",
        "",
        "| group | n paired | n datasets | recurrent negative | recurrent hard tail | MMD risk | mean pp | min pp | max MMD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        s = summaries[group]
        lines.append(
            f"| `{group}` | {s['n_seed_paired_conditions']} | {s['n_datasets']} | "
            f"{s['n_recurrent_negative_tail']} | {s['n_recurrent_hard_tail_pp_le_minus_0p5']} | "
            f"{s['n_mmd_risk_ge_0p05']} | {fmt(s['mean_pp'])} | {fmt(s['min_pp'])} | {fmt(s['max_mmd'])} |"
        )
    lines += [
        "",
        "## Top Recurrent Hard Tails",
        "",
        "| group | dataset | condition | gene | pp mean | max MMD |",
        "|---|---|---|---|---:|---:|",
    ]
    for group in GROUPS:
        for row in summaries[group]["top_hard_tail_conditions"][:8]:
            lines.append(
                f"| `{group}` | `{row['dataset']}` | `{row['condition']}` | `{row['gene']}` | "
                f"{fmt(row['pp_mean'])} | {fmt(row['mmd_max'])} |"
            )
    lines += [
        "",
        "## Decision",
        "",
        "- Recurrent tails are real failure-localization evidence, not a candidate model.",
        "- No GPU is authorized because this gate produces no paired candidate-vs-anchor MMD/no-harm evidence.",
        "- Future candidate mechanisms should pre-register these recurrent hard tails as a tail-protection set and must improve them without harming exact `simple_single_unseen`, exact cross-background, `test_single`, or `family_gene` no-harm metrics.",
        "",
        "## Outputs",
        "",
        f"- Rows: `{OUT_ROWS}`",
        f"- Dataset summary: `{OUT_SUMMARY}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
