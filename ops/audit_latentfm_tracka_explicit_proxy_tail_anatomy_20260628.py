#!/usr/bin/env python3
"""Tail anatomy for explicit Track A proxy groups.

CPU/report-only over the frozen-row explicit proxy benchmark. This script looks
for stable tail structure that could motivate a materially new tail/no-harm
repair gate. It does not train, infer, read checkpoints, use canonical multi,
read Track C query, or use GPU.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
IN_ROWS = ROOT / "reports/tracka_explicit_group_proxy_benchmark_20260628/condition_rows.csv"
OUT_DIR = ROOT / "reports/tracka_explicit_proxy_tail_anatomy_20260628"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_EXPLICIT_PROXY_TAIL_ANATOMY_20260628.md"
OUT_JSON = ROOT / "reports/latentfm_tracka_explicit_proxy_tail_anatomy_20260628.json"


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def finite(value: Any) -> float | None:
    text = norm(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except (TypeError, ValueError):
        return str(value)


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        r = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = r
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rankdata(xs), rankdata(ys)) if len(xs) == len(ys) and len(xs) >= 3 else None


def load_rows() -> list[dict[str, Any]]:
    rows = []
    with IN_ROWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pp = finite(row.get("pearson_pert"))
            if pp is None:
                continue
            train_bgs = [x for x in norm(row.get("train_backgrounds_for_gene")).split(";") if x]
            rec = dict(row)
            rec["pearson_pert"] = pp
            rec["test_mmd_clamped"] = finite(row.get("test_mmd_clamped"))
            rec["n_src_eval"] = finite(row.get("n_src_eval"))
            rec["train_background_count"] = len(set(train_bgs))
            rec["is_unseen_gene"] = int(bool(norm(row.get("single_gene"))) and not train_bgs)
            rec["is_seen_crossbg_gene"] = int(bool(norm(row.get("single_gene"))) and len(set(train_bgs)) >= 2)
            rows.append(rec)
    return rows


def summarize_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["seed"]), str(row["explicit_group"]), str(row["dataset"]))].append(row)
    out = []
    for (seed, group, ds), vals in sorted(by_key.items()):
        pp = [float(v["pearson_pert"]) for v in vals]
        mmd = [float(v["test_mmd_clamped"]) for v in vals if v.get("test_mmd_clamped") is not None]
        out.append(
            {
                "seed": seed,
                "group": group,
                "dataset": ds,
                "n": len(vals),
                "mean_pp": mean(pp),
                "min_pp": min(pp),
                "frac_pp_below_0": sum(1 for v in pp if v < 0) / len(pp),
                "frac_pp_below_0p05": sum(1 for v in pp if v < 0.05) / len(pp),
                "mean_mmd": mean(mmd) if mmd else None,
                "max_mmd": max(mmd) if mmd else None,
            }
        )
    return sorted(out, key=lambda r: (r["seed"], r["group"], float(r["mean_pp"])))


def recurrent_bad(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["explicit_group"]) == "family_gene":
            continue
        by_key[(str(row["dataset"]), str(row["condition"]))].append(row)
    out = []
    for (ds, cond), vals in by_key.items():
        seed_groups = {(str(v["seed"]), str(v["explicit_group"])) for v in vals}
        pps = [float(v["pearson_pert"]) for v in vals]
        if sum(1 for v in pps if v < 0) < 2:
            continue
        out.append(
            {
                "dataset": ds,
                "condition": cond,
                "gene": norm(vals[0].get("single_gene")),
                "cell_background": norm(vals[0].get("cell_background")),
                "seed_group_hits": len(seed_groups),
                "mean_pp": mean(pps),
                "min_pp": min(pps),
                "max_mmd": max([float(v["test_mmd_clamped"]) for v in vals if v.get("test_mmd_clamped") is not None], default=None),
                "train_background_count": vals[0].get("train_background_count"),
                "groups": ";".join(sorted({str(v["explicit_group"]) for v in vals})),
            }
        )
    return sorted(out, key=lambda r: (float(r["mean_pp"]), str(r["dataset"]), str(r["condition"])))


def feature_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pp = [float(r["pearson_pert"]) for r in rows]
    out: dict[str, Any] = {
        "n": len(rows),
        "tail_pp_below_0": sum(1 for v in pp if v < 0),
        "tail_pp_below_0p05": sum(1 for v in pp if v < 0.05),
    }
    for key in ("test_mmd_clamped", "n_src_eval", "train_background_count", "is_unseen_gene", "is_seen_crossbg_gene"):
        pairs = [(float(r[key]), float(r["pearson_pert"])) for r in rows if r.get(key) is not None]
        xs = [x for x, _ in pairs]
        ys = [y for _, y in pairs]
        out[f"{key}_spearman_vs_pp"] = spearman(xs, ys) if len(xs) >= 3 else None
        tail_vals = [float(r[key]) for r in rows if r.get(key) is not None and float(r["pearson_pert"]) < 0]
        ok_vals = [float(r[key]) for r in rows if r.get(key) is not None and float(r["pearson_pert"]) >= 0]
        out[f"{key}_tail_median"] = median(tail_vals) if tail_vals else None
        out[f"{key}_nontail_median"] = median(ok_vals) if ok_vals else None
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    rows = load_rows()
    dataset_rows = summarize_dataset(rows)
    bad_rows = recurrent_bad(rows)
    features = feature_summary(rows)
    top_bad_datasets = dataset_rows[:30]
    top_bad_conditions = bad_rows[:50]

    write_csv(
        OUT_DIR / "dataset_tail_summary.csv",
        dataset_rows,
        ["seed", "group", "dataset", "n", "mean_pp", "min_pp", "frac_pp_below_0", "frac_pp_below_0p05", "mean_mmd", "max_mmd"],
    )
    write_csv(
        OUT_DIR / "recurrent_bad_conditions.csv",
        bad_rows,
        ["dataset", "condition", "gene", "cell_background", "seed_group_hits", "mean_pp", "min_pp", "max_mmd", "train_background_count", "groups"],
    )

    # This audit is diagnostic only. It can motivate subagent review, not GPU.
    reasons = []
    if features.get("test_mmd_clamped_spearman_vs_pp") is not None and float(features["test_mmd_clamped_spearman_vs_pp"]) < -0.25:
        reasons.append("mmd_correlates_with_bad_pp")
    if top_bad_conditions:
        reasons.append("recurrent_condition_tails_present")
    status = "tracka_explicit_proxy_tail_anatomy_ready_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_report_only": True,
            "input_rows": str(IN_ROWS),
            "training_or_inference": False,
            "checkpoints": False,
            "canonical_multi_selection": False,
            "trackc_query": False,
            "gpu": False,
        },
        "feature_summary": features,
        "top_bad_datasets": top_bad_datasets,
        "top_bad_conditions": top_bad_conditions,
        "diagnostic_reasons": reasons,
        "outputs": {
            "markdown": str(OUT_MD),
            "json": str(OUT_JSON),
            "dataset_tail_summary": str(OUT_DIR / "dataset_tail_summary.csv"),
            "recurrent_bad_conditions": str(OUT_DIR / "recurrent_bad_conditions.csv"),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Track A Explicit Proxy Tail Anatomy",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only over explicit Track A proxy condition rows.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C query, or GPU.",
        "",
        "## Feature Summary",
        "",
        f"- rows: `{features['n']}`",
        f"- pp<0 rows: `{features['tail_pp_below_0']}`",
        f"- pp<0.05 rows: `{features['tail_pp_below_0p05']}`",
        f"- MMD Spearman vs pp: `{fmt(features.get('test_mmd_clamped_spearman_vs_pp'))}`",
        f"- n_src Spearman vs pp: `{fmt(features.get('n_src_eval_spearman_vs_pp'))}`",
        f"- train-background-count Spearman vs pp: `{fmt(features.get('train_background_count_spearman_vs_pp'))}`",
        f"- unseen-gene Spearman vs pp: `{fmt(features.get('is_unseen_gene_spearman_vs_pp'))}`",
        f"- seen-crossbg-gene Spearman vs pp: `{fmt(features.get('is_seen_crossbg_gene_spearman_vs_pp'))}`",
        "",
        "## Worst Dataset Slices",
        "",
        "| seed | group | dataset | n | mean pp | min pp | frac pp<0 | max MMD |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in top_bad_datasets[:20]:
        lines.append(
            f"| `{row['seed']}` | `{row['group']}` | `{row['dataset']}` | {row['n']} | {fmt(row['mean_pp'])} | {fmt(row['min_pp'])} | {fmt(row['frac_pp_below_0'])} | {fmt(row['max_mmd'])} |"
        )
    lines.extend(
        [
            "",
            "## Recurrent Bad Conditions",
            "",
            "| dataset | condition | gene | background | hits | mean pp | min pp | max MMD | train bg count | groups |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in top_bad_conditions[:25]:
        lines.append(
            f"| `{row['dataset']}` | `{row['condition']}` | `{row['gene']}` | `{row['cell_background']}` | {row['seed_group_hits']} | {fmt(row['mean_pp'])} | {fmt(row['min_pp'])} | {fmt(row['max_mmd'])} | {row['train_background_count']} | `{row['groups']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Diagnostic only; no GPU is authorized.",
            "- Use this table to design a materially new CPU gate, not another closed reweighting/normalization/OT/low-rank/lookahead route.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- dataset tails: `{OUT_DIR / 'dataset_tail_summary.csv'}`",
            f"- recurrent bad conditions: `{OUT_DIR / 'recurrent_bad_conditions.csv'}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "recurrent_bad": len(bad_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
