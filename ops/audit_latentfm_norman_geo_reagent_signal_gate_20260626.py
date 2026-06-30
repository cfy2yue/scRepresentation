#!/usr/bin/env python3
"""Signal gate for Norman GEO reagent artifacts against train-only row outcomes.

Short CPU task. Joins condition-level Norman GEO artifacts to completed
train-only/internal row metrics and tests whether artifact values explain tails.
It does not train, infer, read checkpoints, read canonical multi, read Track C
query, or use GPU.
"""

from __future__ import annotations

import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
ARTIFACT_DIR = ROOT / "reports/norman_geo_reagent_artifacts_20260626"
OUTCOME_FILES = [
    ROOT / "reports/latentfm_condition_exposure_row_bootstrap_rows_20260625.csv",
    ROOT / "reports/latentfm_qc_support_reliability_rows_20260625.csv",
    ROOT / "reports/latentfm_response_program_projection_rows_20260625.csv",
    ROOT / "reports/latentfm_lodo_domain_conflict_rows_20260625.csv",
    ROOT / "reports/latentfm_background_target_actionability_rows_20260625.csv",
    ROOT / "reports/latentfm_truecell_riskrow_complementarity_rows_20260625.csv",
]
OUT_JSON = ROOT / "reports/latentfm_norman_geo_reagent_signal_gate_20260626.json"
OUT_MD = ROOT / "reports/LATENTFM_NORMAN_GEO_REAGENT_SIGNAL_GATE_20260626.md"
OUT_CSV = ROOT / "reports/latentfm_norman_geo_reagent_signal_gate_rows_20260626.csv"
DATASET = "NormanWeissman2019_filtered"
SEED = 20260626
N_PERM = 2000


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


def load_outcomes() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in OUTCOME_FILES:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if "dataset" not in fields or "condition" not in fields:
                continue
            for row in reader:
                if norm(row.get("dataset")) != DATASET:
                    continue
                condition = norm(row.get("condition"))
                if not condition:
                    continue
                rec = rows.setdefault(condition, {"condition": condition, "pp_values": [], "mmd_values": []})
                for key in ("cross_pp_diff", "pp_delta", "pp_mean", "truecell_pp_delta_mean"):
                    val = to_float(row.get(key))
                    if val is not None:
                        rec["pp_values"].append(val)
                for key in ("cross_mmd_diff", "mmd_delta", "mmd_mean", "truecell_mmd_delta_mean"):
                    val = to_float(row.get(key))
                    if val is not None:
                        rec["mmd_values"].append(val)
    out = {}
    for condition, rec in rows.items():
        if rec["pp_values"]:
            out[condition] = {
                "condition": condition,
                "pp_proxy_mean": mean(rec["pp_values"]),
                "mmd_proxy_max": max(rec["mmd_values"]) if rec["mmd_values"] else None,
            }
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


def summarize_artifact(path: Path, outcomes: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            condition = norm(row.get("condition"))
            value = to_float(row.get("artifact_value"))
            outcome = outcomes.get(condition)
            if value is None or outcome is None:
                continue
            rows.append(
                {
                    "artifact": path.stem,
                    "condition": condition,
                    "artifact_value": value,
                    "pp_proxy_mean": outcome["pp_proxy_mean"],
                    "mmd_proxy_max": outcome["mmd_proxy_max"],
                }
            )
    xs = [r["artifact_value"] for r in rows]
    ys = [r["pp_proxy_mean"] for r in rows]
    spearman = pearson(rank(xs), rank(ys)) if len(rows) >= 3 else None
    pear = pearson(xs, ys)
    sorted_rows = sorted(rows, key=lambda r: r["artifact_value"])
    k = max(1, len(rows) // 3) if rows else 0
    low = sorted_rows[:k]
    high = sorted_rows[-k:] if k else []
    high_low_pp = (mean([r["pp_proxy_mean"] for r in high]) - mean([r["pp_proxy_mean"] for r in low])) if k else None
    rng = random.Random(SEED + sum(ord(c) for c in path.stem))
    perm_vals = []
    if k and len(rows) >= 6:
        ycopy = ys[:]
        for _ in range(N_PERM):
            rng.shuffle(ycopy)
            perm_vals.append(mean(ycopy[-k:]) - mean(ycopy[:k]))
    if perm_vals and high_low_pp is not None:
        p_abs = (sum(abs(v) >= abs(high_low_pp) for v in perm_vals) + 1) / (len(perm_vals) + 1)
        p_pos = (sum(v >= high_low_pp for v in perm_vals) + 1) / (len(perm_vals) + 1)
    else:
        p_abs = None
        p_pos = None
    reasons = []
    if len(rows) < 30:
        reasons.append("overlap_rows_lt_30")
    if spearman is None or abs(spearman) < 0.30:
        reasons.append("abs_spearman_lt_0p30")
    if high_low_pp is None or high_low_pp <= 0.02:
        reasons.append("high_minus_low_pp_lte_0p020")
    if p_pos is None or p_pos > 0.10:
        reasons.append("permutation_p_pos_gt_0p10")
    if max((r["mmd_proxy_max"] or -999.0) for r in rows) > 0.001:
        reasons.append("mmd_proxy_max_above_0p001")
    reasons.append("single_dataset_preview_only")
    status = "preview_signal_no_gpu" if len(reasons) == 1 else "fail_no_gpu"
    summary = {
        "artifact": path.stem,
        "status": status,
        "gpu_authorized": False,
        "overlap_rows": len(rows),
        "spearman_artifact_vs_pp": spearman,
        "pearson_artifact_vs_pp": pear,
        "high_minus_low_pp": high_low_pp,
        "permutation_p_abs": p_abs,
        "permutation_p_pos": p_pos,
        "mmd_proxy_max": max((r["mmd_proxy_max"] or -999.0) for r in rows) if rows else None,
        "reasons": reasons,
    }
    return summary, rows


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):+.6f}"


def main() -> int:
    outcomes = load_outcomes()
    summaries = []
    all_rows = []
    for path in sorted(ARTIFACT_DIR.glob("norman_geo_*.csv")):
        summary, rows = summarize_artifact(path, outcomes)
        summaries.append(summary)
        all_rows.extend(rows)

    pass_like = [s["artifact"] for s in summaries if s["status"] == "preview_signal_no_gpu"]
    status = "norman_geo_reagent_signal_preview_no_gpu" if pass_like else "norman_geo_reagent_signal_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": "Norman source metadata preview only; no checkpoint/canonical multi/Track C query/training/inference/GPU",
        "dataset": DATASET,
        "pass_like_preview_artifacts": pass_like,
        "summaries": summaries,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        fields = ["artifact", "condition", "artifact_value", "pp_proxy_mean", "mmd_proxy_max"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    lines = [
        "# LatentFM Norman GEO Reagent Signal Gate",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M CST')}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only one-dataset source preview over Norman GEO condition artifacts and completed train-only/internal row metrics.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "",
        "## Summary",
        "",
        "| artifact | status | n | spearman | pearson | high-low pp | p_pos | MMD max | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['artifact']}` | `{row['status']}` | {row['overlap_rows']} | "
            f"{fmt(row['spearman_artifact_vs_pp'])} | {fmt(row['pearson_artifact_vs_pp'])} | "
            f"{fmt(row['high_minus_low_pp'])} | {fmt(row['permutation_p_pos'])} | "
            f"{fmt(row['mmd_proxy_max'])} | {', '.join(row['reasons'])} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- preview pass-like artifacts: `{pass_like}`",
        "- No GPU is authorized because this is one dataset and MMD/tail controls remain required.",
        "- If no artifact is pass-like, do not prioritize more reagent-quality acquisition from this exact Norman source family.",
        "- If one is pass-like, acquire Frangieh/Dixit equivalents and rerun a multi-dataset preflight/control gate before any training.",
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
