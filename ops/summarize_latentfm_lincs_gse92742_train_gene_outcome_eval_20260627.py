#!/usr/bin/env python3
"""Summarize eval-only GSE92742 train/gene outcome materialization."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OVERLAP = ROOT / "reports/lincs_l1000_gse92742_condition_join_gate_20260627/gse92742_s0_overlap_rows.csv"

FEATURES = (
    "tas_mean",
    "distil_cc_q75_mean",
    "sig_count_mean",
    "sig_count_sum",
    "exact_bg_frac",
    "unique_lincs_cells",
    "frac_trt_sh",
    "frac_trt_oe",
    "frac_trt_sh_cgs",
    "frac_trt_lig",
)


def norm_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def norm_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm_text(value).lower())


def fnum(value: object) -> float | None:
    text = norm_text(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return None if math.isnan(out) or math.isinf(out) else out


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    xm = mean(x)
    ym = mean(y)
    xv = [v - xm for v in x]
    yv = [v - ym for v in y]
    denom = math.sqrt(sum(v * v for v in xv) * sum(v * v for v in yv))
    return None if denom == 0 else sum(a * b for a, b in zip(xv, yv)) / denom


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(rankdata(x), rankdata(y))


def read_lincs_features() -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with OVERLAP.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("membership") != "train" or row.get("modality") != "gene":
                continue
            grouped[(norm_text(row.get("dataset")), norm_text(row.get("condition")))].append(row)

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, part in grouped.items():
        def vals(field: str) -> list[float]:
            return [v for v in (fnum(row.get(field)) for row in part) if v is not None]

        type_counts = Counter(norm_text(row.get("lincs_pert_type")) for row in part)
        total = max(1, len(part))
        exact = [
            1.0
            if norm_key(row.get("s0_cell_background"))
            and norm_key(row.get("s0_cell_background")) == norm_key(row.get("lincs_cell_id"))
            else 0.0
            for row in part
        ]
        out[key] = {
            "dataset": key[0],
            "condition": key[1],
            "n_lincs_overlap_rows": len(part),
            "tas_mean": mean(vals("tas_mean")) if vals("tas_mean") else math.nan,
            "distil_cc_q75_mean": mean(vals("distil_cc_q75_mean")) if vals("distil_cc_q75_mean") else math.nan,
            "sig_count_mean": mean(vals("lincs_sig_count")) if vals("lincs_sig_count") else math.nan,
            "sig_count_sum": sum(vals("lincs_sig_count")) if vals("lincs_sig_count") else math.nan,
            "exact_bg_frac": mean(exact) if exact else 0.0,
            "unique_lincs_cells": len({norm_text(row.get("lincs_cell_id")) for row in part if norm_text(row.get("lincs_cell_id"))}),
            "frac_trt_sh": type_counts.get("trt_sh", 0) / total,
            "frac_trt_oe": type_counts.get("trt_oe", 0) / total,
            "frac_trt_sh_cgs": type_counts.get("trt_sh.cgs", 0) / total,
            "frac_trt_lig": type_counts.get("trt_lig", 0) / total,
        }
    return out


def condition_metrics(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    group = obj.get("groups", {}).get("test", {})
    rows = group.get("condition_metrics") if isinstance(group, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"No groups.test.condition_metrics in {path}")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (norm_text(row.get("dataset")), norm_text(row.get("condition")))
        if key[0] and key[1]:
            out[key] = row
    return out


def feature_result(rows: list[dict[str, Any]], feature: str, target: str, *, n_perm: int = 1000) -> dict[str, Any]:
    pairs = [(fnum(row.get(feature)), fnum(row.get(target)), norm_text(row.get("dataset"))) for row in rows]
    pairs = [(float(x), float(y), dataset) for x, y, dataset in pairs if x is not None and y is not None]
    if len(pairs) < 10 or len({x for x, _, _ in pairs}) < 2 or len({y for _, y, _ in pairs}) < 2:
        return {"feature": feature, "target": target, "n": len(pairs), "rho": None, "shuffle_p_abs": None}
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    actual = spearman(x, y)
    if actual is None:
        return {"feature": feature, "target": target, "n": len(pairs), "rho": None, "shuffle_p_abs": None}
    rng = random.Random(20260627)
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for i, (_, _, dataset) in enumerate(pairs):
        by_dataset[dataset].append(i)
    hits = 0
    total = 0
    for _ in range(n_perm):
        shuffled = x[:]
        for idxs in by_dataset.values():
            values = [shuffled[i] for i in idxs]
            rng.shuffle(values)
            for i, value in zip(idxs, values):
                shuffled[i] = value
        rho = spearman(shuffled, y)
        if rho is None:
            continue
        total += 1
        if abs(rho) >= abs(actual):
            hits += 1
    return {
        "feature": feature,
        "target": target,
        "n": len(pairs),
        "rho": actual,
        "shuffle_p_abs": (hits + 1) / (total + 1) if total else 1.0,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "condition",
        "candidate_pp",
        "anchor_pp",
        "pp_delta",
        "candidate_mmd",
        "anchor_mmd",
        "mmd_delta",
        "n_lincs_overlap_rows",
        *FEATURES,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor-json", type=Path, required=True)
    ap.add_argument("--candidate-json", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--candidate-label", default="candidate")
    args = ap.parse_args()

    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_prefix.with_suffix(".json")
    out_md = out_prefix.with_suffix(".md")
    out_rows = out_prefix.with_suffix(".csv")

    boundary = {
        "gpu_used_by_summarizer": False,
        "training_or_checkpoint_selection_used": False,
        "canonical_multi_selection_used": False,
        "trackc_heldout_query_used": False,
        "eval_split": "GSE92742 strict S0 train/gene overlap",
        "candidate_label": args.candidate_label,
    }
    missing = [str(p) for p in (OVERLAP, args.anchor_json, args.candidate_json) if not p.is_file()]
    if missing:
        payload = {
            "status": "lincs_gse92742_train_gene_outcome_eval_missing_inputs_no_gpu",
            "gpu_authorized": False,
            "boundary": boundary,
            "missing": missing,
        }
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        out_md.write_text("# LINCS GSE92742 Train Gene Outcome Eval\n\nMissing inputs.\n", encoding="utf-8")
        print(json.dumps({"status": payload["status"], "gpu_authorized": False}, indent=2))
        return 0

    lincs = read_lincs_features()
    anchor = condition_metrics(args.anchor_json)
    candidate = condition_metrics(args.candidate_json)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(lincs) & set(anchor) & set(candidate)):
        app = fnum(anchor[key].get("pearson_pert"))
        cpp = fnum(candidate[key].get("pearson_pert"))
        ammd = fnum(anchor[key].get("test_mmd_clamped"))
        cmmd = fnum(candidate[key].get("test_mmd_clamped"))
        if app is None or cpp is None or ammd is None or cmmd is None:
            continue
        rows.append(
            {
                **lincs[key],
                "candidate_pp": cpp,
                "anchor_pp": app,
                "pp_delta": cpp - app,
                "candidate_mmd": cmmd,
                "anchor_mmd": ammd,
                "mmd_delta": cmmd - ammd,
            }
        )
    write_rows(out_rows, rows)

    results = [feature_result(rows, feature, target) for feature in FEATURES for target in ("pp_delta", "mmd_delta")]
    pp_results = [r for r in results if r["target"] == "pp_delta" and r["rho"] is not None]
    mmd_results = [r for r in results if r["target"] == "mmd_delta" and r["rho"] is not None]
    best_pp = max(pp_results, key=lambda r: abs(float(r["rho"])), default=None)
    max_abs_mmd = max((abs(float(r["rho"])) for r in mmd_results), default=None)
    pp_vals = [float(row["pp_delta"]) for row in rows]
    mmd_vals = [float(row["mmd_delta"]) for row in rows]
    dataset_counts = Counter(row["dataset"] for row in rows)
    exact_bg_conditions = sum(1 for row in rows if fnum(row.get("exact_bg_frac")) and float(row["exact_bg_frac"]) > 0.0)

    reasons: list[str] = []
    if len(rows) < 50:
        reasons.append("eval_overlap_condition_count_below_50")
    if len(dataset_counts) < 3:
        reasons.append("eval_dataset_count_below_3")
    if exact_bg_conditions < 3:
        reasons.append("eval_exact_background_condition_count_below_3")
    if best_pp is None or best_pp.get("rho") is None or abs(float(best_pp["rho"])) < 0.25:
        reasons.append("best_lincs_pp_signal_abs_rho_below_0p25")
    if best_pp is None or best_pp.get("shuffle_p_abs") is None or float(best_pp["shuffle_p_abs"]) > 0.05:
        reasons.append("best_lincs_pp_signal_shuffle_p_above_0p05")
    if max_abs_mmd is not None and max_abs_mmd > 0.30:
        reasons.append("lincs_mmd_correlation_too_large")
    if mmd_vals and mean(mmd_vals) > 0.001:
        reasons.append("mean_mmd_delta_above_0p001")
    reasons.append("eval_only_not_training_or_promotion")

    status = "lincs_gse92742_train_gene_outcome_eval_pass_review_only_no_gpu" if len(reasons) == 1 else "lincs_gse92742_train_gene_outcome_eval_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": boundary,
        "summary": {
            "conditions": len(rows),
            "datasets": len(dataset_counts),
            "dataset_counts_top20": dataset_counts.most_common(20),
            "exact_background_conditions": exact_bg_conditions,
            "mean_pp_delta": mean(pp_vals) if pp_vals else None,
            "mean_mmd_delta": mean(mmd_vals) if mmd_vals else None,
            "best_pp_signal": best_pp,
            "max_abs_mmd_signal_rho": max_abs_mmd,
            "all_signal_results": results,
        },
        "reasons": reasons,
        "outputs": {
            "json": str(out_json),
            "markdown": str(out_md),
            "rows": str(out_rows),
        },
        "next_action": (
            "If this gate passes, send for external/protocol review before any "
            "new training. If it fails, close this GSE92742 outcome-materialized "
            "diagnostic as non-launchable."
        ),
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def fmt_signal(item: dict[str, Any] | None) -> str:
        if not item:
            return "`None`"
        rho = item.get("rho")
        pval = item.get("shuffle_p_abs")
        rho_s = "None" if rho is None else f"{float(rho):+.4f}"
        p_s = "None" if pval is None else f"{float(pval):.4f}"
        return f"`{item['feature']} -> {item['target']}: rho={rho_s}, shuffle_p={p_s}, n={item['n']}`"

    lines = [
        "# LINCS GSE92742 Train Gene Outcome Eval",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- Summarizes eval-only frozen anchor/candidate outputs on the strict GSE92742 S0 train/gene split.",
        "- No training, checkpoint selection, canonical multi selection, or Track C query.",
        "- Passing would authorize external/protocol review only, not promotion.",
        "",
        "## Summary",
        "",
        f"- conditions: `{len(rows)}`",
        f"- datasets: `{len(dataset_counts)}`",
        f"- exact-background conditions: `{exact_bg_conditions}`",
        f"- mean pp delta: `{payload['summary']['mean_pp_delta']}`",
        f"- mean MMD delta: `{payload['summary']['mean_mmd_delta']}`",
        f"- best pp signal: {fmt_signal(best_pp)}",
        f"- max |MMD signal rho|: `{max_abs_mmd}`",
        "",
        "## Decision",
        "",
        "This is an eval-only diagnostic. It cannot authorize training or model replacement by itself.",
        "",
        "## Reasons",
        "",
        *[f"- `{reason}`" for reason in reasons],
        "",
        "## Outputs",
        "",
        f"- JSON: `{out_json}`",
        f"- rows: `{out_rows}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": False, "out_md": str(out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
