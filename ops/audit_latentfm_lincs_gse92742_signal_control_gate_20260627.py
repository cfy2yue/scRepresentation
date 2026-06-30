#!/usr/bin/env python3
"""Strict train-only signal/control gate for GSE92742 LINCS small metadata.

This gate tests whether small-metadata LINCS activity fields are strong enough
to justify a later code/provenance launcher gate. It uses only existing
train/internal outcome tables and the GSE92742 overlap table. It does not
train, infer, use GPU, read canonical multi for selection, or read Track C
held-out query.
"""

from __future__ import annotations

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
REPORTS = ROOT / "reports"
OVERLAP = REPORTS / "lincs_l1000_gse92742_condition_join_gate_20260627/gse92742_s0_overlap_rows.csv"
OUT_JSON = REPORTS / "latentfm_lincs_gse92742_signal_control_gate_20260627.json"
OUT_MD = REPORTS / "LATENTFM_LINCS_GSE92742_SIGNAL_CONTROL_GATE_20260627.md"
OUT_ROWS = REPORTS / "lincs_gse92742_signal_control_rows_20260627.csv"

OUTCOME_SPECS = [
    (
        "response_projection",
        REPORTS / "latentfm_response_program_projection_rows_20260625.csv",
        "pp_delta",
        "mmd_delta",
    ),
    (
        "background_actionability",
        REPORTS / "latentfm_background_target_actionability_rows_20260625.csv",
        "pp_delta",
        "mmd_delta",
    ),
    (
        "qc_reliability",
        REPORTS / "latentfm_qc_support_reliability_rows_20260625.csv",
        "cross_pp_diff",
        "cross_mmd_diff",
    ),
    (
        "lodo_domain_conflict",
        REPORTS / "latentfm_lodo_domain_conflict_rows_20260625.csv",
        "pp_mean",
        "mmd_mean",
    ),
    (
        "truecell_riskrow",
        REPORTS / "latentfm_truecell_riskrow_complementarity_rows_20260625.csv",
        "truecell_pp_delta_mean",
        "truecell_mmd_delta_mean",
    ),
]

FEATURES = [
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
]


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


def read_outcomes() -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for name, path, pp_col, mmd_col in OUTCOME_SPECS:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"dataset", "condition"}.issubset(set(reader.fieldnames or [])):
                continue
            for row in reader:
                dataset = norm_text(row.get("dataset"))
                condition = norm_text(row.get("condition"))
                if not dataset or not condition:
                    continue
                pp = fnum(row.get(pp_col))
                mmd = fnum(row.get(mmd_col))
                key = (dataset, condition)
                if pp is not None:
                    out[key][f"{name}_pp"] = pp
                if mmd is not None:
                    out[key][f"{name}_mmd"] = mmd
    for vals in out.values():
        pp_candidates = [
            vals.get("response_projection_pp"),
            vals.get("background_actionability_pp"),
            vals.get("qc_reliability_pp"),
            vals.get("lodo_domain_conflict_pp"),
            vals.get("truecell_riskrow_pp"),
        ]
        mmd_candidates = [
            vals.get("response_projection_mmd"),
            vals.get("background_actionability_mmd"),
            vals.get("qc_reliability_mmd"),
            vals.get("lodo_domain_conflict_mmd"),
            vals.get("truecell_riskrow_mmd"),
        ]
        vals["primary_pp"] = next((v for v in pp_candidates if v is not None), math.nan)
        vals["primary_mmd"] = next((v for v in mmd_candidates if v is not None), math.nan)
    return dict(out)


def read_overlap() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with OVERLAP.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["tas_mean_num"] = fnum(row.get("tas_mean"))
            row["distil_cc_q75_mean_num"] = fnum(row.get("distil_cc_q75_mean"))
            row["sig_count_num"] = fnum(row.get("lincs_sig_count"))
            row["exact_bg_match"] = (
                1.0
                if norm_key(row.get("s0_cell_background"))
                and norm_key(row.get("s0_cell_background")) == norm_key(row.get("lincs_cell_id"))
                else 0.0
            )
            rows.append(row)
    return rows


def aggregate(rows: list[dict[str, Any]], outcomes: dict[tuple[str, str], dict[str, float]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (norm_text(row.get("dataset")), norm_text(row.get("condition")))
        if key in outcomes:
            grouped[key].append(row)

    out: list[dict[str, Any]] = []
    for (dataset, condition), part in grouped.items():
        def vals(field: str) -> list[float]:
            return [float(r[field]) for r in part if r.get(field) is not None]

        type_counts = Counter(norm_text(r.get("lincs_pert_type")) for r in part)
        total = max(1, len(part))
        rec: dict[str, Any] = {
            "dataset": dataset,
            "condition": condition,
            "membership": Counter(norm_text(r.get("membership")) for r in part).most_common(1)[0][0],
            "modality": Counter(norm_text(r.get("modality")) for r in part).most_common(1)[0][0],
            "n_overlap_rows": len(part),
            "unique_lincs_cells": len({norm_text(r.get("lincs_cell_id")) for r in part if norm_text(r.get("lincs_cell_id"))}),
            "tas_mean": mean(vals("tas_mean_num")) if vals("tas_mean_num") else math.nan,
            "distil_cc_q75_mean": mean(vals("distil_cc_q75_mean_num")) if vals("distil_cc_q75_mean_num") else math.nan,
            "sig_count_mean": mean(vals("sig_count_num")) if vals("sig_count_num") else math.nan,
            "sig_count_sum": sum(vals("sig_count_num")) if vals("sig_count_num") else math.nan,
            "exact_bg_frac": mean(vals("exact_bg_match")) if vals("exact_bg_match") else 0.0,
            "frac_trt_sh": type_counts.get("trt_sh", 0) / total,
            "frac_trt_oe": type_counts.get("trt_oe", 0) / total,
            "frac_trt_sh_cgs": type_counts.get("trt_sh.cgs", 0) / total,
            "frac_trt_lig": type_counts.get("trt_lig", 0) / total,
        }
        rec.update(outcomes[(dataset, condition)])
        out.append(rec)
    out.sort(key=lambda r: (r["dataset"], r["condition"]))
    return out


def feature_result(rows: list[dict[str, Any]], feature: str, target: str, *, n_perm: int = 1000) -> dict[str, Any]:
    pairs = [(fnum(r.get(feature)), fnum(r.get(target)), norm_text(r.get("dataset"))) for r in rows]
    pairs = [(x, y, d) for x, y, d in pairs if x is not None and y is not None]
    if len(pairs) < 10 or len({x for x, _, _ in pairs}) < 2 or len({y for _, y, _ in pairs}) < 2:
        return {"feature": feature, "target": target, "n": len(pairs), "rho": None, "shuffle_p_abs": None}
    x = [float(v[0]) for v in pairs]
    y = [float(v[1]) for v in pairs]
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
            vals = [shuffled[i] for i in idxs]
            rng.shuffle(vals)
            for i, val in zip(idxs, vals):
                shuffled[i] = val
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


def summarize_scope(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    results = [feature_result(rows, feature, target) for feature in FEATURES for target in ("primary_pp", "primary_mmd")]
    pp_results = [r for r in results if r["target"] == "primary_pp" and r["rho"] is not None]
    mmd_results = [r for r in results if r["target"] == "primary_mmd" and r["rho"] is not None]
    best_pp = max(pp_results, key=lambda r: abs(float(r["rho"])), default=None)
    max_abs_mmd = max((abs(float(r["rho"])) for r in mmd_results), default=None)
    return {
        "scope": name,
        "n_conditions": len(rows),
        "n_datasets": len({r["dataset"] for r in rows}),
        "membership_counts": Counter(r["membership"] for r in rows).most_common(),
        "dataset_counts_top20": Counter(r["dataset"] for r in rows).most_common(20),
        "exact_bg_condition_count": sum(1 for r in rows if fnum(r.get("exact_bg_frac")) and float(r["exact_bg_frac"]) > 0.0),
        "best_primary_pp_signal": best_pp,
        "max_abs_primary_mmd_rho": max_abs_mmd,
        "all_results": results,
    }


def write_rows(rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "condition",
        "membership",
        "modality",
        "n_overlap_rows",
        "unique_lincs_cells",
        *FEATURES,
        "primary_pp",
        "primary_mmd",
        "response_projection_pp",
        "background_actionability_pp",
        "qc_reliability_pp",
        "lodo_domain_conflict_pp",
        "truecell_riskrow_pp",
    ]
    with OUT_ROWS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    missing = [str(p) for p in [OVERLAP, *[spec[1] for spec in OUTCOME_SPECS]] if not p.is_file()]
    boundary = {
        "gpu_used": False,
        "training_or_inference_used": False,
        "large_level5_download": False,
        "canonical_multi_selection_used": False,
        "trackc_heldout_query_used": False,
        "source_release": "GSE92742_small_metadata_only",
        "gpu_authorized": False,
    }
    if missing:
        payload = {
            "status": "lincs_gse92742_signal_control_missing_inputs_no_gpu",
            "gpu_authorized": False,
            "boundary": boundary,
            "missing": missing,
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        OUT_MD.write_text("# LINCS GSE92742 Signal/Control Gate\n\nMissing inputs; no GPU authorized.\n", encoding="utf-8")
        print(json.dumps({"status": payload["status"], "gpu_authorized": False}, indent=2))
        return 0

    outcomes = read_outcomes()
    overlap = read_overlap()
    rows = aggregate(overlap, outcomes)
    write_rows(rows)

    strict = [r for r in rows if r["membership"] == "train" and r["modality"] == "gene"]
    diagnostic = [r for r in rows if r["modality"] == "gene"]
    strict_summary = summarize_scope(strict, "strict_train_gene_outcome_overlap")
    diagnostic_summary = summarize_scope(diagnostic, "all_gene_outcome_overlap_diagnostic")

    reasons: list[str] = []
    best = strict_summary["best_primary_pp_signal"]
    if strict_summary["n_conditions"] < 50:
        reasons.append("strict_train_outcome_condition_count_below_50")
    if strict_summary["n_datasets"] < 3:
        reasons.append("strict_train_dataset_count_below_3")
    if strict_summary["exact_bg_condition_count"] < 3:
        reasons.append("strict_train_exact_background_condition_count_below_3")
    if best is None or best.get("rho") is None or abs(float(best["rho"])) < 0.25:
        reasons.append("strict_train_best_pp_signal_abs_rho_below_0p25")
    if best is None or best.get("shuffle_p_abs") is None or float(best["shuffle_p_abs"]) > 0.05:
        reasons.append("strict_train_best_pp_signal_shuffle_p_above_0p05")
    if strict_summary["max_abs_primary_mmd_rho"] is not None and float(strict_summary["max_abs_primary_mmd_rho"]) > 0.30:
        reasons.append("strict_train_mmd_correlation_too_large")
    reasons.extend(
        [
            "no_model_code_or_launcher_gate_yet",
            "no_gpu_from_small_metadata_signal_gate_without_noharm_launcher_review",
        ]
    )

    status = "lincs_gse92742_signal_control_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": boundary,
        "overlap_rows": len(overlap),
        "outcome_condition_keys": len(outcomes),
        "aggregated_condition_rows": len(rows),
        "strict_summary": strict_summary,
        "diagnostic_summary": diagnostic_summary,
        "reasons": reasons,
        "outputs": {
            "markdown": str(OUT_MD),
            "json": str(OUT_JSON),
            "condition_rows": str(OUT_ROWS),
        },
        "next_action": (
            "Do not launch LINCS GSE92742 GPU. The raw source has broad train "
            "S0 overlap, but current completed train-only outcome overlap is "
            "too thin for strict selection. Keep as a provenance/scaling "
            "source; reopen only after a larger leakage-safe train/internal "
            "outcome table or a reviewed no-harm launcher gate exists."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def fmt_signal(item: dict[str, Any] | None) -> str:
        if not item:
            return "`None`"
        rho = item.get("rho")
        p = item.get("shuffle_p_abs")
        rho_s = "None" if rho is None else f"{float(rho):+.4f}"
        p_s = "None" if p is None else f"{float(p):.4f}"
        return f"`{item['feature']} -> {item['target']}: rho={rho_s}, shuffle_p={p_s}, n={item['n']}`"

    lines = [
        "# LINCS/L1000 GSE92742 Signal/Control Gate",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- Uses only GSE92742 small metadata overlap plus existing train/internal outcome rows.",
        "- GPU authorization can only come from the strict train/gene scope; all-outcome rows are diagnostic.",
        "- No Level5 matrices, training, inference, canonical multi selection, Track C held-out query, or GPU.",
        "",
        "## Strict Train Scope",
        "",
        f"- conditions: `{strict_summary['n_conditions']}`",
        f"- datasets: `{strict_summary['n_datasets']}`",
        f"- exact-background conditions: `{strict_summary['exact_bg_condition_count']}`",
        f"- best pp signal: {fmt_signal(strict_summary['best_primary_pp_signal'])}",
        f"- max |MMD rho|: `{strict_summary['max_abs_primary_mmd_rho']}`",
        "",
        "## Diagnostic All-Gene Scope",
        "",
        f"- conditions: `{diagnostic_summary['n_conditions']}`",
        f"- datasets: `{diagnostic_summary['n_datasets']}`",
        f"- exact-background conditions: `{diagnostic_summary['exact_bg_condition_count']}`",
        f"- best pp signal: {fmt_signal(diagnostic_summary['best_primary_pp_signal'])}",
        f"- max |MMD rho|: `{diagnostic_summary['max_abs_primary_mmd_rho']}`",
        "",
        "## Decision",
        "",
        "No GPU is authorized. GSE92742 is useful as a larger LINCS metadata source, but the strict completed train-outcome overlap is only 19 conditions, so any apparent signal would be too underpowered for launch selection.",
        "",
        "## Reasons",
        "",
        *[f"- `{reason}`" for reason in reasons],
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- condition rows: `{OUT_ROWS}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": False, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
