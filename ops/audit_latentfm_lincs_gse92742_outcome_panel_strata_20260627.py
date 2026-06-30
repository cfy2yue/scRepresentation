#!/usr/bin/env python3
"""Stratified failure analysis for the GSE92742 outcome panel."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
PANEL_DIR = ROOT / "reports/lincs_gse92742_train_gene_candidate_panel_20260627"
TRUECELL = ROOT / "reports/lincs_gse92742_train_gene_outcome_eval_20260627/xverse_truecell_budget128_vs_anchor.csv"
OUT_JSON = ROOT / "reports/latentfm_lincs_gse92742_outcome_panel_strata_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_LINCS_GSE92742_OUTCOME_PANEL_STRATA_20260627.md"


def fnum(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) or math.isinf(out) else out


def read_rows(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pp = fnum(row.get("pp_delta"))
            mmd = fnum(row.get("mmd_delta"))
            if pp is None or mmd is None:
                continue
            row["pp_delta"] = pp
            row["mmd_delta"] = mmd
            row["exact_bg_frac"] = fnum(row.get("exact_bg_frac")) or 0.0
            out.append(row)
    return out


def ci(values: list[float], *, alpha: float = 0.05) -> list[float | None]:
    if not values:
        return [None, None]
    vals = sorted(values)
    lo = vals[max(0, int((alpha / 2) * len(vals)))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return [lo, hi]


def dataset_bootstrap(rows: list[dict[str, Any]], field: str, *, n_boot: int = 2000) -> list[float]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row.get("dataset", ""))].append(row)
    datasets = sorted(k for k in by_dataset if k)
    if not datasets:
        return []
    rng = random.Random(20260627)
    vals: list[float] = []
    for _ in range(n_boot):
        sample: list[float] = []
        for _ in datasets:
            ds = rng.choice(datasets)
            part = by_dataset[ds]
            if part:
                sample.append(float(rng.choice(part)[field]))
        if sample:
            vals.append(mean(sample))
    return vals


def label_from_csv(path: Path) -> str:
    name = path.name.removesuffix(".csv")
    if name == "xverse_truecell_budget128_vs_anchor":
        return "xverse_truecell_nested_budget128_tailstable_seed42_6000"
    return name.removesuffix("_vs_anchor")


def summarize(path: Path) -> dict[str, Any]:
    rows = read_rows(path)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row.get("dataset", ""))].append(row)
    dataset_pp = {
        ds: mean(float(row["pp_delta"]) for row in part) for ds, part in by_dataset.items() if part
    }
    dataset_mmd = {
        ds: mean(float(row["mmd_delta"]) for row in part) for ds, part in by_dataset.items() if part
    }
    exact = [row for row in rows if float(row.get("exact_bg_frac", 0.0)) > 0.0]
    nonexact = [row for row in rows if float(row.get("exact_bg_frac", 0.0)) <= 0.0]
    pp_boot = dataset_bootstrap(rows, "pp_delta")
    mmd_boot = dataset_bootstrap(rows, "mmd_delta")
    return {
        "label": label_from_csv(path),
        "path": str(path),
        "conditions": len(rows),
        "datasets": len(by_dataset),
        "mean_pp_delta": mean(float(row["pp_delta"]) for row in rows) if rows else None,
        "mean_mmd_delta": mean(float(row["mmd_delta"]) for row in rows) if rows else None,
        "dataset_pp_min": min(dataset_pp.values()) if dataset_pp else None,
        "dataset_pp_max": max(dataset_pp.values()) if dataset_pp else None,
        "dataset_mmd_max": max(dataset_mmd.values()) if dataset_mmd else None,
        "negative_dataset_pp_count": sum(1 for value in dataset_pp.values() if value < 0),
        "pp_dataset_bootstrap_ci95": ci(pp_boot),
        "mmd_dataset_bootstrap_ci95": ci(mmd_boot),
        "exact_bg_conditions": len(exact),
        "exact_bg_mean_pp_delta": mean(float(row["pp_delta"]) for row in exact) if exact else None,
        "exact_bg_mean_mmd_delta": mean(float(row["mmd_delta"]) for row in exact) if exact else None,
        "nonexact_mean_pp_delta": mean(float(row["pp_delta"]) for row in nonexact) if nonexact else None,
        "top_dataset_pp": sorted(dataset_pp.items(), key=lambda kv: kv[1], reverse=True)[:5],
        "bottom_dataset_pp": sorted(dataset_pp.items(), key=lambda kv: kv[1])[:5],
    }


def main() -> int:
    paths = [TRUECELL, *sorted(PANEL_DIR.glob("*_vs_anchor.csv"))]
    paths = [path for path in paths if path.is_file()]
    rows = [summarize(path) for path in paths]
    cap60 = next((row for row in rows if row["label"] == "xverse_scaling_cap60_6k_seed42"), None)
    reasons: list[str] = []
    if not cap60:
        reasons.append("cap60_row_missing")
    else:
        lo = cap60["pp_dataset_bootstrap_ci95"][0]
        if lo is None or float(lo) <= 0:
            reasons.append("cap60_dataset_bootstrap_pp_ci_crosses_zero")
        if cap60["dataset_pp_min"] is None or float(cap60["dataset_pp_min"]) < -0.01:
            reasons.append("cap60_dataset_tail_pp_harm")
        if cap60["negative_dataset_pp_count"] > 0:
            reasons.append("cap60_negative_dataset_tails_present")
        if cap60["exact_bg_mean_pp_delta"] is None or float(cap60["exact_bg_mean_pp_delta"]) <= 0:
            reasons.append("cap60_exact_background_mean_not_positive")
        if cap60["dataset_mmd_max"] is not None and float(cap60["dataset_mmd_max"]) > 0.001:
            reasons.append("cap60_dataset_mmd_tail_harm")
    status = (
        "lincs_gse92742_outcome_panel_strata_fail_close_no_gpu"
        if reasons
        else "lincs_gse92742_outcome_panel_strata_pass_review_only_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorized": False,
        "training_authorized": False,
        "promotion_authorized": False,
        "reasons": reasons,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def fmt(value: Any) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):+.6f}"
        return "`None`" if value is None else f"`{value}`"

    lines = [
        "# LatentFM LINCS GSE92742 Outcome Panel Strata",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only stratified analysis over existing eval-only CSVs.",
        "- No training, checkpoint selection, canonical multi selection, or Track C query.",
        "",
        "## Candidate Strata Summary",
        "",
        "| candidate | n | datasets | mean pp | pp boot CI95 | dataset pp min | neg datasets | exact-bg n | exact-bg mean pp | dataset MMD max |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        pp_ci = row["pp_dataset_bootstrap_ci95"]
        pp_ci_s = f"[{fmt(pp_ci[0])}, {fmt(pp_ci[1])}]"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['label']}`",
                    str(row["conditions"]),
                    str(row["datasets"]),
                    fmt(row["mean_pp_delta"]),
                    pp_ci_s,
                    fmt(row["dataset_pp_min"]),
                    str(row["negative_dataset_pp_count"]),
                    str(row["exact_bg_conditions"]),
                    fmt(row["exact_bg_mean_pp_delta"]),
                    fmt(row["dataset_mmd_max"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Decision", ""])
    if reasons:
        lines.append(
            "The only positive mean candidate, cap60 6k, fails dataset-tail / "
            "bootstrap / exact-background robustness. Do not reopen cap60 or "
            "GSE92742 training from this evidence."
        )
    else:
        lines.append(
            "Cap60 strata passed this CPU robustness check, but still requires "
            "external/protocol review because the LINCS signal gate failed."
        )
    lines.extend(["", "## Reasons", "", *[f"- `{reason}`" for reason in reasons], "", "## Outputs", "", f"- JSON: `{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
