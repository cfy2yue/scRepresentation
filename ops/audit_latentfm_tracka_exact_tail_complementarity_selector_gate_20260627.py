#!/usr/bin/env python3
"""Exact-tail complementarity oracle/selector gate for closed Track A candidates.

CPU/report-only. This tests whether already-closed Track A candidate outputs
contain enough row-level complementarity to justify a new selector/adapter
branch. It is intentionally optimistic: the row oracle uses held-out paired
rows and therefore can never authorize GPU by itself. If even this oracle fails,
the selector/adapter branch should close before any code or training work.

No training, inference, checkpoint selection, canonical multi selection, Track C
query, or GPU.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
GATE_DIR = ROOT / "reports/tracka_exact_tail_candidate_gate_20260627"
OUT_DIR = ROOT / "reports/tracka_exact_tail_complementarity_selector_gate_20260627"
OUT_ROWS = OUT_DIR / "tracka_exact_tail_complementarity_oracle_rows.csv"
OUT_SUMMARY = OUT_DIR / "tracka_exact_tail_complementarity_summary.csv"
OUT_JSON = ROOT / "reports/latentfm_tracka_exact_tail_complementarity_selector_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_EXACT_TAIL_COMPLEMENTARITY_SELECTOR_GATE_20260627.md"

GROUPS = [
    "exact_simple_single_unseen",
    "exact_cross_background_seen_gene",
    "recurrent_simple_hard_tail",
    "recurrent_cross_background_hard_tail",
    "canonical_test_single",
    "canonical_family_gene",
]
PRIMARY_GROUPS = {
    "exact_simple_single_unseen",
    "exact_cross_background_seen_gene",
    "recurrent_cross_background_hard_tail",
}
EXCLUDE_PATTERNS = ("anchor_selfcheck", "seed43")
MMD_CAP = 0.001
N_BOOT = 5000


def fnum(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def bootstrap(vals: list[float], seed: int) -> dict[str, float | None]:
    if not vals:
        return {"ci_low": None, "ci_high": None, "p_gt0": None, "p_lt0": None}
    arr = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(N_BOOT, len(arr)))
    boots = arr[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"ci_low": float(lo), "ci_high": float(hi), "p_gt0": float(np.mean(boots > 0.0)), "p_lt0": float(np.mean(boots < 0.0))}


def load_candidate_rows() -> dict[str, dict[tuple[str, str], list[dict[str, Any]]]]:
    by_group: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(GATE_DIR.glob("*_paired_rows.csv")):
        name = path.name.replace("_paired_rows.csv", "")
        if any(pattern in name for pattern in EXCLUDE_PATTERNS):
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                group = row.get("group", "")
                if group not in GROUPS:
                    continue
                pp = fnum(row.get("delta_pearson_pert"))
                mmd = fnum(row.get("delta_test_mmd_clamped"))
                if pp is None or mmd is None:
                    continue
                key = (row["dataset"], row["condition"])
                by_group[group][key].append(
                    {
                        "candidate": name,
                        "delta_pearson_pert": pp,
                        "delta_test_mmd_clamped": mmd,
                    }
                )
    return by_group


def choose_oracle(rows: dict[str, dict[tuple[str, str], list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    out = []
    for group in GROUPS:
        for (dataset, condition), opts in sorted(rows.get(group, {}).items()):
            choices = [{"candidate": "anchor", "delta_pearson_pert": 0.0, "delta_test_mmd_clamped": 0.0}]
            choices.extend([item for item in opts if float(item["delta_test_mmd_clamped"]) <= MMD_CAP])
            best = max(choices, key=lambda item: (float(item["delta_pearson_pert"]), -float(item["delta_test_mmd_clamped"])))
            out.append(
                {
                    "group": group,
                    "dataset": dataset,
                    "condition": condition,
                    "selected_candidate": best["candidate"],
                    "delta_pearson_pert": float(best["delta_pearson_pert"]),
                    "delta_test_mmd_clamped": float(best["delta_test_mmd_clamped"]),
                    "n_candidate_options": len(opts),
                    "n_options_after_mmd_cap": len(choices) - 1,
                }
            )
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for group in GROUPS:
        sub = [r for r in rows if r["group"] == group]
        vals = [float(r["delta_pearson_pert"]) for r in sub]
        mmds = [float(r["delta_test_mmd_clamped"]) for r in sub]
        bs = bootstrap(vals, seed=61027 + GROUPS.index(group))
        selected = Counter(r["selected_candidate"] for r in sub)
        summaries.append(
            {
                "group": group,
                "n": len(sub),
                "n_datasets": len({r["dataset"] for r in sub}),
                "oracle_pp_delta_mean": mean(vals) if vals else None,
                "oracle_pp_ci_low": bs["ci_low"],
                "oracle_pp_ci_high": bs["ci_high"],
                "oracle_pp_p_harm": bs["p_lt0"],
                "oracle_mmd_delta_mean": mean(mmds) if mmds else None,
                "oracle_mmd_max": max(mmds) if mmds else None,
                "anchor_selected_frac": (selected.get("anchor", 0) / len(sub)) if sub else None,
                "top_selected": ";".join(f"{k}:{v}" for k, v in selected.most_common(6)),
            }
        )
    return summaries


def decide(summaries: list[dict[str, Any]]) -> tuple[str, list[str]]:
    by_group = {row["group"]: row for row in summaries}
    reasons = ["row_oracle_is_heldout_diagnostic_not_gpu_authorizing"]
    exact_simple = by_group["exact_simple_single_unseen"]
    exact_cross = by_group["exact_cross_background_seen_gene"]
    recur_cross = by_group["recurrent_cross_background_hard_tail"]
    if (exact_cross["oracle_pp_delta_mean"] or -999.0) < 0.020:
        reasons.append("oracle_exact_cross_pp_below_0p020")
    if (recur_cross["oracle_pp_delta_mean"] or -999.0) < 0.050:
        reasons.append("oracle_recurrent_cross_tail_pp_below_0p050")
    if (exact_simple["oracle_pp_delta_mean"] or -999.0) < 0.0:
        reasons.append("oracle_exact_simple_pp_negative")
    for group in PRIMARY_GROUPS:
        row = by_group[group]
        if (row["oracle_mmd_delta_mean"] or 999.0) > MMD_CAP:
            reasons.append(f"{group}_oracle_mmd_mean_above_cap")
        if (row["oracle_pp_p_harm"] or 0.0) > 0.35:
            reasons.append(f"{group}_oracle_pp_p_harm_above_0p35")
    if len(reasons) == 1:
        status = "tracka_exact_tail_complementarity_oracle_signal_selector_needed_no_gpu"
    else:
        status = "tracka_exact_tail_complementarity_oracle_fail_no_gpu"
    return status, reasons


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_group = load_candidate_rows()
    oracle_rows = choose_oracle(rows_by_group)
    summaries = summarize(oracle_rows)
    status, reasons = decide(summaries)

    with OUT_ROWS.open("w", newline="", encoding="utf-8") as handle:
        fields = ["group", "dataset", "condition", "selected_candidate", "delta_pearson_pert", "delta_test_mmd_clamped", "n_candidate_options", "n_options_after_mmd_cap"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in oracle_rows])
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        fields = list(summaries[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    payload = {
        "status": status,
        "gpu_authorized": False,
        "reasons": reasons,
        "boundary": {
            "mode": "cpu_report_only_heldout_row_oracle_diagnostic",
            "mmd_cap_per_selected_row": MMD_CAP,
            "canonical_multi_used": False,
            "trackc_query_used": False,
            "training_or_inference_run": False,
            "seed43_excluded": True,
        },
        "summaries": summaries,
        "outputs": {"rows": str(OUT_ROWS), "summary": str(OUT_SUMMARY), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Track A Exact-Tail Complementarity Selector Gate 2026-06-27",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only over existing closed-candidate paired rows from the exact-tail gate directory.",
        "- Excludes anchor self-check and seed43 rows; this is a seed42 diagnostic oracle over comparable candidate files.",
        f"- Per row, the oracle may choose any candidate whose MMD delta is `<= {MMD_CAP:+.3f}`, or anchor (`0` delta).",
        "- This is an optimistic held-out row oracle, so it cannot authorize GPU even if it passes.",
        "- No training, inference, checkpoint selection, canonical multi selection, or Track C query.",
        "",
        "## Decision",
        "",
        f"- reasons: `{';'.join(reasons)}`",
        "",
        "## Oracle Summary",
        "",
        "| group | n | pp delta | pp CI | MMD delta | anchor frac | top selected |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| {group} | {n} | {pp} | [{lo}, {hi}] | {mmd} | {anchor} | `{top}` |".format(
                group=row["group"],
                n=row["n"],
                pp=fmt(row["oracle_pp_delta_mean"]),
                lo=fmt(row["oracle_pp_ci_low"]),
                hi=fmt(row["oracle_pp_ci_high"]),
                mmd=fmt(row["oracle_mmd_delta_mean"]),
                anchor=fmt(row["anchor_selected_frac"]),
                top=row["top_selected"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A failing oracle closes the complementarity-selector route before any GPU. An oracle-only signal would still require a train-only selector with shuffled controls and external audit before a bounded smoke.",
            "",
            "## Outputs",
            "",
            f"- rows: `{OUT_ROWS}`",
            f"- summary: `{OUT_SUMMARY}`",
            f"- json: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "reasons": reasons, "report": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
