#!/usr/bin/env python3
"""Retrospective Track A train-only selector feasibility diagnostic.

This script tests whether the local ``xverse_conddelta_seed42`` Jiang/recurrent
hard-tail clue can be expressed by simple metadata-only selectors. It reads
existing paired posthoc rows only. It does not train, infer, select a
checkpoint, read canonical multi for selection, or read Track C query.

Important: because several selectors are compared on exact/canonical rows, this
report is diagnostic and cannot itself authorize GPU. A launch would require a
separate train-only selector/proxy gate or an external audit that freezes the
selector before any held-out evaluation.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
PAIRED = ROOT / "reports/tracka_exact_tail_candidate_gate_20260627/xverse_conddelta_seed42_paired_rows.csv"
OUT_JSON = ROOT / "reports/latentfm_tracka_trainonly_selector_feasibility_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_TRAINONLY_SELECTOR_FEASIBILITY_20260627.md"

GROUPS = (
    "canonical_test_single",
    "canonical_family_gene",
    "exact_simple_single_unseen",
    "exact_cross_background_seen_gene",
    "recurrent_simple_hard_tail",
    "recurrent_cross_background_hard_tail",
)


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def bootstrap(vals: list[float], *, seed: int, n_boot: int = 5000) -> dict[str, float | None]:
    if not vals:
        return {"ci_low": None, "ci_high": None, "p_gt0": None, "p_lt0": None}
    arr = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = arr[idx].mean(axis=1)
    return {
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
        "p_gt0": float(np.mean(boots > 0.0)),
        "p_lt0": float(np.mean(boots < 0.0)),
    }


def is_jiang(row: dict[str, Any]) -> bool:
    return str(row["dataset"]).startswith("Jiang_")


def is_jiang_cytokine(row: dict[str, Any]) -> bool:
    return str(row["dataset"]) in {"Jiang_IFNB", "Jiang_IFNG", "Jiang_TGFB", "Jiang_TNFA"}


def is_jiang_ifn(row: dict[str, Any]) -> bool:
    return str(row["dataset"]) in {"Jiang_IFNB", "Jiang_IFNG"}


def dataset_is(name: str) -> Callable[[dict[str, Any]], bool]:
    return lambda row: str(row["dataset"]) == name


SELECTORS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "jiang_all": is_jiang,
    "jiang_cytokine_no_ins": is_jiang_cytokine,
    "jiang_ifn": is_jiang_ifn,
    "jiang_ifnb_only": dataset_is("Jiang_IFNB"),
    "jiang_ifng_only": dataset_is("Jiang_IFNG"),
    "jiang_tnfa_only": dataset_is("Jiang_TNFA"),
    "jiang_tgfb_only": dataset_is("Jiang_TGFB"),
    "jiang_ins_only": dataset_is("Jiang_INS"),
}


def read_rows() -> list[dict[str, Any]]:
    out = []
    with PAIRED.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("group") not in GROUPS:
                continue
            pp = fnum(row.get("delta_pearson_pert"))
            mmd = fnum(row.get("delta_test_mmd_clamped"))
            if pp is None or mmd is None:
                continue
            out.append(
                {
                    "group": str(row["group"]),
                    "dataset": str(row["dataset"]),
                    "condition": str(row["condition"]),
                    "delta_pearson_pert": pp,
                    "delta_test_mmd_clamped": mmd,
                }
            )
    return out


def summarize_group(selector: Callable[[dict[str, Any]], bool], group: str, rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    group_rows = [row for row in rows if row["group"] == group]
    selected = [selector(row) for row in group_rows]
    out = []
    for metric, lower_is_better in (("delta_pearson_pert", False), ("delta_test_mmd_clamped", True)):
        vals = [float(row[metric]) if active else 0.0 for row, active in zip(group_rows, selected)]
        bs = bootstrap(vals, seed=seed + (1 if lower_is_better else 0))
        out.append(
            {
                "group": group,
                "metric": metric.replace("delta_", ""),
                "n": len(vals),
                "n_active": int(sum(selected)),
                "delta_mean": float(np.mean(vals)) if vals else None,
                "ci_low": bs["ci_low"],
                "ci_high": bs["ci_high"],
                "p_improve": bs["p_lt0"] if lower_is_better else bs["p_gt0"],
                "p_harm": bs["p_gt0"] if lower_is_better else bs["p_lt0"],
                "lower_is_better": lower_is_better,
            }
        )
    return out


def per_dataset(rows: list[dict[str, Any]], selector: Callable[[dict[str, Any]], bool], group: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["group"] == group and selector(row):
            buckets[str(row["dataset"])].append(row)
    out = []
    for dataset, ds_rows in sorted(buckets.items()):
        out.append(
            {
                "dataset": dataset,
                "n": len(ds_rows),
                "pp_mean": float(np.mean([float(row["delta_pearson_pert"]) for row in ds_rows])),
                "mmd_mean": float(np.mean([float(row["delta_test_mmd_clamped"]) for row in ds_rows])),
            }
        )
    return out


def decide_selector(summary: list[dict[str, Any]], ds_exact: list[dict[str, Any]]) -> list[str]:
    lookup = {(row["group"], row["metric"]): row for row in summary}
    reasons = []
    for group in ("canonical_test_single", "canonical_family_gene", "exact_simple_single_unseen"):
        pp = lookup[(group, "pearson_pert")]
        mmd = lookup[(group, "test_mmd_clamped")]
        if float(pp["delta_mean"] or 0.0) < -0.002 or float(pp["p_harm"] or 0.0) > 0.35:
            reasons.append(f"{group}_pp_noharm_fail")
        if float(mmd["delta_mean"] or 0.0) > 0.001 or float(mmd["p_harm"] or 0.0) > 0.80:
            reasons.append(f"{group}_mmd_noharm_fail")
    exact_pp = lookup[("exact_cross_background_seen_gene", "pearson_pert")]
    exact_mmd = lookup[("exact_cross_background_seen_gene", "test_mmd_clamped")]
    recurrent_pp = lookup[("recurrent_cross_background_hard_tail", "pearson_pert")]
    recurrent_mmd = lookup[("recurrent_cross_background_hard_tail", "test_mmd_clamped")]
    if int(exact_pp["n_active"]) < 10:
        reasons.append("exact_cross_active_n_lt_10")
    if float(exact_pp["delta_mean"] or 0.0) < 0.01 or float(exact_pp["p_improve"] or 0.0) < 0.90:
        reasons.append("exact_cross_material_gain_fail")
    if float(exact_mmd["delta_mean"] or 0.0) > 0.001:
        reasons.append("exact_cross_mmd_noharm_fail")
    if float(recurrent_pp["delta_mean"] or 0.0) < 0.005:
        reasons.append("recurrent_cross_tail_gain_fail")
    if float(recurrent_mmd["delta_mean"] or 0.0) > 0.001:
        reasons.append("recurrent_cross_mmd_noharm_fail")
    if ds_exact and min(float(row["pp_mean"]) for row in ds_exact) < -0.01:
        reasons.append("exact_cross_active_dataset_min_lt_minus_0p01")
    return reasons


def main() -> None:
    rows = read_rows()
    selector_reports = []
    for sidx, (name, selector) in enumerate(SELECTORS.items()):
        summary: list[dict[str, Any]] = []
        for gidx, group in enumerate(GROUPS):
            summary.extend(summarize_group(selector, group, rows, seed=20260627 + sidx * 101 + gidx * 13))
        ds_exact = per_dataset(rows, selector, "exact_cross_background_seen_gene")
        reasons = decide_selector(summary, ds_exact)
        lookup = {(row["group"], row["metric"]): row for row in summary}
        selector_reports.append(
            {
                "selector": name,
                "status": "retrospective_selector_pass_needs_trainonly_freeze" if not reasons else "retrospective_selector_fail_no_gpu",
                "gpu_authorized": False,
                "reasons": reasons,
                "summary": summary,
                "exact_cross_dataset_breakdown": ds_exact,
                "score": {
                    "exact_cross_pp": lookup[("exact_cross_background_seen_gene", "pearson_pert")]["delta_mean"],
                    "exact_cross_mmd": lookup[("exact_cross_background_seen_gene", "test_mmd_clamped")]["delta_mean"],
                    "recurrent_cross_pp": lookup[("recurrent_cross_background_hard_tail", "pearson_pert")]["delta_mean"],
                    "canonical_test_single_pp": lookup[("canonical_test_single", "pearson_pert")]["delta_mean"],
                },
            }
        )

    ranked = sorted(
        selector_reports,
        key=lambda item: (
            item["status"].startswith("retrospective_selector_pass"),
            float(item["score"]["exact_cross_pp"] or -999),
            float(item["score"]["recurrent_cross_pp"] or -999),
        ),
        reverse=True,
    )
    any_pass = any(item["status"].startswith("retrospective_selector_pass") for item in selector_reports)
    payload = {
        "status": "tracka_trainonly_selector_feasibility_has_retrospective_pass_no_gpu" if any_pass else "tracka_trainonly_selector_feasibility_fail_close_no_gpu",
        "gpu_authorized": False,
        "boundary": {
            "cpu_report_only": True,
            "training": False,
            "inference": False,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
            "retrospective_multiple_selector_comparison": True,
            "launch_requires_independent_trainonly_freeze": True,
        },
        "selector_reports": selector_reports,
        "ranked_selectors": [item["selector"] for item in ranked],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track A Train-Only Selector Feasibility",
        "",
        f"Status: `{payload['status']}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU/report-only retrospective diagnostic over existing `xverse_conddelta_seed42` paired rows. Selectors are metadata-only, but comparing several selectors on exact/canonical rows means this report cannot authorize GPU by itself.",
        "",
        "## Selector Ranking",
        "",
        "| selector | status | exact cross pp | exact cross MMD | recurrent cross pp | canonical single pp | reasons |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in ranked:
        reasons = ";".join(item["reasons"]) or "none"
        lines.append(
            f"| `{item['selector']}` | `{item['status']}` | "
            f"{float(item['score']['exact_cross_pp'] or 0):+.6f} | "
            f"{float(item['score']['exact_cross_mmd'] or 0):+.6f} | "
            f"{float(item['score']['recurrent_cross_pp'] or 0):+.6f} | "
            f"{float(item['score']['canonical_test_single_pp'] or 0):+.6f} | `{reasons}` |"
        )
    lines.extend(["", "## Best Selector Detail", ""])
    best = ranked[0] if ranked else None
    if best is not None:
        lines.append(f"Best retrospective selector: `{best['selector']}`.")
        lines.append("")
        lines.append("| group | metric | n | active | delta | CI95 | p improve | p harm |")
        lines.append("|---|---|---:|---:|---:|---|---:|---:|")
        for row in best["summary"]:
            lines.append(
                f"| `{row['group']}` | `{row['metric']}` | {row['n']} | {row['n_active']} | "
                f"{float(row['delta_mean'] or 0):+.6f} | [{row['ci_low']}, {row['ci_high']}] | "
                f"{float(row['p_improve'] or 0):.4f} | {float(row['p_harm'] or 0):.4f} |"
            )
        lines.extend(["", "Exact-cross active dataset breakdown:", ""])
        lines.append("| dataset | n | pp mean | MMD mean |")
        lines.append("|---|---:|---:|---:|")
        for row in best["exact_cross_dataset_breakdown"]:
            lines.append(f"| `{row['dataset']}` | {row['n']} | {row['pp_mean']:+.6f} | {row['mmd_mean']:+.6f} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "No GPU is authorized. A positive retrospective selector would still need an independently frozen train-only gate before launch; a failed result closes the current Jiang metadata-selector route as a GPU candidate.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
