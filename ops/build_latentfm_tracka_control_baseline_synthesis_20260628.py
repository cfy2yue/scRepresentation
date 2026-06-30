#!/usr/bin/env python3
"""Build Track A control-baseline synthesis tables.

Combines explicit Track A proxy condition rows with exact cap2048 control MMD
rows. This is report-only: no model, no training, no threshold selection.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
EXPLICIT = ROOT / "reports/tracka_explicit_group_proxy_benchmark_20260628/condition_rows.csv"
CTRL_MMD = ROOT / "reports/tracka_ctrl_mmd_gate_20260628/tracka_ctrl_mmd_rows_cap2048.csv"
OUT_DIR = ROOT / "reports/tracka_control_baseline_synthesis_20260628"
OUT_JSON = ROOT / "reports/latentfm_tracka_control_baseline_synthesis_20260628.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_CONTROL_BASELINE_SYNTHESIS_20260628.md"

GROUP_ORDER = (
    "all_test_single_proxy",
    "cross_background_seen_gene_proxy",
    "family_gene",
    "simple_single_unseen_global_gene_proxy",
)


def is_composite_background(text: str) -> bool:
    parts = [p.strip() for p in re.split(r"[/;,|]+", str(text or "")) if p.strip()]
    return len(parts) > 1


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def bootstrap(vals: list[float], *, seed: int = 20260628) -> dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"ci_low": 0.0, "ci_high": 0.0, "p_gt0": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(5000, arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_gt0": float(np.mean(means > 0.0)),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_ds_pp: dict[str, list[float]] = defaultdict(list)
    by_ds_mmd: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_ds_pp[str(row["dataset"])].append(float(row["ctrl_minus_anchor_pp"]))
        by_ds_mmd[str(row["dataset"])].append(float(row["ctrl_minus_anchor_mmd"]))
    ds_pp = [float(np.mean(v)) for v in by_ds_pp.values()]
    ds_mmd = [float(np.mean(v)) for v in by_ds_mmd.values()]
    neg = [r for r in rows if float(r["anchor_pp"]) < 0.0]
    composite = [r for r in rows if bool(r["is_composite_background"])]
    return {
        "n": len(rows),
        "n_datasets": len(by_ds_pp),
        "anchor_pp": float(np.mean([r["anchor_pp"] for r in rows])) if rows else 0.0,
        "ctrl_pp": float(np.mean([r["ctrl_pp"] for r in rows])) if rows else 0.0,
        "ctrl_minus_anchor_pp": float(np.mean(ds_pp)) if ds_pp else 0.0,
        "ctrl_minus_anchor_pp_ci": bootstrap(ds_pp),
        "anchor_mmd": float(np.mean([r["anchor_mmd"] for r in rows])) if rows else 0.0,
        "ctrl_mmd": float(np.mean([r["ctrl_mmd"] for r in rows])) if rows else 0.0,
        "ctrl_minus_anchor_mmd": float(np.mean(ds_mmd)) if ds_mmd else 0.0,
        "ctrl_minus_anchor_mmd_ci": bootstrap(ds_mmd),
        "dataset_min_pp_delta": float(min(ds_pp)) if ds_pp else 0.0,
        "dataset_max_mmd_delta": float(max(ds_mmd)) if ds_mmd else 0.0,
        "anchor_negative_rows": len(neg),
        "anchor_negative_fraction": float(len(neg) / len(rows)) if rows else 0.0,
        "negative_rows_ctrl_better_fraction": float(np.mean([r["ctrl_pp"] > r["anchor_pp"] for r in neg])) if neg else 0.0,
        "negative_rows_ctrl_pp_delta": float(np.mean([r["ctrl_minus_anchor_pp"] for r in neg])) if neg else 0.0,
        "ctrl_better_and_mmd_nonharm_fraction": float(
            np.mean([(r["ctrl_pp"] > r["anchor_pp"]) and (r["ctrl_minus_anchor_mmd"] <= 0.003) for r in rows])
        ) if rows else 0.0,
        "composite_n": len(composite),
        "composite_anchor_pp": float(np.mean([r["anchor_pp"] for r in composite])) if composite else 0.0,
        "composite_ctrl_pp": float(np.mean([r["ctrl_pp"] for r in composite])) if composite else 0.0,
        "composite_mmd_delta": float(np.mean([r["ctrl_minus_anchor_mmd"] for r in composite])) if composite else 0.0,
    }


def main() -> None:
    explicit_rows = load_csv(EXPLICIT)
    mmd_rows = load_csv(CTRL_MMD)
    mmd_by_key = {
        (r["seed"], r["explicit_group"], r["dataset"], r["condition"]): r
        for r in mmd_rows
    }
    rows: list[dict[str, Any]] = []
    missing = []
    for erow in explicit_rows:
        key = (erow["seed"], erow["explicit_group"], erow["dataset"], erow["condition"])
        mrow = mmd_by_key.get(key)
        if mrow is None:
            missing.append(key)
            continue
        anchor_pp = float(erow["pearson_pert"])
        ctrl_pp = float(erow["pearson_ctrl"])
        anchor_mmd = float(mrow["anchor_mmd_clamped"])
        ctrl_mmd = float(mrow["ctrl_mmd_clamped"])
        rows.append(
            {
                "seed": erow["seed"],
                "explicit_group": erow["explicit_group"],
                "dataset": erow["dataset"],
                "condition": erow["condition"],
                "cell_background": erow["cell_background"],
                "single_gene": erow["single_gene"],
                "is_composite_background": is_composite_background(erow["cell_background"]),
                "anchor_pp": anchor_pp,
                "ctrl_pp": ctrl_pp,
                "ctrl_minus_anchor_pp": ctrl_pp - anchor_pp,
                "anchor_mmd": anchor_mmd,
                "ctrl_mmd": ctrl_mmd,
                "ctrl_minus_anchor_mmd": ctrl_mmd - anchor_mmd,
                "ctrl_better_pp": ctrl_pp > anchor_pp,
                "ctrl_mmd_nonharm": ctrl_mmd - anchor_mmd <= 0.003,
            }
        )

    group_summary: dict[str, Any] = {}
    for seed in sorted({r["seed"] for r in rows}):
        group_summary[seed] = {}
        for group in GROUP_ORDER:
            part = [r for r in rows if r["seed"] == seed and r["explicit_group"] == group]
            if part:
                group_summary[seed][group] = summarize_rows(part)

    recurrent: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["anchor_pp"] < 0.0:
            recurrent[(row["dataset"], row["condition"])].append(row)
    recurrent_rows = []
    for (dataset, condition), part in recurrent.items():
        recurrent_rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "negative_row_count": len(part),
                "groups": ";".join(sorted({r["explicit_group"] for r in part})),
                "mean_anchor_pp": float(np.mean([r["anchor_pp"] for r in part])),
                "mean_ctrl_pp": float(np.mean([r["ctrl_pp"] for r in part])),
                "mean_ctrl_minus_anchor_pp": float(np.mean([r["ctrl_minus_anchor_pp"] for r in part])),
                "mean_ctrl_minus_anchor_mmd": float(np.mean([r["ctrl_minus_anchor_mmd"] for r in part])),
                "all_ctrl_mmd_nonharm": bool(all(r["ctrl_mmd_nonharm"] for r in part)),
            }
        )
    recurrent_rows.sort(key=lambda r: (-int(r["negative_row_count"]), -float(r["mean_ctrl_minus_anchor_pp"]), r["dataset"], r["condition"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    group_csv = OUT_DIR / "group_summary.csv"
    recurrent_csv = OUT_DIR / "recurrent_control_baseline_failures.csv"
    row_csv = OUT_DIR / "joined_rows.csv"
    with group_csv.open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "seed",
            "explicit_group",
            "n",
            "n_datasets",
            "anchor_pp",
            "ctrl_pp",
            "ctrl_minus_anchor_pp",
            "pp_ci_low",
            "pp_ci_high",
            "anchor_mmd",
            "ctrl_mmd",
            "ctrl_minus_anchor_mmd",
            "mmd_ci_low",
            "mmd_ci_high",
            "dataset_min_pp_delta",
            "dataset_max_mmd_delta",
            "anchor_negative_fraction",
            "negative_rows_ctrl_better_fraction",
            "negative_rows_ctrl_pp_delta",
            "ctrl_better_and_mmd_nonharm_fraction",
            "composite_n",
            "composite_anchor_pp",
            "composite_ctrl_pp",
            "composite_mmd_delta",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for seed, groups in group_summary.items():
            for group, s in groups.items():
                writer.writerow(
                    {
                        "seed": seed,
                        "explicit_group": group,
                        "n": s["n"],
                        "n_datasets": s["n_datasets"],
                        "anchor_pp": s["anchor_pp"],
                        "ctrl_pp": s["ctrl_pp"],
                        "ctrl_minus_anchor_pp": s["ctrl_minus_anchor_pp"],
                        "pp_ci_low": s["ctrl_minus_anchor_pp_ci"]["ci_low"],
                        "pp_ci_high": s["ctrl_minus_anchor_pp_ci"]["ci_high"],
                        "anchor_mmd": s["anchor_mmd"],
                        "ctrl_mmd": s["ctrl_mmd"],
                        "ctrl_minus_anchor_mmd": s["ctrl_minus_anchor_mmd"],
                        "mmd_ci_low": s["ctrl_minus_anchor_mmd_ci"]["ci_low"],
                        "mmd_ci_high": s["ctrl_minus_anchor_mmd_ci"]["ci_high"],
                        "dataset_min_pp_delta": s["dataset_min_pp_delta"],
                        "dataset_max_mmd_delta": s["dataset_max_mmd_delta"],
                        "anchor_negative_fraction": s["anchor_negative_fraction"],
                        "negative_rows_ctrl_better_fraction": s["negative_rows_ctrl_better_fraction"],
                        "negative_rows_ctrl_pp_delta": s["negative_rows_ctrl_pp_delta"],
                        "ctrl_better_and_mmd_nonharm_fraction": s["ctrl_better_and_mmd_nonharm_fraction"],
                        "composite_n": s["composite_n"],
                        "composite_anchor_pp": s["composite_anchor_pp"],
                        "composite_ctrl_pp": s["composite_ctrl_pp"],
                        "composite_mmd_delta": s["composite_mmd_delta"],
                    }
                )
    with recurrent_csv.open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "dataset",
            "condition",
            "negative_row_count",
            "groups",
            "mean_anchor_pp",
            "mean_ctrl_pp",
            "mean_ctrl_minus_anchor_pp",
            "mean_ctrl_minus_anchor_mmd",
            "all_ctrl_mmd_nonharm",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(recurrent_rows)
    with row_csv.open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "seed",
            "explicit_group",
            "dataset",
            "condition",
            "cell_background",
            "single_gene",
            "is_composite_background",
            "anchor_pp",
            "ctrl_pp",
            "ctrl_minus_anchor_pp",
            "anchor_mmd",
            "ctrl_mmd",
            "ctrl_minus_anchor_mmd",
            "ctrl_better_pp",
            "ctrl_mmd_nonharm",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "status": "tracka_control_baseline_synthesis_ready_no_gpu",
        "gpu_authorized": False,
        "boundary": {
            "report_only": True,
            "selection_weight": 0,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
            "inputs": {"explicit_rows": str(EXPLICIT), "ctrl_mmd_rows": str(CTRL_MMD)},
        },
        "n_rows": len(rows),
        "missing_join_keys": len(missing),
        "group_summary": group_summary,
        "top_recurrent_anchor_negative": recurrent_rows[:30],
        "outputs": {
            "group_summary_csv": str(group_csv),
            "recurrent_failures_csv": str(recurrent_csv),
            "joined_rows_csv": str(row_csv),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track A Control Baseline Synthesis",
        "",
        "Status: `tracka_control_baseline_synthesis_ready_no_gpu`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "Report-only synthesis joining frozen explicit Track A proxy rows with exact cap2048 control/source MMD rows. No model, training, threshold selection, canonical multi selection, or Track C query is used.",
        "",
        "## Group Summary",
        "",
        "| seed | group | n | anchor pp | ctrl pp | ctrl-anchor pp | pp 95% CI | anchor MMD | ctrl MMD | ctrl-anchor MMD | MMD 95% CI | neg rows | neg ctrl better | ctrl better + MMD nonharm |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for seed, groups in sorted(group_summary.items()):
        for group in GROUP_ORDER:
            if group not in groups:
                continue
            s = groups[group]
            lines.append(
                f"| `{seed}` | `{group}` | {s['n']} | {s['anchor_pp']:+.6f} | {s['ctrl_pp']:+.6f} | "
                f"{s['ctrl_minus_anchor_pp']:+.6f} | [{s['ctrl_minus_anchor_pp_ci']['ci_low']:+.6f},{s['ctrl_minus_anchor_pp_ci']['ci_high']:+.6f}] | "
                f"{s['anchor_mmd']:+.6f} | {s['ctrl_mmd']:+.6f} | {s['ctrl_minus_anchor_mmd']:+.6f} | "
                f"[{s['ctrl_minus_anchor_mmd_ci']['ci_low']:+.6f},{s['ctrl_minus_anchor_mmd_ci']['ci_high']:+.6f}] | "
                f"{s['anchor_negative_fraction']:.3f} | {s['negative_rows_ctrl_better_fraction']:.3f} | "
                f"{s['ctrl_better_and_mmd_nonharm_fraction']:.3f} |"
            )
    lines.extend(["", "## Top Recurrent Anchor-Negative Conditions", "", "| dataset | condition | neg rows | groups | anchor pp | ctrl pp | pp delta | MMD delta |", "|---|---|---:|---|---:|---:|---:|---:|"])
    for row in recurrent_rows[:20]:
        lines.append(
            f"| `{row['dataset']}` | `{row['condition']}` | {row['negative_row_count']} | `{row['groups']}` | "
            f"{row['mean_anchor_pp']:+.6f} | {row['mean_ctrl_pp']:+.6f} | {row['mean_ctrl_minus_anchor_pp']:+.6f} | "
            f"{row['mean_ctrl_minus_anchor_mmd']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The source/control endpoint is a mandatory Track A proxy baseline. It often beats the frozen anchor on `pearson_pert` while remaining MMD non-harmful. This is not a model route and does not authorize GPU; it changes benchmark interpretation and failure-analysis requirements.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- group summary: `{group_csv}`",
            f"- recurrent failures: `{recurrent_csv}`",
            f"- joined rows: `{row_csv}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
