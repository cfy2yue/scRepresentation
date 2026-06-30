#!/usr/bin/env python3
"""Consolidate Track A anchor/source-control/dual-baseline requirements."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "tracka_benchmark_control_consolidation_20260630"

INPUTS = {
    "control_joined_rows": REPORTS / "tracka_control_baseline_synthesis_20260628/joined_rows.csv",
    "control_group_summary": REPORTS / "tracka_control_baseline_synthesis_20260628/group_summary.csv",
    "explicit_group_summary": REPORTS / "tracka_explicit_group_proxy_benchmark_20260628/group_summary.csv",
    "explicit_condition_rows": REPORTS / "tracka_explicit_group_proxy_benchmark_20260628/condition_rows.csv",
    "dual_baseline_summary": REPORTS / "dual_baseline_dominance_gate_20260628/dual_baseline_candidate_summary.csv",
    "dual_baseline_rows": REPORTS / "dual_baseline_dominance_gate_20260628/dual_baseline_matched_rows.csv",
}

PRIMARY_GROUPS = [
    "all_test_single_proxy",
    "cross_background_seen_gene_proxy",
    "family_gene",
    "simple_single_unseen_global_gene_proxy",
]


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def fmt(value: Any, digits: int = 6) -> str:
    x = finite_float(value)
    if x is None:
        return "NA"
    return f"{x:+.{digits}f}"


def dataset_equal_mean(df: pd.DataFrame, value_col: str) -> float | None:
    if df.empty or value_col not in df.columns:
        return None
    by_ds = df.groupby("dataset")[value_col].mean()
    if by_ds.empty:
        return None
    return float(by_ds.mean())


def dataset_bootstrap_ci(df: pd.DataFrame, value_col: str, seed: int = 42, repeats: int = 4000) -> tuple[float | None, float | None]:
    if df.empty or value_col not in df.columns:
        return None, None
    by_ds = {str(ds): vals[value_col].dropna().astype(float).to_numpy() for ds, vals in df.groupby("dataset")}
    datasets = sorted(ds for ds, vals in by_ds.items() if vals.size)
    if not datasets:
        return None, None
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(repeats):
        sample = rng.choice(datasets, size=len(datasets), replace=True)
        boot.append(float(np.mean([float(np.mean(by_ds[ds])) for ds in sample])))
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def summarize_control(control_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (seed, group), sub in control_rows.groupby(["seed", "explicit_group"], sort=True):
        if group not in PRIMARY_GROUPS:
            continue
        work = sub.copy()
        work["ctrl_minus_anchor_pp"] = work["ctrl_pp"].astype(float) - work["anchor_pp"].astype(float)
        work["ctrl_minus_anchor_mmd"] = work["ctrl_mmd"].astype(float) - work["anchor_mmd"].astype(float)
        ci_low, ci_high = dataset_bootstrap_ci(work, "ctrl_minus_anchor_pp")
        mmd_ci_low, mmd_ci_high = dataset_bootstrap_ci(work, "ctrl_minus_anchor_mmd", seed=43)
        ds_delta = work.groupby("dataset")["ctrl_minus_anchor_pp"].mean()
        ds_mmd = work.groupby("dataset")["ctrl_minus_anchor_mmd"].mean()
        rows.append(
            {
                "seed": seed,
                "group": group,
                "n_rows": int(len(work)),
                "n_datasets": int(work["dataset"].nunique()),
                "anchor_pp_row_mean": float(work["anchor_pp"].mean()),
                "ctrl_pp_row_mean": float(work["ctrl_pp"].mean()),
                "ctrl_minus_anchor_pp_row_mean": float(work["ctrl_minus_anchor_pp"].mean()),
                "anchor_pp_dataset_equal": dataset_equal_mean(work, "anchor_pp"),
                "ctrl_pp_dataset_equal": dataset_equal_mean(work, "ctrl_pp"),
                "ctrl_minus_anchor_pp_dataset_equal": dataset_equal_mean(work, "ctrl_minus_anchor_pp"),
                "ctrl_minus_anchor_pp_dataset_ci_low": ci_low,
                "ctrl_minus_anchor_pp_dataset_ci_high": ci_high,
                "anchor_mmd_row_mean": float(work["anchor_mmd"].mean()),
                "ctrl_mmd_row_mean": float(work["ctrl_mmd"].mean()),
                "ctrl_minus_anchor_mmd_row_mean": float(work["ctrl_minus_anchor_mmd"].mean()),
                "ctrl_minus_anchor_mmd_dataset_equal": dataset_equal_mean(work, "ctrl_minus_anchor_mmd"),
                "ctrl_minus_anchor_mmd_dataset_ci_low": mmd_ci_low,
                "ctrl_minus_anchor_mmd_dataset_ci_high": mmd_ci_high,
                "dataset_min_ctrl_minus_anchor_pp": float(ds_delta.min()),
                "dataset_max_ctrl_minus_anchor_mmd": float(ds_mmd.max()),
                "anchor_negative_fraction": float((work["anchor_pp"] < 0).mean()),
                "ctrl_better_pp_fraction": float((work["ctrl_minus_anchor_pp"] > 0).mean()),
                "ctrl_better_and_mmd_nonharm_fraction": float(
                    ((work["ctrl_minus_anchor_pp"] > 0) & (work["ctrl_minus_anchor_mmd"] <= 0.001)).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_seed_replication(control_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, sub in control_summary.groupby("group", sort=True):
        by_seed = {str(row["seed"]): row for _, row in sub.iterrows()}
        s42 = by_seed.get("seed42")
        s43 = by_seed.get("seed43")
        rows.append(
            {
                "group": group,
                "seed42_ctrl_minus_anchor_pp_dataset_equal": None
                if s42 is None
                else float(s42["ctrl_minus_anchor_pp_dataset_equal"]),
                "seed43_ctrl_minus_anchor_pp_dataset_equal": None
                if s43 is None
                else float(s43["ctrl_minus_anchor_pp_dataset_equal"]),
                "seed43_minus_seed42": None
                if s42 is None or s43 is None
                else float(s43["ctrl_minus_anchor_pp_dataset_equal"] - s42["ctrl_minus_anchor_pp_dataset_equal"]),
                "min_dataset_tail_across_seeds": float(sub["dataset_min_ctrl_minus_anchor_pp"].min()),
                "max_mmd_tail_across_seeds": float(sub["dataset_max_ctrl_minus_anchor_mmd"].max()),
            }
        )
    return pd.DataFrame(rows)


def summarize_dual_baseline(dual: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, sub in dual.groupby("candidate", sort=True):
        pass_rows = sub[sub["status"].astype(str).str.endswith("_pass")]
        rows.append(
            {
                "candidate": candidate,
                "n_groups": int(len(sub)),
                "pass_groups": int(len(pass_rows)),
                "best_dominance_pp": float(sub["dominance_pp"].max()),
                "worst_dataset_min_dominance_pp": float(sub["dataset_min_dominance_pp"].min()),
                "max_mmd_harm": float(sub["max_mmd_harm"].max()),
                "max_dataset_mmd_harm": float(sub["dataset_max_mmd_harm"].max()),
                "status": "pass_any" if len(pass_rows) else "all_fail",
            }
        )
    return pd.DataFrame(rows)


def make_decision(control_summary: pd.DataFrame, dual_candidate_summary: pd.DataFrame) -> dict[str, Any]:
    core = control_summary[control_summary["group"].isin(PRIMARY_GROUPS)].copy()
    control_positive_groups = int((core["ctrl_minus_anchor_pp_dataset_equal"] > 0).sum())
    control_ci_positive = int((core["ctrl_minus_anchor_pp_dataset_ci_low"] > 0).sum())
    no_dual_pass = bool((dual_candidate_summary["status"] != "pass_any").all())
    return {
        "status": "tracka_benchmark_control_consolidation_complete_no_gpu",
        "gpu_authorized_next": False,
        "default_model": "xverse_8k_anchor",
        "current_model_promotion": False,
        "core_group_rows": int(len(core)),
        "control_positive_group_seed_pairs": control_positive_groups,
        "control_ci_positive_group_seed_pairs": control_ci_positive,
        "dual_baseline_candidates_with_pass": int((dual_candidate_summary["status"] == "pass_any").sum()),
        "interpretation": (
            "Track A success must be measured against both frozen anchor and source/control baseline; "
            "beating anchor alone is insufficient."
        ),
        "next_gate_requirements": [
            "candidate-vs-max(anchor, source/control) pearson_pert dominance > 0 with dataset-bootstrap CI low > 0",
            "dataset-min dominance >= -0.02 on all primary Track A groups",
            "candidate MMD harm versus both anchor and source/control <= +0.001 row-mean and no unsafe dataset tail",
            "seed replication or eval-seed-locked control before promotion",
            "canonical multi remains diagnostic-only and Track C query remains forbidden for Track A selection",
        ],
        "recommended_next_action": (
            "Do not launch a model smoke from recently closed axes. First design a new train-only/exogenous "
            "mechanism that can plausibly beat source/control, then run a small CPU admission gate against "
            "these dual-baseline requirements."
        )
        if no_dual_pass
        else "Review passing candidates before any launch.",
    }


def write_md(
    path: Path,
    decision: dict[str, Any],
    control_summary: pd.DataFrame,
    seed_summary: pd.DataFrame,
    dual_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Track A Benchmark-Control Consolidation",
        "",
        f"Created: `{now_cst()}`",
        "",
        f"Status: `{decision['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only synthesis of frozen Track A anchor, source/control baseline, and dual-baseline candidate artifacts.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C query access, or GPU.",
        "- Reports both row-weighted and dataset-equal statistics to avoid aggregation ambiguity.",
        "",
        "## Decision",
        "",
        f"- default model: `{decision['default_model']}`",
        f"- current model promotion: `{decision['current_model_promotion']}`",
        f"- dual-baseline candidates with any pass: `{decision['dual_baseline_candidates_with_pass']}`",
        f"- interpretation: {decision['interpretation']}",
        f"- recommended next action: {decision['recommended_next_action']}",
        "",
        "## Source/Control Baseline Vs Anchor",
        "",
        "| seed | group | n | datasets | anchor pp row | ctrl pp row | ctrl-anchor pp row | ctrl-anchor pp dataset | CI95 dataset | anchor MMD row | ctrl MMD row | ctrl-anchor MMD dataset | dataset min pp | dataset max MMD | anchor neg frac | ctrl better + MMD nonharm |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in control_summary.sort_values(["seed", "group"]).iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    str(row["group"]),
                    str(int(row["n_rows"])),
                    str(int(row["n_datasets"])),
                    fmt(row["anchor_pp_row_mean"]),
                    fmt(row["ctrl_pp_row_mean"]),
                    fmt(row["ctrl_minus_anchor_pp_row_mean"]),
                    fmt(row["ctrl_minus_anchor_pp_dataset_equal"]),
                    f"[{fmt(row['ctrl_minus_anchor_pp_dataset_ci_low'])}, {fmt(row['ctrl_minus_anchor_pp_dataset_ci_high'])}]",
                    fmt(row["anchor_mmd_row_mean"]),
                    fmt(row["ctrl_mmd_row_mean"]),
                    fmt(row["ctrl_minus_anchor_mmd_dataset_equal"]),
                    fmt(row["dataset_min_ctrl_minus_anchor_pp"]),
                    fmt(row["dataset_max_ctrl_minus_anchor_mmd"]),
                    f"{float(row['anchor_negative_fraction']):.3f}",
                    f"{float(row['ctrl_better_and_mmd_nonharm_fraction']):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Seed Replication",
            "",
            "| group | seed42 dataset pp delta | seed43 dataset pp delta | seed43-seed42 | min dataset tail | max MMD tail |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in seed_summary.sort_values("group").iterrows():
        lines.append(
            f"| {row['group']} | {fmt(row['seed42_ctrl_minus_anchor_pp_dataset_equal'])} | "
            f"{fmt(row['seed43_ctrl_minus_anchor_pp_dataset_equal'])} | {fmt(row['seed43_minus_seed42'])} | "
            f"{fmt(row['min_dataset_tail_across_seeds'])} | {fmt(row['max_mmd_tail_across_seeds'])} |"
        )
    lines.extend(
        [
            "",
            "## Existing Candidate Dual-Baseline Summary",
            "",
            "| candidate | groups | pass groups | best dominance pp | worst dataset min | max MMD harm | max dataset MMD harm | status |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in dual_summary.sort_values(["status", "candidate"]).iterrows():
        lines.append(
            f"| `{row['candidate']}` | {int(row['n_groups'])} | {int(row['pass_groups'])} | "
            f"{fmt(row['best_dominance_pp'])} | {fmt(row['worst_dataset_min_dominance_pp'])} | "
            f"{fmt(row['max_mmd_harm'])} | {fmt(row['max_dataset_mmd_harm'])} | `{row['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Next Gate Requirements",
            "",
            *[f"- {item}" for item in decision["next_gate_requirements"]],
            "",
            "## Outputs",
            "",
            f"- control summary: `{OUT_DIR / 'tracka_benchmark_control_consolidation_control_summary.csv'}`",
            f"- seed replication: `{OUT_DIR / 'tracka_benchmark_control_consolidation_seed_replication.csv'}`",
            f"- dual baseline summary: `{OUT_DIR / 'tracka_benchmark_control_consolidation_dual_candidates.csv'}`",
            f"- JSON: `{OUT_DIR / 'tracka_benchmark_control_consolidation_20260630.json'}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ensure_output_dir(OUT_DIR)
    control_rows = pd.read_csv(INPUTS["control_joined_rows"])
    dual_summary_in = pd.read_csv(INPUTS["dual_baseline_summary"])

    control_summary = summarize_control(control_rows)
    seed_summary = summarize_seed_replication(control_summary)
    dual_summary = summarize_dual_baseline(dual_summary_in)
    decision = make_decision(control_summary, dual_summary)

    control_path = OUT_DIR / "tracka_benchmark_control_consolidation_control_summary.csv"
    seed_path = OUT_DIR / "tracka_benchmark_control_consolidation_seed_replication.csv"
    dual_path = OUT_DIR / "tracka_benchmark_control_consolidation_dual_candidates.csv"
    json_path = OUT_DIR / "tracka_benchmark_control_consolidation_20260630.json"
    md_path = OUT_DIR / "LATENTFM_TRACKA_BENCHMARK_CONTROL_CONSOLIDATION_20260630.md"

    control_summary.to_csv(control_path, index=False)
    seed_summary.to_csv(seed_path, index=False)
    dual_summary.to_csv(dual_path, index=False)
    payload = {
        "timestamp_cst": now_cst(),
        "boundary": {
            "cpu_report_only": True,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
            "gpu_authorized_next": False,
        },
        "inputs": {k: str(v) for k, v in INPUTS.items()},
        "decision": decision,
        "outputs": {
            "control_summary": str(control_path),
            "seed_replication": str(seed_path),
            "dual_candidates": str(dual_path),
            "markdown_report": str(md_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_md(md_path, decision, control_summary, seed_summary, dual_summary)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
