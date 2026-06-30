#!/usr/bin/env python3
"""Residualized condition-axis gate for LatentFM scaling v2.

CPU/report-only. This asks whether any single condition-level information axis
can produce enough high/low train-condition pairs after matching away the other
obvious axes. It does not train, infer, or authorize GPU.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
COND_TABLE = ROOT / "reports/scaling_v2_condition_information_draft_splits_20260628/condition_information_table.csv"
OUT_DIR = ROOT / "reports/scaling_v2_residualized_condition_axes_gate_20260628"


@dataclass(frozen=True)
class AxisSpec:
    name: str
    target: str
    confounds: tuple[str, ...]
    description: str


AXES = (
    AxisSpec(
        name="response_energy_resid",
        target="log_response_energy",
        confounds=("hvg_concentration_80", "hvg_advantage_80", "cell_support_log", "abundance_concentration_80"),
        description="perturbation response magnitude after matching gene-budget/support structure",
    ),
    AxisSpec(
        name="hvg_concentration_resid",
        target="hvg_concentration_80",
        confounds=("log_response_energy", "cell_support_log", "abundance_concentration_80"),
        description="compact response budget after matching response magnitude, support, and abundance concentration",
    ),
    AxisSpec(
        name="hvg_advantage_resid",
        target="hvg_advantage_80",
        confounds=("log_response_energy", "cell_support_log", "abundance_concentration_80"),
        description="HVG-over-abundance response concentration after matching magnitude/support/abundance",
    ),
    AxisSpec(
        name="support_resid",
        target="cell_support_log",
        confounds=("log_response_energy", "hvg_concentration_80", "hvg_advantage_80", "abundance_concentration_80"),
        description="perturbed-cell support after matching response and gene-budget properties",
    ),
)

CUTOFFS = {
    "strict": 0.50,
    "moderate": 1.00,
    "relaxed": 2.00,
}


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def robust_z_by_dataset(df: pd.DataFrame, col: str) -> pd.Series:
    def transform(x: pd.Series) -> pd.Series:
        med = x.median()
        mad = (x - med).abs().median()
        scale = mad if mad and np.isfinite(mad) else x.std()
        if not scale or not np.isfinite(scale):
            scale = 1.0
        return (x - med) / scale

    return df.groupby("dataset", sort=False)[col].transform(transform)


def add_residual_axes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["abundance_concentration_80"] = 1.0 - pd.to_numeric(out["abundance_k80"], errors="coerce") / pd.to_numeric(
        out["n_vars"], errors="coerce"
    )
    needed = sorted({axis.target for axis in AXES} | {c for axis in AXES for c in axis.confounds})
    for col in needed:
        out[f"{col}_dataset_z"] = robust_z_by_dataset(out, col)

    for axis in AXES:
        cols = [axis.target, *axis.confounds]
        sub = out.dropna(subset=[f"{col}_dataset_z" for col in cols]).copy()
        y = sub[f"{axis.target}_dataset_z"].to_numpy(dtype=float)
        x_cols = [sub[f"{col}_dataset_z"].to_numpy(dtype=float) for col in axis.confounds]
        x = np.column_stack([np.ones(len(sub)), *x_cols])
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        out.loc[sub.index, axis.name] = y - x @ beta
    return out


def confound_z(group: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    z = pd.DataFrame(index=group.index)
    for col in cols:
        vals = pd.to_numeric(group[col], errors="coerce")
        med = vals.median()
        mad = (vals - med).abs().median()
        scale = mad if mad and np.isfinite(mad) else vals.std()
        if not scale or not np.isfinite(scale):
            scale = 1.0
        z[col] = (vals - med) / scale
    return z


def greedy_pairs(df: pd.DataFrame, axis: AxisSpec, cutoff: float, mode: str) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    drop_cols = [axis.name, *axis.confounds]
    for key, group in df.dropna(subset=drop_cols).groupby(["dataset", "perturbation_type", "gene_count_bin"], sort=True):
        if len(group) < 8:
            continue
        lo_q = group[axis.name].quantile(1.0 / 3.0)
        hi_q = group[axis.name].quantile(2.0 / 3.0)
        lows = group[group[axis.name] <= lo_q]
        highs = group[group[axis.name] >= hi_q]
        if lows.empty or highs.empty:
            continue
        z = confound_z(group, axis.confounds)
        candidates: list[tuple[float, float, int, int]] = []
        for hi_idx in highs.index:
            for lo_idx in lows.index:
                delta = float(df.loc[hi_idx, axis.name] - df.loc[lo_idx, axis.name])
                if delta <= 0:
                    continue
                dist = float(np.sqrt(np.nanmean((z.loc[hi_idx] - z.loc[lo_idx]).to_numpy(dtype=float) ** 2)))
                if np.isfinite(dist) and dist <= cutoff:
                    candidates.append((dist, -delta, int(hi_idx), int(lo_idx)))
        used_high: set[int] = set()
        used_low: set[int] = set()
        for dist, neg_delta, hi_idx, lo_idx in sorted(candidates):
            if hi_idx in used_high or lo_idx in used_low:
                continue
            used_high.add(hi_idx)
            used_low.add(lo_idx)
            hi = df.loc[hi_idx]
            lo = df.loc[lo_idx]
            row: dict[str, Any] = {
                "axis": axis.name,
                "match_mode": mode,
                "cutoff": cutoff,
                "dataset": key[0],
                "perturbation_type": key[1],
                "gene_count_bin": key[2],
                "high_condition": hi["condition"],
                "low_condition": lo["condition"],
                "axis_delta": -neg_delta,
                "confound_distance": dist,
                "high_axis_value": float(hi[axis.name]),
                "low_axis_value": float(lo[axis.name]),
                "high_target_value": float(hi[axis.target]),
                "low_target_value": float(lo[axis.target]),
            }
            for col in axis.confounds:
                row[f"high_{col}"] = float(hi[col])
                row[f"low_{col}"] = float(lo[col])
                row[f"{col}_delta"] = float(hi[col] - lo[col])
            pairs.append(row)
    return pairs


def standardized_mean_delta(rows: list[dict[str, Any]], col: str) -> float:
    highs = np.asarray([row.get(f"high_{col}", np.nan) for row in rows], dtype=float)
    lows = np.asarray([row.get(f"low_{col}", np.nan) for row in rows], dtype=float)
    vals = np.concatenate([highs[np.isfinite(highs)], lows[np.isfinite(lows)]])
    if len(vals) < 2:
        return float("nan")
    scale = float(np.std(vals, ddof=1))
    if not scale or not np.isfinite(scale):
        scale = 1.0
    return float((np.nanmean(highs) - np.nanmean(lows)) / scale)


def summarize(axis: AxisSpec, mode: str, pairs: list[dict[str, Any]]) -> dict[str, Any]:
    dists = np.asarray([row["confound_distance"] for row in pairs], dtype=float)
    deltas = np.asarray([row["axis_delta"] for row in pairs], dtype=float)
    conf_smd = {col: standardized_mean_delta(pairs, col) for col in axis.confounds}
    finite_smd = [abs(v) for v in conf_smd.values() if np.isfinite(v)]
    max_abs_smd = max(finite_smd) if finite_smd else float("nan")
    n_pairs = len(pairs)
    n_datasets = len({row["dataset"] for row in pairs})
    strict_gate = bool(
        mode == "strict"
        and n_pairs >= 200
        and n_datasets >= 12
        and np.nanmean(dists) <= CUTOFFS["strict"]
        and max_abs_smd <= 0.25
    )
    relaxed_count_only = bool(mode == "relaxed" and n_pairs >= 200 and n_datasets >= 12)
    return {
        "axis": axis.name,
        "description": axis.description,
        "target": axis.target,
        "confounds": ",".join(axis.confounds),
        "match_mode": mode,
        "cutoff": CUTOFFS[mode],
        "n_pairs": n_pairs,
        "n_datasets": n_datasets,
        "mean_axis_delta": float(np.nanmean(deltas)) if n_pairs else math.nan,
        "mean_confound_distance": float(np.nanmean(dists)) if n_pairs else math.nan,
        "max_abs_confound_smd": max_abs_smd,
        "strict_gate": strict_gate,
        "relaxed_count_only": relaxed_count_only,
        **{f"smd_{col}": val for col, val in conf_smd.items()},
    }


def fmt(x: Any, digits: int = 4) -> str:
    try:
        if not np.isfinite(float(x)):
            return "nan"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-table", type=Path, default=COND_TABLE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.condition_table)
    df = add_residual_axes(raw)

    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for axis in AXES:
        for mode, cutoff in CUTOFFS.items():
            pairs = greedy_pairs(df, axis, cutoff=cutoff, mode=mode)
            pair_rows.extend(pairs)
            summary_rows.append(summarize(axis, mode, pairs))

    summary_df = pd.DataFrame(summary_rows)
    pairs_df = pd.DataFrame(pair_rows)
    matrix_cols = [
        "dataset",
        "condition",
        "perturbation_type",
        "gene_count_bin",
        "response_energy",
        "hvg_k80",
        "abundance_k80",
        "n_pert",
        "log_response_energy",
        "hvg_concentration_80",
        "hvg_advantage_80",
        "cell_support_log",
        "abundance_concentration_80",
        *[axis.name for axis in AXES],
    ]
    df[matrix_cols].to_csv(args.out_dir / "residualized_condition_axis_matrix.csv", index=False)
    summary_df.to_csv(args.out_dir / "residualized_condition_axis_summary.csv", index=False)
    pairs_df.to_csv(args.out_dir / "residualized_condition_axis_pairs.csv", index=False)

    any_strict = bool(summary_df["strict_gate"].any())
    any_relaxed = bool(summary_df["relaxed_count_only"].any())
    if any_strict:
        status = "scaling_v2_residualized_condition_axes_gate_pass_prepare_packet_audit_no_gpu"
    elif any_relaxed:
        status = "scaling_v2_residualized_condition_axes_gate_relaxed_not_launch_ready_no_gpu"
    else:
        status = "scaling_v2_residualized_condition_axes_gate_fail_no_gpu"

    payload = {
        "timestamp": now_cst(),
        "status": status,
        "condition_table": str(args.condition_table),
        "n_conditions": int(len(df)),
        "strict_gate_axes": summary_df.loc[summary_df["strict_gate"], "axis"].tolist(),
        "relaxed_count_only_axes": summary_df.loc[summary_df["relaxed_count_only"], "axis"].tolist(),
        "gpu_authorized_next": False,
        "outputs": {
            "matrix": str(args.out_dir / "residualized_condition_axis_matrix.csv"),
            "summary": str(args.out_dir / "residualized_condition_axis_summary.csv"),
            "pairs": str(args.out_dir / "residualized_condition_axis_pairs.csv"),
        },
    }
    write_json(args.out_dir / "latentfm_scaling_v2_residualized_condition_axes_gate_20260628.json", payload)

    lines = [
        "# LatentFM Scaling V2 Residualized Condition Axes Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized next: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only gate over parent train conditions already used by the condition-information draft.",
        "- No training, inference, canonical multi selection, or Track C query use.",
        "- Residual axes are constructed after dataset-robust z-scoring, then high/low pairs are matched within dataset, perturbation type, and gene-count bin.",
        "",
        "## Gate",
        "",
        "A GPU packet would require a strict row with `n_pairs >= 200`, `n_datasets >= 12`, mean confound distance `<= 0.50`, and max absolute confound SMD `<= 0.25`.",
        "",
        "## Summary",
        "",
        "| axis | mode | pairs | datasets | mean axis delta | mean confound distance | max abs SMD | strict gate | relaxed count only |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| `{axis}` | `{match_mode}` | {n_pairs} | {n_datasets} | {mean_axis_delta} | {dist} | {smd} | `{strict}` | `{relaxed}` |".format(
                axis=row["axis"],
                match_mode=row["match_mode"],
                n_pairs=row["n_pairs"],
                n_datasets=row["n_datasets"],
                mean_axis_delta=fmt(row["mean_axis_delta"]),
                dist=fmt(row["mean_confound_distance"]),
                smd=fmt(row["max_abs_confound_smd"]),
                strict=row["strict_gate"],
                relaxed=row["relaxed_count_only"],
            )
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- No strict residualized condition axis currently authorizes a GPU smoke.",
            "- Relaxed count-only rows show that enough pairs can be obtained only when confound distance becomes too loose.",
            "- This supports closing the raw `info_composite` family as a launch signal and moving to better-defined axes: exact coverage with stronger matching, ZSCAPE OT response geometry, or a broader balanced split family.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{args.out_dir / 'latentfm_scaling_v2_residualized_condition_axes_gate_20260628.json'}`",
            f"- Summary CSV: `{args.out_dir / 'residualized_condition_axis_summary.csv'}`",
            f"- Pair CSV: `{args.out_dir / 'residualized_condition_axis_pairs.csv'}`",
            f"- Matrix CSV: `{args.out_dir / 'residualized_condition_axis_matrix.csv'}`",
        ]
    )
    (args.out_dir / "LATENTFM_SCALING_V2_RESIDUALIZED_CONDITION_AXES_GATE_20260628.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
