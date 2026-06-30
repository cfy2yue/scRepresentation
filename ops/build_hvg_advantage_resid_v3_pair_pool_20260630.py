#!/usr/bin/env python3
"""Build a v3 HVG-advantage residual high/low pair pool.

CPU/report-only. This script searches a small, predeclared set of
condition-level high/low matching modes for the post-audit nonstatic observable
information branch. It writes a candidate pair pool and high/low train split
artifacts only if the best mode passes the CPU design gate.

It does not train, infer, read canonical multi for Track A selection, read
Track C query, or use GPU.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path("/data/cyx/1030/scLatent")
MATRIX_CSV = ROOT / "reports/scaling_v2_residualized_condition_axes_gate_20260628/residualized_condition_axis_matrix.csv"
PARENT_SPLIT = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
OUT_DIR = ROOT / "reports/hvg_advantage_resid_v3_pair_pool_20260630"
REPORT = OUT_DIR / "LATENTFM_HVG_ADVANTAGE_RESID_V3_PAIR_POOL_20260630.md"
JSON_OUT = OUT_DIR / "hvg_advantage_resid_v3_pair_pool_20260630.json"
SUMMARY_CSV = OUT_DIR / "hvg_advantage_resid_v3_mode_summary_20260630.csv"
PAIR_CSV = OUT_DIR / "hvg_advantage_resid_v3_selected_pairs_20260630.csv"
PACKET_JSON = OUT_DIR / "hvg_advantage_resid_v3_packet_audit_20260630.json"

HIGH_SPLIT = OUT_DIR / "split_seed42_xverse_hvg_advantage_resid_v3_high_from_cap120_all_v2.json"
LOW_SPLIT = OUT_DIR / "split_seed42_xverse_hvg_advantage_resid_v3_low_from_cap120_all_v2.json"

AXIS = "hvg_advantage_resid"
CONFOUNDS = ("log_response_energy", "cell_support_log", "abundance_concentration_80")
GROUP_KEYS = ("dataset", "perturbation_type", "gene_count_bin")

MODES = (
    # Original strict axis separation with strict pair distance.
    ("tertile_cut050", 1.0 / 3.0, 2.0 / 3.0, 0.50),
    # Same axis separation with controlled aggregate mean distance check.
    ("tertile_cut075", 1.0 / 3.0, 2.0 / 3.0, 0.75),
    # Predeclared v3 broader high/low, still separated by the middle 20%.
    ("q40_q60_cut050", 0.40, 0.60, 0.50),
    ("q40_q60_cut075", 0.40, 0.60, 0.75),
    # Sensitivity only. This is not preferred because the axis gap is narrower.
    ("q45_q55_cut050", 0.45, 0.55, 0.50),
    ("q45_q55_cut060", 0.45, 0.55, 0.60),
)

MIN_PAIRS = 200
MIN_DATASETS = 12
MAX_MEAN_CONFOUND_DISTANCE = 0.50
MAX_SMD = 0.25
MAX_DATASET_FRACTION = 0.25
MAX_TYPE_FRACTION = 0.70
MIN_MEAN_AXIS_DELTA = 2.0
MIN_AXIS_QUANTILE_GAP = 0.20
EPS = 1e-9


@dataclass(frozen=True)
class ModeSummary:
    mode: str
    q_low: float
    q_high: float
    pair_cutoff: float
    n_pairs: int
    n_datasets: int
    mean_axis_delta: float
    mean_confound_distance: float
    max_pair_distance: float
    pair_distance_gt_0p5_fraction: float
    max_abs_confound_smd: float
    max_dataset_fraction: float
    top_dataset: str
    max_perturbation_type_fraction: float
    top_perturbation_type: str
    strict_design_pass: bool
    preferred_candidate: bool
    status: str
    reasons: str


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_float(value: str) -> float:
    if value in {"", "NA", "nan", "None", "null"}:
        return math.nan
    return float(value)


def load_matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with MATRIX_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = dict(row)
            for col in (AXIS, *CONFOUNDS):
                parsed[col] = as_float(row[col])
            rows.append(parsed)
    return rows


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def robust_z(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    med = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - med)))
    scale = mad if mad and np.isfinite(mad) else float(np.nanstd(arr, ddof=1))
    if not scale or not np.isfinite(scale):
        scale = 1.0
    return (arr - med) / scale


def valid_row(row: dict[str, Any]) -> bool:
    return all(np.isfinite(float(row[col])) for col in (AXIS, *CONFOUNDS))


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if valid_row(row):
            grouped[tuple(str(row[key]) for key in GROUP_KEYS)].append(row)
    return grouped


def pair_mode(rows: list[dict[str, Any]], q_low: float, q_high: float, cutoff: float) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for key, group in group_rows(rows).items():
        if len(group) < 8:
            continue
        values = [float(row[AXIS]) for row in group]
        lo_q = quantile(values, q_low)
        hi_q = quantile(values, q_high)
        lows = [row for row in group if float(row[AXIS]) <= lo_q]
        highs = [row for row in group if float(row[AXIS]) >= hi_q]
        if not lows or not highs:
            continue

        z_by_col = {col: robust_z([float(row[col]) for row in group]) for col in CONFOUNDS}
        pos = {id(row): idx for idx, row in enumerate(group)}
        cost = np.full((len(highs), len(lows)), 1e9, dtype=float)
        meta: dict[tuple[int, int], dict[str, Any]] = {}
        for hi_i, hi in enumerate(highs):
            for lo_i, lo in enumerate(lows):
                axis_delta = float(hi[AXIS]) - float(lo[AXIS])
                if axis_delta <= 0:
                    continue
                hi_vec = np.asarray([z_by_col[col][pos[id(hi)]] for col in CONFOUNDS], dtype=float)
                lo_vec = np.asarray([z_by_col[col][pos[id(lo)]] for col in CONFOUNDS], dtype=float)
                dist = float(np.sqrt(np.nanmean((hi_vec - lo_vec) ** 2)))
                if not np.isfinite(dist) or dist > cutoff:
                    continue
                cost[hi_i, lo_i] = dist - 1e-6 * axis_delta
                meta[(hi_i, lo_i)] = {
                    "dataset": key[0],
                    "perturbation_type": key[1],
                    "gene_count_bin": key[2],
                    "high_condition": str(hi["condition"]),
                    "low_condition": str(lo["condition"]),
                    "axis": AXIS,
                    "axis_delta": axis_delta,
                    "confound_distance": dist,
                    "high_axis_value": float(hi[AXIS]),
                    "low_axis_value": float(lo[AXIS]),
                    **{f"high_{col}": float(hi[col]) for col in CONFOUNDS},
                    **{f"low_{col}": float(lo[col]) for col in CONFOUNDS},
                    **{f"{col}_delta": float(hi[col]) - float(lo[col]) for col in CONFOUNDS},
                }
        if cost.size == 0:
            continue
        row_idx, col_idx = linear_sum_assignment(cost)
        for hi_i, lo_i in zip(row_idx, col_idx):
            if cost[hi_i, lo_i] >= 1e8:
                continue
            pairs.append(meta[(hi_i, lo_i)])
    return pairs


def standardized_mean_delta(pairs: list[dict[str, Any]], col: str) -> float:
    highs = np.asarray([row[f"high_{col}"] for row in pairs], dtype=float)
    lows = np.asarray([row[f"low_{col}"] for row in pairs], dtype=float)
    vals = np.concatenate([highs, lows])
    scale = float(np.nanstd(vals, ddof=1))
    if not scale or not np.isfinite(scale):
        scale = 1.0
    return float((np.nanmean(highs) - np.nanmean(lows)) / scale)


def summarize_mode(mode: str, q_low: float, q_high: float, cutoff: float, pairs: list[dict[str, Any]]) -> ModeSummary:
    n_pairs = len(pairs)
    datasets = Counter(row["dataset"] for row in pairs)
    types = Counter(row["perturbation_type"] for row in pairs)
    dists = np.asarray([row["confound_distance"] for row in pairs], dtype=float)
    deltas = np.asarray([row["axis_delta"] for row in pairs], dtype=float)
    smds = [standardized_mean_delta(pairs, col) for col in CONFOUNDS] if pairs else [math.nan]
    top_dataset, top_dataset_n = datasets.most_common(1)[0] if datasets else ("NA", 0)
    top_type, top_type_n = types.most_common(1)[0] if types else ("NA", 0)
    axis_gap = q_high - q_low

    reasons: list[str] = []
    if n_pairs < MIN_PAIRS:
        reasons.append(f"pairs_below_{MIN_PAIRS}")
    if len(datasets) < MIN_DATASETS:
        reasons.append(f"datasets_below_{MIN_DATASETS}")
    if not pairs or float(np.nanmean(dists)) > MAX_MEAN_CONFOUND_DISTANCE:
        reasons.append(f"mean_confound_distance_gt_{MAX_MEAN_CONFOUND_DISTANCE}")
    if not pairs or max(abs(x) for x in smds if np.isfinite(x)) > MAX_SMD:
        reasons.append(f"max_smd_gt_{MAX_SMD}")
    if n_pairs and top_dataset_n / n_pairs > MAX_DATASET_FRACTION:
        reasons.append(f"dataset_fraction_gt_{MAX_DATASET_FRACTION}")
    if n_pairs and top_type_n / n_pairs > MAX_TYPE_FRACTION:
        reasons.append(f"perturbation_type_fraction_gt_{MAX_TYPE_FRACTION}")
    if not pairs or float(np.nanmean(deltas)) < MIN_MEAN_AXIS_DELTA:
        reasons.append(f"mean_axis_delta_below_{MIN_MEAN_AXIS_DELTA}")
    if axis_gap + EPS < MIN_AXIS_QUANTILE_GAP:
        reasons.append(f"axis_quantile_gap_below_{MIN_AXIS_QUANTILE_GAP}")

    pass_gate = not reasons
    preferred = pass_gate and mode == "q40_q60_cut075"
    if preferred:
        status = "preferred_v3_pair_pool_pass_prepare_packet_no_gpu"
    elif pass_gate:
        status = "sensitivity_pass_not_preferred_no_gpu"
    elif n_pairs >= MIN_PAIRS:
        status = "count_only_or_confounded_no_gpu"
    else:
        status = "fail_no_gpu"

    return ModeSummary(
        mode=mode,
        q_low=q_low,
        q_high=q_high,
        pair_cutoff=cutoff,
        n_pairs=n_pairs,
        n_datasets=len(datasets),
        mean_axis_delta=float(np.nanmean(deltas)) if pairs else math.nan,
        mean_confound_distance=float(np.nanmean(dists)) if pairs else math.nan,
        max_pair_distance=float(np.nanmax(dists)) if pairs else math.nan,
        pair_distance_gt_0p5_fraction=float(np.mean(dists > 0.5)) if pairs else math.nan,
        max_abs_confound_smd=max(abs(x) for x in smds if np.isfinite(x)) if pairs else math.nan,
        max_dataset_fraction=float(top_dataset_n / n_pairs) if n_pairs else math.nan,
        top_dataset=str(top_dataset),
        max_perturbation_type_fraction=float(top_type_n / n_pairs) if n_pairs else math.nan,
        top_perturbation_type=str(top_type),
        strict_design_pass=pass_gate,
        preferred_candidate=preferred,
        status=status,
        reasons=";".join(reasons) if reasons else "none",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = columns or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_splits(parent: dict[str, Any], pairs: list[dict[str, Any]]) -> None:
    high = json.loads(json.dumps(parent))
    low = json.loads(json.dumps(parent))
    by_dataset: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"high": [], "low": []})
    for row in pairs:
        by_dataset[row["dataset"]]["high"].append(str(row["high_condition"]))
        by_dataset[row["dataset"]]["low"].append(str(row["low_condition"]))
    for dataset, groups in parent.items():
        high[dataset]["train"] = sorted(set(by_dataset.get(dataset, {}).get("high", [])))
        low[dataset]["train"] = sorted(set(by_dataset.get(dataset, {}).get("low", [])))
    write_json(HIGH_SPLIT, high)
    write_json(LOW_SPLIT, low)


def split_packet(parent: dict[str, Any], pairs: list[dict[str, Any]], summary: ModeSummary) -> dict[str, Any]:
    parent_train = {
        (str(dataset), str(condition))
        for dataset, groups in parent.items()
        for condition in groups.get("train", [])
    }
    high_keys = {(row["dataset"], row["high_condition"]) for row in pairs}
    low_keys = {(row["dataset"], row["low_condition"]) for row in pairs}
    dataset_rows = []
    for dataset, rows in sorted(defaultdict(list, {d: [r for r in pairs if r["dataset"] == d] for d in {p["dataset"] for p in pairs}}).items()):
        dataset_rows.append(
            {
                "dataset": dataset,
                "pairs": len(rows),
                "mean_axis_delta": float(np.mean([r["axis_delta"] for r in rows])),
                "mean_confound_distance": float(np.mean([r["confound_distance"] for r in rows])),
            }
        )

    return {
        "timestamp": now_cst(),
        "status": "hvg_advantage_resid_v3_pair_pool_pass_prepare_gpu_smoke"
        if summary.preferred_candidate
        else "hvg_advantage_resid_v3_pair_pool_no_gpu",
        "gpu_authorized": False,
        "axis": AXIS,
        "selected_mode": summary.mode,
        "summary": asdict(summary),
        "parent_split": str(PARENT_SPLIT),
        "high_split": str(HIGH_SPLIT),
        "low_split": str(LOW_SPLIT),
        "pair_csv": str(PAIR_CSV),
        "checks": {
            "high_subset_parent_train": high_keys.issubset(parent_train),
            "low_subset_parent_train": low_keys.issubset(parent_train),
            "high_low_disjoint": len(high_keys & low_keys) == 0,
            "n_high_conditions": len(high_keys),
            "n_low_conditions": len(low_keys),
            "n_pairs": len(pairs),
            "n_datasets": len({row["dataset"] for row in pairs}),
        },
        "dataset_rows": dataset_rows,
        "boundary": "CPU/report-only pair pool; launcher owns any later bounded smoke protocol",
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.6g}"
            vals.append(str(val).replace("\n", " "))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def main() -> int:
    for path in [MATRIX_CSV, PARENT_SPLIT]:
        if not path.exists():
            raise FileNotFoundError(path)
    if REPORT.exists() or JSON_OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing outputs in {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_matrix()
    parent = load_json(PARENT_SPLIT)
    mode_pairs: dict[str, list[dict[str, Any]]] = {}
    summaries: list[ModeSummary] = []
    for mode, q_low, q_high, cutoff in MODES:
        pairs = pair_mode(rows, q_low, q_high, cutoff)
        for row in pairs:
            row["mode"] = mode
            row["q_low"] = q_low
            row["q_high"] = q_high
            row["pair_cutoff"] = cutoff
        mode_pairs[mode] = pairs
        summaries.append(summarize_mode(mode, q_low, q_high, cutoff, pairs))

    preferred = [summary for summary in summaries if summary.preferred_candidate]
    selected_summary = preferred[0] if preferred else None
    selected_pairs = mode_pairs[selected_summary.mode] if selected_summary else []

    summary_rows = [asdict(summary) for summary in summaries]
    write_csv(SUMMARY_CSV, summary_rows)
    if selected_pairs:
        pair_columns = [
            "mode",
            "axis",
            "dataset",
            "perturbation_type",
            "gene_count_bin",
            "high_condition",
            "low_condition",
            "axis_delta",
            "confound_distance",
            "q_low",
            "q_high",
            "pair_cutoff",
            "high_axis_value",
            "low_axis_value",
            *[f"high_{col}" for col in CONFOUNDS],
            *[f"low_{col}" for col in CONFOUNDS],
            *[f"{col}_delta" for col in CONFOUNDS],
        ]
        write_csv(PAIR_CSV, selected_pairs, pair_columns)
        write_splits(parent, selected_pairs)
        packet = split_packet(parent, selected_pairs, selected_summary)
        write_json(PACKET_JSON, packet)
    else:
        packet = {
            "timestamp": now_cst(),
            "status": "hvg_advantage_resid_v3_pair_pool_fail_no_gpu",
            "gpu_authorized": False,
            "axis": AXIS,
            "selected_mode": None,
            "checks": {},
        }
        write_json(PACKET_JSON, packet)

    status = packet["status"]
    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_authorized": False,
        "selected_mode": selected_summary.mode if selected_summary else None,
        "mode_summaries": summary_rows,
        "outputs": {
            "summary_csv": str(SUMMARY_CSV),
            "selected_pairs": str(PAIR_CSV) if selected_pairs else None,
            "packet_json": str(PACKET_JSON),
            "high_split": str(HIGH_SPLIT) if selected_pairs else None,
            "low_split": str(LOW_SPLIT) if selected_pairs else None,
        },
        "decision": {
            "bounded_smoke_candidate": bool(selected_pairs),
            "promotion_authorized": False,
            "canonical_multi_selection": False,
            "trackc_query_use": False,
        },
    }
    write_json(JSON_OUT, payload)

    report_lines = [
        "# LatentFM HVG-Advantage Residual V3 Pair Pool 20260630",
        "",
        f"Created: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only v3 pair-pool construction.",
        "- Uses only the completed residualized condition-axis matrix and parent train split.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C query, or GPU.",
        "",
        "## Hypothesis",
        "",
        "`hvg_advantage_resid` may capture nonstatic condition-level information:",
        "HVG-over-abundance response concentration after matching response magnitude,",
        "cell support, abundance concentration, dataset, perturbation type, and gene-count bin.",
        "",
        "## Mode Summary",
        "",
        markdown_table(
            summary_rows,
            [
                "mode",
                "n_pairs",
                "n_datasets",
                "mean_axis_delta",
                "mean_confound_distance",
                "pair_distance_gt_0p5_fraction",
                "max_abs_confound_smd",
                "max_dataset_fraction",
                "top_dataset",
                "max_perturbation_type_fraction",
                "top_perturbation_type",
                "status",
                "reasons",
            ],
        ),
        "",
        "## Decision",
        "",
    ]
    if selected_summary:
        report_lines.extend(
            [
                f"- Selected mode: `{selected_summary.mode}`.",
                f"- Selected pairs: `{selected_summary.n_pairs}` across `{selected_summary.n_datasets}` datasets.",
                f"- Mean confound distance: `{selected_summary.mean_confound_distance:.6f}`; max SMD: `{selected_summary.max_abs_confound_smd:.6f}`.",
                f"- Pair-distance `>0.5` fraction: `{selected_summary.pair_distance_gt_0p5_fraction:.6f}`.",
                "- This passes the CPU pair-pool design gate, but remains GPU-authorized `False` until a bounded launcher protocol is explicitly invoked.",
                "- Promotion remains blocked until high beats low, placebo/random controls collapse, and a later dual-baseline no-harm gate passes.",
            ]
        )
    else:
        report_lines.extend(
            [
                "- No v3 mode passes the CPU pair-pool design gate.",
                "- Close local observable-information as descriptor-only unless a new external small table appears.",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- JSON: `{JSON_OUT}`",
            f"- packet: `{PACKET_JSON}`",
            f"- mode summary: `{SUMMARY_CSV}`",
            f"- selected pairs: `{PAIR_CSV if selected_pairs else 'NA'}`",
            f"- high split: `{HIGH_SPLIT if selected_pairs else 'NA'}`",
            f"- low split: `{LOW_SPLIT if selected_pairs else 'NA'}`",
        ]
    )
    REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "selected_mode": payload["selected_mode"], "report": str(REPORT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
