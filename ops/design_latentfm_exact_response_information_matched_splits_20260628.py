#!/usr/bin/env python3
"""Find matched split candidates for exact response-information scaling tests."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
MATRIX_CSV = ROOT / "reports/exact_response_information_posthoc_20260628/exact_response_information_split_matrix.csv"
OUT_DIR = ROOT / "reports/exact_response_information_matched_split_design_20260628"
OUT_MD = OUT_DIR / "LATENTFM_EXACT_RESPONSE_INFORMATION_MATCHED_SPLIT_DESIGN_20260628.md"
OUT_JSON = OUT_DIR / "latentfm_exact_response_information_matched_split_design_20260628.json"
OUT_PAIRS = OUT_DIR / "exact_response_information_matched_pair_candidates.csv"

AXIS = "exact_condition_fraction"
CONFOUNDS = [
    "n_train_conditions",
    "base_dataset_effective_count",
    "base_background_effective_count",
    "base_perturbation_type_effective_count",
    "base_target_gene_effective_count",
    "exact_abundance_share_top1000_mean",
    "exact_abundance_k90_mean",
]


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt_float(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def zscores(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    z = pd.DataFrame(index=frame.index)
    for col in cols:
        vals = frame[col].astype(float)
        std = float(vals.std(ddof=0))
        if std <= 0 or not np.isfinite(std):
            z[col] = 0.0
        else:
            z[col] = (vals - float(vals.mean())) / std
    return z


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(MATRIX_CSV)
    frame = (
        raw.sort_values(["has_downstream_outcome", AXIS], ascending=[False, False])
        .drop_duplicates("split_name")
        .reset_index(drop=True)
    )
    numeric_cols = [AXIS] + CONFOUNDS
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=numeric_cols)
    z = zscores(frame, CONFOUNDS)

    rows: list[dict[str, Any]] = []
    for i in range(len(frame)):
        for j in range(i + 1, len(frame)):
            left = frame.iloc[i]
            right = frame.iloc[j]
            axis_delta = float(left[AXIS] - right[AXIS])
            confound_z = {col: abs(float(z.loc[i, col] - z.loc[j, col])) for col in CONFOUNDS}
            max_confound_z = max(confound_z.values()) if confound_z else 0.0
            mean_confound_z = float(np.mean(list(confound_z.values()))) if confound_z else 0.0
            rows.append(
                {
                    "left_split": left["split_name"],
                    "right_split": right["split_name"],
                    "axis_delta": axis_delta,
                    "abs_axis_delta": abs(axis_delta),
                    "left_axis": float(left[AXIS]),
                    "right_axis": float(right[AXIS]),
                    "max_confound_z": max_confound_z,
                    "mean_confound_z": mean_confound_z,
                    "both_have_downstream_outcome": bool(left["has_downstream_outcome"] and right["has_downstream_outcome"]),
                    "candidate_strict": bool(abs(axis_delta) >= 0.15 and max_confound_z <= 1.0),
                    "candidate_relaxed": bool(abs(axis_delta) >= 0.15 and max_confound_z <= 1.5),
                    **{f"delta_z_{col}": confound_z[col] for col in CONFOUNDS},
                }
            )
    rows = sorted(rows, key=lambda r: (not r["candidate_strict"], not r["candidate_relaxed"], r["max_confound_z"], -r["abs_axis_delta"]))

    fields = [
        "left_split",
        "right_split",
        "axis_delta",
        "abs_axis_delta",
        "left_axis",
        "right_axis",
        "max_confound_z",
        "mean_confound_z",
        "both_have_downstream_outcome",
        "candidate_strict",
        "candidate_relaxed",
    ] + [f"delta_z_{col}" for col in CONFOUNDS]
    write_csv(OUT_PAIRS, rows, fields)

    strict = [row for row in rows if row["candidate_strict"]]
    relaxed = [row for row in rows if row["candidate_relaxed"]]
    status = "exact_response_information_matched_split_design_no_clean_pair_no_gpu"
    if strict:
        status = "exact_response_information_matched_split_design_strict_candidates_no_gpu"
    elif relaxed:
        status = "exact_response_information_matched_split_design_relaxed_candidates_no_gpu"

    payload = {
        "created_at": now_cst(),
        "status": status,
        "unique_split_rows": int(frame.shape[0]),
        "pair_rows": len(rows),
        "strict_candidates": len(strict),
        "relaxed_candidates": len(relaxed),
        "pair_csv": str(OUT_PAIRS),
        "axis": AXIS,
        "confounds": CONFOUNDS,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    top_rows = strict[:8] if strict else relaxed[:8] if relaxed else rows[:8]
    lines = [
        "# LatentFM Exact Response-Information Matched Split Design",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only candidate search over existing split rows.",
        "* This does not train, infer, use canonical multi, use Track C query, select checkpoints, or authorize GPU.",
        "",
        "## Summary",
        "",
        f"* Unique split rows: `{payload['unique_split_rows']}`.",
        f"* Pair rows: `{payload['pair_rows']}`.",
        f"* Strict candidates: `{payload['strict_candidates']}` (`abs(axis_delta)>=0.15`, max confound z<=1.0).",
        f"* Relaxed candidates: `{payload['relaxed_candidates']}` (`abs(axis_delta)>=0.15`, max confound z<=1.5).",
        "",
        "## Top Candidate Rows",
        "",
        "| left | right | axis delta | max confound z | relaxed | strict |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in top_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["left_split"]),
                    str(row["right_split"]),
                    fmt_float(row["axis_delta"]),
                    fmt_float(row["max_confound_z"]),
                    str(row["candidate_relaxed"]),
                    str(row["candidate_strict"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"* Status: `{status}`.",
            "* A future GPU smoke would still require a separate launcher, leakage-safe split protocol, dual baseline, and no-harm gate.",
            "* If only relaxed/no candidates exist, keep this as design evidence and create new matched split definitions before GPU.",
            "",
            "## Outputs",
            "",
            f"* Pair candidates: `{OUT_PAIRS}`",
            f"* JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"status {status}")


if __name__ == "__main__":
    main()
