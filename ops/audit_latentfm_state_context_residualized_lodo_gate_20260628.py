#!/usr/bin/env python3
"""Residualized/LODO gate for state-context scaling associations."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("/data/cyx/1030/scLatent")
JOIN = ROOT / "reports/state_context_split_association_gate_20260628/state_context_split_join_rows.csv"
OUT_DIR = ROOT / "reports/state_context_residualized_lodo_gate_20260628"


CONTROL_SETS = {
    "cells_only": ["mean_gt_cells"],
    "cells_exact": ["mean_gt_cells", "exact_condition_fraction"],
    "cells_exact_dataset": ["mean_gt_cells", "exact_condition_fraction", "base_dataset_effective_count"],
    "cells_exact_hvg_dataset": [
        "mean_gt_cells",
        "exact_condition_fraction",
        "exact_hvg_share_top1000_mean",
        "base_dataset_effective_count",
    ],
}
PREDICTORS = ["state_signal_fraction", "mean_state_entropy", "state_dataset_effective_count"]
OUTCOMES = ["family_mmd_delta", "tail_score", "cross_pp_delta", "family_pp_delta"]


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    cols = []
    for j in range(x.shape[1]):
        col = x[:, j]
        if np.nanstd(col) > 1e-12:
            cols.append((col - np.nanmean(col)) / np.nanstd(col))
    if cols:
        X = np.column_stack([np.ones(len(y)), *cols])
    else:
        X = np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def source_family(split_file: str) -> str:
    path = Path(str(split_file))
    return path.parent.name or "root"


def assoc(df: pd.DataFrame, pred: str, outcome: str, controls: list[str], control_name: str) -> dict[str, Any]:
    cols = [pred, outcome, *controls]
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < len(controls) + 6 or sub[pred].nunique() < 2 or sub[outcome].nunique() < 2:
        rho = pval = float("nan")
    else:
        pr = residualize(sub[pred].to_numpy(), sub[controls].to_numpy())
        yr = residualize(sub[outcome].to_numpy(), sub[controls].to_numpy())
        if np.std(pr) <= 1e-12 or np.std(yr) <= 1e-12:
            rho = pval = float("nan")
        else:
            rho, pval = spearmanr(pr, yr)
    return {
        "predictor": pred,
        "outcome": outcome,
        "control_set": control_name,
        "controls": ";".join(controls),
        "n": int(len(sub)),
        "residual_spearman_rho": float(rho) if math.isfinite(float(rho)) else float("nan"),
        "p_value": float(pval) if math.isfinite(float(pval)) else float("nan"),
        "abs_rho": abs(float(rho)) if math.isfinite(float(rho)) else float("nan"),
    }


def lodo_rows(df: pd.DataFrame, pred: str, outcome: str, controls: list[str], control_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fam in sorted(df["source_family"].dropna().unique()):
        sub = df[df["source_family"] != fam]
        row = assoc(sub, pred, outcome, controls, control_name)
        row["left_out_source_family"] = fam
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(out_dir: Path, rows: pd.DataFrame, lodo: pd.DataFrame) -> None:
    top = rows.dropna(subset=["abs_rho"]).sort_values("abs_rho", ascending=False).head(10)
    primary = rows[
        (rows["predictor"].isin(["state_signal_fraction", "mean_state_entropy"]))
        & (rows["outcome"] == "family_mmd_delta")
        & (rows["control_set"] == "cells_exact_dataset")
    ]
    lodo_primary = lodo[
        (lodo["predictor"].isin(["state_signal_fraction", "mean_state_entropy"]))
        & (lodo["outcome"] == "family_mmd_delta")
        & (lodo["control_set"] == "cells_exact_dataset")
    ]
    same_sign = float("nan")
    if not primary.empty and not lodo_primary.empty:
        primary_sign = np.sign(primary["residual_spearman_rho"].mean())
        vals = lodo_primary["residual_spearman_rho"].dropna().to_numpy(dtype=float)
        same_sign = float((np.sign(vals) == primary_sign).mean()) if vals.size else float("nan")
    gpu = False
    status = "state_context_residualized_lodo_partial_no_gpu"
    lines: list[str] = []
    lines.append("# LatentFM State/Context Residualized LODO Gate")
    lines.append("")
    lines.append(f"Timestamp: `{now_cst()}`")
    lines.append("")
    lines.append(f"Status: `{status}`")
    lines.append("")
    lines.append(f"GPU authorized: `{gpu}`")
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("- CPU/report-only residual association over frozen split-level rows.")
    lines.append("- No training, inference, checkpoint selection, canonical multi selection, or Track C query.")
    lines.append("")
    lines.append("## Primary Control")
    lines.append("")
    lines.append("- Primary control set: `cells_exact_dataset` = mean GT cells, exact condition fraction, base dataset effective count.")
    lines.append(f"- Primary LODO same-sign rate: `{fmt(same_sign)}`.")
    lines.append("")
    lines.append("## Top Residual Associations")
    lines.append("")
    lines.append("| predictor | outcome | controls | n | rho | p |")
    lines.append("|---|---|---|---:|---:|---:|")
    for _, row in top.iterrows():
        lines.append(
            f"| {row['predictor']} | {row['outcome']} | {row['control_set']} | {int(row['n'])} | {fmt(row['residual_spearman_rho'])} | {fmt(row['p_value'])} |"
        )
    lines.append("")
    lines.append("## Primary Rows")
    lines.append("")
    lines.append("| predictor | outcome | control set | n | rho | p |")
    lines.append("|---|---|---|---:|---:|---:|")
    for _, row in primary.iterrows():
        lines.append(
            f"| {row['predictor']} | {row['outcome']} | {row['control_set']} | {int(row['n'])} | {fmt(row['residual_spearman_rho'])} | {fmt(row['p_value'])} |"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("- State/context support does not reopen GPU at this gate.")
    lines.append("- Treat any residual signal as hypothesis-generating until it survives source-family LODO, dataset/cell-count controls, and dual-baseline no-harm.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- residual associations: `{out_dir / 'state_context_residualized_association_rows.csv'}`")
    lines.append(f"- LODO rows: `{out_dir / 'state_context_residualized_lodo_rows.csv'}`")
    lines.append(f"- JSON: `{out_dir / 'state_context_residualized_lodo_gate_20260628.json'}`")
    (out_dir / "LATENTFM_STATE_CONTEXT_RESIDUALIZED_LODO_GATE_20260628.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(JOIN)
    df = df[df["has_downstream_outcome"].astype(bool)].copy()
    df["source_family"] = df["split_file"].map(source_family)
    rows = []
    lodo_frames = []
    for cname, controls in CONTROL_SETS.items():
        for pred in PREDICTORS:
            for outcome in OUTCOMES:
                rows.append(assoc(df, pred, outcome, controls, cname))
                if cname == "cells_exact_dataset" and outcome == "family_mmd_delta":
                    lodo_frames.append(lodo_rows(df, pred, outcome, controls, cname))
    out = pd.DataFrame(rows)
    lodo = pd.concat(lodo_frames, ignore_index=True) if lodo_frames else pd.DataFrame()
    assoc_path = OUT_DIR / "state_context_residualized_association_rows.csv"
    lodo_path = OUT_DIR / "state_context_residualized_lodo_rows.csv"
    out.to_csv(assoc_path, index=False)
    lodo.to_csv(lodo_path, index=False)
    primary = out[
        (out["predictor"].isin(["state_signal_fraction", "mean_state_entropy"]))
        & (out["outcome"] == "family_mmd_delta")
        & (out["control_set"] == "cells_exact_dataset")
    ]
    obj = {
        "timestamp": now_cst(),
        "status": "state_context_residualized_lodo_partial_no_gpu",
        "gpu_authorized_next": False,
        "n_outcome_rows": int(len(df)),
        "primary_rows": primary.to_dict(orient="records"),
        "outputs": {
            "associations": str(assoc_path),
            "lodo": str(lodo_path),
            "report": str(OUT_DIR / "LATENTFM_STATE_CONTEXT_RESIDUALIZED_LODO_GATE_20260628.md"),
        },
    }
    write_json(OUT_DIR / "state_context_residualized_lodo_gate_20260628.json", obj)
    write_report(OUT_DIR, out, lodo)


if __name__ == "__main__":
    main()
