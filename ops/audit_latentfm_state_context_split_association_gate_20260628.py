#!/usr/bin/env python3
"""Split-level association gate for state/context support scaling variables."""

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
COND_MATRIX = ROOT / "reports/state_context_condition_matrix_20260628/state_context_condition_matrix.csv"
SPLIT_MATRIX = ROOT / "reports/exact_response_information_posthoc_parent_train_complete_20260628/exact_response_information_split_matrix.csv"
OUT_DIR = ROOT / "reports/state_context_split_association_gate_20260628"


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


def load_split(path_s: str) -> dict[str, Any]:
    path = Path(path_s)
    if not path.is_absolute():
        path = ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def effective_count(values: pd.Series) -> float:
    vc = values.dropna().astype(str)
    if vc.empty:
        return 0.0
    counts = vc.value_counts().to_numpy(dtype=float)
    p = counts / counts.sum()
    return float(1.0 / np.sum(p * p))


def aggregate_for_split(split_file: str, cond_lookup: pd.DataFrame) -> dict[str, Any]:
    split = load_split(split_file)
    rows: list[pd.DataFrame] = []
    n_missing = 0
    for dataset, spec in split.items():
        train_conds = [str(c) for c in spec.get("train", [])]
        if not train_conds:
            continue
        key = pd.MultiIndex.from_product([[str(dataset)], train_conds], names=["dataset", "condition"])
        hit = cond_lookup.reindex(key).reset_index()
        n_missing += int(hit["n_gt_cells"].isna().sum())
        rows.append(hit)
    if not rows:
        return {"split_file": split_file, "state_context_rows": 0}
    df = pd.concat(rows, ignore_index=True)
    for col in ["n_gt_cells", "max_state_unique", "max_state_entropy", "max_context_unique"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    n = len(df)
    state_rows = int((df["max_state_unique"] > 0).sum())
    context_rows = int((df["max_context_unique"] > 0).sum())
    return {
        "split_file": split_file,
        "state_context_rows": int(n),
        "state_context_missing_rows": int(n_missing),
        "state_signal_fraction": float(state_rows / max(n, 1)),
        "context_signal_fraction": float(context_rows / max(n, 1)),
        "mean_state_entropy": float(df["max_state_entropy"].mean()),
        "max_state_entropy": float(df["max_state_entropy"].max()),
        "mean_gt_cells": float(df["n_gt_cells"].mean()),
        "median_gt_cells": float(df["n_gt_cells"].median()),
        "state_dataset_effective_count": effective_count(df.loc[df["max_state_unique"] > 0, "dataset"]),
        "all_dataset_effective_count_from_matrix": effective_count(df["dataset"]),
    }


def association_rows(joined: pd.DataFrame) -> pd.DataFrame:
    predictors = [
        "state_signal_fraction",
        "context_signal_fraction",
        "mean_state_entropy",
        "max_state_entropy",
        "mean_gt_cells",
        "median_gt_cells",
        "state_dataset_effective_count",
        "all_dataset_effective_count_from_matrix",
    ]
    outcomes = ["cross_pp_delta", "family_pp_delta", "family_mmd_delta", "tail_score"]
    rows: list[dict[str, Any]] = []
    df = joined[joined["has_downstream_outcome"].astype(bool)].copy()
    for pred in predictors:
        for outcome in outcomes:
            sub = df[[pred, outcome]].dropna()
            if len(sub) < 6 or sub[pred].nunique() < 2 or sub[outcome].nunique() < 2:
                rho = pval = float("nan")
            else:
                rho, pval = spearmanr(sub[pred], sub[outcome])
            rows.append(
                {
                    "predictor": pred,
                    "outcome": outcome,
                    "n": int(len(sub)),
                    "spearman_rho": float(rho) if math.isfinite(float(rho)) else float("nan"),
                    "p_value": float(pval) if math.isfinite(float(pval)) else float("nan"),
                    "abs_rho": abs(float(rho)) if math.isfinite(float(rho)) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def write_report(out_dir: Path, joined: pd.DataFrame, assoc: pd.DataFrame) -> None:
    top = assoc.dropna(subset=["abs_rho"]).sort_values("abs_rho", ascending=False).head(8)
    gpu = False
    lines: list[str] = []
    lines.append("# LatentFM State/Context Split Association Gate")
    lines.append("")
    lines.append(f"Timestamp: `{now_cst()}`")
    lines.append("")
    lines.append("Status: `state_context_split_association_hypothesis_only_no_gpu`")
    lines.append("")
    lines.append(f"GPU authorized: `{gpu}`")
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("- CPU/report-only split-level aggregation over frozen condition metadata and outcome matrices.")
    lines.append("- Reads split JSONs but does not modify them; no training/inference/GPU/canonical multi selection/Track C query.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Split rows: `{len(joined)}`.")
    lines.append(f"- Split rows with downstream outcomes: `{int(joined['has_downstream_outcome'].astype(bool).sum())}`.")
    lines.append(f"- State signal fraction range: `{fmt(joined['state_signal_fraction'].min())}` to `{fmt(joined['state_signal_fraction'].max())}`.")
    lines.append(f"- Context signal fraction range: `{fmt(joined['context_signal_fraction'].min())}` to `{fmt(joined['context_signal_fraction'].max())}`.")
    lines.append("")
    lines.append("## Top Associations")
    lines.append("")
    lines.append("| predictor | outcome | n | rho | p |")
    lines.append("|---|---|---:|---:|---:|")
    for _, row in top.iterrows():
        lines.append(
            f"| {row['predictor']} | {row['outcome']} | {int(row['n'])} | {fmt(row['spearman_rho'])} | {fmt(row['p_value'])} |"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("- This is a hypothesis-only association screen, not a GPU packet.")
    lines.append("- State/context variables are dataset-concentrated and require source-family LODO plus dual-baseline no-harm before model use.")
    lines.append("- Keep state/context support as a covariate and candidate curriculum axis; do not replay old support-only GPU routes.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- joined split rows: `{out_dir / 'state_context_split_join_rows.csv'}`")
    lines.append(f"- associations: `{out_dir / 'state_context_split_association_rows.csv'}`")
    lines.append(f"- JSON: `{out_dir / 'state_context_split_association_gate_20260628.json'}`")
    (out_dir / "LATENTFM_STATE_CONTEXT_SPLIT_ASSOCIATION_GATE_20260628.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cond = pd.read_csv(COND_MATRIX)
    cond_lookup = cond.set_index(["dataset", "condition"])
    split_matrix = pd.read_csv(SPLIT_MATRIX)
    agg_rows = [aggregate_for_split(str(p), cond_lookup) for p in split_matrix["split_file"].tolist()]
    agg = pd.DataFrame(agg_rows)
    joined = split_matrix.merge(agg, on="split_file", how="left")
    assoc = association_rows(joined)
    joined_path = OUT_DIR / "state_context_split_join_rows.csv"
    assoc_path = OUT_DIR / "state_context_split_association_rows.csv"
    joined.to_csv(joined_path, index=False)
    assoc.to_csv(assoc_path, index=False)
    obj = {
        "timestamp": now_cst(),
        "status": "state_context_split_association_hypothesis_only_no_gpu",
        "gpu_authorized_next": False,
        "n_split_rows": int(len(joined)),
        "n_outcome_rows": int(joined["has_downstream_outcome"].astype(bool).sum()),
        "top_associations": assoc.dropna(subset=["abs_rho"]).sort_values("abs_rho", ascending=False).head(8).to_dict(orient="records"),
        "outputs": {
            "joined": str(joined_path),
            "associations": str(assoc_path),
            "report": str(OUT_DIR / "LATENTFM_STATE_CONTEXT_SPLIT_ASSOCIATION_GATE_20260628.md"),
        },
    }
    write_json(OUT_DIR / "state_context_split_association_gate_20260628.json", obj)
    write_report(OUT_DIR, joined, assoc)


if __name__ == "__main__":
    main()
