#!/usr/bin/env python3
"""Strict CPU admission gate for SciPlex pharmacogenomic source overlap.

This script consumes the preflight produced by
materialize_sciplex_pharmacogenomic_source_bundle_20260630.py. It checks
whether external pharmacogenomic viability has enough source-block consistency
to justify a later outcome/no-harm CPU gate.

It does not train, run inference, select checkpoints, read canonical multi for
selection, read Track C query, or use GPU. Passing this script still does not
authorize GPU.
"""

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
PREFLIGHT_DIR = ROOT / "reports" / "sciplex_pharmacogenomic_source_preflight_20260630"
PREFLIGHT_JSON = PREFLIGHT_DIR / "sciplex_pharmacogenomic_source_preflight_20260630.json"
OVERLAP_ROWS = PREFLIGHT_DIR / "sciplex_pharmacogenomic_source_overlap_rows_20260630.csv"
OUT_DIR = ROOT / "reports" / "sciplex_pharmacogenomic_source_admission_gate_20260630"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def spearman(x: pd.Series, y: pd.Series) -> float:
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    mask = xv.notna() & yv.notna()
    if int(mask.sum()) < 5:
        return float("nan")
    return float(xv[mask].rank().corr(yv[mask].rank()))


def zscore_by_source(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, sub in df.groupby("source", sort=False):
        vals = pd.to_numeric(sub["sensitivity_score"], errors="coerce")
        mean = float(vals.mean())
        std = float(vals.std(ddof=0))
        if not math.isfinite(std) or std <= 1e-12:
            std = 1.0
        tmp = sub.copy()
        tmp["source_sensitivity_z"] = (vals - mean) / std
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else df


def main() -> int:
    preflight = read_json(PREFLIGHT_JSON)
    summary = preflight.get("summary", {})
    preflight_ready = bool(summary.get("ready_for_cpu_admission"))

    blocked_reasons = []
    if not PREFLIGHT_JSON.exists():
        blocked_reasons.append("preflight_json_missing")
    if not OVERLAP_ROWS.exists():
        blocked_reasons.append("overlap_rows_missing")
    if not preflight_ready:
        blocked_reasons.extend(summary.get("reasons") or ["preflight_not_ready"])

    status = "sciplex_pharmacogenomic_source_admission_gate_blocked_preflight_no_gpu"
    result: dict[str, Any] = {
        "timestamp": now(),
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_report_only": True,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
            "uses_gpu": False,
        },
        "preflight_json": str(PREFLIGHT_JSON),
        "overlap_rows": str(OVERLAP_ROWS),
        "blocked_reasons": blocked_reasons,
    }

    rows_path = OUT_DIR / "sciplex_pharmacogenomic_source_admission_rows_20260630.csv"
    source_block_path = OUT_DIR / "sciplex_pharmacogenomic_source_lodo_rows_20260630.csv"
    json_path = OUT_DIR / "sciplex_pharmacogenomic_source_admission_gate_20260630.json"
    md_path = OUT_DIR / "LATENTFM_SCIPLEX_PHARMACOGENOMIC_SOURCE_ADMISSION_GATE_20260630.md"

    admission_rows = pd.DataFrame()
    lodo_rows: list[dict[str, Any]] = []

    if not blocked_reasons:
        df = pd.read_csv(OVERLAP_ROWS)
        if df.empty:
            blocked_reasons.append("overlap_rows_empty")
        else:
            # AUC is viability area under curve; lower AUC implies stronger response.
            auc = pd.to_numeric(df["AUC_mode"], errors="coerce")
            auc = auc.fillna(pd.to_numeric(df["AUC_all"], errors="coerce"))
            df = df.copy()
            df["auc_used"] = auc
            df = df[df["auc_used"].notna() & np.isfinite(df["auc_used"])]
            df["sensitivity_score"] = 1.0 - df["auc_used"].astype(float)
            df = zscore_by_source(df)
            train = df[df["split_group"] == "train"].copy()
            key_cols = ["dataset", "cell_line", "sciplex_condition"]
            agg = (
                train.groupby(key_cols, as_index=False)
                .agg(
                    drug_norm=("drug_norm", "first"),
                    source_count=("source", "nunique"),
                    mean_sensitivity_z=("source_sensitivity_z", "mean"),
                    sd_sensitivity_z=("source_sensitivity_z", "std"),
                    mean_auc=("auc_used", "mean"),
                    n_source_rows=("source", "size"),
                )
            )
            agg["sd_sensitivity_z"] = agg["sd_sensitivity_z"].fillna(0.0)
            admission_rows = agg

            for heldout in sorted(train["source"].unique()):
                held = train[train["source"] == heldout]
                rest = train[train["source"] != heldout]
                rest_agg = (
                    rest.groupby(key_cols, as_index=False)["source_sensitivity_z"]
                    .mean()
                    .rename(columns={"source_sensitivity_z": "other_source_mean_z"})
                )
                merged = held.merge(rest_agg, on=key_cols, how="inner")
                rho = spearman(merged["source_sensitivity_z"], merged["other_source_mean_z"])
                lodo_rows.append(
                    {
                        "heldout_source": heldout,
                        "n_overlap": int(len(merged)),
                        "spearman_vs_other_sources": rho,
                    }
                )

            lodo_df = pd.DataFrame(lodo_rows)
            n_train_multi = int((agg["source_count"] >= 2).sum())
            n_backgrounds = int(agg.loc[agg["source_count"] >= 2, "dataset"].nunique())
            lodo_pass_rows = int(
                (
                    (pd.to_numeric(lodo_df["n_overlap"], errors="coerce") >= 20)
                    & (pd.to_numeric(lodo_df["spearman_vs_other_sources"], errors="coerce") > 0.20)
                ).sum()
            )
            reasons: list[str] = []
            if n_train_multi < 50:
                reasons.append("multi_source_train_rows_lt_50")
            if n_backgrounds < 3:
                reasons.append("backgrounds_lt_3")
            if len(lodo_df) < 2:
                reasons.append("source_blocks_lt_2")
            if lodo_pass_rows < 2:
                reasons.append("lodo_source_consistency_pass_rows_lt_2")
            status = (
                "sciplex_pharmacogenomic_source_admission_gate_pass_outcome_gate_next_no_gpu"
                if not reasons
                else "sciplex_pharmacogenomic_source_admission_gate_fail_no_gpu"
            )
            result.update(
                {
                    "status": status,
                    "blocked_reasons": reasons,
                    "n_train_rows": int(len(agg)),
                    "n_train_multisource_rows": n_train_multi,
                    "n_backgrounds_multisource": n_backgrounds,
                    "n_source_blocks": int(train["source"].nunique()),
                    "lodo_pass_rows": lodo_pass_rows,
                    "decision": (
                        "Proceed only to a later outcome/no-harm CPU gate; GPU remains blocked."
                        if not reasons
                        else "Do not proceed to GPU or outcome gate without fixing source consistency/overlap."
                    ),
                }
            )

    admission_rows.to_csv(rows_path, index=False)
    pd.DataFrame(lodo_rows).to_csv(source_block_path, index=False)
    result["outputs"] = {
        "rows": str(rows_path),
        "lodo_rows": str(source_block_path),
        "json": str(json_path),
        "markdown": str(md_path),
    }
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# SciPlex Pharmacogenomic Source Admission Gate",
        "",
        f"Created: `{result['timestamp']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only source consistency gate.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C query, or GPU.",
        "- Passing this gate only permits a later outcome/no-harm CPU gate.",
        "",
        "## Decision",
        "",
        result.get("decision", "Blocked before admission because preflight outputs are missing or insufficient."),
        "",
        "## Reasons",
        "",
    ]
    reasons = result.get("blocked_reasons", [])
    if reasons:
        for reason in reasons:
            lines.append(f"- `{reason}`")
    else:
        lines.append("- `none`")
    lines += [
        "",
        "## Metrics",
        "",
        f"- train rows: `{result.get('n_train_rows', 'NA')}`",
        f"- train multi-source rows: `{result.get('n_train_multisource_rows', 'NA')}`",
        f"- multi-source backgrounds: `{result.get('n_backgrounds_multisource', 'NA')}`",
        f"- source blocks: `{result.get('n_source_blocks', 'NA')}`",
        f"- LODO pass rows: `{result.get('lodo_pass_rows', 'NA')}`",
        "",
        "## Outputs",
        "",
        f"- rows: `{rows_path}`",
        f"- LODO rows: `{source_block_path}`",
        f"- JSON: `{json_path}`",
        f"- Markdown: `{md_path}`",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "json": str(json_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
