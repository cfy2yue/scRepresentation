#!/usr/bin/env python3
"""Gate condition-level QC/support reliability as a scaling-training signal.

Short CPU task. Reads AnnData ``.obs`` metadata and completed train-only
condition-exposure row metrics only. It does not read expression matrices,
checkpoints, canonical multi, Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OBS_SCHEMA_JSON = REPORTS / "latentfm_dataset_obs_schema_audit_20260624.json"
EXPOSURE_ROWS = REPORTS / "latentfm_condition_exposure_row_bootstrap_rows_20260625.csv"
OUT_JSON = REPORTS / "latentfm_qc_support_reliability_gate_20260625.json"
OUT_CSV = REPORTS / "latentfm_qc_support_reliability_rows_20260625.csv"
OUT_MD = REPORTS / "LATENTFM_QC_SUPPORT_RELIABILITY_GATE_20260625.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return text


def pick_first(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def infer_condition_col(columns: list[str], modality: str) -> str | None:
    if modality == "chemical":
        return pick_first(columns, ["cov_drug_dose_name", "drug_dose_name", "cov_drug", "condition", "cov"])
    return pick_first(columns, ["perturbation", "condition", "gene", "target"])


def is_control_condition(text: str) -> bool:
    return text.lower() in {
        "control",
        "ctrl",
        "ntc",
        "non-targeting",
        "non_targeting",
        "dmso",
        "vehicle",
    }


def numeric_series(frame: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    col = pick_first([str(c) for c in frame.columns], candidates)
    if col is None:
        return None
    return pd.to_numeric(frame[col], errors="coerce")


def aggregate_qc_rows() -> pd.DataFrame:
    obs_payload = read_json(OBS_SCHEMA_JSON)
    rows = []
    for file_row in obs_payload["rows"]:
        if file_row.get("status") != "ok":
            continue
        path = Path(str(file_row["path"]))
        if not path.is_file():
            continue
        modality = str(file_row.get("modality", ""))
        dataset = str(file_row.get("dataset", ""))
        bucket = str(file_row.get("bucket", ""))
        obj = ad.read_h5ad(path, backed="r")
        try:
            obs = obj.obs.copy()
        finally:
            if getattr(obj, "file", None) is not None:
                obj.file.close()
        condition_col = infer_condition_col([str(c) for c in obs.columns], modality)
        if condition_col is None or condition_col not in obs.columns:
            continue
        obs["_condition_norm"] = obs[condition_col].map(norm)
        obs = obs.loc[~obs["_condition_norm"].map(is_control_condition)].copy()
        n_genes = numeric_series(obs, ["n_genes_by_counts", "n_genes", "ngenes"])
        total_counts = numeric_series(obs, ["total_counts", "ncounts"])
        pct_mito = numeric_series(obs, ["pct_counts_mt", "percent_mito"])
        pct_ribo = numeric_series(obs, ["percent_ribo"])
        obs["_n_genes_metric"] = n_genes if n_genes is not None else np.nan
        obs["_total_counts_metric"] = total_counts if total_counts is not None else np.nan
        obs["_pct_mito_metric"] = pct_mito if pct_mito is not None else np.nan
        obs["_pct_ribo_metric"] = pct_ribo if pct_ribo is not None else np.nan
        grouped = obs.groupby("_condition_norm", observed=True, dropna=False)
        for condition, frame in grouped:
            condition_s = norm(condition)
            if not condition_s:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition_s,
                    "bucket": bucket,
                    "modality": modality,
                    "n_cells": int(len(frame)),
                    "mean_n_genes": float(frame["_n_genes_metric"].mean()),
                    "mean_total_counts": float(frame["_total_counts_metric"].mean()),
                    "mean_pct_mito": float(frame["_pct_mito_metric"].mean()),
                    "mean_pct_ribo": float(frame["_pct_ribo_metric"].mean()),
                }
            )
    qc = pd.DataFrame(rows)
    return add_reliability_score(qc)


def zscore_by_dataset(df: pd.DataFrame, col: str) -> pd.Series:
    vals = pd.to_numeric(df[col], errors="coerce")
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in df.groupby("dataset").groups.items():
        v = vals.loc[idx]
        mu = float(v.mean(skipna=True))
        sd = float(v.std(skipna=True))
        if not np.isfinite(sd) or sd == 0:
            out.loc[idx] = 0.0
        else:
            out.loc[idx] = (v - mu) / sd
    return out


def add_reliability_score(qc: pd.DataFrame) -> pd.DataFrame:
    qc = qc.copy()
    qc["log_n_cells"] = np.log1p(qc["n_cells"].astype(float))
    qc["log_total_counts"] = np.log1p(pd.to_numeric(qc["mean_total_counts"], errors="coerce"))
    components = {
        "z_log_n_cells": zscore_by_dataset(qc, "log_n_cells"),
        "z_mean_n_genes": zscore_by_dataset(qc, "mean_n_genes"),
        "z_log_total_counts": zscore_by_dataset(qc, "log_total_counts"),
        "z_mean_pct_mito_negative": -zscore_by_dataset(qc, "mean_pct_mito"),
    }
    for key, value in components.items():
        qc[key] = value
    component_cols = list(components)
    qc["reliability_score"] = qc[component_cols].mean(axis=1, skipna=True)
    return qc


def load_primary_exposure_rows() -> pd.DataFrame:
    rows = pd.read_csv(EXPOSURE_ROWS)
    rows = rows[(rows["comparison"] == "cap120_minus_cap30") & (rows["group"] == "cross")].copy()
    rows["cross_pp_diff"] = pd.to_numeric(rows["cross_pp_diff"], errors="coerce")
    rows["cross_mmd_diff"] = pd.to_numeric(rows["cross_mmd_diff"], errors="coerce")
    return rows


def bootstrap_diff(values_high: np.ndarray, values_low: np.ndarray, *, n_boot: int = 4000) -> dict[str, float]:
    rng = np.random.default_rng(42)
    values_high = values_high[np.isfinite(values_high)]
    values_low = values_low[np.isfinite(values_low)]
    if len(values_high) == 0 or len(values_low) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    obs = float(values_high.mean() - values_low.mean())
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        hi = rng.choice(values_high, size=len(values_high), replace=True)
        lo = rng.choice(values_low, size=len(values_low), replace=True)
        boots[i] = hi.mean() - lo.mean()
    return {
        "mean": obs,
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
    }


def sign_permutation_p(joined: pd.DataFrame, *, n_perm: int = 4000) -> float:
    rng = np.random.default_rng(43)
    df = joined[["dataset", "high_reliability", "cross_pp_diff"]].dropna().copy()
    if df.empty:
        return float("nan")
    actual = group_diff(df)
    count = 0
    for _ in range(n_perm):
        perm = []
        for _, block in df.groupby("dataset"):
            labels = block["high_reliability"].to_numpy().copy()
            rng.shuffle(labels)
            b = block.copy()
            b["high_reliability"] = labels
            perm.append(b)
        diff = group_diff(pd.concat(perm, ignore_index=True))
        if diff >= actual:
            count += 1
    return float((count + 1) / (n_perm + 1))


def group_diff(df: pd.DataFrame) -> float:
    hi = df.loc[df["high_reliability"], "cross_pp_diff"].astype(float)
    lo = df.loc[~df["high_reliability"], "cross_pp_diff"].astype(float)
    if len(hi) == 0 or len(lo) == 0:
        return float("nan")
    return float(hi.mean() - lo.mean())


def summarize(joined: pd.DataFrame) -> dict[str, Any]:
    score_median = float(joined["reliability_score"].median())
    joined["high_reliability"] = joined["reliability_score"] >= score_median
    hi = joined.loc[joined["high_reliability"], "cross_pp_diff"].to_numpy(dtype=float)
    lo = joined.loc[~joined["high_reliability"], "cross_pp_diff"].to_numpy(dtype=float)
    boot = bootstrap_diff(hi, lo)
    p_perm = sign_permutation_p(joined)
    hi_rows = joined.loc[joined["high_reliability"]]
    lo_rows = joined.loc[~joined["high_reliability"]]
    summary = {
        "n_joined_rows": int(len(joined)),
        "n_datasets": int(joined["dataset"].nunique()),
        "score_median": score_median,
        "high_n": int(len(hi_rows)),
        "low_n": int(len(lo_rows)),
        "high_mean_pp": float(hi_rows["cross_pp_diff"].mean()),
        "low_mean_pp": float(lo_rows["cross_pp_diff"].mean()),
        "high_minus_low_pp": boot["mean"],
        "high_minus_low_pp_ci_low": boot["ci_low"],
        "high_minus_low_pp_ci_high": boot["ci_high"],
        "dataset_shuffle_p_ge_actual": p_perm,
        "high_tail_frac_lt_minus_0p02": float((hi_rows["cross_pp_diff"] < -0.02).mean()),
        "low_tail_frac_lt_minus_0p02": float((lo_rows["cross_pp_diff"] < -0.02).mean()),
        "high_dataset_min": float(hi_rows.groupby("dataset")["cross_pp_diff"].mean().min()),
        "low_dataset_min": float(lo_rows.groupby("dataset")["cross_pp_diff"].mean().min()),
        "corr_score_pp": float(joined["reliability_score"].corr(joined["cross_pp_diff"], method="spearman")),
    }
    reasons = []
    if not (summary["high_minus_low_pp_ci_low"] > 0):
        reasons.append("bootstrap_ci_low_not_positive")
    if not (summary["dataset_shuffle_p_ge_actual"] < 0.05):
        reasons.append("dataset_shuffle_control_not_separated")
    if not (summary["high_dataset_min"] >= -0.02):
        reasons.append("high_reliability_dataset_tail_below_minus_0p02")
    if not (summary["high_tail_frac_lt_minus_0p02"] <= summary["low_tail_frac_lt_minus_0p02"]):
        reasons.append("high_reliability_tail_fraction_not_lower")
    if summary["high_minus_low_pp"] < 0.01:
        reasons.append("effect_size_below_0p01")
    summary["gate_pass"] = not reasons
    summary["reasons"] = reasons
    summary["status"] = "qc_support_reliability_gate_pass_cpu_only" if not reasons else "qc_support_reliability_gate_fail_no_gpu"
    return summary


def write_rows(joined: pd.DataFrame) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "dataset",
        "condition",
        "n_cells",
        "mean_n_genes",
        "mean_total_counts",
        "mean_pct_mito",
        "reliability_score",
        "high_reliability",
        "cross_pp_diff",
        "cross_mmd_diff",
    ]
    joined[cols].to_csv(OUT_CSV, index=False)


def render_md(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# LatentFM QC/Support Reliability Gate",
        "",
        "Timestamp: `2026-06-25`",
        "",
        f"Status: `{s['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only condition-level QC/support reliability audit.",
        "- Reads raw h5ad `.obs` metadata and completed `cap120_minus_cap30` train-only row metrics.",
        "- Does not read expression matrices, checkpoints, canonical multi, Track C query, train, infer, or use GPU.",
        "",
        "## Primary Result",
        "",
        "| n rows | datasets | high mean pp | low mean pp | high-low pp CI95 | dataset-shuffle p | high tail frac | low tail frac | gate |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---|",
        (
            f"| {s['n_joined_rows']} | {s['n_datasets']} | {s['high_mean_pp']:+.6f} | "
            f"{s['low_mean_pp']:+.6f} | {s['high_minus_low_pp']:+.6f} "
            f"[{s['high_minus_low_pp_ci_low']:+.6f}, {s['high_minus_low_pp_ci_high']:+.6f}] | "
            f"{s['dataset_shuffle_p_ge_actual']:.4f} | {s['high_tail_frac_lt_minus_0p02']:.3f} | "
            f"{s['low_tail_frac_lt_minus_0p02']:.3f} | `{s['status']}` |"
        ),
        "",
        "## Decision",
        "",
    ]
    if s["gate_pass"]:
        lines.append(
            "- QC/support reliability is a viable CPU-passed training-set construction signal. "
            "Next step is a bounded leakage-safe filter/weighting GPU smoke after resource audit."
        )
    else:
        lines.append(
            "- QC/support reliability is useful for failure analysis but does not pass the CPU gate for training changes."
        )
    lines += [
        f"- fail/pass reasons: `{s['reasons']}`",
        f"- Spearman reliability-score vs pp: `{s['corr_score_pp']:+.6f}`",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- rows: `{OUT_CSV}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    qc = aggregate_qc_rows()
    exposure = load_primary_exposure_rows()
    joined = exposure.merge(qc, on=["dataset", "condition"], how="inner", validate="many_to_one")
    summary = summarize(joined)
    payload = {
        "boundary": {
            "cpu_only": True,
            "reads_obs_metadata": True,
            "reads_train_only_row_metrics": True,
            "reads_expression_matrix": False,
            "reads_model_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
        },
        "summary": summary,
        "outputs": {"json": str(OUT_JSON), "csv": str(OUT_CSV), "md": str(OUT_MD)},
    }
    write_rows(joined)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "gpu_authorized": False, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
