#!/usr/bin/env python3
"""Gate Jiang guide/cytokine/mixscale context as a scaling signal.

Short CPU task. Reads raw AnnData ``.obs`` metadata and completed train-only
condition-exposure row metrics only. It does not read expression matrices,
checkpoints, canonical multi, Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

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
OUT_JSON = REPORTS / "latentfm_jiang_guide_cytokine_context_gate_20260625.json"
OUT_CSV = REPORTS / "latentfm_jiang_guide_cytokine_context_rows_20260625.csv"
OUT_MD = REPORTS / "LATENTFM_JIANG_GUIDE_CYTOKINE_CONTEXT_GATE_20260625.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return text


def is_control_condition(text: str) -> bool:
    return text.lower() in {
        "control",
        "ctrl",
        "ntc",
        "non-targeting",
        "non_targeting",
    }


def cytokine_from_dataset(dataset: str) -> str:
    if dataset.startswith("Jiang_"):
        return dataset.split("Jiang_", 1)[1]
    return "unknown"


def collect_jiang_context() -> pd.DataFrame:
    payload = read_json(OBS_SCHEMA_JSON)
    rows = []
    for file_row in payload["rows"]:
        dataset = str(file_row.get("dataset", ""))
        if not dataset.startswith("Jiang_") or file_row.get("status") != "ok":
            continue
        path = Path(str(file_row["path"]))
        if not path.is_file():
            continue
        obj = ad.read_h5ad(path, backed="r")
        try:
            obs = obj.obs.copy()
        finally:
            if getattr(obj, "file", None) is not None:
                obj.file.close()
        required = {"perturbation", "guide", "mixscale_score"}
        if not required.issubset(set(obs.columns)):
            continue
        obs["_condition"] = obs["perturbation"].map(norm)
        obs = obs.loc[~obs["_condition"].map(is_control_condition)].copy()
        obs["_guide"] = obs["guide"].map(norm)
        obs["_mixscale"] = pd.to_numeric(obs["mixscale_score"], errors="coerce")
        if "cytokine_treatment" in obs.columns:
            obs["_cytokine"] = obs["cytokine_treatment"].map(norm)
        else:
            obs["_cytokine"] = cytokine_from_dataset(dataset)
        for condition, frame in obs.groupby("_condition", observed=True, dropna=False):
            condition_s = norm(condition)
            if not condition_s:
                continue
            guide_counts = frame["_guide"].value_counts(dropna=True)
            top_guide_frac = float(guide_counts.iloc[0] / len(frame)) if len(guide_counts) else float("nan")
            rows.append(
                {
                    "dataset": dataset,
                    "cytokine": cytokine_from_dataset(dataset),
                    "obs_cytokine": norm(frame["_cytokine"].mode().iloc[0]) if len(frame["_cytokine"].dropna()) else "",
                    "condition": condition_s,
                    "n_cells": int(len(frame)),
                    "n_guides": int(frame["_guide"].replace("", np.nan).nunique(dropna=True)),
                    "top_guide_fraction": top_guide_frac,
                    "mean_mixscale_score": float(frame["_mixscale"].mean()),
                    "std_mixscale_score": float(frame["_mixscale"].std()),
                }
            )
    return pd.DataFrame(rows)


def load_exposure_rows() -> pd.DataFrame:
    rows = pd.read_csv(EXPOSURE_ROWS)
    rows = rows[(rows["comparison"] == "cap120_minus_cap30") & (rows["group"] == "cross")].copy()
    rows["cross_pp_diff"] = pd.to_numeric(rows["cross_pp_diff"], errors="coerce")
    rows["cross_mmd_diff"] = pd.to_numeric(rows["cross_mmd_diff"], errors="coerce")
    return rows


def bootstrap_high_low(high: np.ndarray, low: np.ndarray, *, n_boot: int = 4000) -> dict[str, float]:
    rng = np.random.default_rng(44)
    high = high[np.isfinite(high)]
    low = low[np.isfinite(low)]
    if len(high) == 0 or len(low) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    obs = float(high.mean() - low.mean())
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = rng.choice(high, len(high), replace=True).mean() - rng.choice(low, len(low), replace=True).mean()
    return {
        "mean": obs,
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
    }


def group_diff(df: pd.DataFrame, label_col: str) -> float:
    high = df.loc[df[label_col], "cross_pp_diff"].astype(float)
    low = df.loc[~df[label_col], "cross_pp_diff"].astype(float)
    if len(high) == 0 or len(low) == 0:
        return float("nan")
    return float(high.mean() - low.mean())


def permute_within_dataset(df: pd.DataFrame, label_col: str, *, n_perm: int = 4000) -> float:
    rng = np.random.default_rng(45)
    actual = group_diff(df, label_col)
    if not np.isfinite(actual):
        return float("nan")
    ge = 0
    for _ in range(n_perm):
        pieces = []
        for _, block in df.groupby("dataset"):
            labels = block[label_col].to_numpy().copy()
            rng.shuffle(labels)
            b = block.copy()
            b[label_col] = labels
            pieces.append(b)
        if group_diff(pd.concat(pieces, ignore_index=True), label_col) >= actual:
            ge += 1
    return float((ge + 1) / (n_perm + 1))


def summarize(joined: pd.DataFrame) -> dict[str, Any]:
    joined = joined.copy()
    joined["high_mixscale"] = joined["mean_mixscale_score"] >= joined["mean_mixscale_score"].median()
    joined["multi_guide"] = joined["n_guides"] > 1
    high = joined.loc[joined["high_mixscale"], "cross_pp_diff"].to_numpy(dtype=float)
    low = joined.loc[~joined["high_mixscale"], "cross_pp_diff"].to_numpy(dtype=float)
    boot = bootstrap_high_low(high, low)
    p_perm = permute_within_dataset(joined, "high_mixscale")
    cytokine_means = (
        joined.groupby("cytokine", observed=True)["cross_pp_diff"]
        .agg(["count", "mean", "min"])
        .reset_index()
        .to_dict("records")
    )
    summary = {
        "n_joined_rows": int(len(joined)),
        "n_datasets": int(joined["dataset"].nunique()),
        "n_cytokines": int(joined["cytokine"].nunique()),
        "mixscale_spearman_pp": float(joined["mean_mixscale_score"].corr(joined["cross_pp_diff"], method="spearman")),
        "high_mixscale_mean_pp": float(joined.loc[joined["high_mixscale"], "cross_pp_diff"].mean()),
        "low_mixscale_mean_pp": float(joined.loc[~joined["high_mixscale"], "cross_pp_diff"].mean()),
        "high_minus_low_pp": boot["mean"],
        "high_minus_low_pp_ci_low": boot["ci_low"],
        "high_minus_low_pp_ci_high": boot["ci_high"],
        "dataset_shuffle_p_ge_actual": p_perm,
        "high_tail_frac_lt_minus_0p02": float((joined.loc[joined["high_mixscale"], "cross_pp_diff"] < -0.02).mean()),
        "low_tail_frac_lt_minus_0p02": float((joined.loc[~joined["high_mixscale"], "cross_pp_diff"] < -0.02).mean()),
        "dataset_min": float(joined.groupby("dataset")["cross_pp_diff"].mean().min()),
        "multi_guide_mean_pp": float(joined.loc[joined["multi_guide"], "cross_pp_diff"].mean()),
        "single_guide_mean_pp": float(joined.loc[~joined["multi_guide"], "cross_pp_diff"].mean()),
        "cytokine_means": cytokine_means,
    }
    reasons = []
    if not (summary["n_joined_rows"] >= 40 and summary["n_datasets"] >= 4):
        reasons.append("overlap_too_small_for_general_claim")
    if not (summary["high_minus_low_pp_ci_low"] > 0):
        reasons.append("mixscale_bootstrap_ci_low_not_positive")
    if not (summary["dataset_shuffle_p_ge_actual"] < 0.05):
        reasons.append("dataset_shuffle_control_not_separated")
    if not (summary["dataset_min"] >= -0.02):
        reasons.append("dataset_tail_below_minus_0p02")
    if not (summary["high_tail_frac_lt_minus_0p02"] <= summary["low_tail_frac_lt_minus_0p02"]):
        reasons.append("high_mixscale_tail_fraction_not_lower")
    summary["gate_pass"] = not reasons
    summary["status"] = "jiang_guide_cytokine_context_gate_pass_cpu_only" if not reasons else "jiang_guide_cytokine_context_gate_fail_no_gpu"
    summary["reasons"] = reasons
    return summary


def render_md(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# LatentFM Jiang Guide/Cytokine Context Gate",
        "",
        "Timestamp: `2026-06-25`",
        "",
        f"Status: `{s['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only Jiang obs-metadata stratification over completed train-only row metrics.",
        "- Reads raw h5ad `.obs` columns: `perturbation`, `guide`, `cytokine_treatment`, `mixscale_score`.",
        "- Does not read expression matrices, checkpoints, canonical multi, Track C query, train, infer, or use GPU.",
        "",
        "## Primary Result",
        "",
        "| rows | datasets | cytokines | high mixscale pp | low mixscale pp | high-low CI95 | shuffle p | tail frac high/low | gate |",
        "|---:|---:|---:|---:|---:|---|---:|---:|---|",
        (
            f"| {s['n_joined_rows']} | {s['n_datasets']} | {s['n_cytokines']} | "
            f"{s['high_mixscale_mean_pp']:+.6f} | {s['low_mixscale_mean_pp']:+.6f} | "
            f"{s['high_minus_low_pp']:+.6f} [{s['high_minus_low_pp_ci_low']:+.6f}, "
            f"{s['high_minus_low_pp_ci_high']:+.6f}] | {s['dataset_shuffle_p_ge_actual']:.4f} | "
            f"{s['high_tail_frac_lt_minus_0p02']:.3f}/{s['low_tail_frac_lt_minus_0p02']:.3f} | "
            f"`{s['status']}` |"
        ),
        "",
        "## Cytokine Means",
        "",
        "| cytokine | n | mean pp | min pp |",
        "|---|---:|---:|---:|",
    ]
    for row in s["cytokine_means"]:
        lines.append(f"| `{row['cytokine']}` | {int(row['count'])} | {row['mean']:+.6f} | {row['min']:+.6f} |")
    lines += [
        "",
        "## Decision",
        "",
    ]
    if s["gate_pass"]:
        lines.append(
            "- Jiang guide/cytokine context passes as a narrow CPU mechanism signal; it still needs external review before any GPU smoke."
        )
    else:
        lines.append(
            "- Jiang guide/cytokine context is useful for supplemental failure analysis only; it does not authorize a GPU training branch."
        )
    lines += [
        f"- reasons: `{s['reasons']}`",
        f"- mixscale Spearman vs pp: `{s['mixscale_spearman_pp']:+.6f}`",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- rows: `{OUT_CSV}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    context = collect_jiang_context()
    exposure = load_exposure_rows()
    joined = exposure.merge(context, on=["dataset", "condition"], how="inner", validate="many_to_one")
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
    joined.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "gpu_authorized": False, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
