#!/usr/bin/env python3
"""Build train-only observable-information condition loss weights.

The output is a bounded exploratory artifact: a real continuous weight file and
a same-marginal stratified random control.  It does not train or evaluate.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
PARENT_SPLIT = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
OBS_CSV = ROOT / "reports/exact_analog_observability_matched_feasibility_20260629/condition_observability_rows.csv"
SUPPORT_CSV = ROOT / "reports/condition_neighborhood_support_gate_20260629/condition_neighborhood_support_rows.csv"
COVERAGE_CSV = ROOT / "reports/exact_response_information_combined_coverage_20260628/exact_response_information_condition_rows.csv"
NULL_PANEL_JSON = ROOT / "reports/condition_neighborhood_response_resid_null_variance_panel_20260629/latentfm_condition_neighborhood_response_resid_null_variance_panel_20260629.json"
OUT_DIR = ROOT / "reports/observable_information_condition_loss_weights_20260629"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rank01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    return values.rank(method="average", pct=True).fillna(0.5).astype(float)


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if std <= 1e-12 or not np.isfinite(std):
        return pd.Series(0.0, index=series.index, dtype=float)
    return (values - mean) / std


def residualize(y: pd.Series, frame: pd.DataFrame, controls: list[str]) -> pd.Series:
    mats = [np.ones((len(frame), 1), dtype=float)]
    for col in controls:
        if col in {"log1p_n_gt", "response_norm"}:
            mats.append(zscore(frame[col]).to_numpy()[:, None])
        else:
            dummies = pd.get_dummies(frame[col].astype(str), prefix=col, drop_first=True, dtype=float)
            if dummies.shape[1] > 0:
                mats.append(dummies.to_numpy(dtype=float))
    x = np.concatenate(mats, axis=1)
    yy = pd.to_numeric(y, errors="coerce").fillna(float(pd.to_numeric(y, errors="coerce").median())).to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(x, yy, rcond=None)
    resid = yy - x @ coef
    return pd.Series(resid, index=frame.index, dtype=float)


def smd(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce").astype(float)
    bb = pd.to_numeric(b, errors="coerce").astype(float)
    pooled = float(np.sqrt((aa.var(ddof=1) + bb.var(ddof=1)) / 2.0))
    if pooled <= 1e-12 or not np.isfinite(pooled):
        return 0.0
    return float((aa.mean() - bb.mean()) / pooled)


def corr(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce").astype(float)
    bb = pd.to_numeric(b, errors="coerce").astype(float)
    mask = aa.notna() & bb.notna()
    if mask.sum() <= 2:
        return 0.0
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


def split_train_rows(split: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for dataset, groups in split.items():
        for condition in groups.get("train", []) or []:
            rows.append({"dataset": str(dataset), "condition": str(condition)})
    return pd.DataFrame(rows)


def randomize_within_strata(df: pd.DataFrame, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    out = pd.Series(index=df.index, dtype=float)
    for _, idx in df.groupby(["dataset", "perturbation_type", "gene_count_bin"], dropna=False).groups.items():
        idx = list(idx)
        values = df.loc[idx, "weight"].to_numpy(dtype=float).copy()
        rng.shuffle(values)
        out.loc[idx] = values
    return out.astype(float)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-split", type=Path, default=PARENT_SPLIT)
    parser.add_argument("--obs-csv", type=Path, default=OBS_CSV)
    parser.add_argument("--support-csv", type=Path, default=SUPPORT_CSV)
    parser.add_argument("--coverage-csv", type=Path, default=COVERAGE_CSV)
    parser.add_argument("--null-panel-json", type=Path, default=NULL_PANEL_JSON)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--amplitude", type=float, default=0.25)
    parser.add_argument("--random-seed", type=int, default=43)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split = load_json(args.parent_split)
    train = split_train_rows(split)
    obs = pd.read_csv(args.obs_csv)
    support = pd.read_csv(args.support_csv)
    coverage = pd.read_csv(args.coverage_csv)
    null_panel = load_json(args.null_panel_json) if args.null_panel_json.is_file() else {}

    cov_cols = [
        "dataset",
        "condition",
        "response_energy",
        "hvg_k80",
        "abundance_k80",
    ]
    cov_small = coverage[cov_cols].drop_duplicates(["dataset", "condition"], keep="first")
    support_cols = [
        "dataset",
        "condition",
        "perturbation_type_raw",
        "n_ctrl",
        "n_gt",
        "response_norm",
        "neighbor_support_score",
    ]
    df = (
        train.merge(obs, on=["dataset", "condition"], how="left")
        .merge(support[support_cols], on=["dataset", "condition"], how="left")
        .merge(cov_small, on=["dataset", "condition"], how="left")
    )
    if df[["perturbation_type", "gene_count_bin", "exact_train_covered"]].isna().any().any():
        raise SystemExit("observability rows do not cover all train conditions")
    df["perturbation_type"] = df["perturbation_type"].astype(str)
    df["gene_count_bin"] = df["gene_count_bin"].astype(str)
    df["perturbation_type_raw"] = df["perturbation_type_raw"].fillna(df["perturbation_type"])
    df["n_gt"] = pd.to_numeric(df["n_gt"], errors="coerce").fillna(0.0)
    df["n_ctrl"] = pd.to_numeric(df["n_ctrl"], errors="coerce").fillna(0.0)
    df["response_norm"] = pd.to_numeric(df["response_norm"], errors="coerce").fillna(df["response_norm"].median())
    df["log1p_n_gt"] = np.log1p(df["n_gt"].clip(lower=0))

    hvg_adv = pd.to_numeric(df["hvg_k80"], errors="coerce") - pd.to_numeric(df["abundance_k80"], errors="coerce")
    components = pd.DataFrame(
        {
            "exact_component": df["exact_train_covered"].astype(float),
            "analog_component": rank01(np.log1p(pd.to_numeric(df["analog_support_dataset_count"], errors="coerce").fillna(0.0))),
            "support_component": rank01(df["neighbor_support_score"]),
            "response_component": rank01(np.log1p(pd.to_numeric(df["response_energy"], errors="coerce"))),
            "hvg_adv_component": rank01(hvg_adv),
        },
        index=df.index,
    )
    # Exact/analog/local support carry the main observable-information idea.
    # Response and HVG-advantage are downweighted because they are more confound-prone.
    composite = (
        0.30 * components["exact_component"]
        + 0.25 * components["analog_component"]
        + 0.25 * components["support_component"]
        + 0.10 * components["response_component"]
        + 0.10 * components["hvg_adv_component"]
    )
    resid = residualize(
        composite,
        df,
        controls=["dataset", "perturbation_type", "gene_count_bin", "log1p_n_gt", "response_norm"],
    )
    percentile = rank01(resid)
    weights = 1.0 + float(args.amplitude) * (2.0 * percentile - 1.0)
    weights = weights.clip(lower=1.0 - float(args.amplitude), upper=1.0 + float(args.amplitude))
    weights = weights / float(weights.mean())
    df["observable_information_score_raw"] = composite
    df["observable_information_score_resid"] = resid
    df["observable_information_percentile"] = percentile
    df["weight"] = weights
    df["random_weight"] = randomize_within_strata(df, args.random_seed)
    df["random_weight"] = df["random_weight"] / float(df["random_weight"].mean())

    high = df[df["weight"] >= df["weight"].quantile(0.75)]
    low = df[df["weight"] <= df["weight"].quantile(0.25)]
    dataset_mean_abs_dev = float((df.groupby("dataset")["weight"].mean() - 1.0).abs().max())
    ptype_mean_abs_dev = float((df.groupby("perturbation_type")["weight"].mean() - 1.0).abs().max())
    diagnostics = {
        "n_train_conditions": int(len(df)),
        "datasets": int(df["dataset"].nunique()),
        "weight_min": float(df["weight"].min()),
        "weight_median": float(df["weight"].median()),
        "weight_mean": float(df["weight"].mean()),
        "weight_max": float(df["weight"].max()),
        "effective_sample_size_ratio": float((df["weight"].sum() ** 2) / (len(df) * (df["weight"] ** 2).sum())),
        "dataset_mean_abs_dev_max": dataset_mean_abs_dev,
        "ptype_mean_abs_dev_max": ptype_mean_abs_dev,
        "corr_weight_response_norm": corr(df["weight"], df["response_norm"]),
        "corr_weight_log1p_n_gt": corr(df["weight"], df["log1p_n_gt"]),
        "corr_weight_neighbor_support": corr(df["weight"], df["neighbor_support_score"]),
        "high_low_response_norm_smd": smd(high["response_norm"], low["response_norm"]),
        "high_low_log1p_n_gt_smd": smd(high["log1p_n_gt"], low["log1p_n_gt"]),
        "high_low_neighbor_support_smd": smd(high["neighbor_support_score"], low["neighbor_support_score"]),
        "random_weight_same_marginal_sorted_equal": bool(
            np.allclose(np.sort(df["weight"].to_numpy()), np.sort(df["random_weight"].to_numpy()))
        ),
        "null_panel_status": (null_panel.get("decision") or {}).get("status"),
        "null_future_axis_required_cross_gap": (null_panel.get("decision") or {}).get("future_axis_required_cross_gap"),
        "null_future_axis_required_family_gap": (null_panel.get("decision") or {}).get("future_axis_required_family_gap"),
    }
    reasons = []
    if diagnostics["n_train_conditions"] < 1000:
        reasons.append("too_few_train_conditions")
    if diagnostics["effective_sample_size_ratio"] < 0.94:
        reasons.append("weight_ess_ratio_below_0p94")
    if diagnostics["dataset_mean_abs_dev_max"] > 0.08:
        reasons.append("dataset_weight_mean_imbalance_gt_0p08")
    if diagnostics["ptype_mean_abs_dev_max"] > 0.08:
        reasons.append("ptype_weight_mean_imbalance_gt_0p08")
    if abs(diagnostics["corr_weight_response_norm"]) > 0.12:
        reasons.append("weight_response_norm_corr_gt_0p12")
    if abs(diagnostics["corr_weight_log1p_n_gt"]) > 0.12:
        reasons.append("weight_cellcount_corr_gt_0p12")
    if not diagnostics["random_weight_same_marginal_sorted_equal"]:
        reasons.append("random_control_not_same_marginal")

    status = (
        "observable_information_condition_loss_weights_gate_pass_bounded_smoke_ready"
        if not reasons
        else "observable_information_condition_loss_weights_gate_fail_no_gpu"
    )
    real_cols = [
        "dataset",
        "condition",
        "weight",
        "observable_information_percentile",
        "observable_information_score_raw",
        "observable_information_score_resid",
        "exact_train_covered",
        "analog_support_dataset_count",
        "neighbor_support_score",
        "response_norm",
        "n_ctrl",
        "n_gt",
        "perturbation_type",
        "gene_count_bin",
    ]
    real_csv = args.out_dir / "observable_information_condition_loss_weights.csv"
    rand_csv = args.out_dir / f"observable_information_condition_loss_weights_random_seed{args.random_seed}.csv"
    df[real_cols].to_csv(real_csv, index=False)
    rand = df[real_cols].copy()
    rand["weight"] = df["random_weight"]
    rand_csv = args.out_dir / f"observable_information_condition_loss_weights_random_seed{args.random_seed}.csv"
    rand.to_csv(rand_csv, index=False)

    dataset_csv = args.out_dir / "observable_information_condition_loss_weight_dataset_summary.csv"
    (
        df.groupby("dataset")
        .agg(
            n=("condition", "size"),
            mean_weight=("weight", "mean"),
            mean_random_weight=("random_weight", "mean"),
            exact_fraction=("exact_train_covered", "mean"),
            mean_response_norm=("response_norm", "mean"),
            mean_neighbor_support=("neighbor_support_score", "mean"),
        )
        .reset_index()
        .to_csv(dataset_csv, index=False)
    )
    payload = {
        "created_at": now_cst(),
        "status": status,
        "gpu_authorized": status.endswith("_ready"),
        "boundary": "train_only_weight_artifact_no_training_no_inference_no_canonical_multi_no_trackc_query",
        "hypothesis": "continuous observable-information condition loss weights reduce harmful low-information updates better than same-marginal random weights",
        "parent_split": str(args.parent_split),
        "inputs": {
            "obs_csv": str(args.obs_csv),
            "support_csv": str(args.support_csv),
            "coverage_csv": str(args.coverage_csv),
            "null_panel_json": str(args.null_panel_json),
        },
        "diagnostics": diagnostics,
        "reasons": reasons,
        "outputs": {
            "real_weights": str(real_csv),
            "random_weights": str(rand_csv),
            "dataset_summary": str(dataset_csv),
        },
        "promotion_gate_after_smoke": [
            "observable arm beats same-marginal random arm on internal cross and family pp",
            "observable arm family MMD delta <= +0.001 vs anchor",
            "no canonical no-harm unless internal/random-control gate passes",
            "future axis gaps must be compared to the seed43-46 null panel before any scaling-law claim",
        ],
        "fail_close": [
            "if CPU gate fails, do not launch GPU",
            "if random-weight arm matches or beats observable arm, close this weighting mechanism",
            "if pp improves only with MMD/canonical harm, keep as mechanism-only negative evidence",
        ],
    }
    out_json = args.out_dir / "latentfm_observable_information_condition_loss_weights_20260629.json"
    out_md = args.out_dir / "LATENTFM_OBSERVABLE_INFORMATION_CONDITION_LOSS_WEIGHTS_20260629.md"
    write_json(out_json, payload)

    lines = [
        "# LatentFM Observable-Information Condition Loss Weights",
        "",
        f"Created: `{payload['created_at']}`",
        "",
        f"Status: `{status}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU artifact/gate only; no training, inference, canonical multi selection, or Track C query use.",
        "- Weights are built from parent-train condition metadata and train-only exact/support/coverage rows.",
        "- The random control preserves the exact weight marginal by shuffling within dataset/ptype/gene-count strata.",
        "",
        "## Diagnostics",
        "",
    ]
    for key, value in diagnostics.items():
        lines.append(f"- {key}: `{value}`")
    if reasons:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {reason}" for reason in reasons)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- If launched, this is a bounded mechanism smoke only: observable weights vs same-marginal random weights.",
            "- No canonical no-harm or promotion is authorized unless the internal/random-control gate passes.",
            "",
            "## Outputs",
            "",
            f"- real weights: `{real_csv}`",
            f"- random weights: `{rand_csv}`",
            f"- dataset summary: `{dataset_csv}`",
            f"- JSON: `{out_json}`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": payload["gpu_authorized"], "report": str(out_md)}, indent=2))
    return 0 if payload["gpu_authorized"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
