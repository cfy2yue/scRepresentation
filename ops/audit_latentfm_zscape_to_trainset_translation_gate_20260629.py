#!/usr/bin/env python3
"""ZSCAPE-to-LatentFM train-set translation gate.

CPU/report-only. This asks whether the ZSCAPE state-preserved time-vector
insight has a train-safe analogue in the current LatentFM split/outcome
matrix. It is deliberately a translation gate, not a model launcher.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("/data/cyx/1030/scLatent")
BASE_JOIN = ROOT / "reports/multiaxis_information_scaling_incremental_gate_20260629/multiaxis_information_scaling_join_rows.csv"
CLUSTER_JOIN = ROOT / "reports/trainset_cluster_density_information_gate_20260629/trainset_cluster_density_join_rows.csv"
OT_JOIN = ROOT / "reports/trainset_ot_coverage_information_gate_20260629/trainset_ot_coverage_join_rows.csv"
ZSCAPE_ROWS = ROOT / "reports/zscape_state_preserved_time_vector_gate_20260629/zscape_state_preserved_time_vector_rows.csv"
META_INVENTORY = ROOT / "reports/state_context_support_scaling_gate_20260628/state_context_metadata_inventory.csv"
OUT_DIR = ROOT / "reports/zscape_to_latentfm_trainset_translation_gate_20260629"
OUT_MD = OUT_DIR / "LATENTFM_ZSCAPE_TO_LATENTFM_TRAINSET_TRANSLATION_GATE_20260629.md"
OUT_JSON = OUT_DIR / "latentfm_zscape_to_latentfm_trainset_translation_gate_20260629.json"
OUT_JOIN = OUT_DIR / "zscape_to_latentfm_trainset_translation_join_rows.csv"
OUT_ASSOC = OUT_DIR / "zscape_to_latentfm_trainset_translation_association_rows.csv"
OUT_FEATURES = OUT_DIR / "zscape_to_latentfm_translation_feature_readiness.csv"

OUTCOMES = ["cross_pp_delta", "family_pp_delta", "family_mmd_delta", "tail_score"]
CONTROL_SETS = {
    "base": ["n_train_conditions", "base_dataset_effective_count", "base_background_effective_count"],
    "base_exact": [
        "n_train_conditions",
        "base_dataset_effective_count",
        "base_background_effective_count",
        "exact_condition_fraction",
    ],
    "base_exact_type_target": [
        "n_train_conditions",
        "base_dataset_effective_count",
        "base_background_effective_count",
        "base_perturbation_type_effective_count",
        "base_target_gene_effective_count",
        "exact_condition_fraction",
    ],
}
PREDICTORS = {
    "state_signal_fraction": {
        "family": "state_preservation_proxy",
        "primary_control_set": "base_exact_type_target",
        "zscape_link": "state identity is preserved while response moves within lineage/substate",
    },
    "mean_state_entropy": {
        "family": "state_preservation_proxy",
        "primary_control_set": "base_exact_type_target",
        "zscape_link": "richer within-condition state structure could support within-state displacement learning",
    },
    "state_dataset_effective_count": {
        "family": "state_preservation_proxy",
        "primary_control_set": "base_exact_type_target",
        "zscape_link": "state signal replicated across datasets instead of one source",
    },
    "residual_cluster_effective_count": {
        "family": "support_density_proxy",
        "primary_control_set": "base_exact_type_target",
        "zscape_link": "more occupied residual-response clusters should approximate more biological support states",
    },
    "cluster_entropy_norm": {
        "family": "support_density_proxy",
        "primary_control_set": "base_exact_type_target",
        "zscape_link": "balanced residual cluster occupancy may approximate nonredundant information",
    },
    "low_density_tail_fraction": {
        "family": "support_density_proxy",
        "primary_control_set": "base_exact_type_target",
        "zscape_link": "rare or low-density response states may identify unsupported extrapolation regimes",
    },
    "density_per_condition": {
        "family": "support_density_proxy",
        "primary_control_set": "base_exact_type_target",
        "zscape_link": "local response density per condition is a train-set support proxy",
    },
    "residual_sliced_wasserstein_to_parent": {
        "family": "transport_geometry_proxy",
        "primary_control_set": "base_exact_type_target",
        "zscape_link": "aggregate train residual transport distance to parent split",
    },
    "residual_sliced_wasserstein_delta_vs_random": {
        "family": "transport_geometry_proxy",
        "primary_control_set": "base_exact_type_target",
        "zscape_link": "distance beyond same-n random parent draws",
    },
    "residual_ot_coverage_score": {
        "family": "transport_geometry_proxy",
        "primary_control_set": "base_exact_type_target",
        "zscape_link": "higher is closer to parent residual distribution under same-n random baseline",
    },
    "residual_effective_rank": {
        "family": "latent_vector_capacity_proxy",
        "primary_control_set": "base_exact_type_target",
        "zscape_link": "more latent response directions may approximate richer vector fields",
    },
    "residual_vendi_rbf_effective_count": {
        "family": "latent_vector_capacity_proxy",
        "primary_control_set": "base_exact_type_target",
        "zscape_link": "diverse residual response manifold proxy",
    },
}


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes", "pass"])


def zscore(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    mean = float(vals.mean())
    std = float(vals.std(ddof=0))
    if not math.isfinite(std) or std <= 1e-12:
        return pd.Series(np.zeros(len(vals)), index=vals.index, dtype=float)
    return (vals - mean) / std


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    columns = [np.ones(len(y))]
    for idx in range(x.shape[1]):
        col = x[:, idx].astype(float)
        fill = float(np.nanmean(col)) if np.isfinite(col).any() else 0.0
        col = np.where(np.isfinite(col), col, fill)
        std = float(np.std(col))
        if std > 1e-12:
            columns.append((col - float(np.mean(col))) / std)
    design = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def residual_spearman(df: pd.DataFrame, predictor: str, outcome: str, controls: list[str]) -> dict[str, Any]:
    cols = [predictor, outcome, *controls]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        return {"n": 0, "rho": float("nan"), "p_value": float("nan"), "missing": ";".join(missing)}
    part = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(part) < max(8, len(controls) + 5) or part[predictor].nunique() < 3 or part[outcome].nunique() < 3:
        return {"n": int(len(part)), "rho": float("nan"), "p_value": float("nan"), "missing": ""}
    x_resid = residualize(part[predictor].to_numpy(dtype=float), part[controls].to_numpy(dtype=float))
    y_resid = residualize(part[outcome].to_numpy(dtype=float), part[controls].to_numpy(dtype=float))
    if float(np.std(x_resid)) <= 1e-12 or float(np.std(y_resid)) <= 1e-12:
        return {"n": int(len(part)), "rho": float("nan"), "p_value": float("nan"), "missing": ""}
    rho, p_value = spearmanr(x_resid, y_resid)
    return {"n": int(len(part)), "rho": float(rho), "p_value": float(p_value), "missing": ""}


def bootstrap_ci(
    df: pd.DataFrame,
    predictor: str,
    outcome: str,
    controls: list[str],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    cols = [predictor, outcome, *controls]
    if any(col not in df.columns for col in cols):
        return float("nan"), float("nan")
    part = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(part) < max(8, len(controls) + 5):
        return float("nan"), float("nan")
    values: list[float] = []
    for _ in range(n_boot):
        sample = part.iloc[rng.integers(0, len(part), len(part))].copy()
        if sample[predictor].nunique() < 3 or sample[outcome].nunique() < 3:
            continue
        res = residual_spearman(sample, predictor, outcome, controls)
        rho = float(res["rho"])
        if math.isfinite(rho):
            values.append(rho)
    if len(values) < 20:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def lodo_same_sign(df: pd.DataFrame, predictor: str, outcome: str, controls: list[str], leave_col: str, full_rho: float) -> float:
    if leave_col not in df.columns or not math.isfinite(full_rho) or abs(full_rho) <= 1e-12:
        return float("nan")
    full_sign = math.copysign(1.0, full_rho)
    signs: list[bool] = []
    for leave_value in sorted(df[leave_col].astype(str).unique()):
        sub = df[df[leave_col].astype(str) != leave_value]
        res = residual_spearman(sub, predictor, outcome, controls)
        rho = float(res["rho"])
        if math.isfinite(rho) and abs(rho) > 1e-12:
            signs.append(math.copysign(1.0, rho) == full_sign)
    if not signs:
        return float("nan")
    return float(np.mean(signs))


def stratified_permutation_p(
    df: pd.DataFrame,
    predictor: str,
    outcome: str,
    controls: list[str],
    full_rho: float,
    *,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    if "source_family" not in df.columns or not math.isfinite(full_rho):
        return float("nan")
    cols = [predictor, outcome, "source_family", *controls]
    if any(col not in df.columns for col in cols):
        return float("nan")
    part = df[cols].copy()
    part[[predictor, outcome, *controls]] = part[[predictor, outcome, *controls]].apply(pd.to_numeric, errors="coerce")
    part = part.replace([np.inf, -np.inf], np.nan).dropna()
    if len(part) < max(8, len(controls) + 5):
        return float("nan")
    extreme = 0
    total = 0
    for _ in range(n_perm):
        perm = part.copy()
        shuffled_parts = []
        for _, group in perm.groupby("source_family", sort=False):
            group = group.copy()
            if len(group) > 1:
                group[predictor] = rng.permutation(group[predictor].to_numpy())
            shuffled_parts.append(group)
        shuffled = pd.concat(shuffled_parts, axis=0)
        res = residual_spearman(shuffled, predictor, outcome, controls)
        rho = float(res["rho"])
        if math.isfinite(rho):
            total += 1
            extreme += int(abs(rho) >= abs(full_rho))
    if total == 0:
        return float("nan")
    return float((extreme + 1) / (total + 1))


def load_join() -> pd.DataFrame:
    base = read_csv(BASE_JOIN)
    cluster = read_csv(CLUSTER_JOIN)
    ot = read_csv(OT_JOIN)
    cluster_cols = [
        "split_name",
        "residual_cluster_effective_count",
        "cluster_entropy_norm",
        "parent_cluster_coverage_fraction",
        "density_per_condition",
        "low_density_tail_fraction",
        "random_low_density_tail_fraction_mean",
    ]
    ot_cols = [
        "split_name",
        "residual_sliced_wasserstein_to_parent",
        "residual_sliced_wasserstein_delta_vs_random",
        "residual_sliced_wasserstein_ratio_vs_random",
        "residual_ot_coverage_score",
    ]
    joined = base.copy()
    for col in ["n_train_conditions", "n_train_conditions_y", "n_train_conditions_x"]:
        if col in joined.columns:
            joined["n_train_conditions"] = pd.to_numeric(joined[col], errors="coerce")
            break
    if not cluster.empty:
        joined = joined.merge(cluster[cluster_cols].drop_duplicates("split_name"), on="split_name", how="left", validate="many_to_one")
    if not ot.empty:
        joined = joined.merge(ot[ot_cols].drop_duplicates("split_name"), on="split_name", how="left", validate="many_to_one")
    joined["zscape_proxy_state_support_score"] = (
        zscore(joined.get("state_signal_fraction", pd.Series(index=joined.index, dtype=float)))
        + zscore(joined.get("mean_state_entropy", pd.Series(index=joined.index, dtype=float)))
        + zscore(joined.get("cluster_entropy_norm", pd.Series(index=joined.index, dtype=float)))
        - zscore(joined.get("low_density_tail_fraction", pd.Series(index=joined.index, dtype=float)))
    )
    joined["zscape_proxy_transport_coverage_score"] = (
        zscore(joined.get("residual_ot_coverage_score", pd.Series(index=joined.index, dtype=float)))
        - zscore(joined.get("residual_sliced_wasserstein_delta_vs_random", pd.Series(index=joined.index, dtype=float)))
    )
    joined["zscape_proxy_combined_state_transport_score"] = (
        zscore(joined["zscape_proxy_state_support_score"]) + zscore(joined["zscape_proxy_transport_coverage_score"])
    )
    PREDICTORS["zscape_proxy_state_support_score"] = {
        "family": "zscape_inspired_composite",
        "primary_control_set": "base_exact_type_target",
        "zscape_link": "state support plus balanced residual cluster occupancy",
    }
    PREDICTORS["zscape_proxy_transport_coverage_score"] = {
        "family": "zscape_inspired_composite",
        "primary_control_set": "base_exact_type_target",
        "zscape_link": "transport coverage relative to same-n random parent draws",
    }
    PREDICTORS["zscape_proxy_combined_state_transport_score"] = {
        "family": "zscape_inspired_composite",
        "primary_control_set": "base_exact_type_target",
        "zscape_link": "combined state/support/transport proxy for dynamic-response information",
    }
    return joined


def build_associations(joined: pd.DataFrame, n_boot: int, n_perm: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260629)
    rows: list[dict[str, Any]] = []
    for predictor, meta in PREDICTORS.items():
        for outcome in OUTCOMES:
            for control_name, controls in CONTROL_SETS.items():
                res = residual_spearman(joined, predictor, outcome, controls)
                rho = float(res["rho"])
                ci_low, ci_high = bootstrap_ci(joined, predictor, outcome, controls, n_boot=n_boot, rng=rng)
                source_lodo = lodo_same_sign(joined, predictor, outcome, controls, "source_family", rho)
                axis_lodo = lodo_same_sign(joined, predictor, outcome, controls, "axis_family", rho)
                perm_p = stratified_permutation_p(joined, predictor, outcome, controls, rho, n_perm=n_perm, rng=rng)
                ci_excludes = (
                    math.isfinite(ci_low)
                    and math.isfinite(ci_high)
                    and ((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))
                )
                primary = control_name == meta["primary_control_set"]
                passes = (
                    primary
                    and int(res["n"]) >= 12
                    and math.isfinite(rho)
                    and abs(rho) >= 0.50
                    and float(res["p_value"]) <= 0.05
                    and ci_excludes
                    and math.isfinite(source_lodo)
                    and source_lodo >= 0.80
                    and math.isfinite(axis_lodo)
                    and axis_lodo >= 0.80
                    and (not math.isfinite(perm_p) or perm_p <= 0.10)
                )
                rows.append(
                    {
                        "predictor": predictor,
                        "predictor_family": meta["family"],
                        "zscape_link": meta["zscape_link"],
                        "outcome": outcome,
                        "control_set": control_name,
                        "controls": ";".join(controls),
                        "n": int(res["n"]),
                        "residual_spearman_rho": rho,
                        "p_value": float(res["p_value"]),
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "ci_excludes_zero": bool(ci_excludes),
                        "source_lodo_same_sign": source_lodo,
                        "axis_lodo_same_sign": axis_lodo,
                        "source_stratified_permutation_p": perm_p,
                        "primary_for_predictor": bool(primary),
                        "translation_gate_pass": bool(passes),
                        "missing": res.get("missing", ""),
                    }
                )
    return pd.DataFrame(rows)


def time_metadata_summary(meta: pd.DataFrame) -> tuple[bool, list[str]]:
    if meta.empty or "all_obs_keys" not in meta.columns:
        return False, []
    tokens = ("time", "hour", "day", "stage", "age", "dpi", "hpf")
    hits: list[str] = []
    for _, row in meta.iterrows():
        keys = str(row.get("all_obs_keys", ""))
        matched = [key for key in keys.split(";") if any(tok in key.lower() for tok in tokens)]
        if matched:
            hits.append(f"{row.get('dataset')}:{','.join(matched)}")
    return bool(hits), hits


def feature_readiness(joined: pd.DataFrame, zscape: pd.DataFrame, meta: pd.DataFrame, assoc: pd.DataFrame) -> pd.DataFrame:
    has_time, time_hits = time_metadata_summary(meta)
    zscape_vector_positive = int(bool_series(zscape.get("vector_dynamic_gate", pd.Series(dtype=object))).sum()) if not zscape.empty else 0
    zscape_model_ready = int(bool_series(zscape.get("model_constraint_ready", pd.Series(dtype=object))).sum()) if not zscape.empty else 0
    primary = assoc[assoc["primary_for_predictor"].astype(bool)].copy()
    rows = [
        {
            "translation_feature": "external_zscape_state_time_vector_biology",
            "materialized": True,
            "evidence": f"vector_positive_rows={zscape_vector_positive}; model_ready_rows={zscape_model_ready}",
            "blocker": "module specificity blocked; external zebrafish dynamics cannot be used directly as loss",
            "gpu_authorized": False,
        },
        {
            "translation_feature": "latentfm_explicit_time_axis",
            "materialized": has_time,
            "evidence": "; ".join(time_hits[:8]) if time_hits else "no obvious time/hour/day/stage/age obs keys in current metadata inventory",
            "blocker": "no direct train-set time-vector analogue in current LatentFM split matrix" if not has_time else "requires dataset-specific time QC before use",
            "gpu_authorized": False,
        },
        {
            "translation_feature": "state_preservation_proxy",
            "materialized": "state_signal_fraction" in joined.columns,
            "evidence": _family_best(primary, "state_preservation_proxy"),
            "blocker": "must pass exact/count/source/background/type/target controls and LODO",
            "gpu_authorized": _family_pass(primary, "state_preservation_proxy"),
        },
        {
            "translation_feature": "support_density_proxy",
            "materialized": "cluster_entropy_norm" in joined.columns,
            "evidence": _family_best(primary, "support_density_proxy"),
            "blocker": "cluster-density signals were unstable against bootstrap/random controls",
            "gpu_authorized": _family_pass(primary, "support_density_proxy"),
        },
        {
            "translation_feature": "transport_geometry_proxy",
            "materialized": "residual_sliced_wasserstein_to_parent" in joined.columns,
            "evidence": _family_best(primary, "transport_geometry_proxy"),
            "blocker": "OT coverage did not survive source/LODO/random controls",
            "gpu_authorized": _family_pass(primary, "transport_geometry_proxy"),
        },
        {
            "translation_feature": "zscape_inspired_composite",
            "materialized": "zscape_proxy_combined_state_transport_score" in joined.columns,
            "evidence": _family_best(primary, "zscape_inspired_composite"),
            "blocker": "composite must beat exact/count/source/background/type/target controls without becoming a posthoc mixture",
            "gpu_authorized": _family_pass(primary, "zscape_inspired_composite"),
        },
    ]
    return pd.DataFrame(rows)


def _family_best(primary: pd.DataFrame, family: str) -> str:
    sub = primary[primary["predictor_family"] == family].copy()
    if sub.empty:
        return "no primary rows"
    sub["abs_rho"] = pd.to_numeric(sub["residual_spearman_rho"], errors="coerce").abs()
    row = sub.sort_values("abs_rho", ascending=False).iloc[0]
    return (
        f"best {row['predictor']}->{row['outcome']} "
        f"rho={fmt(row['residual_spearman_rho'])}, p={fmt(row['p_value'])}, "
        f"CI=[{fmt(row['bootstrap_ci_low'])},{fmt(row['bootstrap_ci_high'])}], "
        f"source_lodo={fmt(row['source_lodo_same_sign'])}, axis_lodo={fmt(row['axis_lodo_same_sign'])}, "
        f"perm_p={fmt(row['source_stratified_permutation_p'])}"
    )


def _family_pass(primary: pd.DataFrame, family: str) -> bool:
    sub = primary[primary["predictor_family"] == family]
    return bool((sub["translation_gate_pass"].astype(bool)).any()) if not sub.empty else False


def decide(assoc: pd.DataFrame, features: pd.DataFrame) -> tuple[str, list[str]]:
    passing = assoc[assoc["translation_gate_pass"].astype(bool)].copy()
    direct_time_ready = bool(features.loc[features["translation_feature"].eq("latentfm_explicit_time_axis"), "materialized"].any())
    zscape_direct_blocked = not bool(features.loc[features["translation_feature"].eq("external_zscape_state_time_vector_biology"), "gpu_authorized"].any())
    if passing.empty and zscape_direct_blocked:
        status = "zscape_to_latentfm_translation_gate_no_passing_axis_no_gpu"
    elif not direct_time_ready:
        status = "zscape_to_latentfm_translation_gate_proxy_only_no_gpu"
    else:
        status = "zscape_to_latentfm_translation_gate_review_required"
    reasons = []
    if passing.empty:
        reasons.append("no train-set proxy passes strict incremental association controls")
    if not direct_time_ready:
        reasons.append("no explicit LatentFM train-set time-vector analogue is available")
    if zscape_direct_blocked:
        reasons.append("external ZSCAPE module/pathway specificity is blocked")
    return status, reasons


def write_report(joined: pd.DataFrame, assoc: pd.DataFrame, features: pd.DataFrame, payload: dict[str, Any]) -> None:
    primary = assoc[assoc["primary_for_predictor"].astype(bool)].copy()
    primary["abs_rho"] = pd.to_numeric(primary["residual_spearman_rho"], errors="coerce").abs()
    top = primary.sort_values("abs_rho", ascending=False).head(12)
    lines = [
        "# ZSCAPE-To-LatentFM Train-Set Translation Gate",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only synthesis over frozen split-level LatentFM outcome tables, train-set state/context proxies, cluster-density proxies, OT-coverage proxies, and the ZSCAPE state-time-vector report.",
        "* No training, inference, new OT pairing, GPU, canonical multi, Track C query, or checkpoint selection.",
        "* This is a proxy translation gate. The current LatentFM train-set matrix does not contain a direct time-vector analogue comparable to ZSCAPE 24h->36h periderm.",
        "",
        "## Decision",
        "",
        f"* Reasons: `{'; '.join(payload['reasons'])}`.",
        "* Keep ZSCAPE as biology insight and diagnostic framing.",
        "* Do not launch a GPU sampling/curriculum/loss smoke from this translation packet.",
        "",
        "## Feature Readiness",
        "",
        "| feature | materialized | gpu | evidence | blocker |",
        "|---|---:|---:|---|---|",
    ]
    for _, row in features.iterrows():
        lines.append(
            f"| `{row['translation_feature']}` | `{bool(row['materialized'])}` | `{bool(row['gpu_authorized'])}` | "
            f"{row['evidence']} | {row['blocker']} |"
        )
    lines.extend(
        [
            "",
            "## Strongest Primary Associations",
            "",
            "| predictor | family | outcome | rho | p | CI | source LODO | axis LODO | perm p | pass |",
            "|---|---|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in top.iterrows():
        lines.append(
            f"| `{row['predictor']}` | `{row['predictor_family']}` | `{row['outcome']}` | "
            f"`{fmt(row['residual_spearman_rho'])}` | `{fmt(row['p_value'])}` | "
            f"`[{fmt(row['bootstrap_ci_low'])}, {fmt(row['bootstrap_ci_high'])}]` | "
            f"`{fmt(row['source_lodo_same_sign'])}` | `{fmt(row['axis_lodo_same_sign'])}` | "
            f"`{fmt(row['source_stratified_permutation_p'])}` | `{bool(row['translation_gate_pass'])}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "* ZSCAPE gives a useful biological prior: state-preserved perturbation responses can align with a normal developmental vector while preserving lineage/substate identity.",
            "* In current LatentFM train-set artifacts, that prior only maps to coarse proxies: state/context availability, residual cluster support, and aggregate transport coverage.",
            "* Those proxies do not pass strict incremental gates beyond exact coverage, counts, source/background, perturbation type, and target-gene controls.",
            "* The strongest proxy associations are therefore hypothesis-generating covariates, not launchable training axes.",
            "",
            "## Next Gate",
            "",
            "* If continuing the biology route, materialize a condition-level response-neighborhood table rather than another split-level proxy: per-condition local analog density, source/background-matched neighbor count, residual-vector alignment to matched training neighbors, and exact coverage.",
            "* Require a prospective high/low split with enough matched pairs and LODO/source stability before GPU.",
            "",
            "## Outputs",
            "",
            f"* Join rows: `{OUT_JOIN}`",
            f"* Association rows: `{OUT_ASSOC}`",
            f"* Feature readiness: `{OUT_FEATURES}`",
            f"* JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joined = load_join()
    zscape = read_csv(ZSCAPE_ROWS)
    meta = read_csv(META_INVENTORY)
    assoc = build_associations(joined, n_boot=300, n_perm=300)
    features = feature_readiness(joined, zscape, meta, assoc)
    status, reasons = decide(assoc, features)
    joined.to_csv(OUT_JOIN, index=False)
    assoc.to_csv(OUT_ASSOC, index=False)
    features.to_csv(OUT_FEATURES, index=False)
    payload = {
        "created_at": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "reasons": reasons,
        "n_split_rows": int(len(joined)),
        "n_translation_gate_pass_rows": int(assoc["translation_gate_pass"].astype(bool).sum()),
        "inputs": {
            "base_join": str(BASE_JOIN),
            "cluster_join": str(CLUSTER_JOIN),
            "ot_join": str(OT_JOIN),
            "zscape_rows": str(ZSCAPE_ROWS),
            "metadata_inventory": str(META_INVENTORY),
        },
        "outputs": {
            "report": str(OUT_MD),
            "json": str(OUT_JSON),
            "join_rows": str(OUT_JOIN),
            "association_rows": str(OUT_ASSOC),
            "feature_readiness": str(OUT_FEATURES),
        },
        "boundary": "cpu_report_only_no_training_no_inference_no_new_ot_no_gpu_no_canonical_multi_no_trackc_query",
    }
    write_json(OUT_JSON, payload)
    write_report(joined, assoc, features, payload)
    print(json.dumps({"status": status, "n_split_rows": len(joined), "n_pass_rows": payload["n_translation_gate_pass_rows"], "report": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
