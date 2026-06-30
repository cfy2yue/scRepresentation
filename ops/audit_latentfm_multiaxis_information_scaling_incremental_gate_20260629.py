#!/usr/bin/env python3
"""Incremental multiaxis information-scaling gate.

CPU/report-only. Tests whether HVG, state/context, and train-mean geometry axes
add split-level explanatory signal beyond exact response-information coverage
and source/background/count controls.
"""

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
STATE_JOIN = ROOT / "reports/state_context_split_association_gate_20260628/state_context_split_join_rows.csv"
EXACT_JOIN = (
    ROOT
    / "reports/exact_response_information_clustered_ci_combined_20260628/"
    / "exact_response_information_outcome_join_rows.csv"
)
HVG_JOIN = ROOT / "reports/hvg_response_scaling_design_matrix_20260628/hvg_response_scaling_design_matrix.csv"
ZSCAPE_PANEL = ROOT / "reports/zscape_response_law_panel_20260628/LATENTFM_ZSCAPE_RESPONSE_LAW_PANEL_20260628.md"
OUT_DIR = ROOT / "reports/multiaxis_information_scaling_incremental_gate_20260629"
OUT_MD = OUT_DIR / "LATENTFM_MULTIAXIS_INFORMATION_SCALING_INCREMENTAL_GATE_20260629.md"
OUT_JSON = OUT_DIR / "latentfm_multiaxis_information_scaling_incremental_gate_20260629.json"
OUT_JOIN = OUT_DIR / "multiaxis_information_scaling_join_rows.csv"
OUT_ASSOC = OUT_DIR / "multiaxis_information_scaling_incremental_association_rows.csv"
OUT_LODO = OUT_DIR / "multiaxis_information_scaling_incremental_lodo_rows.csv"

OUTCOMES = ["family_mmd_delta", "tail_score", "family_pp_delta", "cross_pp_delta"]
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
    "exact_condition_fraction": {
        "family": "exact_response_coverage",
        "primary_control_set": "base",
        "model_use": "scaling_descriptor",
    },
    "exact_hvg_minus_abundance_top1000_mean": {
        "family": "hvg_minus_abundance",
        "primary_control_set": "base_exact",
        "model_use": "gene_budget_axis",
    },
    "hvg_top1000_advantage_group_or_dataset_prior_mean": {
        "family": "hvg_group_prior_advantage",
        "primary_control_set": "base_exact",
        "model_use": "gene_budget_axis",
    },
    "state_signal_fraction": {
        "family": "state_context",
        "primary_control_set": "base_exact",
        "model_use": "sampling_covariate_candidate",
    },
    "mean_state_entropy": {
        "family": "state_context",
        "primary_control_set": "base_exact",
        "model_use": "sampling_covariate_candidate",
    },
    "state_dataset_effective_count": {
        "family": "state_context",
        "primary_control_set": "base_exact",
        "model_use": "sampling_covariate_candidate",
    },
    "residual_effective_rank": {
        "family": "latent_residual_geometry",
        "primary_control_set": "base_exact",
        "model_use": "geometry_covariate_candidate",
    },
    "residual_vendi_rbf_effective_count": {
        "family": "latent_residual_geometry",
        "primary_control_set": "base_exact",
        "model_use": "geometry_covariate_candidate",
    },
    "residual_pairwise_l2_mean": {
        "family": "latent_residual_geometry",
        "primary_control_set": "base_exact",
        "model_use": "geometry_covariate_candidate",
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


def split_source_family(split_file: Any) -> str:
    text = "" if split_file is None else str(split_file)
    path = Path(text)
    return path.parent.name or "root"


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    cols = []
    for j in range(x.shape[1]):
        col = x[:, j]
        mask = np.isfinite(col)
        if mask.any():
            fill = float(np.nanmean(col))
            col = np.where(np.isfinite(col), col, fill)
        if float(np.nanstd(col)) > 1e-12:
            cols.append((col - float(np.nanmean(col))) / float(np.nanstd(col)))
    if cols:
        design = np.column_stack([np.ones(len(y)), *cols])
    else:
        design = np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def residual_spearman(df: pd.DataFrame, predictor: str, outcome: str, controls: list[str]) -> dict[str, Any]:
    cols = [predictor, outcome, *controls]
    part = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(part) < max(8, len(controls) + 5) or part[predictor].nunique() < 3 or part[outcome].nunique() < 3:
        return {"n": int(len(part)), "rho": float("nan"), "p_value": float("nan")}
    x_resid = residualize(part[predictor].to_numpy(dtype=float), part[controls].to_numpy(dtype=float))
    y_resid = residualize(part[outcome].to_numpy(dtype=float), part[controls].to_numpy(dtype=float))
    if float(np.std(x_resid)) <= 1e-12 or float(np.std(y_resid)) <= 1e-12:
        return {"n": int(len(part)), "rho": float("nan"), "p_value": float("nan")}
    rho, p_value = spearmanr(x_resid, y_resid)
    return {"n": int(len(part)), "rho": float(rho), "p_value": float(p_value)}


def load_join() -> pd.DataFrame:
    state = pd.read_csv(STATE_JOIN)
    exact = pd.read_csv(EXACT_JOIN)
    hvg = pd.read_csv(HVG_JOIN)
    state_cols = [
        "split_name",
        "n_train_conditions",
        "state_signal_fraction",
        "mean_state_entropy",
        "state_dataset_effective_count",
        "exact_hvg_minus_abundance_top1000_mean",
    ]
    hvg_cols = [
        "split_name",
        "hvg_top1000_group_or_dataset_prior_mean",
        "hvg_top1000_random_group_or_dataset_prior_mean",
        "hvg_top1000_advantage_group_or_dataset_prior_mean",
        "hvg_top1000_oracle_group_or_dataset_prior_mean",
    ]
    state_one = state[state_cols].drop_duplicates(subset=["split_name"], keep="first")
    hvg_one = hvg[hvg_cols].drop_duplicates(subset=["split_name"], keep="first")
    joined = exact.copy()
    if "n_train_conditions" not in joined.columns:
        train_cols = [col for col in ["n_train_conditions_y", "n_train_conditions_x"] if col in joined.columns]
        if train_cols:
            joined["n_train_conditions"] = joined[train_cols[0]]
    joined = joined.merge(state_one, on="split_name", how="left", suffixes=("", "_state"), validate="many_to_one")
    if "n_train_conditions_state" in joined.columns:
        joined["n_train_conditions"] = joined["n_train_conditions"].fillna(joined["n_train_conditions_state"])
    joined = joined.merge(hvg_one, on="split_name", how="left", validate="many_to_one")
    joined["axis_family"] = joined["axis_family"].fillna("unclassified")
    return joined


def build_assoc(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assoc_rows: list[dict[str, Any]] = []
    lodo_rows: list[dict[str, Any]] = []
    for predictor, meta in PREDICTORS.items():
        for outcome in OUTCOMES:
            for control_name, controls in CONTROL_SETS.items():
                if predictor == "exact_condition_fraction" and "exact_condition_fraction" in controls:
                    continue
                if predictor not in joined.columns:
                    continue
                result = residual_spearman(joined, predictor, outcome, controls)
                row = {
                    "predictor": predictor,
                    "predictor_family": meta["family"],
                    "outcome": outcome,
                    "control_set": control_name,
                    "controls": ";".join(controls),
                    "n": result["n"],
                    "residual_spearman_rho": result["rho"],
                    "p_value": result["p_value"],
                    "abs_rho": abs(result["rho"]) if math.isfinite(result["rho"]) else float("nan"),
                    "primary_for_predictor": control_name == meta["primary_control_set"],
                    "model_use": meta["model_use"],
                }
                assoc_rows.append(row)
                if control_name == meta["primary_control_set"]:
                    full_sign = math.copysign(1.0, result["rho"]) if math.isfinite(result["rho"]) and result["rho"] != 0 else 0.0
                    for leave_col in ["source_family", "axis_family"]:
                        for leave_value in sorted(joined[leave_col].astype(str).unique()):
                            sub = joined[joined[leave_col].astype(str) != leave_value]
                            leave = residual_spearman(sub, predictor, outcome, controls)
                            leave_sign = (
                                math.copysign(1.0, leave["rho"])
                                if math.isfinite(leave["rho"]) and leave["rho"] != 0
                                else 0.0
                            )
                            lodo_rows.append(
                                {
                                    "predictor": predictor,
                                    "predictor_family": meta["family"],
                                    "outcome": outcome,
                                    "control_set": control_name,
                                    "leave_col": leave_col,
                                    "leave_value": leave_value,
                                    "n": leave["n"],
                                    "rho_full": result["rho"],
                                    "rho_leaveout": leave["rho"],
                                    "same_sign": bool(full_sign != 0.0 and leave_sign == full_sign),
                                }
                            )
    return pd.DataFrame(assoc_rows), pd.DataFrame(lodo_rows)


def decide(assoc: pd.DataFrame, lodo: pd.DataFrame) -> tuple[str, list[str], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    reasons: list[str] = []
    primary = assoc[assoc["primary_for_predictor"].astype(bool)].copy()
    for _, row in primary.iterrows():
        if row["predictor"] == "exact_condition_fraction":
            continue
        if not math.isfinite(float(row["residual_spearman_rho"])):
            continue
        sub = lodo[
            (lodo["predictor"] == row["predictor"])
            & (lodo["outcome"] == row["outcome"])
            & (lodo["control_set"] == row["control_set"])
        ]
        same_sign = float(sub["same_sign"].mean()) if not sub.empty else float("nan")
        pass_gate = (
            abs(float(row["residual_spearman_rho"])) >= 0.55
            and float(row["p_value"]) <= 0.05
            and math.isfinite(same_sign)
            and same_sign >= 0.75
        )
        if pass_gate:
            candidates.append(
                {
                    "predictor": row["predictor"],
                    "predictor_family": row["predictor_family"],
                    "outcome": row["outcome"],
                    "rho": float(row["residual_spearman_rho"]),
                    "p_value": float(row["p_value"]),
                    "lodo_same_sign_rate": same_sign,
                    "model_use": row["model_use"],
                }
            )
    if not candidates:
        reasons.append("no_nonexact_axis_survives_exact_source_background_controls")
    status = (
        "multiaxis_information_scaling_incremental_pass_no_gpu"
        if candidates
        else "multiaxis_information_scaling_incremental_no_incremental_axis_no_gpu"
    )
    return status, reasons, candidates


def write_report(joined: pd.DataFrame, assoc: pd.DataFrame, lodo: pd.DataFrame, payload: dict[str, Any]) -> None:
    primary = assoc[assoc["primary_for_predictor"].astype(bool)].copy()
    primary = primary.sort_values(["abs_rho"], ascending=False).head(18)
    lines = [
        "# LatentFM Multiaxis Information-Scaling Incremental Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only split-level association gate.",
        "- Tests whether HVG, state/context, and train-mean geometry axes add signal beyond exact coverage, condition count, and source/background controls.",
        "- Does not train, infer, use canonical multi, use Track C query, or select checkpoints.",
        "",
        "## Summary",
        "",
        f"- Outcome rows: `{len(joined)}`.",
        f"- Candidate incremental axes passing gate: `{len(payload['passing_incremental_axes'])}`.",
        "",
        "## Primary Associations",
        "",
        "| predictor | family | outcome | controls | n | rho | p | LODO same-sign |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in primary.iterrows():
        sub = lodo[
            (lodo["predictor"] == row["predictor"])
            & (lodo["outcome"] == row["outcome"])
            & (lodo["control_set"] == row["control_set"])
        ]
        same_sign = float(sub["same_sign"].mean()) if not sub.empty else float("nan")
        lines.append(
            f"| `{row['predictor']}` | `{row['predictor_family']}` | `{row['outcome']}` | `{row['control_set']}` | "
            f"`{int(row['n'])}` | `{fmt(row['residual_spearman_rho'])}` | `{fmt(row['p_value'])}` | `{fmt(same_sign)}` |"
        )
    lines.extend(["", "## Decision", ""])
    if payload["passing_incremental_axes"]:
        lines.append("- At least one non-exact axis survives the incremental gate, but this still authorizes CPU design only.")
    else:
        lines.append("- No non-exact axis currently survives exact-coverage/source/background controls with stable LODO.")
    lines.extend(
        [
            "- Exact response-information coverage remains the leading unadjusted/clustered descriptor, but its family-MMD signal is not independent of count/source/background controls in this 17-row residual gate.",
            "- This strengthens the need for exact/analog matched split feasibility before any scaling-law or model-training claim.",
            "- ZSCAPE state/OT evidence remains biologically useful, but should next be tested as independent variables or matched-split design inputs, not as a direct GPU loss.",
            "",
            "## Closed/Blocked Routes",
            "",
            "- Do not convert current ZSCAPE module/pathway rows into a LatentFM loss: module specificity failed.",
            "- Do not launch generic HVG/full-gene or state/context GPU smokes from this gate alone.",
            "- Do not use canonical multi or Track C query for any selection.",
            "",
            "## Outputs",
            "",
            f"- Join rows: `{OUT_JOIN}`",
            f"- Association rows: `{OUT_ASSOC}`",
            f"- LODO rows: `{OUT_LODO}`",
            f"- JSON: `{OUT_JSON}`",
            f"- ZSCAPE law context: `{ZSCAPE_PANEL}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joined = load_join()
    assoc, lodo = build_assoc(joined)
    status, reasons, candidates = decide(assoc, lodo)
    joined.to_csv(OUT_JOIN, index=False)
    assoc.to_csv(OUT_ASSOC, index=False)
    lodo.to_csv(OUT_LODO, index=False)
    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "reasons": reasons,
        "n_outcome_rows": int(len(joined)),
        "passing_incremental_axes": candidates,
        "inputs": {
            "state_join": str(STATE_JOIN),
            "exact_join": str(EXACT_JOIN),
            "hvg_join": str(HVG_JOIN),
            "zscape_panel": str(ZSCAPE_PANEL),
        },
        "outputs": {
            "join_rows": str(OUT_JOIN),
            "association_rows": str(OUT_ASSOC),
            "lodo_rows": str(OUT_LODO),
            "report": str(OUT_MD),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(joined, assoc, lodo, payload)
    print(json.dumps({"status": status, "reasons": reasons, "report": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
