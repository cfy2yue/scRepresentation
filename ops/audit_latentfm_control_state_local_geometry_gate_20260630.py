#!/usr/bin/env python3
"""CPU gate for control-state local geometry as a LatentFM routing signal.

This is CPU/report-only. It asks whether local control/GT manifold geometry in
train-only latent H5 files can identify conditions where the existing cap120
train-only update helps over the anchor. It does not train, infer, select
checkpoints, read canonical multi, or read Track C query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_key, "8")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path("/data/cyx/1030/scLatent")
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
RUN_DIR = ROOT / "runs/latentfm_xverse_scaling_count_smokes_20260624/xverse_scaling_cap120_all_3k_seed42/posthoc_eval_internal"
ANCHOR_SPLIT = RUN_DIR / "split_group_eval_anchor_internal_ode20.json"
CANDIDATE_SPLIT = RUN_DIR / "split_group_eval_candidate_internal_ode20.json"
OUT_DIR = ROOT / "reports/control_state_local_geometry_gate_20260630"
OUT_JSON = OUT_DIR / "control_state_local_geometry_gate_20260630.json"
OUT_MD = OUT_DIR / "LATENTFM_CONTROL_STATE_LOCAL_GEOMETRY_GATE_20260630.md"
OUT_FEATURES = OUT_DIR / "control_state_local_geometry_features_20260630.csv"
OUT_ROUTE_ROWS = OUT_DIR / "control_state_local_geometry_route_rows_20260630.csv"
OUT_POLICIES = OUT_DIR / "control_state_local_geometry_policy_summary_20260630.csv"
OUT_CONTROLS = OUT_DIR / "control_state_local_geometry_controls_20260630.csv"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
PRIMARY_FEATURES = (
    "support_score_rank",
    "coverage_q90_rank",
    "coverage_q95_rank",
    "neg_gt_to_ctrl_q50_ratio_rank",
    "neg_gt_to_ctrl_q90_ratio_rank",
    "neg_cov_mismatch_rank",
    "ctrl_eff_rank_frac_rank",
    "ctrl_density_rank",
)
COUNT_CONTROL_FEATURES = (
    "log_n_ctrl_rank",
    "log_n_gt_rank",
    "log_response_norm_rank",
)
QUANTILES = (0.25, 0.40, 0.50, 0.60, 0.75)


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def fmt(value: Any, digits: int = 6) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def stable_seed(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def decode(values: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def sample_block(emb: h5py.Dataset, start: int, end: int, cap: int, seed_text: str) -> np.ndarray:
    n = int(end - start)
    if n <= 0:
        return np.empty((0, int(emb.shape[1])), dtype=np.float32)
    if cap > 0 and n > cap:
        rng = np.random.default_rng(stable_seed(seed_text))
        rel = np.sort(rng.choice(n, size=cap, replace=False))
        return np.asarray(emb[start + rel], dtype=np.float32)
    return np.asarray(emb[start:end], dtype=np.float32)


def sqdist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)
    return np.maximum(
        np.sum(x64 * x64, axis=1, keepdims=True)
        + np.sum(y64 * y64, axis=1, keepdims=True).T
        - 2.0 * (x64 @ y64.T),
        0.0,
    )


def effective_rank(centered: np.ndarray) -> tuple[float, float, float]:
    if centered.shape[0] < 3:
        return (float("nan"), float("nan"), float("nan"))
    singular = np.linalg.svd(centered.astype(np.float64, copy=False), full_matrices=False, compute_uv=False)
    eig = np.maximum(singular * singular / max(1, centered.shape[0] - 1), 0.0)
    total = float(eig.sum())
    if total <= 1e-12:
        return (0.0, 0.0, 0.0)
    p = eig / total
    entropy_rank = float(math.exp(-float(np.sum(p * np.log(p + 1e-12)))))
    top_frac = float(eig[0] / total)
    return (entropy_rank, entropy_rank / max(1, centered.shape[0] - 1), top_frac)


def condition_geometry(dataset: str, condition: str, h5: h5py.File, index: dict[str, int], cap: int) -> dict[str, Any] | None:
    idx = index.get(condition)
    if idx is None:
        return None
    ctrl_key = "ctrl" if "ctrl/emb" in h5 else "ir"
    ctrl_offsets = np.asarray(h5[f"{ctrl_key}/offsets"])
    gt_offsets = np.asarray(h5["gt/offsets"])
    c0, c1 = int(ctrl_offsets[idx]), int(ctrl_offsets[idx + 1])
    g0, g1 = int(gt_offsets[idx]), int(gt_offsets[idx + 1])
    ctrl = sample_block(h5[f"{ctrl_key}/emb"], c0, c1, cap, f"ctrl|{dataset}|{condition}|{cap}")
    gt = sample_block(h5["gt/emb"], g0, g1, cap, f"gt|{dataset}|{condition}|{cap}")
    if ctrl.shape[0] < 3 or gt.shape[0] < 3:
        return None

    ctrl_mean = ctrl.mean(axis=0)
    gt_mean = gt.mean(axis=0)
    response_norm = float(np.linalg.norm(gt_mean - ctrl_mean))
    ctrl_centered = ctrl - ctrl_mean[None, :]
    gt_centered = gt - gt_mean[None, :]
    ctrl_radius = np.linalg.norm(ctrl_centered, axis=1)
    gt_to_ctrl = np.linalg.norm(gt - ctrl_mean[None, :], axis=1)
    d2 = sqdist(gt, ctrl)
    nn = np.sqrt(np.min(d2, axis=1))
    ctrl_q50 = float(np.quantile(ctrl_radius, 0.50))
    ctrl_q90 = float(np.quantile(ctrl_radius, 0.90))
    ctrl_q95 = float(np.quantile(ctrl_radius, 0.95))
    gt_q50 = float(np.quantile(gt_to_ctrl, 0.50))
    gt_q90 = float(np.quantile(gt_to_ctrl, 0.90))
    gt_q95 = float(np.quantile(gt_to_ctrl, 0.95))
    ctrl_var = float(np.mean(np.var(ctrl, axis=0)))
    gt_var = float(np.mean(np.var(gt, axis=0)))
    ctrl_eff, ctrl_eff_frac, ctrl_top_frac = effective_rank(ctrl_centered)
    gt_eff, gt_eff_frac, gt_top_frac = effective_rank(gt_centered)
    eps = 1e-8
    coverage_q90 = float(np.mean(gt_to_ctrl <= ctrl_q90))
    coverage_q95 = float(np.mean(gt_to_ctrl <= ctrl_q95))
    nn_ratio = float(np.mean(nn) / (ctrl_q50 + eps))
    gt_q50_ratio = float(gt_q50 / (ctrl_q50 + eps))
    gt_q90_ratio = float(gt_q90 / (ctrl_q90 + eps))
    cov_mismatch = float(abs(math.log((gt_var + eps) / (ctrl_var + eps))))
    support_score = (
        coverage_q95
        + coverage_q90
        - gt_q50_ratio
        - 0.5 * gt_q90_ratio
        - 0.5 * cov_mismatch
        - 0.25 * nn_ratio
    )
    return {
        "dataset": dataset,
        "condition": condition,
        "key": f"{dataset}||{condition}",
        "n_ctrl_actual": int(c1 - c0),
        "n_gt_actual": int(g1 - g0),
        "n_ctrl_sampled": int(ctrl.shape[0]),
        "n_gt_sampled": int(gt.shape[0]),
        "response_norm": response_norm,
        "log_response_norm": float(math.log1p(response_norm)),
        "log_n_ctrl": float(math.log1p(c1 - c0)),
        "log_n_gt": float(math.log1p(g1 - g0)),
        "ctrl_radius_q50": ctrl_q50,
        "ctrl_radius_q90": ctrl_q90,
        "ctrl_radius_q95": ctrl_q95,
        "gt_to_ctrl_q50": gt_q50,
        "gt_to_ctrl_q90": gt_q90,
        "gt_to_ctrl_q95": gt_q95,
        "gt_to_ctrl_q50_ratio": gt_q50_ratio,
        "gt_to_ctrl_q90_ratio": gt_q90_ratio,
        "gt_nn_ctrl_mean_ratio": nn_ratio,
        "coverage_q90": coverage_q90,
        "coverage_q95": coverage_q95,
        "ctrl_var_mean": ctrl_var,
        "gt_var_mean": gt_var,
        "cov_mismatch": cov_mismatch,
        "ctrl_eff_rank": ctrl_eff,
        "ctrl_eff_rank_frac": ctrl_eff_frac,
        "ctrl_top_eig_frac": ctrl_top_frac,
        "gt_eff_rank": gt_eff,
        "gt_eff_rank_frac": gt_eff_frac,
        "gt_top_eig_frac": gt_top_frac,
        "support_score": support_score,
        "ctrl_density": float(1.0 / (ctrl_q90 + eps)),
    }


def load_outcomes(anchor_path: Path, candidate_path: Path) -> pd.DataFrame:
    anchor = load_json(anchor_path)
    cand = load_json(candidate_path)
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        a_rows = {
            (str(r["dataset"]), str(r["condition"])): r
            for r in ((anchor.get("groups") or {}).get(group) or {}).get("condition_metrics", [])
        }
        c_rows = {
            (str(r["dataset"]), str(r["condition"])): r
            for r in ((cand.get("groups") or {}).get(group) or {}).get("condition_metrics", [])
        }
        for key in sorted(set(a_rows) & set(c_rows)):
            a = a_rows[key]
            c = c_rows[key]
            rows.append(
                {
                    "group": group,
                    "dataset": key[0],
                    "condition": key[1],
                    "key": f"{key[0]}||{key[1]}",
                    "anchor_pearson_pert": float(a.get("pearson_pert")),
                    "candidate_pearson_pert": float(c.get("pearson_pert")),
                    "pp_delta": float(c.get("pearson_pert")) - float(a.get("pearson_pert")),
                    "anchor_mmd": float(a.get("test_mmd")),
                    "candidate_mmd": float(c.get("test_mmd")),
                    "mmd_delta": float(c.get("test_mmd")) - float(a.get("test_mmd")),
                }
            )
    return pd.DataFrame(rows)


def materialize_features(outcomes: pd.DataFrame, data_dir: Path, cap: int) -> pd.DataFrame:
    needed = outcomes[["dataset", "condition", "key"]].drop_duplicates()
    feature_rows: list[dict[str, Any]] = []
    for dataset, part in needed.groupby("dataset", sort=True):
        h5_path = data_dir / f"{dataset}.h5"
        if not h5_path.is_file():
            continue
        with h5py.File(h5_path, "r") as h5:
            index = {condition: idx for idx, condition in enumerate(decode(np.asarray(h5["conditions"])))}
            for condition in sorted(part["condition"].astype(str).unique()):
                row = condition_geometry(str(dataset), str(condition), h5, index, cap)
                if row is not None:
                    feature_rows.append(row)
    features = pd.DataFrame(feature_rows)
    if features.empty:
        return features
    for col in [
        "support_score",
        "coverage_q90",
        "coverage_q95",
        "ctrl_eff_rank_frac",
        "ctrl_density",
        "log_n_ctrl",
        "log_n_gt",
        "log_response_norm",
    ]:
        features[f"{col}_rank"] = features.groupby("dataset")[col].rank(pct=True, method="average")
    for col in ["gt_to_ctrl_q50_ratio", "gt_to_ctrl_q90_ratio", "cov_mismatch"]:
        features[f"neg_{col}_rank"] = features.groupby("dataset")[col].rank(pct=True, method="average", ascending=False)
    return features


def policy_grid(features: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in features:
        for q in QUANTILES:
            rows.append({"feature": feature, "op": ">=", "threshold": float(q), "select_candidate_on_hit": True})
            rows.append({"feature": feature, "op": "<=", "threshold": float(q), "select_candidate_on_hit": True})
    return rows


def apply_policy(df: pd.DataFrame, policy: dict[str, Any]) -> pd.Series:
    val = pd.to_numeric(df[policy["feature"]], errors="coerce")
    if policy["op"] == ">=":
        hit = val >= float(policy["threshold"])
    else:
        hit = val <= float(policy["threshold"])
    return hit.fillna(False)


def summarize_route(df: pd.DataFrame, selected: pd.Series) -> dict[str, Any]:
    work = df.copy()
    work["selected"] = selected.to_numpy(dtype=bool)
    work["route_pp_delta"] = np.where(work["selected"], work["pp_delta"], 0.0)
    work["route_mmd_delta"] = np.where(work["selected"], work["mmd_delta"], 0.0)
    out: dict[str, Any] = {
        "n": int(len(work)),
        "selected_fraction": float(work["selected"].mean()) if len(work) else 0.0,
        "mean_pp_delta": float(work["route_pp_delta"].mean()) if len(work) else float("nan"),
        "mean_mmd_delta": float(work["route_mmd_delta"].mean()) if len(work) else float("nan"),
        "p_harm_condition": float((work["route_pp_delta"] < 0.0).mean()) if len(work) else float("nan"),
        "dataset_min_pp_delta": float(work.groupby("dataset")["route_pp_delta"].mean().min()) if len(work) else float("nan"),
        "selected_dataset_count": int(work.loc[work["selected"], "dataset"].nunique()),
        "max_selected_dataset_fraction": max_dataset_fraction(work.loc[work["selected"]]),
    }
    return out


def max_dataset_fraction(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    counts = df["dataset"].astype(str).value_counts()
    return float(counts.iloc[0] / counts.sum())


def score_summary(summary_by_group: dict[str, dict[str, Any]]) -> tuple[float, float, float, float]:
    cross = summary_by_group.get(GROUPS[0], {})
    family = summary_by_group.get(GROUPS[1], {})
    cross_pp = float(cross.get("mean_pp_delta", -999.0))
    family_pp = float(family.get("mean_pp_delta", -999.0))
    worst_min = min(float(cross.get("dataset_min_pp_delta", -999.0)), float(family.get("dataset_min_pp_delta", -999.0)))
    worst_mmd = max(float(cross.get("mean_mmd_delta", 999.0)), float(family.get("mean_mmd_delta", 999.0)))
    return (min(cross_pp, family_pp), cross_pp + family_pp, worst_min, -worst_mmd)


def evaluate_policy(df: pd.DataFrame, policy: dict[str, Any]) -> dict[str, Any]:
    selected = apply_policy(df, policy)
    by_group: dict[str, dict[str, Any]] = {}
    for group, part in df.groupby("group", sort=True):
        by_group[str(group)] = summarize_route(part, selected.loc[part.index])
    return {"policy": policy, "groups": by_group, "score": score_summary(by_group)}


def select_lodo(df: pd.DataFrame, policies: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    route_rows: list[pd.DataFrame] = []
    policy_rows: list[dict[str, Any]] = []
    for heldout in sorted(df["dataset"].astype(str).unique()):
        train = df[df["dataset"].astype(str) != heldout].copy()
        test = df[df["dataset"].astype(str) == heldout].copy()
        ranked = sorted((evaluate_policy(train, p) for p in policies), key=lambda x: x["score"], reverse=True)
        best = ranked[0]
        selected = apply_policy(test, best["policy"])
        test = test.copy()
        test["heldout_dataset"] = heldout
        test["selected"] = selected.to_numpy(dtype=bool)
        test["route_pp_delta"] = np.where(test["selected"], test["pp_delta"], 0.0)
        test["route_mmd_delta"] = np.where(test["selected"], test["mmd_delta"], 0.0)
        route_rows.append(test)
        policy_rows.append(
            {
                "heldout_dataset": heldout,
                **best["policy"],
                "train_score_min_pp": best["score"][0],
                "train_score_sum_pp": best["score"][1],
                "train_score_dataset_min": best["score"][2],
                "train_score_neg_mmd": best["score"][3],
                "test_selected_fraction": float(test["selected"].mean()) if len(test) else 0.0,
                "test_mean_pp_delta": float(test["route_pp_delta"].mean()) if len(test) else float("nan"),
                "test_mean_mmd_delta": float(test["route_mmd_delta"].mean()) if len(test) else float("nan"),
            }
        )
    return pd.concat(route_rows, ignore_index=True), pd.DataFrame(policy_rows)


def summarize_lodo(route: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {str(group): summarize_route(part, part["selected"].astype(bool)) for group, part in route.groupby("group", sort=True)}


def shuffle_within_dataset(df: pd.DataFrame, features: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    for feature in features:
        values = out[feature].to_numpy(copy=True)
        for _, idx in out.groupby("dataset").groups.items():
            arr_idx = np.asarray(list(idx), dtype=int)
            values[arr_idx] = rng.permutation(values[arr_idx])
        out[feature] = values
    return out


def control_panel(df: pd.DataFrame, primary_policies: list[dict[str, Any]], count_policies: list[dict[str, Any]], n_shuffle: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary_route, _ = select_lodo(df, primary_policies)
    primary_summary = summarize_lodo(primary_route)
    for group, summary in primary_summary.items():
        rows.append({"control": "primary", "seed": seed, "group": group, **summary})

    count_route, _ = select_lodo(df, count_policies)
    count_summary = summarize_lodo(count_route)
    for group, summary in count_summary.items():
        rows.append({"control": "count_only", "seed": seed, "group": group, **summary})

    inverted = [
        {**p, "op": "<=" if p["op"] == ">=" else ">="}
        for p in primary_policies
    ]
    inv_route, _ = select_lodo(df, inverted)
    inv_summary = summarize_lodo(inv_route)
    for group, summary in inv_summary.items():
        rows.append({"control": "inverted", "seed": seed, "group": group, **summary})

    for i in range(n_shuffle):
        shuf = shuffle_within_dataset(df, list(PRIMARY_FEATURES), seed + 1000 + i)
        shuf_route, _ = select_lodo(shuf, primary_policies)
        shuf_summary = summarize_lodo(shuf_route)
        for group, summary in shuf_summary.items():
            rows.append({"control": "within_dataset_shuffle", "seed": seed + 1000 + i, "group": group, **summary})
    return pd.DataFrame(rows)


def gate_decision(primary_summary: dict[str, dict[str, Any]], controls: pd.DataFrame) -> tuple[str, list[str]]:
    reasons: list[str] = []
    for group in GROUPS:
        row = primary_summary.get(group, {})
        if float(row.get("mean_pp_delta", -999.0)) < 0.010:
            reasons.append(f"{group}_mean_pp_delta_lt_0p010")
        if float(row.get("dataset_min_pp_delta", -999.0)) < -0.020:
            reasons.append(f"{group}_dataset_min_lt_neg0p020")
        if float(row.get("mean_mmd_delta", 999.0)) > 0.001:
            reasons.append(f"{group}_mean_mmd_delta_gt_0p001")
        if float(row.get("selected_dataset_count", 0)) < 6:
            reasons.append(f"{group}_selected_dataset_count_lt_6")
        if float(row.get("max_selected_dataset_fraction", 1.0)) > 0.30:
            reasons.append(f"{group}_max_selected_dataset_fraction_gt_0p30")
    for group in GROUPS:
        primary = float(primary_summary.get(group, {}).get("mean_pp_delta", -999.0))
        for control in ("count_only", "inverted"):
            vals = controls[(controls["control"] == control) & (controls["group"] == group)]["mean_pp_delta"]
            if not vals.empty and primary - float(vals.max()) < 0.005:
                reasons.append(f"{group}_{control}_not_0p005_below_primary")
        vals = controls[(controls["control"] == "within_dataset_shuffle") & (controls["group"] == group)]["mean_pp_delta"]
        if not vals.empty:
            p95 = float(np.quantile(vals, 0.95))
            if primary - p95 < 0.005:
                reasons.append(f"{group}_shuffle_p95_not_0p005_below_primary")
    status = "control_state_local_geometry_gate_pass_prepare_bounded_smoke" if not reasons else "control_state_local_geometry_gate_fail_no_gpu"
    return status, reasons


def render_report(payload: dict[str, Any], primary_summary: dict[str, dict[str, Any]], policy_rows: pd.DataFrame, controls: pd.DataFrame) -> str:
    lines = [
        "# LatentFM Control-State Local Geometry Gate 20260630",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized next: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only gate.",
        "- Uses train-only/internal cap120 candidate-vs-anchor condition metrics plus latent H5 control/GT geometry.",
        "- Does not train, infer, select checkpoints, read canonical multi, or read Track C query.",
        "",
        "## Primary LODO Route",
        "",
        "| group | n | selected frac | mean pp delta | mean MMD delta | condition harm | dataset min | selected datasets | max selected dataset frac |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        row = primary_summary.get(group, {})
        lines.append(
            f"| `{group}` | {row.get('n', 0)} | {fmt(row.get('selected_fraction'))} | "
            f"{fmt(row.get('mean_pp_delta'))} | {fmt(row.get('mean_mmd_delta'))} | "
            f"{fmt(row.get('p_harm_condition'))} | {fmt(row.get('dataset_min_pp_delta'))} | "
            f"{row.get('selected_dataset_count', 0)} | {fmt(row.get('max_selected_dataset_fraction'))} |"
        )
    lines.extend(["", "## Controls", "", "| control | group | mean pp delta max/mean | p95 |", "|---|---|---:|---:|"])
    for control in sorted(controls["control"].astype(str).unique()):
        for group in GROUPS:
            vals = controls[(controls["control"] == control) & (controls["group"] == group)]["mean_pp_delta"]
            if vals.empty:
                continue
            lines.append(
                f"| `{control}` | `{group}` | {fmt(vals.max())}/{fmt(vals.mean())} | {fmt(np.quantile(vals, 0.95))} |"
            )
    lines.extend(["", "## Decision", ""])
    if payload["reasons"]:
        lines.append("Close this CPU route; no GPU smoke is authorized.")
        lines.extend(f"- reason: `{reason}`" for reason in payload["reasons"])
    else:
        lines.append("This CPU gate passes and may be externally audited before any bounded GPU smoke.")
    lines.extend(
        [
            "",
            "## Top Heldout Policies",
            "",
            "| heldout dataset | feature | op | threshold | test selected frac | test pp delta |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for _, row in policy_rows.head(20).iterrows():
        lines.append(
            f"| `{row['heldout_dataset']}` | `{row['feature']}` | `{row['op']}` | "
            f"{fmt(row['threshold'])} | {fmt(row['test_selected_fraction'])} | {fmt(row['test_mean_pp_delta'])} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- features: `{OUT_FEATURES}`",
            f"- route rows: `{OUT_ROUTE_ROWS}`",
            f"- policies: `{OUT_POLICIES}`",
            f"- controls: `{OUT_CONTROLS}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    global OUT_DIR, OUT_JSON, OUT_MD, OUT_FEATURES, OUT_ROUTE_ROWS, OUT_POLICIES, OUT_CONTROLS

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--anchor-split", type=Path, default=ANCHOR_SPLIT)
    parser.add_argument("--candidate-split", type=Path, default=CANDIDATE_SPLIT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--cap", type=int, default=128)
    parser.add_argument("--n-shuffle", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260630)
    args = parser.parse_args()

    OUT_DIR = args.out_dir
    OUT_JSON = OUT_DIR / "control_state_local_geometry_gate_20260630.json"
    OUT_MD = OUT_DIR / "LATENTFM_CONTROL_STATE_LOCAL_GEOMETRY_GATE_20260630.md"
    OUT_FEATURES = OUT_DIR / "control_state_local_geometry_features_20260630.csv"
    OUT_ROUTE_ROWS = OUT_DIR / "control_state_local_geometry_route_rows_20260630.csv"
    OUT_POLICIES = OUT_DIR / "control_state_local_geometry_policy_summary_20260630.csv"
    OUT_CONTROLS = OUT_DIR / "control_state_local_geometry_controls_20260630.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    outcomes = load_outcomes(args.anchor_split, args.candidate_split)
    features = materialize_features(outcomes, args.data_dir, args.cap)
    joined = outcomes.merge(features, on=["dataset", "condition", "key"], how="inner")
    primary_policies = policy_grid(list(PRIMARY_FEATURES))
    count_policies = policy_grid(list(COUNT_CONTROL_FEATURES))
    route, policy_rows = select_lodo(joined, primary_policies)
    primary_summary = summarize_lodo(route)
    controls = control_panel(joined, primary_policies, count_policies, args.n_shuffle, args.seed)
    status, reasons = gate_decision(primary_summary, controls)

    features.to_csv(OUT_FEATURES, index=False)
    route.to_csv(OUT_ROUTE_ROWS, index=False)
    policy_rows.to_csv(OUT_POLICIES, index=False)
    controls.to_csv(OUT_CONTROLS, index=False)

    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "boundary": {
            "cpu_report_only": True,
            "train_internal_only": True,
            "canonical_multi_read": False,
            "trackc_query_read": False,
            "training_or_inference": False,
        },
        "inputs": {
            "data_dir": str(args.data_dir),
            "anchor_split": str(args.anchor_split),
            "candidate_split": str(args.candidate_split),
            "file_hashes": {
                str(args.anchor_split): sha256(args.anchor_split),
                str(args.candidate_split): sha256(args.candidate_split),
            },
        },
        "feature_cap_cells": int(args.cap),
        "n_outcome_rows": int(len(outcomes)),
        "n_feature_rows": int(len(features)),
        "n_joined_rows": int(len(joined)),
        "primary_summary": primary_summary,
        "reasons": reasons,
        "controls": {
            "n_shuffle": int(args.n_shuffle),
            "summary_csv": str(OUT_CONTROLS),
        },
        "outputs": {
            "features": str(OUT_FEATURES),
            "route_rows": str(OUT_ROUTE_ROWS),
            "policies": str(OUT_POLICIES),
            "controls": str(OUT_CONTROLS),
            "report": str(OUT_MD),
        },
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_report(payload, primary_summary, policy_rows, controls), encoding="utf-8")
    print(json.dumps({"status": status, "reasons": reasons, "out": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
