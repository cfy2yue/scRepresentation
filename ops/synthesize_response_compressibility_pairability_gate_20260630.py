#!/usr/bin/env python3
"""Response-compressibility / pairability-proxy scaling gate.

This is a CPU-only synthesis over completed report artifacts. It asks whether
observable response concentration, mean-matched specificity, or reproducibility
can serve as a useful scaling x for Track A behavior. It does not train, infer,
select checkpoints, or read held-out Track C query data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "response_compressibility_pairability_gate_20260630"

INPUTS = {
    "budget_rows": REPORTS / "raw_expression_hvg_budget_expanded_gate_20260629" / "condition_budget_rows.csv",
    "meanmatched_rows": REPORTS / "hvg_meanmatched_expanded_controls_20260629" / "condition_negative_control_rows.csv",
    "observable_scaling_json": REPORTS
    / "observable_gene_budget_scaling_law_gate_20260630"
    / "latentfm_observable_gene_budget_scaling_law_gate_20260630.json",
    "all_test_single": REPORTS / "tracka_all_test_single_exact_20260627" / "all_test_single_rows.csv",
    "cross_background_seen_gene": REPORTS
    / "tracka_cross_background_seen_gene_exact_20260627"
    / "cross_background_seen_gene_rows.csv",
    "simple_single_unseen": REPORTS / "tracka_simple_single_unseen_exact_20260627" / "condition_rows.csv",
    "explicit_group_proxy": REPORTS / "tracka_explicit_group_proxy_benchmark_20260628" / "condition_rows.csv",
    "ot_pairing_gate": REPORTS / "latentfm_ot_pairing_gate_20260630" / "latentfm_ot_pairing_gate_20260630.json",
}

BUDGETS = [500, 1000, 2000]
PRIMARY_TASKS = {
    "all_test_single",
    "cross_background_seen_gene",
    "simple_cross_background_seen_gene_exact",
    "simple_test_single_gene_exact",
    "proxy_all_test_single_proxy",
}

FEATURE_DIRECTIONS = {
    "share_top500": "higher_easier",
    "share_top1000": "higher_easier",
    "share_top2000": "higher_easier",
    "oracle_share_top1000": "higher_easier",
    "hvg_minus_random_top1000": "higher_easier",
    "hvg_minus_meanmatched_top1000": "higher_easier",
    "hvg_minus_shuffled_top1000": "higher_easier",
    "split_half_jaccard_top1000": "higher_easier",
    "response_energy_over_shuffled_top1000": "higher_easier",
    "k80_hvg": "lower_easier",
    "k90_hvg": "lower_easier",
    "oracle_gap_top1000": "lower_easier",
    "response_energy": "unknown",
    "n_control": "nuisance",
    "n_pert": "nuisance",
    "n_vars": "nuisance",
    "random_share_top1000": "nuisance",
}


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def rank_corr(x: pd.Series, y: pd.Series) -> float | None:
    xy = pd.concat([x, y], axis=1).dropna()
    if len(xy) < 3:
        return None
    xr = xy.iloc[:, 0].rank(method="average")
    yr = xy.iloc[:, 1].rank(method="average")
    if float(xr.std(ddof=0)) == 0.0 or float(yr.std(ddof=0)) == 0.0:
        return None
    value = xr.corr(yr)
    return finite_float(value)


def within_dataset_rank_corr(df: pd.DataFrame, feature: str, metric: str) -> float | None:
    cols = ["dataset", feature, metric]
    sub = df[cols].dropna().copy()
    if len(sub) < 6 or sub["dataset"].nunique() < 2:
        return None
    sub[feature] = sub.groupby("dataset")[feature].transform(lambda s: s - s.mean())
    sub[metric] = sub.groupby("dataset")[metric].transform(lambda s: s - s.mean())
    return rank_corr(sub[feature], sub[metric])


def bootstrap_ci(df: pd.DataFrame, feature: str, metric: str, n_boot: int = 300) -> tuple[float | None, float | None]:
    sub = df[[feature, metric]].dropna()
    if len(sub) < 10:
        return None, None
    rng = np.random.default_rng(42)
    vals: list[float] = []
    arr = sub.to_numpy()
    n = len(arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        x = pd.Series(arr[idx, 0])
        y = pd.Series(arr[idx, 1])
        r = rank_corr(x, y)
        if r is not None:
            vals.append(r)
    if len(vals) < 20:
        return None, None
    lo, hi = np.quantile(np.array(vals), [0.025, 0.975])
    return finite_float(lo), finite_float(hi)


def expected_sign_ok(value: float | None, direction: str, metric: str, threshold: float) -> bool:
    if value is None:
        return False
    if direction == "higher_easier":
        return value >= threshold if metric == "pearson_pert_mean" else value <= -threshold
    if direction == "lower_easier":
        return value <= -threshold if metric == "pearson_pert_mean" else value >= threshold
    return False


def expected_direction_value(direction: str, metric: str) -> int:
    if direction == "higher_easier":
        return 1 if metric == "pearson_pert_mean" else -1
    if direction == "lower_easier":
        return -1 if metric == "pearson_pert_mean" else 1
    return 0


def sign_matches(value: float | None, expected: int) -> bool:
    if value is None or expected == 0:
        return False
    return value * expected > 0


def interpolate_budget(rows: pd.DataFrame, share_col: str, threshold: float) -> float | None:
    clean = rows[["effective_budget", share_col]].dropna().sort_values("effective_budget")
    if clean.empty:
        return None
    points = [(0.0, 0.0)]
    for _, row in clean.iterrows():
        x = finite_float(row["effective_budget"])
        y = finite_float(row[share_col])
        if x is not None and y is not None:
            points.append((x, y))
    best_by_x: dict[float, float] = {}
    for x, y in points:
        best_by_x[x] = max(best_by_x.get(x, -math.inf), y)
    points = sorted(best_by_x.items())
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        if y1 < threshold:
            continue
        if y1 == y0:
            return finite_float(x1)
        frac = (threshold - y0) / (y1 - y0)
        frac = min(1.0, max(0.0, frac))
        return finite_float(x0 + frac * (x1 - x0))
    if points and points[-1][1] >= threshold:
        return finite_float(points[-1][0])
    return None


def value_at_budget(rows: pd.DataFrame, budget: int, column: str) -> float | None:
    hit = rows[rows["budget"] == budget]
    if hit.empty:
        return None
    return finite_float(hit.iloc[0].get(column))


def build_condition_features() -> pd.DataFrame:
    budget = pd.read_csv(INPUTS["budget_rows"])
    meanmatched = pd.read_csv(INPUTS["meanmatched_rows"])

    features: list[dict[str, Any]] = []
    key_cols = ["group", "dataset", "condition"]
    for key, rows in budget.groupby(key_cols, sort=True):
        group, dataset, condition = key
        base = rows.iloc[0]
        item: dict[str, Any] = {
            "group": group,
            "dataset": dataset,
            "condition": condition,
            "n_vars": finite_float(base.get("n_vars")),
            "n_control": finite_float(base.get("n_control")),
            "n_pert": finite_float(base.get("n_pert")),
            "response_energy": finite_float(base.get("response_energy")),
            "matrix_source": base.get("matrix_source"),
            "log1p_policy": base.get("log1p_policy"),
            "k80_hvg": interpolate_budget(rows, "control_hvg_share", 0.80),
            "k90_hvg": interpolate_budget(rows, "control_hvg_share", 0.90),
        }
        for budget_value in BUDGETS:
            suffix = f"top{budget_value}"
            item[f"share_{suffix}"] = value_at_budget(rows, budget_value, "control_hvg_share")
            item[f"random_share_{suffix}"] = value_at_budget(rows, budget_value, "random_share_mean")
            item[f"oracle_share_{suffix}"] = value_at_budget(rows, budget_value, "oracle_response_share")
            item[f"hvg_minus_random_{suffix}"] = value_at_budget(rows, budget_value, "hvg_minus_random_mean")
        share = item.get("share_top1000")
        oracle = item.get("oracle_share_top1000")
        item["oracle_gap_top1000"] = (
            finite_float(oracle - share) if isinstance(oracle, float) and isinstance(share, float) else None
        )
        features.append(item)

    feat = pd.DataFrame(features)
    neg_keep = meanmatched[meanmatched["budget"].isin([500, 1000])].copy()
    neg_features: list[dict[str, Any]] = []
    for key, rows in neg_keep.groupby(key_cols, sort=True):
        item = {"group": key[0], "dataset": key[1], "condition": key[2]}
        for budget_value in [500, 1000]:
            hit = rows[rows["budget"] == budget_value]
            if hit.empty:
                continue
            row = hit.iloc[0]
            suffix = f"top{budget_value}"
            item[f"hvg_minus_meanmatched_{suffix}"] = finite_float(row.get("hvg_minus_mean_matched_mean"))
            item[f"hvg_minus_shuffled_{suffix}"] = finite_float(row.get("hvg_minus_shuffled_label_hvg_mean"))
            item[f"response_energy_over_shuffled_{suffix}"] = finite_float(
                row.get("response_energy_over_shuffled_mean")
            )
            item[f"split_half_jaccard_{suffix}"] = finite_float(row.get("split_half_jaccard"))
            item[f"split_half_overlap_fold_random_{suffix}"] = finite_float(
                row.get("split_half_overlap_fold_random")
            )
        neg_features.append(item)
    neg = pd.DataFrame(neg_features)
    return feat.merge(neg, on=key_cols, how="left")


def load_tracka_rows() -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []

    all_single = pd.read_csv(INPUTS["all_test_single"])
    all_single["task"] = "all_test_single"
    chunks.append(all_single)

    cross = pd.read_csv(INPUTS["cross_background_seen_gene"])
    cross["task"] = "cross_background_seen_gene"
    chunks.append(cross)

    simple = pd.read_csv(INPUTS["simple_single_unseen"])
    simple["task"] = "simple_" + simple["group"].astype(str)
    chunks.append(simple)

    proxy = pd.read_csv(INPUTS["explicit_group_proxy"])
    proxy["task"] = "proxy_" + proxy["explicit_group"].astype(str)
    chunks.append(proxy)

    raw = pd.concat(chunks, ignore_index=True, sort=False)
    raw = raw[["task", "seed", "dataset", "condition", "pearson_pert", "test_mmd_clamped"]].copy()
    raw = raw.dropna(subset=["dataset", "condition", "pearson_pert", "test_mmd_clamped"])
    out = (
        raw.groupby(["task", "dataset", "condition"], as_index=False)
        .agg(
            pearson_pert_mean=("pearson_pert", "mean"),
            test_mmd_clamped_mean=("test_mmd_clamped", "mean"),
            metric_rows=("pearson_pert", "size"),
            seed_n=("seed", "nunique"),
        )
        .sort_values(["task", "dataset", "condition"])
    )
    return out


def association_rows(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task, task_df in joined.groupby("task", sort=True):
        for feature, direction in FEATURE_DIRECTIONS.items():
            if feature not in task_df.columns:
                continue
            sub = task_df.dropna(subset=[feature, "pearson_pert_mean", "test_mmd_clamped_mean"])
            n = len(sub)
            dataset_n = sub["dataset"].nunique()
            if n < 10:
                continue
            pp = rank_corr(sub[feature], sub["pearson_pert_mean"])
            mmd = rank_corr(sub[feature], sub["test_mmd_clamped_mean"])
            pp_within = within_dataset_rank_corr(sub, feature, "pearson_pert_mean")
            mmd_within = within_dataset_rank_corr(sub, feature, "test_mmd_clamped_mean")
            pp_ci = bootstrap_ci(sub, feature, "pearson_pert_mean")
            mmd_ci = bootstrap_ci(sub, feature, "test_mmd_clamped_mean")

            pp_lodo: list[float] = []
            mmd_lodo: list[float] = []
            for dataset in sorted(sub["dataset"].dropna().unique()):
                rest = sub[sub["dataset"] != dataset]
                if len(rest) < 8 or rest["dataset"].nunique() < 1:
                    continue
                pp_val = rank_corr(rest[feature], rest["pearson_pert_mean"])
                mmd_val = rank_corr(rest[feature], rest["test_mmd_clamped_mean"])
                if pp_val is not None:
                    pp_lodo.append(pp_val)
                if mmd_val is not None:
                    mmd_lodo.append(mmd_val)
            pp_expected = expected_direction_value(direction, "pearson_pert_mean")
            mmd_expected = expected_direction_value(direction, "test_mmd_clamped_mean")
            pp_lodo_frac = (
                sum(sign_matches(v, pp_expected) for v in pp_lodo) / len(pp_lodo) if pp_lodo else None
            )
            mmd_lodo_frac = (
                sum(sign_matches(v, mmd_expected) for v in mmd_lodo) / len(mmd_lodo) if mmd_lodo else None
            )
            pp_ok = expected_sign_ok(pp, direction, "pearson_pert_mean", 0.15)
            mmd_ok = expected_sign_ok(mmd, direction, "test_mmd_clamped_mean", 0.10)
            within_ok = (
                sign_matches(pp_within, pp_expected)
                and sign_matches(mmd_within, mmd_expected)
                if direction in {"higher_easier", "lower_easier"} and pp_within is not None and mmd_within is not None
                else False
            )
            lodo_ok = (
                pp_lodo_frac is not None
                and mmd_lodo_frac is not None
                and pp_lodo_frac >= 0.66
                and mmd_lodo_frac >= 0.66
            )
            if direction in {"unknown", "nuisance"}:
                signal = "covariate_only"
            elif n >= 30 and dataset_n >= 3 and pp_ok and mmd_ok and lodo_ok and within_ok:
                signal = "robust_expected_direction"
            elif n >= 30 and dataset_n >= 3 and pp_ok and (mmd_ok or lodo_ok):
                signal = "mixed_expected_direction"
            else:
                signal = "weak_or_no_expected_direction"
            rows.append(
                {
                    "task": task,
                    "feature": feature,
                    "direction": direction,
                    "n_conditions": n,
                    "n_datasets": dataset_n,
                    "rho_pearson_pert": pp,
                    "rho_test_mmd": mmd,
                    "within_dataset_rho_pearson_pert": pp_within,
                    "within_dataset_rho_test_mmd": mmd_within,
                    "rho_pearson_pert_ci025": pp_ci[0],
                    "rho_pearson_pert_ci975": pp_ci[1],
                    "rho_test_mmd_ci025": mmd_ci[0],
                    "rho_test_mmd_ci975": mmd_ci[1],
                    "lodo_sign_frac_pearson_pert": pp_lodo_frac,
                    "lodo_sign_frac_test_mmd": mmd_lodo_frac,
                    "signal": signal,
                }
            )
    return pd.DataFrame(rows)


def top_rows(assoc: pd.DataFrame, signal: str | None = None, n: int = 12) -> pd.DataFrame:
    df = assoc.copy()
    if signal is not None:
        df = df[df["signal"] == signal]
    df = df[df["direction"].isin(["higher_easier", "lower_easier"])]
    if df.empty:
        return df
    df["_score"] = (
        df["rho_pearson_pert"].abs().fillna(0)
        + df["rho_test_mmd"].abs().fillna(0)
        + 0.5 * df["within_dataset_rho_pearson_pert"].abs().fillna(0)
        + 0.5 * df["within_dataset_rho_test_mmd"].abs().fillna(0)
    )
    return df.sort_values(["signal", "_score"], ascending=[True, False]).drop(columns=["_score"]).head(n)


def decide(assoc: pd.DataFrame, joined: pd.DataFrame) -> dict[str, Any]:
    primary = assoc[
        assoc["task"].isin(PRIMARY_TASKS)
        & assoc["direction"].isin(["higher_easier", "lower_easier"])
        & (assoc["n_conditions"] >= 30)
        & (assoc["n_datasets"] >= 3)
    ].copy()
    robust = primary[primary["signal"] == "robust_expected_direction"].copy()
    mixed = primary[primary["signal"] == "mixed_expected_direction"].copy()
    pairability_proxy = primary[
        primary["feature"].isin(
            [
                "split_half_jaccard_top1000",
                "response_energy_over_shuffled_top1000",
                "oracle_gap_top1000",
                "k90_hvg",
            ]
        )
        & primary["signal"].isin(["robust_expected_direction", "mixed_expected_direction"])
    ].copy()

    if not robust.empty:
        status = "response_compressibility_descriptor_pass_no_gpu"
        descriptor_pass = True
        reasons = [
            "at_least_one_primary_tracka_association_has_expected_pp_and_mmd_direction",
            "leave_one_dataset_and_within_dataset_checks_support_the_signal",
        ]
    elif not mixed.empty:
        status = "response_compressibility_mixed_signal_no_gpu"
        descriptor_pass = True
        reasons = [
            "primary_tracka_associations_show_expected_pp_or_mmd_direction_but_not_full_robustness",
            "use_as_scaling_descriptor_candidate_not_training_intervention_yet",
        ]
    else:
        status = "response_compressibility_no_stable_tracka_signal"
        descriptor_pass = False
        reasons = [
            "no_primary_tracka_feature_met_expected_direction_and_stability_gate",
            "condition_level_pairability_requires_new_ot_or_timecourse_measurement",
        ]

    observable_json = json.loads(INPUTS["observable_scaling_json"].read_text(encoding="utf-8"))
    ot_json = json.loads(INPUTS["ot_pairing_gate"].read_text(encoding="utf-8"))
    return {
        "status": status,
        "descriptor_pass": descriptor_pass,
        "gpu_authorized_next": False,
        "reasons": reasons,
        "joined_tracka_conditions": int(joined[["dataset", "condition"]].drop_duplicates().shape[0]),
        "joined_tracka_rows": int(len(joined)),
        "robust_primary_associations": robust[
            ["task", "feature", "rho_pearson_pert", "rho_test_mmd", "within_dataset_rho_pearson_pert", "within_dataset_rho_test_mmd"]
        ].head(20).to_dict(orient="records"),
        "mixed_primary_associations": mixed[
            ["task", "feature", "rho_pearson_pert", "rho_test_mmd", "within_dataset_rho_pearson_pert", "within_dataset_rho_test_mmd"]
        ].head(20).to_dict(orient="records"),
        "pairability_proxy_associations": pairability_proxy[
            ["task", "feature", "signal", "rho_pearson_pert", "rho_test_mmd"]
        ].head(20).to_dict(orient="records"),
        "observable_budget_descriptor_status": observable_json.get("status"),
        "observable_budget_descriptor_pass": observable_json.get("descriptor_pass"),
        "ot_pairing_status": (ot_json.get("decision") or {}).get("status"),
        "ot_pairing_default_reduction_vs_random": (ot_json.get("decision") or {}).get(
            "default_cost_reduction_vs_random"
        ),
        "next_action": (
            "if descriptor signal is robust, design a response-compressibility-stratified "
            "train-only sampling/weighting smoke with canonical no-harm gate; otherwise keep "
            "the result as scaling-law/manuscript evidence and move to ZSCAPE frozen validation"
        ),
    }


def fmt(value: Any, digits: int = 4) -> str:
    x = finite_float(value)
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def markdown_table(df: pd.DataFrame, cols: list[str], n: int = 12) -> str:
    if df.empty:
        return "_None._"
    use = df[cols].head(n).copy()
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in use.iterrows():
        vals = []
        for col in cols:
            val = row.get(col)
            if isinstance(val, float):
                vals.append(fmt(val))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(features: pd.DataFrame, joined: pd.DataFrame, assoc: pd.DataFrame, decision: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features_path = OUT_DIR / "response_compressibility_condition_features_20260630.csv"
    joined_path = OUT_DIR / "response_compressibility_tracka_joined_20260630.csv"
    assoc_path = OUT_DIR / "response_compressibility_associations_20260630.csv"
    json_path = OUT_DIR / "response_compressibility_pairability_gate_20260630.json"
    md_path = OUT_DIR / "LATENTFM_RESPONSE_COMPRESSIBILITY_PAIRABILITY_GATE_20260630.md"

    features.to_csv(features_path, index=False)
    joined.to_csv(joined_path, index=False)
    assoc.to_csv(assoc_path, index=False)

    robust = top_rows(assoc, "robust_expected_direction", 16)
    mixed = top_rows(assoc, "mixed_expected_direction", 16)
    covariates = assoc[assoc["direction"].isin(["unknown", "nuisance"])].copy()
    if not covariates.empty:
        covariates["_score"] = covariates["rho_pearson_pert"].abs().fillna(0) + covariates["rho_test_mmd"].abs().fillna(0)
        covariates = covariates.sort_values("_score", ascending=False).drop(columns=["_score"]).head(10)

    payload = {
        "boundary": {
            "reads_completed_reports_only": True,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
            "gpu_authorized_next": False,
        },
        "inputs": {k: str(v) for k, v in INPUTS.items()},
        "outputs": {
            "condition_features": str(features_path),
            "joined_rows": str(joined_path),
            "association_rows": str(assoc_path),
            "markdown_report": str(md_path),
        },
        "decision": decision,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    cols = [
        "task",
        "feature",
        "n_conditions",
        "n_datasets",
        "rho_pearson_pert",
        "rho_test_mmd",
        "within_dataset_rho_pearson_pert",
        "within_dataset_rho_test_mmd",
        "lodo_sign_frac_pearson_pert",
        "lodo_sign_frac_test_mmd",
        "signal",
    ]
    text = f"""# LatentFM Response-Compressibility / Pairability-Proxy Gate

## Boundary

- CPU/report-only synthesis of completed artifacts.
- No training, inference, checkpoint selection, canonical-multi selection, or Track C held-out query access.
- Track A rows are joined only by `(dataset, condition)` to completed observable-response descriptors.
- GPU authorization from this gate is `False`; any training intervention still needs a separate launcher, split boundary, no-harm gate, and stop rule.

## Decision

- Status: `{decision['status']}`
- Descriptor pass: `{decision['descriptor_pass']}`
- GPU authorized next: `{decision['gpu_authorized_next']}`
- Joined Track A condition rows: `{decision['joined_tracka_rows']}` rows, `{decision['joined_tracka_conditions']}` unique `(dataset, condition)` pairs.
- Observable-gene budget descriptor context: `{decision['observable_budget_descriptor_status']}`
- OT minibatch context: `{decision['ot_pairing_status']}`, default reduction vs random `{fmt(decision['ot_pairing_default_reduction_vs_random'])}`.

Reasons:

{chr(10).join(f'- {r}' for r in decision['reasons'])}

## Robust Expected-Direction Associations

{markdown_table(robust, cols)}

## Mixed Expected-Direction Associations

{markdown_table(mixed, cols)}

## Strong Nuisance / Unknown Covariates

{markdown_table(covariates, cols)}

## Interpretation

This gate separates two questions. First, whether response concentration is a
manuscript-grade scaling descriptor for single-cell perturbation prediction.
Second, whether it is already sufficient to justify a GPU training change. The
second answer remains no here: this report only authorizes a follow-up design
step if the descriptor is robust, not a direct checkpoint-producing run.

Pairability is represented only by proxies available in completed reports
(`split_half_jaccard`, response-over-shuffled, `k90_hvg`, and oracle gap). A
true condition-level OT pairability x still needs a dedicated train-only or
time-course OT measurement before it can become a modeling intervention.

## Outputs

- Condition features: `{features_path}`
- Joined Track A rows: `{joined_path}`
- Association rows: `{assoc_path}`
- JSON decision: `{json_path}`
"""
    md_path.write_text(text, encoding="utf-8")


def main() -> None:
    for name, path in INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    features = build_condition_features()
    tracka = load_tracka_rows()
    joined = tracka.merge(features, on=["dataset", "condition"], how="inner")
    assoc = association_rows(joined)
    decision = decide(assoc, joined)
    write_report(features, joined, assoc, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
