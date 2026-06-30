#!/usr/bin/env python3
"""CPU-only failure-strata audit for LatentFM prior-correction results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_IN = Path("/data/cyx/1030/scLatent/reports/latentfm_prior_correction_eval_scf_inject_20260620.csv")
DEFAULT_OUT_CSV = Path("/data/cyx/1030/scLatent/reports/latentfm_priorcorr_failure_strata_20260621.csv")
DEFAULT_OUT_JSON = Path("/data/cyx/1030/scLatent/reports/latentfm_priorcorr_failure_strata_20260621.json")
DEFAULT_OUT_MD = Path("/data/cyx/1030/scLatent/reports/LATENTFM_PRIORCORR_FAILURE_STRATA_20260621.md")


def fnum(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def bucket_components(n_components: float) -> str:
    if n_components <= 2:
        return "01_<=2"
    if n_components <= 4:
        return "02_3-4"
    return "03_>=5"


def bucket_missing(n_missing: float) -> str:
    if n_missing <= 0:
        return "01_none"
    if n_missing <= 2:
        return "02_1-2"
    return "03_>=3"


def bucket_similarity(series: pd.Series) -> pd.Series:
    out = pd.Series(["NA"] * len(series), index=series.index, dtype="object")
    valid = series.dropna()
    if valid.empty:
        return out
    if valid.nunique() < 3:
        median = valid.median()
        out.loc[series.notna() & (series < median)] = "01_low"
        out.loc[series.notna() & (series >= median)] = "02_high"
        return out
    try:
        bins = pd.qcut(valid, 3, labels=["01_low", "02_mid", "03_high"], duplicates="drop")
    except ValueError:
        return out
    out.loc[bins.index] = bins.astype(str)
    return out


def summarize_stratum(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    grouped = []
    for keys, g in df.groupby(cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(cols, keys))
        row.update(
            {
                "n_conditions": int(len(g)),
                "base_pp_mean": g["base_pp"].mean(),
                "best_pp_mean": g["best_pp"].mean(),
                "delta_pp_mean": g["delta_pp"].mean(),
                "base_pc_mean": g["base_pc"].mean(),
                "best_pc_mean": g["best_pc"].mean(),
                "delta_pc_mean": g["delta_pc"].mean(),
                "improved_pp_rate": (g["delta_pp"] >= 0.02).mean(),
                "worsened_pp_rate": (g["delta_pp"] <= -0.02).mean(),
                "turned_positive_rate": ((g["base_pp"] <= 0) & (g["best_pp"] > 0)).mean(),
                "stayed_negative_rate": (g["best_pp"] <= 0).mean(),
                "correction_available_rate": g["correction_available"].mean(),
                "prior_available_rate": g["prior_available"].mean(),
                "mean_missing_components": g["n_missing"].mean(),
                "mean_knn_similarity": g["median_knn_similarity"].mean(),
            }
        )
        grouped.append(row)
    return pd.DataFrame(grouped)


def build_condition_deltas(raw: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "condition",
        "group",
        "alpha",
        "k",
        "pp",
        "pc",
        "direct",
        "prior_available",
        "n_components",
        "n_seen",
        "n_knn",
        "n_missing",
        "median_knn_similarity",
        "components",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    group_cols = ["dataset", "group", "condition", "k"]
    for (dataset, group, condition, k), g in raw.groupby(group_cols, dropna=False):
        base_rows = g[g["alpha"].astype(float) == 0.0]
        cand = g[g["alpha"].astype(float) > 0.0]
        if base_rows.empty:
            continue
        base = base_rows.iloc[0]
        if cand.empty:
            best_pp = base
            best_pc = base
            best_alpha_by_pp = None
            best_alpha_by_pc = None
            correction_available = False
        else:
            best_pp = cand.sort_values(["pp", "pc", "direct"], ascending=False).iloc[0]
            best_pc = cand.sort_values(["pc", "pp", "direct"], ascending=False).iloc[0]
            best_alpha_by_pp = float(best_pp["alpha"])
            best_alpha_by_pc = float(best_pc["alpha"])
            correction_available = True
        rows.append(
            {
                "dataset": dataset,
                "group": group,
                "condition": condition,
                "k": int(k),
                "base_pp": float(base["pp"]),
                "base_pc": float(base["pc"]),
                "base_direct": float(base["direct"]),
                "correction_available": correction_available,
                "best_alpha_by_pp": best_alpha_by_pp,
                "best_pp": float(best_pp["pp"]),
                "best_pc_at_best_pp": float(best_pp["pc"]),
                "best_direct_at_best_pp": float(best_pp["direct"]),
                "delta_pp": float(best_pp["pp"] - base["pp"]),
                "delta_pc_at_best_pp": float(best_pp["pc"] - base["pc"]),
                "best_alpha_by_pc": best_alpha_by_pc,
                "best_pc": float(best_pc["pc"]),
                "delta_pc": float(best_pc["pc"] - base["pc"]),
                "prior_available": float(base["prior_available"]),
                "n_components": float(base["n_components"]),
                "n_seen": float(base["n_seen"]),
                "n_knn": float(base["n_knn"]),
                "n_missing": float(base["n_missing"]),
                "median_knn_similarity": float(base["median_knn_similarity"]),
                "components": str(base["components"]),
            }
        )
    out = pd.DataFrame(rows)
    out["component_bucket"] = out["n_components"].map(bucket_components)
    out["missing_bucket"] = out["n_missing"].map(bucket_missing)
    out["similarity_bucket"] = out.groupby(["dataset", "group", "k"], group_keys=False)[
        "median_knn_similarity"
    ].apply(bucket_similarity)
    out["pp_outcome"] = "flat"
    out.loc[out["delta_pp"] >= 0.02, "pp_outcome"] = "improved"
    out.loc[out["delta_pp"] <= -0.02, "pp_outcome"] = "worsened"
    out.loc[(out["base_pp"] <= 0) & (out["best_pp"] > 0), "pp_outcome"] = "rescued_positive"
    out.loc[out["best_pp"] <= 0, "pp_outcome"] = "nonpositive_after_best"
    return out


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))


def write_markdown(
    condition_deltas: pd.DataFrame,
    group_summary: pd.DataFrame,
    missing_summary: pd.DataFrame,
    similarity_summary: pd.DataFrame,
    out_md: Path,
) -> None:
    lines = [
        "# LatentFM Prior-Correction Failure Strata",
        "",
        "Status: `complete`",
        "",
        "This is a CPU-only condition-level audit of the strict scFoundation",
        "prior-correction evaluator. It is diagnostic only: it does not add MMD",
        "or family-gene evidence and does not justify GPU promotion by itself.",
        "",
        "## Route-Level Interpretation",
        "",
        "- Norman has broad condition-level rescue and is the cleanest signal.",
        "- Wessels has rescue in seen/unseen1 but remains blocked in unseen2.",
        "- Gasperini has too few held-out multi-unseen2 conditions and worsens under this prior.",
        "- The failure pattern argues for a no-leakage model-design audit of condition/prior logic, not broader training.",
        "",
        "## Group Summary",
        "",
        "| dataset | group | k | n | base pp | best pp | delta pp | improved | worsened | turned positive | stayed negative | prior avail | correction avail | missing | knn sim |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in group_summary.sort_values(["dataset", "group", "k"]).to_dict(orient="records"):
        lines.append(
            f"| `{row['dataset']}` | `{row['group']}` | {int(row['k'])} | {int(row['n_conditions'])} | "
            f"{fnum(row['base_pp_mean'])} | {fnum(row['best_pp_mean'])} | {fnum(row['delta_pp_mean'])} | "
            f"{fnum(row['improved_pp_rate'])} | {fnum(row['worsened_pp_rate'])} | "
            f"{fnum(row['turned_positive_rate'])} | {fnum(row['stayed_negative_rate'])} | "
            f"{fnum(row['prior_available_rate'])} | {fnum(row['correction_available_rate'])} | "
            f"{fnum(row['mean_missing_components'])} | {fnum(row['mean_knn_similarity'])} |"
        )

    focus = condition_deltas[
        condition_deltas["group"].eq("test_multi_unseen2")
        & condition_deltas["dataset"].isin(["Wessels", "GasperiniShendure2019_lowMOI"])
    ]
    worst = focus.sort_values(["delta_pp", "best_pp"], ascending=True).head(12)
    lines += [
        "",
        "## Worst Unseen2 Prior-Correction Cases",
        "",
        "| dataset | k | condition | base pp | best alpha | best pp | delta pp | pc delta | correction avail | missing | knn sim | components |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in worst.to_dict(orient="records"):
        cond = str(row["condition"]).replace("|", "/")
        comps = str(row["components"]).replace("|", "/")
        lines.append(
            f"| `{row['dataset']}` | {int(row['k'])} | `{cond}` | {fnum(row['base_pp'])} | "
            f"{fnum(row['best_alpha_by_pp'])} | {fnum(row['best_pp'])} | {fnum(row['delta_pp'])} | "
            f"{fnum(row['delta_pc_at_best_pp'])} | `{row['correction_available']}` | "
            f"{fnum(row['n_missing'])} | "
            f"{fnum(row['median_knn_similarity'])} | `{comps}` |"
        )

    lines += [
        "",
        "## Missing-Component Strata",
        "",
        "| dataset | group | k | missing bucket | n | delta pp | worsened | stayed negative | knn sim |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in missing_summary.sort_values(["dataset", "group", "k", "missing_bucket"]).to_dict(orient="records"):
        lines.append(
            f"| `{row['dataset']}` | `{row['group']}` | {int(row['k'])} | `{row['missing_bucket']}` | "
            f"{int(row['n_conditions'])} | {fnum(row['delta_pp_mean'])} | "
            f"{fnum(row['worsened_pp_rate'])} | {fnum(row['stayed_negative_rate'])} | "
            f"{fnum(row['mean_knn_similarity'])} |"
        )

    lines += [
        "",
        "## Similarity Strata",
        "",
        "| dataset | group | k | similarity bucket | n | delta pp | worsened | stayed negative | missing |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in similarity_summary.sort_values(["dataset", "group", "k", "similarity_bucket"]).to_dict(orient="records"):
        lines.append(
            f"| `{row['dataset']}` | `{row['group']}` | {int(row['k'])} | `{row['similarity_bucket']}` | "
            f"{int(row['n_conditions'])} | {fnum(row['delta_pp_mean'])} | "
            f"{fnum(row['worsened_pp_rate'])} | {fnum(row['stayed_negative_rate'])} | "
            f"{fnum(row['mean_missing_components'])} |"
        )

    lines += [
        "",
        "## Next Gate",
        "",
        "- Do not use this posthoc correction directly as a promoted result.",
        "- If converted into a model mechanism, it must be default-off, no-leakage, and evaluated with MMD/family-gene gates.",
        "- A useful next CPU check is whether unseen2 failures concentrate in conditions with zero seen components, low KNN similarity, or dataset-specific naming artifacts.",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    condition_deltas = build_condition_deltas(raw)
    group_summary = summarize_stratum(condition_deltas, ["dataset", "group", "k"])
    missing_summary = summarize_stratum(condition_deltas, ["dataset", "group", "k", "missing_bucket"])
    similarity_summary = summarize_stratum(condition_deltas, ["dataset", "group", "k", "similarity_bucket"])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    condition_deltas.to_csv(args.out_csv, index=False)
    payload = {
        "input": str(args.input),
        "condition_delta_rows": int(len(condition_deltas)),
        "condition_deltas": records(condition_deltas),
        "group_summary": records(group_summary),
        "missing_summary": records(missing_summary),
        "similarity_summary": records(similarity_summary),
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(condition_deltas, group_summary, missing_summary, similarity_summary, args.out_md)
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
