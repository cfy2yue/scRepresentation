#!/usr/bin/env python3
"""CPU-only negative controls for the HVG/response-energy scaling axis."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path("/data/cyx/1030/scLatent")
INPUT_CONDITION_CSV = ROOT / "reports/raw_expression_hvg_budget_predictability_gate_20260628/condition_budget_rows.csv"
OUT_DIR = ROOT / "reports/hvg_meanmatched_negative_controls_20260628"
OUT_MD = OUT_DIR / "LATENTFM_HVG_MEANMATCHED_NEGATIVE_CONTROLS_20260628.md"
OUT_JSON = OUT_DIR / "latentfm_hvg_meanmatched_negative_controls_20260628.json"
OUT_CONDITION_CSV = OUT_DIR / "condition_negative_control_rows.csv"
OUT_SUMMARY_CSV = OUT_DIR / "negative_control_summary_rows.csv"

BUDGETS = (500, 1000)
RANDOM_REPEATS = 30
MEAN_MATCHED_REPEATS = 30
SHUFFLED_LABEL_REPEATS = 10
N_MEAN_BINS = 20
BOOT_REPEATS = 500
SEED = 43


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt_float(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def raw_path(dataset: str, group: str) -> Path:
    if group == "chemicalpert_bench" or dataset.startswith("sciplex3_"):
        return ROOT / f"dataset/raw/chemicalpert_bench/{dataset}.h5ad"
    return ROOT / f"dataset/raw/genepert_DE5000/{dataset}.h5ad"


def get_matrix(adata: ad.AnnData, group: str) -> tuple[sparse.spmatrix | np.ndarray, str, str]:
    if group == "chemicalpert_bench" and "logNor" in adata.layers:
        matrix = adata.layers["logNor"]
        source = "layer:logNor"
        policy = "use_existing_log_normalized_matrix_no_second_log1p"
    elif adata.X is not None:
        matrix = adata.X
        source = "X"
        policy = "use_existing_X_no_second_log1p"
    else:
        raise ValueError("no expression matrix")
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
    else:
        matrix = np.asarray(matrix)
    return matrix, source, policy


def dense_mean(matrix: sparse.spmatrix | np.ndarray) -> np.ndarray:
    return np.asarray(matrix.mean(axis=0)).ravel().astype(np.float64)


def dense_var(matrix: sparse.spmatrix | np.ndarray) -> np.ndarray:
    mean = dense_mean(matrix)
    if sparse.issparse(matrix):
        mean_sq = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel().astype(np.float64)
    else:
        arr = np.asarray(matrix)
        mean_sq = np.mean(arr * arr, axis=0).astype(np.float64)
    return np.maximum(mean_sq - mean * mean, 0.0)


def control_mask(obs: pd.DataFrame, group: str) -> np.ndarray:
    if group == "chemicalpert_bench" and "control" in obs.columns:
        vals = obs["control"].astype(str).str.lower()
        return vals.isin({"1", "true", "yes", "control", "ctrl"}).to_numpy()
    vals = obs["perturbation"].astype(str).str.lower()
    return vals.isin({"control", "ctrl", "vehicle", "dmso", "non-targeting", "non_targeting"}).to_numpy()


def make_mean_bins(control_mean: np.ndarray, n_bins: int) -> np.ndarray:
    order = np.argsort(control_mean, kind="mergesort")
    bins = np.zeros(control_mean.shape[0], dtype=np.int32)
    for bin_id, idx in enumerate(np.array_split(order, n_bins)):
        bins[idx] = bin_id
    return bins


def matched_sample_indices(
    hvg_idx: np.ndarray,
    bins: np.ndarray,
    rng: np.random.Generator,
    repeats: int,
) -> list[np.ndarray]:
    hvg_set = set(int(i) for i in hvg_idx)
    hvg_bins = CounterLike(bins[hvg_idx])
    out: list[np.ndarray] = []
    all_indices = np.arange(bins.shape[0])
    for _ in range(repeats):
        parts = []
        for bin_id, count in hvg_bins.items():
            candidates = all_indices[bins == bin_id]
            non_hvg = np.asarray([i for i in candidates if int(i) not in hvg_set], dtype=np.int64)
            pool = non_hvg if non_hvg.size >= count else candidates
            replace = pool.size < count
            parts.append(rng.choice(pool, size=count, replace=replace))
        idx = np.concatenate(parts) if parts else np.asarray([], dtype=np.int64)
        if idx.size < hvg_idx.size:
            filler = rng.choice(all_indices, size=hvg_idx.size - idx.size, replace=False)
            idx = np.concatenate([idx, filler])
        out.append(idx[: hvg_idx.size])
    return out


def CounterLike(values: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = defaultdict(int)
    for value in values:
        out[int(value)] += 1
    return dict(out)


def response_share(response_sq: np.ndarray, idx: np.ndarray, total: float) -> float:
    if total <= 0 or idx.size == 0:
        return float("nan")
    return float(response_sq[idx].sum() / total)


def split_half_stats(
    control_matrix: sparse.spmatrix | np.ndarray,
    budgets: tuple[int, ...],
    rng: np.random.Generator,
) -> dict[int, dict[str, float]]:
    n = control_matrix.shape[0]
    perm = rng.permutation(n)
    left = control_matrix[perm[: n // 2]]
    right = control_matrix[perm[n // 2 :]]
    left_order = np.argsort(-dense_var(left), kind="mergesort")
    right_order = np.argsort(-dense_var(right), kind="mergesort")
    out: dict[int, dict[str, float]] = {}
    n_vars = control_matrix.shape[1]
    for budget in budgets:
        k = min(budget, n_vars)
        a = set(int(x) for x in left_order[:k])
        b = set(int(x) for x in right_order[:k])
        overlap = len(a & b)
        union = len(a | b)
        random_expected = k * k / max(n_vars, 1)
        out[budget] = {
            "split_half_overlap": float(overlap),
            "split_half_jaccard": overlap / union if union else float("nan"),
            "split_half_overlap_fold_random": overlap / max(random_expected, 1e-12),
        }
    return out


def bootstrap_ci(values: list[float], rng: np.random.Generator) -> tuple[float | None, float | None]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size < 3:
        return None, None
    samples = rng.choice(arr, size=(BOOT_REPEATS, arr.size), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def dataset_rows(
    dataset: str,
    group: str,
    conditions: list[str],
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = raw_path(dataset, group)
    adata = ad.read_h5ad(path)
    try:
        matrix, matrix_source, log1p_policy = get_matrix(adata, group)
        obs = adata.obs.copy()
        mask_control = control_mask(obs, group)
        control_matrix = matrix[mask_control]
        control_mean = dense_mean(control_matrix)
        control_var = dense_var(control_matrix)
        hvg_order = np.argsort(-control_var, kind="mergesort")
        mean_bins = make_mean_bins(control_mean, N_MEAN_BINS)
        split_half = split_half_stats(control_matrix, BUDGETS, rng)
        perturb_values = obs["perturbation"].astype(str).to_numpy()
        non_control_indices = np.where(~mask_control)[0]
        control_center = control_mean

        rows: list[dict[str, Any]] = []
        for condition in conditions:
            mask_condition = perturb_values == condition
            n_pert = int(mask_condition.sum())
            if n_pert <= 0:
                continue
            pert_mean = dense_mean(matrix[mask_condition])
            response_sq = np.square(pert_mean - control_center)
            total_energy = float(response_sq.sum())
            if total_energy <= 0 or not np.isfinite(total_energy):
                continue
            for budget in BUDGETS:
                k = min(budget, adata.n_vars)
                hvg_idx = hvg_order[:k]
                random_shares = [
                    response_share(response_sq, rng.choice(adata.n_vars, size=k, replace=False), total_energy)
                    for _ in range(RANDOM_REPEATS)
                ]
                mean_matched_shares = [
                    response_share(response_sq, idx, total_energy)
                    for idx in matched_sample_indices(hvg_idx, mean_bins, rng, MEAN_MATCHED_REPEATS)
                ]
                shuffled_hvg_shares = []
                shuffled_energies = []
                for _ in range(SHUFFLED_LABEL_REPEATS):
                    sample = rng.choice(non_control_indices, size=n_pert, replace=non_control_indices.size < n_pert)
                    shuffled_mean = dense_mean(matrix[sample])
                    shuffled_response_sq = np.square(shuffled_mean - control_center)
                    shuffled_total = float(shuffled_response_sq.sum())
                    shuffled_energies.append(shuffled_total)
                    shuffled_hvg_shares.append(response_share(shuffled_response_sq, hvg_idx, shuffled_total))
                hvg_share = response_share(response_sq, hvg_idx, total_energy)
                random_mean = float(np.mean(random_shares))
                mean_matched_mean = float(np.mean(mean_matched_shares))
                shuffled_hvg_mean = float(np.mean(shuffled_hvg_shares))
                rows.append(
                    {
                        "group": group,
                        "dataset": dataset,
                        "condition": condition,
                        "budget": budget,
                        "effective_budget": k,
                        "n_vars": int(adata.n_vars),
                        "n_control": int(mask_control.sum()),
                        "n_pert": n_pert,
                        "matrix_source": matrix_source,
                        "log1p_policy": log1p_policy,
                        "response_energy": total_energy,
                        "control_hvg_share": hvg_share,
                        "random_share_mean": random_mean,
                        "mean_matched_random_share_mean": mean_matched_mean,
                        "shuffled_label_hvg_share_mean": shuffled_hvg_mean,
                        "shuffled_label_response_energy_mean": float(np.mean(shuffled_energies)),
                        "hvg_minus_random_mean": hvg_share - random_mean,
                        "hvg_minus_mean_matched_mean": hvg_share - mean_matched_mean,
                        "hvg_minus_shuffled_label_hvg_mean": hvg_share - shuffled_hvg_mean,
                        "response_energy_over_shuffled_mean": total_energy / max(float(np.mean(shuffled_energies)), 1e-12),
                        **split_half[budget],
                    }
                )
        meta = {
            "dataset": dataset,
            "group": group,
            "path": str(path),
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "n_control": int(mask_control.sum()),
            "conditions": len(conditions),
            "matrix_source": matrix_source,
            "log1p_policy": log1p_policy,
        }
        return rows, meta
    finally:
        del adata


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED + 100)
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    groupings = [
        (["group", "dataset", "budget"], "dataset"),
        (["group", "budget"], "group"),
        (["budget"], "all"),
    ]
    for cols, level in groupings:
        for keys, part in frame.groupby(cols, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_map = dict(zip(cols, keys))
            adv = part["hvg_minus_mean_matched_mean"].astype(float).tolist()
            adv_low, adv_high = bootstrap_ci(adv, rng)
            rand_adv = part["hvg_minus_random_mean"].astype(float).tolist()
            rand_low, rand_high = bootstrap_ci(rand_adv, rng)
            rows_out = {
                "level": level,
                "group": key_map.get("group", "__ALL__"),
                "dataset": key_map.get("dataset", "__ALL__"),
                "budget": int(key_map["budget"]),
                "condition_rows": int(part.shape[0]),
                "control_hvg_share_mean": float(part["control_hvg_share"].mean()),
                "random_share_mean": float(part["random_share_mean"].mean()),
                "mean_matched_random_share_mean": float(part["mean_matched_random_share_mean"].mean()),
                "shuffled_label_hvg_share_mean": float(part["shuffled_label_hvg_share_mean"].mean()),
                "hvg_minus_random_mean": float(part["hvg_minus_random_mean"].mean()),
                "hvg_minus_random_ci95_low": rand_low,
                "hvg_minus_random_ci95_high": rand_high,
                "hvg_minus_mean_matched_mean": float(part["hvg_minus_mean_matched_mean"].mean()),
                "hvg_minus_mean_matched_ci95_low": adv_low,
                "hvg_minus_mean_matched_ci95_high": adv_high,
                "hvg_minus_shuffled_label_hvg_mean": float(part["hvg_minus_shuffled_label_hvg_mean"].mean()),
                "response_energy_over_shuffled_mean": float(part["response_energy_over_shuffled_mean"].mean()),
                "split_half_jaccard_mean": float(part["split_half_jaccard"].mean()),
                "split_half_overlap_fold_random_mean": float(part["split_half_overlap_fold_random"].mean()),
            }
            summaries.append(rows_out)
    return summaries


def decide(summary_rows: list[dict[str, Any]]) -> tuple[str, list[str], str]:
    reasons: list[str] = []
    groups = [
        row
        for row in summary_rows
        if row["level"] == "group" and int(row["budget"]) == 1000 and row["group"] in {"genepert_DE5000_small", "chemicalpert_bench"}
    ]
    if len(groups) != 2:
        reasons.append("missing_group_level_budget1000_summaries")
    for row in groups:
        if float(row["hvg_minus_mean_matched_mean"]) <= 0.05:
            reasons.append(f"{row['group']}_top1000_meanmatched_advantage_too_small")
        ci_low = row.get("hvg_minus_mean_matched_ci95_low")
        if ci_low is None or float(ci_low) <= 0:
            reasons.append(f"{row['group']}_top1000_meanmatched_ci_crosses_zero")
        if float(row["split_half_overlap_fold_random_mean"]) <= 2.0:
            reasons.append(f"{row['group']}_top1000_hvg_rank_split_half_unstable")
    if reasons:
        return (
            "hvg_meanmatched_negative_controls_partial_no_gpu",
            reasons,
            "keep HVG response coverage as a promising axis but expand controls before any formal scaling-law claim",
        )
    return (
        "hvg_meanmatched_negative_controls_pass_no_gpu",
        [],
        "use HVG response coverage in the formal CPU scaling-law design matrix; still no direct GPU authorization",
    )


def main() -> None:
    global INPUT_CONDITION_CSV, OUT_DIR, OUT_MD, OUT_JSON, OUT_CONDITION_CSV, OUT_SUMMARY_CSV
    global RANDOM_REPEATS, MEAN_MATCHED_REPEATS, SHUFFLED_LABEL_REPEATS, BOOT_REPEATS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-condition-csv", type=Path, default=INPUT_CONDITION_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--random-repeats", type=int, default=RANDOM_REPEATS)
    parser.add_argument("--mean-matched-repeats", type=int, default=MEAN_MATCHED_REPEATS)
    parser.add_argument("--shuffled-label-repeats", type=int, default=SHUFFLED_LABEL_REPEATS)
    parser.add_argument("--boot-repeats", type=int, default=BOOT_REPEATS)
    args = parser.parse_args()

    INPUT_CONDITION_CSV = args.input_condition_csv
    OUT_DIR = args.out_dir
    OUT_MD = OUT_DIR / "LATENTFM_HVG_MEANMATCHED_NEGATIVE_CONTROLS_20260628.md"
    OUT_JSON = OUT_DIR / "latentfm_hvg_meanmatched_negative_controls_20260628.json"
    OUT_CONDITION_CSV = OUT_DIR / "condition_negative_control_rows.csv"
    OUT_SUMMARY_CSV = OUT_DIR / "negative_control_summary_rows.csv"
    RANDOM_REPEATS = int(args.random_repeats)
    MEAN_MATCHED_REPEATS = int(args.mean_matched_repeats)
    SHUFFLED_LABEL_REPEATS = int(args.shuffled_label_repeats)
    BOOT_REPEATS = int(args.boot_repeats)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    input_rows = pd.read_csv(INPUT_CONDITION_CSV)
    selected = (
        input_rows[input_rows["budget"] == 1000][["group", "dataset", "condition"]]
        .drop_duplicates()
        .sort_values(["group", "dataset", "condition"])
    )
    condition_rows: list[dict[str, Any]] = []
    dataset_meta: list[dict[str, Any]] = []
    for (group, dataset), part in selected.groupby(["group", "dataset"], sort=True):
        rows, meta = dataset_rows(str(dataset), str(group), part["condition"].astype(str).tolist(), rng)
        condition_rows.extend(rows)
        dataset_meta.append(meta)
    summary_rows = summarize(condition_rows)
    status, reasons, next_action = decide(summary_rows)

    condition_fields = [
        "group",
        "dataset",
        "condition",
        "budget",
        "effective_budget",
        "n_vars",
        "n_control",
        "n_pert",
        "matrix_source",
        "log1p_policy",
        "response_energy",
        "control_hvg_share",
        "random_share_mean",
        "mean_matched_random_share_mean",
        "shuffled_label_hvg_share_mean",
        "shuffled_label_response_energy_mean",
        "hvg_minus_random_mean",
        "hvg_minus_mean_matched_mean",
        "hvg_minus_shuffled_label_hvg_mean",
        "response_energy_over_shuffled_mean",
        "split_half_overlap",
        "split_half_jaccard",
        "split_half_overlap_fold_random",
    ]
    summary_fields = [
        "level",
        "group",
        "dataset",
        "budget",
        "condition_rows",
        "control_hvg_share_mean",
        "random_share_mean",
        "mean_matched_random_share_mean",
        "shuffled_label_hvg_share_mean",
        "hvg_minus_random_mean",
        "hvg_minus_random_ci95_low",
        "hvg_minus_random_ci95_high",
        "hvg_minus_mean_matched_mean",
        "hvg_minus_mean_matched_ci95_low",
        "hvg_minus_mean_matched_ci95_high",
        "hvg_minus_shuffled_label_hvg_mean",
        "response_energy_over_shuffled_mean",
        "split_half_jaccard_mean",
        "split_half_overlap_fold_random_mean",
    ]
    write_csv(OUT_CONDITION_CSV, condition_rows, condition_fields)
    write_csv(OUT_SUMMARY_CSV, summary_rows, summary_fields)

    payload = {
        "created_at": now_cst(),
        "status": status,
        "reasons": reasons,
        "next_action": next_action,
        "dataset_meta": dataset_meta,
        "summary_rows": summary_rows,
        "condition_csv": str(OUT_CONDITION_CSV),
        "summary_csv": str(OUT_SUMMARY_CSV),
        "limits": {
            "budgets": BUDGETS,
            "random_repeats": RANDOM_REPEATS,
            "mean_matched_repeats": MEAN_MATCHED_REPEATS,
            "shuffled_label_repeats": SHUFFLED_LABEL_REPEATS,
            "mean_bins": N_MEAN_BINS,
            "seed": SEED,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_by_key = {
        (row["level"], row["group"], row["budget"]): row
        for row in summary_rows
        if row["level"] in {"group", "all"}
    }
    lines = [
        "# LatentFM HVG Mean-Matched Negative Controls",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "* CPU-only negative-control gate over the same 7 real expression datasets and selected conditions as the HVG-budget predictability gate.",
        "* Tests random genes, mean-expression-matched random genes, shuffled-label response proxies, and control split-half HVG stability.",
        "* Does not train, infer, evaluate canonical multi, use Track C query, select checkpoints, or authorize GPU.",
        "",
        "## Group Summary",
        "",
        "| group | budget | rows | HVG share | random | mean-matched random | HVG - mean-matched | CI low | split-half fold over random | shuffled-label HVG |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ["genepert_DE5000_small", "chemicalpert_bench", "__ALL__"]:
        level = "all" if group == "__ALL__" else "group"
        for budget in BUDGETS:
            row = summary_by_key.get((level, group, budget))
            if not row:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        group,
                        str(budget),
                        str(row["condition_rows"]),
                        fmt_float(row["control_hvg_share_mean"]),
                        fmt_float(row["random_share_mean"]),
                        fmt_float(row["mean_matched_random_share_mean"]),
                        fmt_float(row["hvg_minus_mean_matched_mean"]),
                        fmt_float(row["hvg_minus_mean_matched_ci95_low"]),
                        fmt_float(row["split_half_overlap_fold_random_mean"]),
                        fmt_float(row["shuffled_label_hvg_share_mean"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"* Status: `{status}`.",
            f"* Reasons: `{', '.join(reasons) if reasons else 'none'}`.",
            f"* Next action: {next_action}.",
            "",
            "## Outputs",
            "",
            f"* Condition rows: `{OUT_CONDITION_CSV}`",
            f"* Summary rows: `{OUT_SUMMARY_CSV}`",
            f"* JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"status {status}")


if __name__ == "__main__":
    main()
