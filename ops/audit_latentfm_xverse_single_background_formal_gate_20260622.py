#!/usr/bin/env python3
"""Formal no-leak single/background gate for xverse LatentFM anchors.

This is a reporting/gating audit over completed condition-uncapped posthoc
JSONs. It does not train, tune, or select a checkpoint. Held-out metrics are
used only to define the current bottom-line evidence and future promotion gates.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_SEED42_SPLIT = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    "xverse_comp006_endpoint5_8k_seed42_fulleval/posthoc_eval_uncapped_20260621/"
    "split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
)
DEFAULT_SEED43_SPLIT = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_seed_replicate_20260621/"
    "xverse_comp006_endpoint5_8k_seed43_fulleval/posthoc_eval_uncapped_20260621/"
    "split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
)
DEFAULT_SEED42_FAMILY = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    "xverse_comp006_endpoint5_8k_seed42_fulleval/posthoc_eval_uncapped_20260621/"
    "condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
)
DEFAULT_SEED43_FAMILY = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_seed_replicate_20260621/"
    "xverse_comp006_endpoint5_8k_seed43_fulleval/posthoc_eval_uncapped_20260621/"
    "condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
)
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_single_background_formal_gate_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_SINGLE_BACKGROUND_FORMAL_GATE_20260622.md"

METRICS = ("pearson_pert", "pearson_ctrl", "test_mmd_clamped")
LOWER_IS_BETTER = {"test_mmd_clamped"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    try:
        if value is None:
            return None
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except (TypeError, ValueError):
        return None


def group_rows(payload: dict[str, Any], group: str) -> list[dict[str, Any]]:
    rows = (payload.get("groups", {}).get(group, {}) or {}).get("condition_metrics") or []
    return [r for r in rows if isinstance(r, dict) and r.get("dataset") and r.get("condition")]


def condition_table(payload: dict[str, Any], group: str) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(r["dataset"]), str(r["condition"])): r
        for r in group_rows(payload, group)
    }


def genes_for(metadata: dict[str, Any], ds: str, cond: str) -> list[str]:
    entry = (metadata.get(ds) or {}).get(cond) or {}
    return [str(g).strip().upper() for g in entry.get("genes") or [] if str(g).strip()]


def train_single_gene_sets(
    split: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[dict[str, set[str]], set[str]]:
    by_ds: dict[str, set[str]] = defaultdict(set)
    global_genes: set[str] = set()
    for ds, groups in split.items():
        for cond in groups.get("train") or []:
            genes = genes_for(metadata, str(ds), str(cond))
            if len(genes) != 1:
                continue
            gene = genes[0]
            by_ds[str(ds)].add(gene)
            global_genes.add(gene)
    return by_ds, global_genes


def stratum_for(ds: str, gene: str, train_by_ds: dict[str, set[str]], train_global: set[str]) -> str:
    if gene in train_by_ds.get(ds, set()):
        return "same_background_seen_gene"
    if gene in train_global:
        return "cross_background_seen_gene"
    return "globally_unseen_gene"


def build_single_rows(
    payload: dict[str, Any],
    metadata: dict[str, Any],
    train_by_ds: dict[str, set[str]],
    train_global: set[str],
) -> list[dict[str, Any]]:
    rows = []
    for row in group_rows(payload, "test_single"):
        ds = str(row["dataset"])
        cond = str(row["condition"])
        genes = genes_for(metadata, ds, cond)
        if len(genes) != 1:
            continue
        out = {
            "dataset": ds,
            "condition": cond,
            "gene": genes[0],
            "strata": ["all_test_single", stratum_for(ds, genes[0], train_by_ds, train_global)],
        }
        for metric in METRICS:
            out[metric] = fnum(row.get(metric))
        rows.append(out)
    return rows


def equal_dataset_bootstrap_values(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = fnum(row.get(metric))
        if val is not None:
            by_ds[str(row["dataset"])].append(val)
    datasets = sorted(ds for ds, vals in by_ds.items() if vals)
    out = {
        "metric": metric,
        "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
        "n_conditions": int(sum(len(by_ds[ds]) for ds in datasets)),
        "n_datasets": int(len(datasets)),
        "mean": None,
        "ci95": [None, None],
        "status": "ok",
    }
    if not datasets:
        out["status"] = "missing_metric"
        return out
    observed = float(np.mean([np.mean(by_ds[ds]) for ds in datasets]))
    samples = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        picked = rng.choice(datasets, size=len(datasets), replace=True)
        vals = []
        for ds in picked:
            arr = np.asarray(by_ds[str(ds)], dtype=np.float64)
            idx = rng.integers(0, len(arr), size=len(arr))
            vals.append(float(np.mean(arr[idx])))
        samples[i] = float(np.mean(vals))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    out["mean"] = observed
    out["ci95"] = [float(lo), float(hi)]
    return out


def paired_delta_rows(
    base_rows: list[dict[str, Any]],
    cand_rows: list[dict[str, Any]],
    stratum: str,
    metric: str,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    base = {(r["dataset"], r["condition"]): r for r in base_rows if stratum in r["strata"]}
    cand = {(r["dataset"], r["condition"]): r for r in cand_rows if stratum in r["strata"]}
    by_ds: dict[str, list[float]] = defaultdict(list)
    for key in sorted(set(base) & set(cand)):
        b = fnum(base[key].get(metric))
        c = fnum(cand[key].get(metric))
        if b is not None and c is not None:
            by_ds[key[0]].append(float(c) - float(b))
    datasets = sorted(ds for ds, vals in by_ds.items() if vals)
    out = {
        "comparison": "seed43_minus_seed42",
        "stratum": stratum,
        "metric": metric,
        "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
        "n_matched_conditions": int(sum(len(by_ds[ds]) for ds in datasets)),
        "n_matched_datasets": int(len(datasets)),
        "delta_mean": None,
        "ci95": [None, None],
        "p_improve": None,
        "p_harm": None,
        "status": "ok",
    }
    if not datasets:
        out["status"] = "missing_metric"
        return out
    observed = float(np.mean([np.mean(by_ds[ds]) for ds in datasets]))
    samples = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        picked = rng.choice(datasets, size=len(datasets), replace=True)
        vals = []
        for ds in picked:
            arr = np.asarray(by_ds[str(ds)], dtype=np.float64)
            idx = rng.integers(0, len(arr), size=len(arr))
            vals.append(float(np.mean(arr[idx])))
        samples[i] = float(np.mean(vals))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    if metric in LOWER_IS_BETTER:
        improve = samples < 0
        harm = samples > 0
    else:
        improve = samples > 0
        harm = samples < 0
    out.update(
        {
            "delta_mean": observed,
            "ci95": [float(lo), float(hi)],
            "p_improve": float(np.mean(improve)),
            "p_harm": float(np.mean(harm)),
        }
    )
    return out


def summarize_family(payload: dict[str, Any], *, n_boot: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    rows = group_rows(payload, "family_gene")
    return [
        {"group": "family_gene", **equal_dataset_bootstrap_values(rows, metric, n_boot=n_boot, rng=rng)}
        for metric in METRICS
    ]


def build_gate(payload: dict[str, Any]) -> dict[str, Any]:
    def get_value(seed: str, stratum: str, metric: str) -> dict[str, Any] | None:
        for row in payload["single_values"]:
            if row["seed"] == seed and row["stratum"] == stratum and row["metric"] == metric:
                return row
        return None

    seed42_all = get_value("seed42", "all_test_single", "pearson_pert")
    seed42_cross = get_value("seed42", "cross_background_seen_gene", "pearson_pert")
    seed42_global = get_value("seed42", "globally_unseen_gene", "pearson_pert")
    seed43_cross = get_value("seed43", "cross_background_seen_gene", "pearson_pert")
    reasons = []
    if not seed42_all or (seed42_all["ci95"][0] is None or seed42_all["ci95"][0] <= 0):
        reasons.append("seed42_all_test_single_pp_ci_not_positive")
    if not seed42_cross or (seed42_cross["ci95"][0] is None or seed42_cross["ci95"][0] <= 0):
        reasons.append("seed42_cross_background_pp_ci_not_positive")
    if not seed43_cross or (seed43_cross["ci95"][0] is None or seed43_cross["ci95"][0] <= 0):
        reasons.append("seed43_cross_background_pp_ci_not_positive")
    if not seed42_global or (seed42_global["ci95"][0] is None or seed42_global["ci95"][0] <= 0):
        reasons.append("seed42_globally_unseen_pp_ci_not_positive")
    return {
        "status": "formal_baseline_pass" if not reasons else "formal_baseline_partial",
        "reasons": reasons,
        "future_candidate_gate": [
            "condition-uncapped canonical posthoc required; stablecaps-only is insufficient",
            "primary: cross_background_seen_gene pearson_pert paired delta p_improve >= 0.90 or CI lower > 0",
            "no hard harm: all_test_single/family_gene pearson_pert p_harm <= 0.20",
            "MMD no hard harm: test_mmd_clamped p_harm <= 0.20 for all_test_single and family_gene",
            "seed robustness: compare against seed43 anchor or repeat candidate seed before claim",
            "no checkpoint/model selection on canonical test metrics; use train-only validation or predeclared fixed step",
        ],
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Single/Background Formal Gate 2026-06-22",
        "",
        f"Status: `{payload['gate']['status']}`",
        "",
        "## Provenance",
        "",
        f"- seed42 split eval: `{payload['seed42_split_json']}`",
        f"- seed43 split eval: `{payload['seed43_split_json']}`",
        f"- canonical split: `{payload['split_file']}`",
        f"- condition metadata: `{payload['condition_metadata']}`",
        f"- bootstrap: `{payload['n_boot']}` resamples, seed `{payload['seed']}`",
        "- leakage note: this audit reads held-out posthoc metrics only for reporting/gating; it does not train, tune, or select checkpoints.",
        "",
        "## Single/Background Values",
        "",
        "| seed | stratum | metric | n cond | n datasets | mean | 95% CI |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in payload["single_values"]:
        if row["metric"] not in {"pearson_pert", "pearson_ctrl", "test_mmd_clamped"}:
            continue
        ci = row["ci95"]
        lines.append(
            f"| {row['seed']} | {row['stratum']} | {row['metric']} | "
            f"{row['n_conditions']} | {row['n_datasets']} | {fmt(row['mean'])} | "
            f"[{fmt(ci[0])}, {fmt(ci[1])}] |"
        )
    lines += [
        "",
        "## Seed Robustness",
        "",
        "| comparison | stratum | metric | n cond | n datasets | delta | 95% CI | p improve | p harm |",
        "|---|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in payload["paired_seed_deltas"]:
        ci = row["ci95"]
        lines.append(
            f"| {row['comparison']} | {row['stratum']} | {row['metric']} | "
            f"{row['n_matched_conditions']} | {row['n_matched_datasets']} | "
            f"{fmt(row['delta_mean'])} | [{fmt(ci[0])}, {fmt(ci[1])}] | "
            f"{fmt(row['p_improve'])} | {fmt(row['p_harm'])} |"
        )
    lines += [
        "",
        "## Gate Interpretation",
        "",
    ]
    if payload["gate"]["reasons"]:
        lines.append("Partial reasons:")
        for reason in payload["gate"]["reasons"]:
            lines.append(f"- `{reason}`")
    else:
        lines.append("The current xverse anchor passes the formal baseline single/background gate.")
    lines += [
        "",
        "Future candidate promotion requirements:",
    ]
    for item in payload["gate"]["future_candidate_gate"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "Decision:",
        "- Canonical model selection should prioritize `all_test_single`, `cross_background_seen_gene`, and `family_gene` over zero-shot multi-only slices.",
        "- Multi-aware/fine-tuned multi experiments remain separate from this canonical gate.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed42-split-json", type=Path, default=DEFAULT_SEED42_SPLIT)
    parser.add_argument("--seed43-split-json", type=Path, default=DEFAULT_SEED43_SPLIT)
    parser.add_argument("--seed42-family-json", type=Path, default=DEFAULT_SEED42_FAMILY)
    parser.add_argument("--seed43-family-json", type=Path, default=DEFAULT_SEED43_FAMILY)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    seed42_split = load_json(args.seed42_split_json)
    seed43_split = load_json(args.seed43_split_json)
    seed42_family = load_json(args.seed42_family_json)
    seed43_family = load_json(args.seed43_family_json)
    split = load_json(args.split_file)
    metadata_path = args.data_dir / "condition_metadata.json"
    metadata = load_json(metadata_path)
    train_by_ds, train_global = train_single_gene_sets(split, metadata)

    rows_by_seed = {
        "seed42": build_single_rows(seed42_split, metadata, train_by_ds, train_global),
        "seed43": build_single_rows(seed43_split, metadata, train_by_ds, train_global),
    }
    rng = np.random.default_rng(int(args.seed))
    strata = ("all_test_single", "cross_background_seen_gene", "globally_unseen_gene")
    single_values = []
    for seed_name, rows in rows_by_seed.items():
        for stratum in strata:
            subset = [r for r in rows if stratum in r["strata"]]
            for metric in METRICS:
                rec = equal_dataset_bootstrap_values(
                    subset,
                    metric,
                    n_boot=int(args.n_boot),
                    rng=rng,
                )
                rec["seed"] = seed_name
                rec["stratum"] = stratum
                single_values.append(rec)

    paired = []
    for stratum in strata:
        for metric in METRICS:
            paired.append(
                paired_delta_rows(
                    rows_by_seed["seed42"],
                    rows_by_seed["seed43"],
                    stratum,
                    metric,
                    n_boot=int(args.n_boot),
                    rng=rng,
                )
            )

    family_values = []
    for seed_name, fam in (("seed42", seed42_family), ("seed43", seed43_family)):
        for rec in summarize_family(fam, n_boot=int(args.n_boot), rng=rng):
            rec["seed"] = seed_name
            family_values.append(rec)

    out = {
        "seed42_split_json": str(args.seed42_split_json),
        "seed43_split_json": str(args.seed43_split_json),
        "seed42_family_json": str(args.seed42_family_json),
        "seed43_family_json": str(args.seed43_family_json),
        "split_file": str(args.split_file),
        "condition_metadata": str(metadata_path),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "strata_definition": {
            "all_test_single": "canonical test_single gene perturbations",
            "same_background_seen_gene": "test gene appears in same dataset train singles",
            "cross_background_seen_gene": "test gene absent in same dataset train singles but present in another train dataset/background",
            "globally_unseen_gene": "test gene absent from all canonical train singles",
        },
        "train_single_gene_counts": {
            "by_dataset": {ds: len(vals) for ds, vals in sorted(train_by_ds.items())},
            "global": len(train_global),
        },
        "single_values": single_values,
        "family_values": family_values,
        "paired_seed_deltas": paired,
    }
    out["gate"] = build_gate(out)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(out), encoding="utf-8")
    print(json.dumps({"out_md": str(args.out_md), "status": out["gate"]["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
