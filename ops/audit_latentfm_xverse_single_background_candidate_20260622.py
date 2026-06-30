#!/usr/bin/env python3
"""Candidate-vs-anchor single/background promotion gate for xverse LatentFM.

This is a read-only posthoc audit. It must not be used for checkpoint
selection. Inputs are condition-uncapped canonical posthoc JSONs produced by
``eval_split_groups.py`` and ``eval_condition_families.py``.
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
DEFAULT_ANCHOR_SPLIT = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    "xverse_comp006_endpoint5_8k_seed42_fulleval/posthoc_eval_uncapped_20260621/"
    "split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
)
DEFAULT_ANCHOR_FAMILY = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    "xverse_comp006_endpoint5_8k_seed42_fulleval/posthoc_eval_uncapped_20260621/"
    "condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
)
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_single_background_candidate_gate_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_SINGLE_BACKGROUND_CANDIDATE_GATE_20260622.md"

METRICS = ("pearson_pert", "pearson_ctrl", "test_mmd_clamped")
LOWER_IS_BETTER = {"test_mmd_clamped"}
SINGLE_STRATA = (
    "all_test_single",
    "same_background_seen_gene",
    "cross_background_seen_gene",
    "globally_unseen_gene",
)


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


def single_condition_table(
    payload: dict[str, Any],
    metadata: dict[str, Any],
    train_by_ds: dict[str, set[str]],
    train_global: set[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    table: dict[tuple[str, str], dict[str, Any]] = {}
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
        table[(ds, cond)] = out
    return table


def family_condition_table(payload: dict[str, Any], group: str = "family_gene") -> dict[tuple[str, str], dict[str, Any]]:
    table: dict[tuple[str, str], dict[str, Any]] = {}
    for row in group_rows(payload, group):
        ds = str(row["dataset"])
        cond = str(row["condition"])
        out = {"dataset": ds, "condition": cond, "strata": [group]}
        for metric in METRICS:
            out[metric] = fnum(row.get(metric))
        table[(ds, cond)] = out
    return table


def paired_deltas(
    base: dict[tuple[str, str], dict[str, Any]],
    cand: dict[tuple[str, str], dict[str, Any]],
    stratum: str,
    metric: str,
) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for key in sorted(set(base) & set(cand)):
        b = base[key]
        c = cand[key]
        if stratum not in b.get("strata", []) or stratum not in c.get("strata", []):
            continue
        bv = fnum(b.get(metric))
        cv = fnum(c.get(metric))
        if bv is None or cv is None:
            continue
        rows.append((key[0], float(cv) - float(bv)))
    return rows


def equal_dataset_mean(rows: list[tuple[str, float]]) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for ds, val in rows:
        by_ds[ds].append(float(val))
    means = [float(np.mean(vals)) for vals in by_ds.values() if vals]
    if not means:
        return None
    return float(np.mean(means))


def bootstrap_equal_dataset(
    rows: list[tuple[str, float]],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    tmp: dict[str, list[float]] = defaultdict(list)
    for ds, val in rows:
        tmp[str(ds)].append(float(val))
    datasets = sorted(tmp)
    samples = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        picked = rng.choice(datasets, size=len(datasets), replace=True)
        ds_means = []
        for ds in picked:
            arr = np.asarray(tmp[str(ds)], dtype=np.float64)
            idx = rng.integers(0, len(arr), size=len(arr))
            ds_means.append(float(np.mean(arr[idx])))
        samples[i] = float(np.mean(ds_means))
    return samples


def summarize_delta(
    base: dict[tuple[str, str], dict[str, Any]],
    cand: dict[tuple[str, str], dict[str, Any]],
    stratum: str,
    metric: str,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    rows = paired_deltas(base, cand, stratum, metric)
    nds = len({ds for ds, _ in rows})
    out: dict[str, Any] = {
        "stratum": stratum,
        "metric": metric,
        "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
        "n_matched_conditions": len(rows),
        "n_matched_datasets": nds,
        "delta_mean": equal_dataset_mean(rows),
        "ci95": [None, None],
        "p_improve": None,
        "p_harm": None,
        "status": "ok",
    }
    if len(rows) < 2 or nds < 1:
        out["status"] = "insufficient_condition_metrics"
        return out
    boots = bootstrap_equal_dataset(rows, n_boot=n_boot, rng=rng)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    if metric in LOWER_IS_BETTER:
        improve = boots < 0
        harm = boots > 0
    else:
        improve = boots > 0
        harm = boots < 0
    out.update(
        {
            "ci95": [float(lo), float(hi)],
            "p_improve": float(np.mean(improve)),
            "p_harm": float(np.mean(harm)),
        }
    )
    return out


def find_row(rows: list[dict[str, Any]], stratum: str, metric: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("stratum") == stratum and row.get("metric") == metric:
            return row
    return None


def hard_harm(row: dict[str, Any] | None, *, max_p_harm: float = 0.20) -> bool:
    if not row or row.get("status") != "ok":
        return True
    p_harm = fnum(row.get("p_harm"))
    if p_harm is None:
        return True
    return p_harm > max_p_harm


def positive_primary(row: dict[str, Any] | None, *, min_p_improve: float = 0.90) -> bool:
    if not row or row.get("status") != "ok":
        return False
    p_improve = fnum(row.get("p_improve"))
    ci = row.get("ci95") or [None, None]
    lo = fnum(ci[0])
    return bool((p_improve is not None and p_improve >= min_p_improve) or (lo is not None and lo > 0))


def build_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    primary = find_row(rows, "cross_background_seen_gene", "pearson_pert")
    if not positive_primary(primary):
        reasons.append("primary_cross_background_seen_gene_pp_not_improved")
    for stratum in ("all_test_single", "family_gene"):
        if hard_harm(find_row(rows, stratum, "pearson_pert")):
            reasons.append(f"{stratum}_pp_hard_harm_or_missing")
        if hard_harm(find_row(rows, stratum, "test_mmd_clamped")):
            reasons.append(f"{stratum}_mmd_hard_harm_or_missing")
    status = "candidate_gate_pass" if not reasons else "candidate_gate_fail_or_partial"
    return {
        "status": status,
        "reasons": reasons,
        "rules": [
            "condition-uncapped canonical posthoc required",
            "primary cross_background_seen_gene pearson_pert delta p_improve >= 0.90 or CI lower > 0",
            "all_test_single and family_gene pearson_pert p_harm <= 0.20",
            "all_test_single and family_gene test_mmd_clamped p_harm <= 0.20",
            "do not use canonical posthoc metrics for checkpoint selection",
            "seed robustness is required before a strong paper claim",
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
        "# LatentFM xverse Single/Background Candidate Gate 2026-06-22",
        "",
        f"Status: `{payload['gate']['status']}`",
        "",
        "## Provenance",
        "",
        f"- anchor split eval: `{payload['anchor_split_json']}`",
        f"- candidate split eval: `{payload['candidate_split_json']}`",
        f"- anchor family eval: `{payload['anchor_family_json']}`",
        f"- candidate family eval: `{payload['candidate_family_json']}`",
        f"- canonical split: `{payload['split_file']}`",
        f"- condition metadata: `{payload['condition_metadata']}`",
        f"- bootstrap: `{payload['n_boot']}` resamples, seed `{payload['seed']}`",
        "- leakage note: this audit reads held-out canonical posthoc metrics only after training; it is not a checkpoint-selection tool.",
        "",
        "## Paired Candidate Minus Anchor Deltas",
        "",
        "| stratum | metric | direction | n cond | n datasets | delta | 95% CI | p improve | p harm | status |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    order = {
        "cross_background_seen_gene": 0,
        "all_test_single": 1,
        "family_gene": 2,
        "same_background_seen_gene": 3,
        "globally_unseen_gene": 4,
    }
    for row in sorted(payload["paired_deltas"], key=lambda r: (order.get(r["stratum"], 99), r["metric"])):
        ci = row["ci95"]
        lines.append(
            f"| {row['stratum']} | {row['metric']} | {row['direction']} | "
            f"{row['n_matched_conditions']} | {row['n_matched_datasets']} | "
            f"{fmt(row['delta_mean'])} | [{fmt(ci[0])}, {fmt(ci[1])}] | "
            f"{fmt(row['p_improve'])} | {fmt(row['p_harm'])} | {row['status']} |"
        )
    lines += [
        "",
        "## Gate",
        "",
    ]
    if payload["gate"]["reasons"]:
        lines.append("Reasons:")
        for reason in payload["gate"]["reasons"]:
            lines.append(f"- `{reason}`")
    else:
        lines.append("Candidate passes the predeclared single/background paired gate.")
    lines += [
        "",
        "Rules:",
    ]
    for rule in payload["gate"]["rules"]:
        lines.append(f"- {rule}")
    lines += [
        "",
        "Decision note:",
        "- This gate prioritizes canonical single/background/family_gene. Canonical multi remains a separate diagnostic unless a true-multi train/fine-tune protocol is used.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-split-json", type=Path, default=DEFAULT_ANCHOR_SPLIT)
    parser.add_argument("--anchor-family-json", type=Path, default=DEFAULT_ANCHOR_FAMILY)
    parser.add_argument("--candidate-split-json", type=Path, required=True)
    parser.add_argument("--candidate-family-json", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    anchor_split = load_json(args.anchor_split_json)
    cand_split = load_json(args.candidate_split_json)
    anchor_family = load_json(args.anchor_family_json)
    cand_family = load_json(args.candidate_family_json)
    split = load_json(args.split_file)
    metadata_path = args.data_dir / "condition_metadata.json"
    metadata = load_json(metadata_path)
    train_by_ds, train_global = train_single_gene_sets(split, metadata)

    base_single = single_condition_table(anchor_split, metadata, train_by_ds, train_global)
    cand_single = single_condition_table(cand_split, metadata, train_by_ds, train_global)
    base_family = family_condition_table(anchor_family, "family_gene")
    cand_family_table = family_condition_table(cand_family, "family_gene")

    rng = np.random.default_rng(int(args.seed))
    rows = []
    for stratum in SINGLE_STRATA:
        for metric in METRICS:
            rows.append(
                summarize_delta(
                    base_single,
                    cand_single,
                    stratum,
                    metric,
                    n_boot=int(args.n_boot),
                    rng=rng,
                )
            )
    for metric in METRICS:
        rows.append(
            summarize_delta(
                base_family,
                cand_family_table,
                "family_gene",
                metric,
                n_boot=int(args.n_boot),
                rng=rng,
            )
        )

    out = {
        "anchor_split_json": str(args.anchor_split_json),
        "candidate_split_json": str(args.candidate_split_json),
        "anchor_family_json": str(args.anchor_family_json),
        "candidate_family_json": str(args.candidate_family_json),
        "split_file": str(args.split_file),
        "condition_metadata": str(metadata_path),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "aggregation": "paired condition deltas, bootstrap conditions within dataset, then equal-dataset mean",
        "strata_definition": {
            "all_test_single": "canonical test_single gene perturbations",
            "same_background_seen_gene": "test gene appears in same dataset train singles",
            "cross_background_seen_gene": "test gene absent in same dataset train singles but present in another train dataset/background",
            "globally_unseen_gene": "test gene absent from all canonical train singles",
            "family_gene": "condition-family posthoc gene-family group",
        },
        "train_single_gene_counts": {
            "by_dataset": {ds: len(vals) for ds, vals in sorted(train_by_ds.items())},
            "global": len(train_global),
        },
        "paired_deltas": rows,
    }
    out["gate"] = build_gate(rows)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(out), encoding="utf-8")
    print(json.dumps({"out_md": str(args.out_md), "status": out["gate"]["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
