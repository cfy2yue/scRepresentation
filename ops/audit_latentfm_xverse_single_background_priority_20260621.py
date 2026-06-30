#!/usr/bin/env python3
"""Prioritize xverse LatentFM single-gene and cell-background generalization.

This audit reframes model selection around the current bottom line: strong
single-perturbation performance, especially genes unseen in the same dataset
but observed in other cellular backgrounds. Multi-condition metrics remain
reported elsewhere as a harder challenge line.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
RESP_ROOT = ROOT / "runs/latentfm_xverse_response_repair_smoke_20260621/xverse_response_pca32_aux025_replay1_4k/posthoc_eval_stablecaps"
PRIOR_ROOT = ROOT / "runs/latentfm_xverse_condition_prior_adapter_smoke_20260621/xverse_prior_adapter_global_genemean_w005_add002_replay1_4k/posthoc_eval_stablecaps"
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_single_background_priority_20260621.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_SINGLE_BACKGROUND_PRIORITY_20260621.md"

MODELS = ("anchor", "response_aux025", "prior_adapter")
METRICS = ("pearson_pert", "pearson_ctrl", "test_mmd_clamped")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    try:
        if value is None:
            return None
        val = float(value)
        if val != val:
            return None
        return val
    except (TypeError, ValueError):
        return None


def condition_table(payload: dict[str, Any], group: str = "test_single") -> dict[tuple[str, str], dict[str, Any]]:
    rows = (payload.get("groups", {}).get(group, {}) or {}).get("condition_metrics") or []
    return {
        (str(row.get("dataset")), str(row.get("condition"))): row
        for row in rows
        if isinstance(row, dict) and row.get("dataset") and row.get("condition")
    }


def read_bundle(root: Path) -> dict[str, Any]:
    return {
        "anchor": load_json(root / "split_group_eval_anchor_ode20_stablecaps.json"),
        "candidate": load_json(root / "split_group_eval_candidate_ode20_stablecaps.json"),
    }


def genes_for(metadata: dict[str, Any], ds: str, cond: str) -> list[str]:
    meta = (metadata.get(ds) or {}).get(cond) or {}
    return [str(g).strip().upper() for g in meta.get("genes") or [] if str(g).strip()]


def build_train_gene_sets(split: dict[str, Any], metadata: dict[str, Any]) -> tuple[dict[str, set[str]], set[str]]:
    by_ds: dict[str, set[str]] = defaultdict(set)
    global_genes: set[str] = set()
    for ds, obj in split.items():
        for cond in obj.get("train") or []:
            genes = genes_for(metadata, str(ds), str(cond))
            if len(genes) != 1:
                continue
            gene = genes[0]
            by_ds[str(ds)].add(gene)
            global_genes.add(gene)
    return by_ds, global_genes


def strata_for(gene: str, ds: str, by_ds: dict[str, set[str]], global_genes: set[str]) -> list[str]:
    same = gene in by_ds.get(ds, set())
    glob = gene in global_genes
    out = ["all_test_single"]
    if same:
        out.append("same_background_seen_gene")
    elif glob:
        out.append("cross_background_seen_gene")
    else:
        out.append("globally_unseen_gene")
    return out


def build_rows(
    response: dict[str, Any],
    prior: dict[str, Any],
    split: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    train_by_ds, train_global = build_train_gene_sets(split, metadata)
    anchor_tab = condition_table(response["anchor"], "test_single")
    resp_tab = condition_table(response["candidate"], "test_single")
    prior_tab = condition_table(prior["candidate"], "test_single")
    rows = []
    for key in sorted(set(anchor_tab) & set(resp_tab) & set(prior_tab)):
        ds, cond = key
        genes = genes_for(metadata, ds, cond)
        if len(genes) != 1:
            continue
        gene = genes[0]
        row: dict[str, Any] = {
            "dataset": ds,
            "condition": cond,
            "gene": gene,
            "strata": strata_for(gene, ds, train_by_ds, train_global),
        }
        for model_name, table in (
            ("anchor", anchor_tab),
            ("response_aux025", resp_tab),
            ("prior_adapter", prior_tab),
        ):
            src = table[key]
            for metric in METRICS:
                row[f"{model_name}__{metric}"] = fnum(src.get(metric))
        rows.append(row)
    return rows


def summarize_values(
    rows: list[dict[str, Any]],
    *,
    model: str,
    metric: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = row.get(f"{model}__{metric}")
        if val is not None:
            by_ds[row["dataset"]].append(float(val))
    datasets = sorted(ds for ds, vals in by_ds.items() if vals)
    if not datasets:
        return None
    observed = float(np.mean([np.mean(by_ds[ds]) for ds in datasets]))
    samples = []
    for _ in range(n_boot):
        picked = rng.choice(datasets, size=len(datasets), replace=True)
        vals = []
        for ds in picked:
            arr = np.asarray(by_ds[str(ds)], dtype=float)
            idx = rng.integers(0, len(arr), size=len(arr))
            vals.append(float(np.mean(arr[idx])))
        samples.append(float(np.mean(vals)))
    lo, hi = np.quantile(np.asarray(samples), [0.025, 0.975])
    return {
        "model": model,
        "metric": metric,
        "n_conditions": len(rows),
        "n_datasets": len(datasets),
        "value": observed,
        "ci95": [float(lo), float(hi)],
    }


def summarize_delta(
    rows: list[dict[str, Any]],
    *,
    model: str,
    metric: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        base = row.get(f"anchor__{metric}")
        cand = row.get(f"{model}__{metric}")
        if base is None or cand is None:
            continue
        by_ds[row["dataset"]].append(float(cand) - float(base))
    datasets = sorted(ds for ds, vals in by_ds.items() if vals)
    if not datasets:
        return None
    observed = float(np.mean([np.mean(by_ds[ds]) for ds in datasets]))
    samples = []
    for _ in range(n_boot):
        picked = rng.choice(datasets, size=len(datasets), replace=True)
        vals = []
        for ds in picked:
            arr = np.asarray(by_ds[str(ds)], dtype=float)
            idx = rng.integers(0, len(arr), size=len(arr))
            vals.append(float(np.mean(arr[idx])))
        samples.append(float(np.mean(vals)))
    arr = np.asarray(samples, dtype=float)
    lo, hi = np.quantile(arr, [0.025, 0.975])
    if metric == "test_mmd_clamped":
        p_improve = float(np.mean(arr < 0.0))
        p_harm = float(np.mean(arr > 0.0))
    else:
        p_improve = float(np.mean(arr > 0.0))
        p_harm = float(np.mean(arr < 0.0))
    return {
        "model": model,
        "metric": metric,
        "n_conditions": len(rows),
        "n_datasets": len(datasets),
        "delta_vs_anchor": observed,
        "ci95": [float(lo), float(hi)],
        "p_improve": p_improve,
        "p_harm": p_harm,
    }


def summarize(rows: list[dict[str, Any]], n_boot: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    strata = sorted({s for row in rows for s in row["strata"]})
    out: dict[str, Any] = {"values": [], "deltas": [], "dataset_table": []}
    for stratum in strata:
        subset = [row for row in rows if stratum in row["strata"]]
        for model in MODELS:
            for metric in METRICS:
                val = summarize_values(subset, model=model, metric=metric, n_boot=n_boot, rng=rng)
                if val:
                    val["stratum"] = stratum
                    out["values"].append(val)
        for model in ("response_aux025", "prior_adapter"):
            for metric in METRICS:
                delta = summarize_delta(subset, model=model, metric=metric, n_boot=n_boot, rng=rng)
                if delta:
                    delta["stratum"] = stratum
                    out["deltas"].append(delta)

    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ds[row["dataset"]].append(row)
    for ds, subset in sorted(by_ds.items()):
        rec = {"dataset": ds, "n_test_single": len(subset)}
        for metric in METRICS:
            vals = [r.get(f"anchor__{metric}") for r in subset if r.get(f"anchor__{metric}") is not None]
            rec[f"anchor__{metric}"] = float(np.mean(vals)) if vals else None
        rec["cross_background_seen_gene"] = sum("cross_background_seen_gene" in r["strata"] for r in subset)
        rec["globally_unseen_gene"] = sum("globally_unseen_gene" in r["strata"] for r in subset)
        out["dataset_table"].append(rec)
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Single/Background Priority Audit 2026-06-21",
        "",
        "Status: `bottom_line_priority_audit`",
        "",
        "This audit prioritizes single perturbation and cross-cell-background transfer. It uses matched stablecaps condition metrics plus train/deployable metadata for strata.",
        "",
        "## Strata Definitions",
        "",
        "- `same_background_seen_gene`: test_single gene appears in train for the same dataset/background.",
        "- `cross_background_seen_gene`: gene is absent from same-dataset train but appears in train of another dataset/background.",
        "- `globally_unseen_gene`: gene absent from all canonical train singles.",
        "",
        "## Anchor Values",
        "",
        "| stratum | model | metric | n cond | n datasets | value | ci95 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    keep = {"all_test_single", "same_background_seen_gene", "cross_background_seen_gene", "globally_unseen_gene"}
    for row in payload["summary"]["values"]:
        if row["stratum"] not in keep or row["model"] != "anchor":
            continue
        ci = row["ci95"]
        lines.append(
            f"| {row['stratum']} | {row['model']} | {row['metric']} | {row['n_conditions']} | "
            f"{row['n_datasets']} | {fmt(row['value'])} | [{fmt(ci[0])}, {fmt(ci[1])}] |"
        )
    lines.extend([
        "",
        "## Candidate Deltas vs Anchor",
        "",
        "| stratum | model | metric | delta | ci95 | p_improve | p_harm |",
        "|---|---|---|---:|---|---:|---:|",
    ])
    for row in payload["summary"]["deltas"]:
        if row["stratum"] not in keep:
            continue
        ci = row["ci95"]
        lines.append(
            f"| {row['stratum']} | {row['model']} | {row['metric']} | "
            f"{fmt(row['delta_vs_anchor'])} | [{fmt(ci[0])}, {fmt(ci[1])}] | "
            f"{row['p_improve']:.3f} | {row['p_harm']:.3f} |"
        )
    lines.extend([
        "",
        "## Dataset Anchor Table",
        "",
        "| dataset | n test_single | cross-bg seen | globally unseen | pp | pc | MMD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["summary"]["dataset_table"]:
        lines.append(
            f"| {row['dataset']} | {row['n_test_single']} | {row['cross_background_seen_gene']} | "
            f"{row['globally_unseen_gene']} | {fmt(row.get('anchor__pearson_pert'))} | "
            f"{fmt(row.get('anchor__pearson_ctrl'))} | {fmt(row.get('anchor__test_mmd_clamped'))} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Treat this as the main model-selection lens for the current stage; multi/unseen2 remains a challenge metric, not the only bottom line.",
        "- Any new GPU branch should preserve or improve these single/background strata before claiming multi gains.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-root", type=Path, default=RESP_ROOT)
    parser.add_argument("--prior-root", type=Path, default=PRIOR_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    rows = build_rows(
        read_bundle(args.response_root),
        read_bundle(args.prior_root),
        load_json(args.split_file),
        load_json(args.data_dir / "condition_metadata.json"),
    )
    payload = {
        "response_root": str(args.response_root),
        "prior_root": str(args.prior_root),
        "data_dir": str(args.data_dir),
        "split_file": str(args.split_file),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "n_rows": len(rows),
        "leakage_status": "train/deployable metadata for strata; heldout metrics only for scoring",
        "rows": rows,
        "summary": summarize(rows, args.n_boot, args.seed),
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "n_rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
