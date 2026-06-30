#!/usr/bin/env python3
"""Paired Track A single/background gate for a fixed LatentFM candidate.

This reads canonical posthoc metrics only after the candidate checkpoint is
already fixed. It is not a checkpoint-selection script and it ignores canonical
multi groups for the promotion decision.
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
METRICS = ("pearson_pert", "pearson_ctrl", "test_mmd_clamped")
LOWER_IS_BETTER = {"test_mmd_clamped"}
STRATA = ("all_test_single", "cross_background_seen_gene", "globally_unseen_gene")


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


def build_family_rows(payload: dict[str, Any], group: str) -> list[dict[str, Any]]:
    rows = []
    for row in group_rows(payload, group):
        out = {"dataset": str(row["dataset"]), "condition": str(row["condition"]), "strata": [group]}
        for metric in METRICS:
            out[metric] = fnum(row.get(metric))
        rows.append(out)
    return rows


def paired_delta(
    anchor_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    stratum: str,
    metric: str,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    anchor = {(r["dataset"], r["condition"]): r for r in anchor_rows if stratum in r["strata"]}
    candidate = {(r["dataset"], r["condition"]): r for r in candidate_rows if stratum in r["strata"]}
    by_ds: dict[str, list[float]] = defaultdict(list)
    for key in sorted(set(anchor) & set(candidate)):
        a = fnum(anchor[key].get(metric))
        c = fnum(candidate[key].get(metric))
        if a is not None and c is not None:
            by_ds[key[0]].append(float(c) - float(a))
    datasets = sorted(ds for ds, vals in by_ds.items() if vals)
    out = {
        "stratum": stratum,
        "metric": metric,
        "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
        "n_matched_conditions": int(sum(len(by_ds[ds]) for ds in datasets)),
        "n_matched_datasets": int(len(datasets)),
        "delta_mean": None,
        "ci95": [None, None],
        "p_improve": None,
        "p_harm": None,
        "by_dataset": {},
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
            "by_dataset": {ds: float(np.mean(vals)) for ds, vals in by_ds.items()},
        }
    )
    return out


def row_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(r.get("stratum")), str(r.get("metric"))): r for r in rows}


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = row_map(rows)
    reasons = []

    primary = by_key.get(("cross_background_seen_gene", "pearson_pert"), {})
    if primary.get("status") != "ok":
        reasons.append("cross_background_seen_gene_pp_missing")
    else:
        if fnum(primary.get("delta_mean")) is None or float(primary["delta_mean"]) < 0.02:
            reasons.append("cross_background_seen_gene_pp_delta_below_0p02")
        if fnum(primary.get("p_improve")) is None or float(primary["p_improve"]) < 0.90:
            reasons.append("cross_background_seen_gene_pp_p_improve_below_0p90")

    for stratum in ("all_test_single", "family_gene"):
        pp = by_key.get((stratum, "pearson_pert"), {})
        mmd = by_key.get((stratum, "test_mmd_clamped"), {})
        if pp.get("status") != "ok":
            reasons.append(f"{stratum}_pp_missing")
        elif fnum(pp.get("p_harm")) is None or float(pp["p_harm"]) > 0.20:
            reasons.append(f"{stratum}_pp_harm_risk")
        elif any(float(v) < -0.02 for v in (pp.get("by_dataset") or {}).values()):
            reasons.append(f"{stratum}_pp_dataset_level_material_harm")
        if mmd.get("status") != "ok":
            reasons.append(f"{stratum}_mmd_missing")
        elif fnum(mmd.get("p_harm")) is None or float(mmd["p_harm"]) > 0.20:
            reasons.append(f"{stratum}_mmd_harm_risk")
        elif any(float(v) > 0.005 for v in (mmd.get("by_dataset") or {}).values()):
            reasons.append(f"{stratum}_mmd_dataset_level_material_harm")

    return {
        "status": "candidate_gate_pass" if not reasons else "candidate_gate_fail_close_or_nearmiss",
        "reasons": reasons,
        "rules": [
            "primary: cross_background_seen_gene pearson_pert delta_mean >= +0.02",
            "primary: cross_background_seen_gene pearson_pert p_improve >= 0.90",
            "no harm: all_test_single and family_gene pearson_pert p_harm <= 0.20",
            "no hard harm: all_test_single and family_gene test_mmd_clamped p_harm <= 0.20",
            "dataset-level guard: no pp delta < -0.02 and no MMD delta > +0.005 in no-harm strata",
            "canonical multi groups are diagnostic only and have selection weight 0",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-split-json", type=Path, required=True)
    parser.add_argument("--candidate-split-json", type=Path, required=True)
    parser.add_argument("--anchor-family-json", type=Path, required=True)
    parser.add_argument("--candidate-family-json", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=ROOT / "dataset/biFlow_data/split_seed42.json")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    metadata_path = args.data_dir / "condition_metadata.json"
    split = load_json(args.split_file)
    metadata = load_json(metadata_path)
    train_by_ds, train_global = train_single_gene_sets(split, metadata)
    rng = np.random.default_rng(int(args.seed))

    anchor_split = load_json(args.anchor_split_json)
    candidate_split = load_json(args.candidate_split_json)
    anchor_family = load_json(args.anchor_family_json)
    candidate_family = load_json(args.candidate_family_json)
    anchor_rows = build_single_rows(anchor_split, metadata, train_by_ds, train_global)
    candidate_rows = build_single_rows(candidate_split, metadata, train_by_ds, train_global)
    anchor_family_rows = build_family_rows(anchor_family, "family_gene")
    candidate_family_rows = build_family_rows(candidate_family, "family_gene")

    paired = []
    for stratum in STRATA:
        for metric in METRICS:
            paired.append(
                paired_delta(
                    anchor_rows,
                    candidate_rows,
                    stratum,
                    metric,
                    n_boot=int(args.n_boot),
                    rng=rng,
                )
            )
    for metric in METRICS:
        paired.append(
            paired_delta(
                anchor_family_rows,
                candidate_family_rows,
                "family_gene",
                metric,
                n_boot=int(args.n_boot),
                rng=rng,
            )
        )

    payload = {
        "anchor_split_json": str(args.anchor_split_json),
        "candidate_split_json": str(args.candidate_split_json),
        "anchor_family_json": str(args.anchor_family_json),
        "candidate_family_json": str(args.candidate_family_json),
        "split_file": str(args.split_file),
        "condition_metadata": str(metadata_path),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "paired_deltas": paired,
    }
    payload["gate"] = decide(paired)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "status": payload["gate"]["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
