#!/usr/bin/env python3
"""Dataset-level diagnostic for the failed soft-archetype CPU gate.

This reruns the frozen chosen K16 soft-archetype spec on train-only internal
proxy data and reports where the signal helps or harms. It is diagnostic only:
it does not authorize GPU training.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
OPS = ROOT / "ops"
sys.path.insert(0, str(OPS))

from audit_latentfm_soft_archetype_predictive_gate_20260623 import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_GENE_CACHE,
    DEFAULT_PERT_MEANS,
    DEFAULT_SPLIT,
    fit_one_spec,
    load_gene_embeddings,
    load_json,
)
from audit_latentfm_xverse_background_state_residual_consensus_gate_20260622 import collect_rows  # noqa: E402


OUT_JSON = ROOT / "reports/latentfm_soft_archetype_dataset_effects_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_SOFT_ARCHETYPE_DATASET_EFFECTS_20260623.md"
CANDIDATE = "soft_archetype_gene_interact_ridge"
BASELINES = ("dataset_mean", "gene_raw_mean", "gene_only_ridge", "soft_archetype_gene_shuffled_ridge")
FOCUS_NAMES = ("Jiang_IFNG", "Jiang_TNFA", "Jiang_IFNB", "Wessels", "NormanWeissman2019_filtered")


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}"


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["group"]), str(row["dataset"]))].append(row)
    out = []
    for (group, dataset), items in sorted(grouped.items()):
        record: dict[str, Any] = {
            "group": group,
            "dataset": dataset,
            "n_conditions": len(items),
            "candidate_mean": mean(float(row[CANDIDATE]) for row in items),
        }
        for baseline in BASELINES:
            deltas = [float(row[CANDIDATE]) - float(row[baseline]) for row in items]
            record[f"delta_vs_{baseline}"] = mean(deltas)
            record[f"median_delta_vs_{baseline}"] = median(deltas)
            record[f"negative_fraction_vs_{baseline}"] = sum(x < 0.0 for x in deltas) / len(deltas)
        out.append(record)
    return out


def decide(summary: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = ["prior_predictive_gate_failed_dataset_mean_harm_risk"]
    focus = [
        row
        for row in summary
        if row["dataset"] in FOCUS_NAMES and row["group"] in {
            "internal_val_cross_background_seen_gene_proxy",
            "internal_val_family_gene_proxy",
        }
    ]
    harmed_focus = [
        row
        for row in focus
        if float(row.get("delta_vs_dataset_mean") or 0.0) < -0.01
        or float(row.get("negative_fraction_vs_dataset_mean") or 0.0) > 0.50
    ]
    if harmed_focus:
        reasons.append("focus_dataset_dataset_mean_harm_seen")
    helpful_focus = [
        row
        for row in focus
        if float(row.get("delta_vs_dataset_mean") or 0.0) >= 0.02
        and float(row.get("negative_fraction_vs_dataset_mean") or 1.0) <= 0.35
    ]
    if helpful_focus:
        action = "retain_archetype_as_cpu_diagnostic_and_possible_conditioned_fallback_seed"
    else:
        action = "retain_archetype_as_failure_analysis_only"
    return {
        "status": "soft_archetype_dataset_effect_diagnostic_no_gpu",
        "gpu_authorization": "none",
        "action": action,
        "reasons": reasons,
        "n_helpful_focus_rows": len(helpful_focus),
        "n_harmed_focus_rows": len(harmed_focus),
    }


def render(payload: dict[str, Any]) -> str:
    summary = payload["dataset_summary"]
    focus = [
        row
        for row in summary
        if row["dataset"] in FOCUS_NAMES
    ]
    worst = sorted(summary, key=lambda r: float(r["delta_vs_dataset_mean"]))[:12]
    best = sorted(summary, key=lambda r: float(r["delta_vs_dataset_mean"]), reverse=True)[:12]
    lines = [
        "# LatentFM Soft-Archetype Dataset Effects",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"GPU authorization: `{payload['decision']['gpu_authorization']}`",
        f"Action: `{payload['decision']['action']}`",
        "",
        "## Scope",
        "",
        "This is a CPU-only diagnostic rerun of the frozen K16 soft-archetype spec on train-only internal proxy data.",
        "It does not read canonical test outputs or Track C held-out query artifacts.",
        "",
        "## Focus Datasets",
        "",
        "| group | dataset | n | delta vs dataset_mean | neg frac | delta vs gene_only | delta vs shuffled |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in focus:
        lines.append(
            f"| {row['group']} | {row['dataset']} | {row['n_conditions']} | "
            f"{fmt(row['delta_vs_dataset_mean'])} | {fmt(row['negative_fraction_vs_dataset_mean'])} | "
            f"{fmt(row['delta_vs_gene_only_ridge'])} | {fmt(row['delta_vs_soft_archetype_gene_shuffled_ridge'])} |"
        )
    lines += [
        "",
        "## Worst Dataset-Group Effects vs Dataset Mean",
        "",
        "| group | dataset | n | delta | neg frac |",
        "|---|---|---:|---:|---:|",
    ]
    for row in worst:
        lines.append(
            f"| {row['group']} | {row['dataset']} | {row['n_conditions']} | "
            f"{fmt(row['delta_vs_dataset_mean'])} | {fmt(row['negative_fraction_vs_dataset_mean'])} |"
        )
    lines += [
        "",
        "## Best Dataset-Group Effects vs Dataset Mean",
        "",
        "| group | dataset | n | delta | neg frac |",
        "|---|---|---:|---:|---:|",
    ]
    for row in best:
        lines.append(
            f"| {row['group']} | {row['dataset']} | {row['n_conditions']} | "
            f"{fmt(row['delta_vs_dataset_mean'])} | {fmt(row['negative_fraction_vs_dataset_mean'])} |"
        )
    lines += ["", "## Decision Reasons", ""]
    lines.extend(f"- `{reason}`" for reason in payload["decision"].get("reasons") or [])
    lines += [
        "",
        "## Interpretation",
        "",
        "- Dataset-level positive pockets are not sufficient for GPU authorization because the aggregate dataset-mean no-harm gate already failed.",
        "- A future archetype branch would need a separate train-only rule that predicts where to apply archetype conditioning without using validation targets.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    data_dir = DEFAULT_DATA_DIR.resolve()
    split = load_json(DEFAULT_SPLIT)
    manifest = load_json(data_dir / "manifest.json")
    metadata = load_json(Path(manifest["condition_metadata_file"]))
    gene_mapping, gene_emb, unk_index = load_gene_embeddings(DEFAULT_GENE_CACHE)
    pert_means = {k: v.astype(np.float32) for k, v in np.load(DEFAULT_PERT_MEANS).items()}
    train_rows, val_rows = collect_rows(
        data_dir,
        split,
        metadata,
        gene_mapping,
        gene_emb,
        unk_index,
        max_train_per_dataset=160,
        max_cells=128,
    )
    raw = fit_one_spec(
        train_rows,
        val_rows,
        pert_means,
        k=16,
        bg_pcs=24,
        gene_pcs=16,
        interaction_dim=8,
        ridge_alpha=20.0,
        kmeans_seeds=[42, 43, 44],
        max_stability_items=512,
        seed=42,
    )
    dataset_summary = summarize_rows(raw["eval_rows"])
    payload = {
        "leakage_status": "trainonly_internal_proxy_no_canonical_no_query",
        "chosen_k": 16,
        "candidate": CANDIDATE,
        "baselines": BASELINES,
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "stability": raw["stability"],
        "dataset_proxy": raw["dataset_proxy"],
        "dataset_summary": dataset_summary,
    }
    payload["decision"] = decide(dataset_summary)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
