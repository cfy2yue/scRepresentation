#!/usr/bin/env python3
"""Inventory local metadata for LatentFM scaling-effect experiments."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import anndata as ad


ROOT = Path("/data/cyx/1030/scLatent")
BIOFLOW = ROOT / "dataset/biFlow_data"
RAW = ROOT / "dataset/raw"
OUT_JSON = ROOT / "reports/latentfm_scaling_metainfo_inventory_20260624.json"
OUT_CSV = ROOT / "reports/latentfm_scaling_metainfo_inventory_20260624.csv"
OUT_MD = ROOT / "reports/LATENTFM_SCALING_METAINFO_INVENTORY_20260624.md"

SPLITS = {
    "canonical_seed42": BIOFLOW / "split_seed42.json",
    "trainonly_crossbg_v2": BIOFLOW / "split_seed42_xverse_trainonly_crossbg_val_v2.json",
    "cap30_all_v2": BIOFLOW / "xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap30_all_v2.json",
    "cap120_all_v2": BIOFLOW / "xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json",
    "type_balanced_cap120_v2": BIOFLOW / "xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_type_balanced_cap120_v2.json",
    "general_exposure_cap_v2": BIOFLOW / "xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_general_exposure_cap_v2.json",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metainfo() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in (
        RAW / "genepert_DE5000/metainfo.json",
        RAW / "chemicalpert_DE5000/metainfo.json",
        RAW / "genepert_bench/metainfo.json",
        RAW / "chemicalpert_bench/metainfo.json",
    ):
        if not path.exists():
            continue
        obj = load_json(path)
        if not isinstance(obj, list):
            continue
        for row in obj:
            ds = str(row.get("dataset", "")).strip()
            if not ds:
                continue
            out.setdefault(ds, {}).update(row)
    return out


def is_multi_condition(cond: str) -> bool:
    return "+" in str(cond) or "|" in str(cond)


def split_counts(split: dict[str, Any], dataset: str) -> dict[str, int]:
    item = split.get(dataset, {})
    return {
        "train": len(item.get("train", [])),
        "test": len(item.get("test", [])),
        "test_single": len(item.get("test_single", [])),
        "test_multi": len(item.get("test_multi", [])),
        "internal_cross": len(item.get("internal_val_cross_background_seen_gene_proxy", [])),
        "internal_family": len(item.get("internal_val_family_gene_proxy", [])),
        "train_multi": sum(1 for c in item.get("train", []) if is_multi_condition(str(c))),
        "train_single": sum(1 for c in item.get("train", []) if not is_multi_condition(str(c))),
    }


def obs_summary(dataset: str) -> dict[str, Any]:
    path = BIOFLOW / "gt_stack" / f"{dataset}.h5ad"
    if not path.exists():
        return {"h5ad_exists": False}
    adata = ad.read_h5ad(path, backed="r")
    try:
        obs = adata.obs
        cols = list(obs.columns)
        summary: dict[str, Any] = {
            "h5ad_exists": True,
            "n_cells": int(adata.n_obs),
            "obs_columns": cols,
        }
        for col in (
            "perturbation",
            "condition",
            "cell_type",
            "cell_line",
            "cov",
            "cov_drug",
            "cov_drug_dose_name",
            "pathway",
            "pathway_level_1",
            "pathway_level_2",
        ):
            if col in obs.columns:
                vals = obs[col].astype(str)
                counts = vals.value_counts()
                summary[f"{col}_n_unique"] = int(counts.shape[0])
                summary[f"{col}_top"] = {str(k): int(v) for k, v in counts.head(12).items()}
        return summary
    finally:
        adata.file.close()


def main() -> int:
    metainfo = load_metainfo()
    split_objs = {name: load_json(path) for name, path in SPLITS.items() if path.exists()}
    datasets = sorted(set().union(*(set(s.keys()) for s in split_objs.values())))
    rows = []
    for ds in datasets:
        meta = metainfo.get(ds, {})
        obs = obs_summary(ds)
        row: dict[str, Any] = {
            "dataset": ds,
            "perturbation_type": str(meta.get("perturbation_type", "unknown")),
            "cell_line_meta": str(meta.get("cell_line", "unknown")),
            "chemical_screen": str(meta.get("chemical_screen", "")),
            "n_cells_gt_stack": obs.get("n_cells", 0),
            "obs_cell_type_n_unique": obs.get("cell_type_n_unique", 0),
            "obs_cell_line_n_unique": obs.get("cell_line_n_unique", 0),
            "obs_cov_drug_n_unique": obs.get("cov_drug_n_unique", 0),
            "obs_pathway_n_unique": obs.get("pathway_n_unique", 0),
            "obs_columns": obs.get("obs_columns", []),
            "obs_top": {
                k: v for k, v in obs.items() if k.endswith("_top")
            },
        }
        for split_name, split in split_objs.items():
            counts = split_counts(split, ds)
            for k, v in counts.items():
                row[f"{split_name}_{k}"] = v
        rows.append(row)

    type_counts = Counter(r["perturbation_type"] for r in rows)
    cell_counts = Counter(r["cell_line_meta"] for r in rows)
    total_by_split = {
        name: {
            "train": sum(split_counts(split, ds)["train"] for ds in datasets),
            "test": sum(split_counts(split, ds)["test"] for ds in datasets),
            "test_single": sum(split_counts(split, ds)["test_single"] for ds in datasets),
            "test_multi": sum(split_counts(split, ds)["test_multi"] for ds in datasets),
            "internal_cross": sum(split_counts(split, ds)["internal_cross"] for ds in datasets),
            "internal_family": sum(split_counts(split, ds)["internal_family"] for ds in datasets),
        }
        for name, split in split_objs.items()
    }
    train_condition_counts = [r.get("trainonly_crossbg_v2_train", 0) for r in rows]
    payload = {
        "status": "scaling_metainfo_inventory_ready_no_gpu",
        "boundary": {
            "expression_matrix_loaded": False,
            "canonical_metrics_read": False,
            "heldout_query_read": False,
            "gpu_used": False,
            "split_json_read": True,
            "h5ad_obs_read_backed": True,
        },
        "inputs": {
            "metainfo": [
                str(RAW / "genepert_DE5000/metainfo.json"),
                str(RAW / "chemicalpert_DE5000/metainfo.json"),
            ],
            "splits": {k: str(v) for k, v in SPLITS.items()},
        },
        "summary": {
            "n_datasets": len(rows),
            "perturbation_type_counts": dict(type_counts),
            "cell_line_counts_meta": dict(cell_counts),
            "total_by_split": total_by_split,
            "trainonly_crossbg_v2_train_condition_min": min(train_condition_counts),
            "trainonly_crossbg_v2_train_condition_max": max(train_condition_counts),
            "trainonly_crossbg_v2_train_condition_total": sum(train_condition_counts),
            "datasets_with_obs_cell_type": [r["dataset"] for r in rows if r["obs_cell_type_n_unique"]],
            "datasets_with_drug_cov": [r["dataset"] for r in rows if r["obs_cov_drug_n_unique"]],
            "datasets_with_pathway": [r["dataset"] for r in rows if r["obs_pathway_n_unique"]],
        },
        "rows": rows,
        "interpretation": {
            "perturbation_count_scaling": "supported_by_existing_cap30_cap120_full_typebalanced_splits_but_current_model_gates_failed",
            "dataset_count_scaling": "not_yet_isolated_by_existing_splits; requires new leave-dataset-family-in/out protocol",
            "cell_background_scaling": "partly available but strongly confounded with dataset; Jiang has obs cell_type mixtures and sciplex has dataset-level cell lines",
            "perturbation_type_scaling": "available as metadata but confounded with dataset and prior type-balanced training failed",
        },
        "next_gate": {
            "name": "scaling_effect_protocol_gate",
            "requirements": [
                "define independent axes: dataset count, train condition count, cell/background count, perturbation type count",
                "use only train-only/internal validation for model selection",
                "avoid canonical multi in Track A selection",
                "include matched compute/steps and bootstrap CIs",
                "treat cell/background/type effects as confounded unless split design separates them",
            ],
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "dataset",
            "perturbation_type",
            "cell_line_meta",
            "n_cells_gt_stack",
            "obs_cell_type_n_unique",
            "obs_cov_drug_n_unique",
            "canonical_seed42_train",
            "canonical_seed42_test_single",
            "canonical_seed42_test_multi",
            "trainonly_crossbg_v2_train",
            "trainonly_crossbg_v2_internal_cross",
            "trainonly_crossbg_v2_internal_family",
            "cap30_all_v2_train",
            "cap120_all_v2_train",
            "type_balanced_cap120_v2_train",
            "general_exposure_cap_v2_train",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    lines = [
        "# LatentFM Scaling Metainfo Inventory",
        "",
        "Status: `scaling_metainfo_inventory_ready_no_gpu`",
        "",
        "## Boundary",
        "",
        "- Read only local metainfo JSON, split JSON, and `.h5ad.obs` in backed mode.",
        "- Did not load expression matrices, read canonical metric outcomes, read Track C query, or use GPU.",
        "",
        "## Summary",
        "",
        f"- datasets: `{len(rows)}`",
        f"- perturbation types: `{dict(type_counts)}`",
        f"- metadata cell lines/backgrounds: `{dict(cell_counts)}`",
        f"- train-only v2 total train conditions: `{payload['summary']['trainonly_crossbg_v2_train_condition_total']}`",
        f"- train-only v2 per-dataset train condition range: `{payload['summary']['trainonly_crossbg_v2_train_condition_min']}`-`{payload['summary']['trainonly_crossbg_v2_train_condition_max']}`",
        f"- datasets with obs cell_type: `{len(payload['summary']['datasets_with_obs_cell_type'])}`",
        f"- datasets with drug covariates: `{payload['summary']['datasets_with_drug_cov']}`",
        f"- datasets with pathway annotations: `{payload['summary']['datasets_with_pathway']}`",
        "",
        "## Split Totals",
        "",
        "| split | train | test | test_single | test_multi | internal_cross | internal_family |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, counts in total_by_split.items():
        lines.append(
            f"| {name} | {counts['train']} | {counts['test']} | {counts['test_single']} | "
            f"{counts['test_multi']} | {counts['internal_cross']} | {counts['internal_family']} |"
        )
    lines.extend(
        [
            "",
            "## Dataset Rows",
            "",
            "| dataset | type | cell/background | trainonly train | cap30 train | cap120 train | obs cell types | cells |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['perturbation_type']} | {r['cell_line_meta']} | "
            f"{r.get('trainonly_crossbg_v2_train', 0)} | {r.get('cap30_all_v2_train', 0)} | "
            f"{r.get('cap120_all_v2_train', 0)} | {r.get('obs_cell_type_n_unique', 0)} | "
            f"{r.get('n_cells_gt_stack', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Perturbation-count scaling is locally supported by cap30/cap120/full/type-balanced split artifacts, but the already-run model gates did not produce a deployable best model.",
            "- Dataset-count scaling is not isolated yet; existing splits mostly vary condition caps, not the number or family of datasets under matched compute.",
            "- Cell/background scaling is available but confounded: many backgrounds are dataset-level, while Jiang has mixed cell-type obs signatures that require careful handling.",
            "- Perturbation-type scaling is represented in metadata, but type is strongly confounded with dataset and the previous type-balanced cap120 branch failed.",
            "",
            "## Next Gate",
            "",
            "Design a `scaling_effect_protocol_gate` before new GPU: a matched-compute protocol that separates dataset count, condition count, background count, and perturbation-type count as much as the local metadata allows, with bootstrap/CI and no canonical multi selection.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- CSV: `{OUT_CSV}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_csv": str(OUT_CSV)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
