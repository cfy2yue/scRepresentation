#!/usr/bin/env python3
"""Build a train-only gene eval split from GSE92742 strict source overlap.

The split is for diagnostic outcome materialization only. It contains S0
`membership=train`, modality `gene` conditions that overlap GSE92742 small
metadata. It must not be used for canonical selection, canonical multi, or
Track C query.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
OVERLAP = ROOT / "reports/lincs_l1000_gse92742_condition_join_gate_20260627/gse92742_s0_overlap_rows.csv"
OUT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_lincs_gse92742_train_gene_eval_20260627.json"
OUT_JSON = ROOT / "reports/latentfm_lincs_gse92742_train_gene_eval_split_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_LINCS_GSE92742_TRAIN_GENE_EVAL_SPLIT_20260627.md"


def norm_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def main() -> int:
    missing = [str(OVERLAP)] if not OVERLAP.is_file() else []
    boundary = {
        "gpu_used": False,
        "training_or_inference_used": False,
        "canonical_multi_selection_used": False,
        "trackc_heldout_query_used": False,
        "source": "GSE92742_small_metadata_strict_train_gene_overlap",
    }
    if missing:
        payload = {
            "status": "lincs_gse92742_train_gene_eval_split_missing_source",
            "boundary": boundary,
            "missing": missing,
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        OUT_MD.write_text("# LINCS GSE92742 Train Gene Eval Split\n\nMissing source overlap.\n", encoding="utf-8")
        print(json.dumps({"status": payload["status"]}, indent=2))
        return 0

    by_ds: dict[str, set[str]] = defaultdict(set)
    overlap_rows = 0
    exact_bg_keys: set[tuple[str, str]] = set()
    lincs_type_counts: Counter[str] = Counter()
    with OVERLAP.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("membership") != "train" or row.get("modality") != "gene":
                continue
            dataset = norm_text(row.get("dataset"))
            condition = norm_text(row.get("condition"))
            if not dataset or not condition:
                continue
            overlap_rows += 1
            by_ds[dataset].add(condition)
            lincs_type_counts[norm_text(row.get("lincs_pert_type"))] += 1
            if norm_text(row.get("s0_cell_background")).lower() == norm_text(row.get("lincs_cell_id")).lower():
                exact_bg_keys.add((dataset, condition))

    split = {
        dataset: {
            "train": [],
            "test": sorted(conditions),
            "test_single": sorted(conditions),
            "lincs_gse92742_train_gene_eval": sorted(conditions),
        }
        for dataset, conditions in sorted(by_ds.items())
    }
    OUT_SPLIT.parent.mkdir(parents=True, exist_ok=True)
    OUT_SPLIT.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    condition_count = sum(len(v["test"]) for v in split.values())
    payload = {
        "status": "lincs_gse92742_train_gene_eval_split_ready_no_gpu",
        "boundary": boundary,
        "outputs": {
            "split": str(OUT_SPLIT),
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
        },
        "summary": {
            "overlap_rows": overlap_rows,
            "datasets": len(split),
            "conditions": condition_count,
            "dataset_counts": {ds: len(v["test"]) for ds, v in split.items()},
            "exact_background_conditions": len(exact_bg_keys),
            "lincs_type_counts": lincs_type_counts.most_common(),
        },
        "next_action": (
            "Use this split only for bounded eval-only outcome materialization "
            "of frozen checkpoints. It does not authorize training or promotion."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LINCS GSE92742 Train Gene Eval Split",
        "",
        "Status: `lincs_gse92742_train_gene_eval_split_ready_no_gpu`",
        "",
        "## Boundary",
        "",
        "- Built only from GSE92742 strict S0 `membership=train`, modality `gene` overlap.",
        "- No training, inference, GPU, canonical multi selection, or Track C query.",
        "- Intended use: bounded eval-only outcome materialization for a later CPU signal/control gate.",
        "",
        "## Summary",
        "",
        f"- overlap rows: `{overlap_rows}`",
        f"- datasets: `{len(split)}`",
        f"- conditions: `{condition_count}`",
        f"- exact-background conditions: `{len(exact_bg_keys)}`",
        f"- LINCS perturbation types: `{lincs_type_counts.most_common()}`",
        "",
        "## Output",
        "",
        f"- split: `{OUT_SPLIT}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "split": str(OUT_SPLIT), "conditions": condition_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
