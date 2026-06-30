#!/usr/bin/env python3
"""Construct and audit a candidate multi-aware LatentFM split.

This script never overwrites the canonical split. It starts from
``split_seed42.json``, keeps canonical train singles, and creates a separate
candidate split where a small deterministic subset of existing multi-test
conditions is moved into ``train_multi``/``val_multi`` for a future multi
fine-tune protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
DEFAULT_METADATA = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
DEFAULT_OUT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_aware_v1.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_multi_aware_split_v1_audit_20260621.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_MULTI_AWARE_SPLIT_V1_AUDIT_20260621.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def deterministic_shuffle(values: list[str], key: str) -> list[str]:
    vals = list(values)
    vals.sort(key=lambda x: hashlib.sha256(f"{key}|{x}".encode("utf-8")).hexdigest())
    return vals


def genes_for(metadata: dict[str, Any], ds: str, cond: str) -> list[str]:
    meta = (metadata.get(ds) or {}).get(cond) or {}
    return [str(g).strip().upper() for g in meta.get("genes") or [] if str(g).strip()]


def is_multi(metadata: dict[str, Any], ds: str, cond: str) -> bool:
    return len(genes_for(metadata, ds, cond)) >= 2


def train_gene_sets(split_obj: dict[str, Any], metadata: dict[str, Any]) -> tuple[set[str], set[str]]:
    singles: set[str] = set()
    all_genes: set[str] = set()
    for cond in split_obj.get("train") or []:
        genes = genes_for(metadata, split_obj["_dataset"], cond)
        if len(genes) == 1:
            singles.add(genes[0])
        for gene in genes:
            all_genes.add(gene)
    for cond in split_obj.get("train_multi") or []:
        for gene in genes_for(metadata, split_obj["_dataset"], cond):
            all_genes.add(gene)
    return singles, all_genes


def classify_multi(
    metadata: dict[str, Any],
    ds: str,
    cond: str,
    same_train_genes: set[str],
    global_train_genes: set[str],
) -> str:
    genes = genes_for(metadata, ds, cond)
    if not genes:
        return "unknown"
    same_hits = sum(g in same_train_genes for g in genes)
    global_hits = sum(g in global_train_genes for g in genes)
    if same_hits == len(genes):
        return "test_multi_seen_combo"
    if same_hits > 0:
        return "test_multi_unseen1_or_more_same_background"
    if global_hits == len(genes):
        return "test_multi_cross_background_seen"
    return "test_multi_hard_unseen"


def choose_counts(n_multi: int, args: argparse.Namespace) -> tuple[int, int]:
    if n_multi < int(args.min_eligible_multi):
        return 0, 0
    train_n = max(int(args.min_train_multi), int(math.ceil(float(args.train_multi_frac) * n_multi)))
    train_n = min(train_n, int(math.floor(float(args.max_train_multi_frac) * n_multi)))
    val_n = max(int(args.min_val_multi), int(math.ceil(float(args.val_multi_frac) * n_multi)))
    max_val = max(0, n_multi - train_n - int(args.min_test_multi_remaining))
    val_n = min(val_n, max_val)
    if train_n <= 0 or n_multi - train_n - val_n < int(args.min_test_multi_remaining):
        return 0, 0
    return train_n, val_n


def construct(args: argparse.Namespace) -> dict[str, Any]:
    split = load_json(args.canonical_split)
    metadata = load_json(args.metadata_json)
    out_split: dict[str, Any] = {}
    audit_rows = []

    # Global train genes start from canonical train singles only.
    canonical_global_train_genes: set[str] = set()
    for ds, obj in split.items():
        for cond in obj.get("train") or []:
            genes = genes_for(metadata, str(ds), str(cond))
            if len(genes) == 1:
                canonical_global_train_genes.add(genes[0])

    for ds, obj in sorted(split.items()):
        ds = str(ds)
        canonical_train = list(obj.get("train") or [])
        test_single = list(obj.get("test_single") or [])
        multi_pool = [str(c) for c in obj.get("test_multi") or [] if is_multi(metadata, ds, str(c))]
        ordered = deterministic_shuffle(multi_pool, f"multi_aware_v1|{ds}|{args.seed}")
        train_n, val_n = choose_counts(len(ordered), args)
        train_multi = ordered[:train_n]
        val_multi = ordered[train_n : train_n + val_n]
        heldout_multi = ordered[train_n + val_n :]

        same_train_genes = set()
        for cond in canonical_train + train_multi:
            for gene in genes_for(metadata, ds, cond):
                same_train_genes.add(gene)
        global_train_genes = set(canonical_global_train_genes)
        for d2, obj2 in split.items():
            if str(d2) == ds:
                for cond in train_multi:
                    global_train_genes.update(genes_for(metadata, ds, cond))

        strata: dict[str, list[str]] = {
            "test_multi_seen_combo": [],
            "test_multi_unseen1_or_more_same_background": [],
            "test_multi_cross_background_seen": [],
            "test_multi_hard_unseen": [],
            "unknown": [],
        }
        for cond in heldout_multi:
            strata[classify_multi(metadata, ds, cond, same_train_genes, global_train_genes)].append(cond)

        out_obj = {
            "train": canonical_train + train_multi,
            "train_single": canonical_train,
            "train_multi": train_multi,
            "val_multi": val_multi,
            "test": test_single + heldout_multi,
            "test_single": test_single,
            "test_multi": heldout_multi,
            **strata,
            "source_canonical_test_multi": multi_pool,
        }
        out_split[ds] = out_obj
        audit_rows.append(
            {
                "dataset": ds,
                "canonical_train": len(canonical_train),
                "canonical_test_single": len(test_single),
                "canonical_test_multi": len(multi_pool),
                "train_multi": len(train_multi),
                "val_multi": len(val_multi),
                "heldout_multi": len(heldout_multi),
                "eligible_for_multi_finetune": bool(train_multi and heldout_multi),
                "strata_counts": {k: len(v) for k, v in strata.items()},
            }
        )

    return {
        "canonical_split": str(args.canonical_split),
        "metadata_json": str(args.metadata_json),
        "out_split": str(args.out_split),
        "seed": int(args.seed),
        "params": {
            "min_eligible_multi": int(args.min_eligible_multi),
            "train_multi_frac": float(args.train_multi_frac),
            "max_train_multi_frac": float(args.max_train_multi_frac),
            "min_train_multi": int(args.min_train_multi),
            "val_multi_frac": float(args.val_multi_frac),
            "min_val_multi": int(args.min_val_multi),
            "min_test_multi_remaining": int(args.min_test_multi_remaining),
        },
        "leakage_status": "candidate split uses condition metadata and canonical split only; no GT metrics or posthoc outcomes",
        "split": out_split,
        "audit_rows": audit_rows,
        "summary": summarize(audit_rows),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_datasets": len(rows),
        "n_multi_datasets": sum(r["canonical_test_multi"] > 0 for r in rows),
        "n_eligible_multi_finetune_datasets": sum(bool(r["eligible_for_multi_finetune"]) for r in rows),
        "total_train_multi": sum(int(r["train_multi"]) for r in rows),
        "total_val_multi": sum(int(r["val_multi"]) for r in rows),
        "total_heldout_multi": sum(int(r["heldout_multi"]) for r in rows),
    }


def render_md(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# LatentFM Multi-Aware Split v1 Audit 2026-06-21",
        "",
        "Status: `candidate_split_created_not_launched`",
        "",
        "This split is for future multi fine-tune experiments only. It does not replace canonical `split_seed42.json` and does not use outcome metrics.",
        "",
        "## Provenance",
        "",
        f"- canonical_split: `{payload['canonical_split']}`",
        f"- metadata_json: `{payload['metadata_json']}`",
        f"- out_split: `{payload['out_split']}`",
        f"- leakage_status: `{payload['leakage_status']}`",
        f"- params: `{payload['params']}`",
        "",
        "## Summary",
        "",
        f"- datasets: `{s['n_datasets']}`",
        f"- datasets with canonical multi: `{s['n_multi_datasets']}`",
        f"- eligible multi-finetune datasets: `{s['n_eligible_multi_finetune_datasets']}`",
        f"- total train_multi: `{s['total_train_multi']}`",
        f"- total val_multi: `{s['total_val_multi']}`",
        f"- total heldout_multi: `{s['total_heldout_multi']}`",
        "",
        "## Dataset Counts",
        "",
        "| dataset | canon train | test single | canon multi | train multi | val multi | heldout multi | eligible | heldout strata |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["audit_rows"]:
        strata = ",".join(f"{k}:{v}" for k, v in row["strata_counts"].items() if v)
        lines.append(
            f"| {row['dataset']} | {row['canonical_train']} | {row['canonical_test_single']} | "
            f"{row['canonical_test_multi']} | {row['train_multi']} | {row['val_multi']} | "
            f"{row['heldout_multi']} | {row['eligible_for_multi_finetune']} | {strata or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Use this only for a separate multi-aware/fine-tune track.",
            "- Main stage model selection remains canonical single/background/family_gene.",
            "- Launch GPU only after checking that eligible datasets and held-out strata are sufficient for the intended question.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out-split", type=Path, default=DEFAULT_OUT_SPLIT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-eligible-multi", type=int, default=20)
    parser.add_argument("--train-multi-frac", type=float, default=0.10)
    parser.add_argument("--max-train-multi-frac", type=float, default=0.20)
    parser.add_argument("--min-train-multi", type=int, default=5)
    parser.add_argument("--val-multi-frac", type=float, default=0.10)
    parser.add_argument("--min-val-multi", type=int, default=2)
    parser.add_argument("--min-test-multi-remaining", type=int, default=10)
    args = parser.parse_args()

    payload = construct(args)
    args.out_split.write_text(json.dumps(payload["split"], indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_json.write_text(json.dumps({k: v for k, v in payload.items() if k != "split"}, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_split": str(args.out_split), "out_json": str(args.out_json), "out_md": str(args.out_md), "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
