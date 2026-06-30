#!/usr/bin/env python3
"""Construct a train-only cross-background/family proxy validation split.

This split is for checkpoint/model selection only. It selects validation
conditions from canonical train single-gene conditions, preferring genes that
remain present in another dataset after the holdout. Canonical test conditions
are copied only to ``canonical_test_reference`` metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_CANONICAL_SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_OUT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
DEFAULT_ARTIFACT = (
    ROOT
    / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/"
    "xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz"
)
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_trainonly_crossbg_val_v2_audit_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_TRAINONLY_CROSSBG_VAL_V2_AUDIT_20260622.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_drug(entry: dict[str, Any], ds: str) -> bool:
    raw = str(entry.get("perturbation_type_raw", entry.get("perturbation_type", ""))).strip().lower()
    if raw in {"drug", "chemical", "compound", "small molecule", "small-molecule"}:
        return True
    return any(tok in ds.lower() for tok in ("sciplex", "chempert", "chemical", "drug"))


def genes_for(metadata: dict[str, Any], ds: str, cond: str) -> list[str]:
    entry = (metadata.get(ds) or {}).get(cond) or {}
    return [str(g).strip().upper() for g in entry.get("genes") or [] if str(g).strip()]


def stable_score(seed: int, ds: str, cond: str) -> str:
    return hashlib.sha256(f"{seed}\t{ds}\t{cond}".encode("utf-8")).hexdigest()


def train_single_records(split: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for ds, groups in sorted(split.items()):
        for cond in groups.get("train") or []:
            cond_s = str(cond)
            entry = (metadata.get(ds) or {}).get(cond_s) or {}
            genes = genes_for(metadata, ds, cond_s)
            if len(genes) != 1 or is_drug(entry, ds):
                continue
            rows.append({"dataset": str(ds), "condition": cond_s, "gene": genes[0]})
    return rows


def gene_dataset_map(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        out[row["gene"]].add(row["dataset"])
    return out


def construct_split(
    split: dict[str, Any],
    metadata: dict[str, Any],
    *,
    seed: int,
    val_fraction: float,
    min_val_per_dataset: int,
    max_val_per_dataset: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    all_single = train_single_records(split, metadata)
    gene_ds = gene_dataset_map(all_single)
    cross_candidates_by_ds: dict[str, list[str]] = defaultdict(list)
    family_candidates_by_ds: dict[str, list[str]] = defaultdict(list)
    gene_by_key = {(r["dataset"], r["condition"]): r["gene"] for r in all_single}
    for row in all_single:
        ds = row["dataset"]
        cond = row["condition"]
        family_candidates_by_ds[ds].append(cond)
        if len(gene_ds[row["gene"]] - {ds}) > 0:
            cross_candidates_by_ds[ds].append(cond)

    selected_by_ds: dict[str, list[str]] = {}
    for ds, groups in sorted(split.items()):
        cross = sorted(cross_candidates_by_ds.get(str(ds), []), key=lambda c: stable_score(seed, str(ds), c))
        family = sorted(family_candidates_by_ds.get(str(ds), []), key=lambda c: stable_score(seed + 17, str(ds), c))
        base_n = int(math.ceil(len(family) * float(val_fraction))) if family else 0
        target_n = min(int(max_val_per_dataset), max(int(min_val_per_dataset), base_n))
        target_n = min(target_n, len(family))
        chosen = list(cross[: min(target_n, len(cross))])
        if len(chosen) < target_n:
            chosen_set = set(chosen)
            chosen.extend([c for c in family if c not in chosen_set][: target_n - len(chosen)])
        selected_by_ds[str(ds)] = sorted(chosen)

    heldout = {(ds, cond) for ds, conds in selected_by_ds.items() for cond in conds}
    remaining_gene_ds: dict[str, set[str]] = defaultdict(set)
    for row in all_single:
        key = (row["dataset"], row["condition"])
        if key in heldout:
            continue
        remaining_gene_ds[row["gene"]].add(row["dataset"])

    out: dict[str, Any] = {}
    audit_rows = []
    total_cross_proxy = 0
    total_family_only = 0
    for ds, groups in sorted(split.items()):
        ds_s = str(ds)
        train = [str(c) for c in groups.get("train") or []]
        canonical_test = [str(c) for c in groups.get("test") or []]
        val = selected_by_ds.get(ds_s, [])
        val_set = set(val)
        cross_proxy = []
        family_only = []
        for cond in val:
            gene = gene_by_key.get((ds_s, cond))
            if gene and len(remaining_gene_ds.get(gene, set()) - {ds_s}) > 0:
                cross_proxy.append(cond)
            else:
                family_only.append(cond)
        train_new = [c for c in train if c not in val_set]
        out[ds_s] = {
            "train": train_new,
            "test": val,
            "test_single": val,
            "internal_val_cross_background_seen_gene_proxy": sorted(cross_proxy),
            "internal_val_family_gene_proxy": val,
            "canonical_test_reference": canonical_test,
        }
        total_cross_proxy += len(cross_proxy)
        total_family_only += len(family_only)
        audit_rows.append(
            {
                "dataset": ds_s,
                "canonical_train": len(train),
                "canonical_test_reference": len(canonical_test),
                "train_single_family_candidates": len(family_candidates_by_ds.get(ds_s, [])),
                "train_single_cross_candidates_pre": len(cross_candidates_by_ds.get(ds_s, [])),
                "internal_val_total": len(val),
                "internal_val_cross_proxy_post": len(cross_proxy),
                "internal_val_family_only_post": len(family_only),
                "new_train": len(train_new),
            }
        )

    summary = {
        "total_internal_val": int(sum(len(v) for v in selected_by_ds.values())),
        "total_cross_proxy_post": int(total_cross_proxy),
        "total_family_only_post": int(total_family_only),
        "datasets_with_val": int(sum(1 for vals in selected_by_ds.values() if vals)),
        "datasets_with_cross_proxy_post": int(
            sum(1 for row in audit_rows if row["internal_val_cross_proxy_post"] > 0)
        ),
        "max_dataset_val_fraction": float(
            max((row["internal_val_total"] for row in audit_rows), default=0)
            / max(1, sum(row["internal_val_total"] for row in audit_rows))
        ),
    }
    reasons = []
    if summary["datasets_with_cross_proxy_post"] < 16:
        reasons.append("cross_proxy_dataset_coverage_lt_16")
    if summary["total_cross_proxy_post"] < 100:
        reasons.append("cross_proxy_conditions_lt_100")
    if summary["max_dataset_val_fraction"] > 0.25:
        reasons.append("single_dataset_internal_val_fraction_gt_0.25")
    summary["gate_status"] = "pass_cpu_split_gate" if not reasons else "fail_cpu_split_gate"
    summary["gate_reasons"] = reasons
    return out, audit_rows, summary


def condition_index_map(h5: h5py.File) -> dict[str, int]:
    conds = h5["conditions"][:]
    out = {}
    for idx, cond in enumerate(conds):
        key = cond.decode("utf-8") if isinstance(cond, bytes) else str(cond)
        out[key] = int(idx)
    return out


def compute_train_pert_means(data_dir: Path, split: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    means: dict[str, np.ndarray] = {}
    audit = []
    for ds, groups in sorted(split.items()):
        h5_path = data_dir / f"{ds}.h5"
        if not h5_path.is_file():
            raise FileNotFoundError(f"missing dataset H5: {h5_path}")
        train = [str(c) for c in groups.get("train") or []]
        with h5py.File(h5_path, "r") as h5:
            cmap = condition_index_map(h5)
            offsets = h5["gt/offsets"][:]
            emb = h5["gt/emb"]
            total = None
            n_cells = 0
            used = 0
            missing = []
            for cond in train:
                idx = cmap.get(cond)
                if idx is None:
                    missing.append(cond)
                    continue
                lo = int(offsets[idx])
                hi = int(offsets[idx + 1])
                if hi <= lo:
                    continue
                arr = np.asarray(emb[lo:hi], dtype=np.float64)
                total = arr.sum(axis=0, dtype=np.float64) if total is None else total + arr.sum(axis=0, dtype=np.float64)
                n_cells += int(arr.shape[0])
                used += 1
            if total is None or n_cells <= 0:
                raise ValueError(f"no train GT cells found for {ds}")
            means[ds] = (total / float(n_cells)).astype(np.float32)
            audit.append(
                {
                    "dataset": ds,
                    "train_conditions_used": used,
                    "train_cells_used": n_cells,
                    "n_missing_conditions": len(missing),
                    "missing_conditions": missing[:10],
                }
            )
    return means, audit


def render_md(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# LatentFM xverse Train-Only Cross-Background Validation Split v2 Audit",
        "",
        f"Status: `{s['gate_status']}`",
        "",
        "## Provenance",
        "",
        f"- canonical split: `{payload['canonical_split']}`",
        f"- output split: `{payload['output_split']}`",
        f"- train-only pert means: `{payload['pert_means_file']}`",
        f"- data dir: `{payload['data_dir']}`",
        f"- condition metadata: `{payload['condition_metadata']}`",
        f"- seed: `{payload['seed']}`",
        "",
        "Leakage guard:",
        "- validation conditions are selected only from canonical train single-gene conditions;",
        "- cross-background proxy labels are recomputed after validation holdout;",
        "- canonical test is copied only into `canonical_test_reference` and is not used for selection;",
        "- pert means are recomputed only from the new split's `train` conditions.",
        "",
        "## Gate Summary",
        "",
        f"- total internal val: {s['total_internal_val']}",
        f"- cross-background proxy after holdout: {s['total_cross_proxy_post']}",
        f"- family-only after holdout: {s['total_family_only_post']}",
        f"- datasets with val: {s['datasets_with_val']}",
        f"- datasets with cross-background proxy: {s['datasets_with_cross_proxy_post']}",
        f"- max dataset validation fraction: {s['max_dataset_val_fraction']:.3f}",
        "",
    ]
    if s["gate_reasons"]:
        lines.append("Gate reasons:")
        for reason in s["gate_reasons"]:
            lines.append(f"- `{reason}`")
        lines.append("")
    lines += [
        "## Split Counts",
        "",
        "| dataset | canonical train | family candidates | cross candidates pre | val total | cross proxy post | family only post | new train | canonical test ref |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["split_audit"]:
        lines.append(
            f"| {row['dataset']} | {row['canonical_train']} | "
            f"{row['train_single_family_candidates']} | {row['train_single_cross_candidates_pre']} | "
            f"{row['internal_val_total']} | {row['internal_val_cross_proxy_post']} | "
            f"{row['internal_val_family_only_post']} | {row['new_train']} | "
            f"{row['canonical_test_reference']} |"
        )
    lines += [
        "",
        "## Pert-Mean Counts",
        "",
        "| dataset | train conditions used | train cells used | missing conditions |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["pert_mean_audit"]:
        lines.append(
            f"| {row['dataset']} | {row['train_conditions_used']} | "
            f"{row['train_cells_used']} | {row['n_missing_conditions']} |"
        )
    lines += [
        "",
        "Decision:",
        "- `pass_cpu_split_gate` means this split may be used for a single future train-only selection smoke.",
        "- It still does not justify any canonical claim until a frozen checkpoint passes canonical posthoc.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-split", type=Path, default=DEFAULT_CANONICAL_SPLIT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-split", type=Path, default=DEFAULT_OUT_SPLIT)
    parser.add_argument("--pert-means-file", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.08)
    parser.add_argument("--min-val-per-dataset", type=int, default=1)
    parser.add_argument("--max-val-per-dataset", type=int, default=24)
    args = parser.parse_args()

    metadata_path = args.data_dir / "condition_metadata.json"
    split = load_json(args.canonical_split)
    metadata = load_json(metadata_path)
    out_split, split_audit, summary = construct_split(
        split,
        metadata,
        seed=int(args.seed),
        val_fraction=float(args.val_fraction),
        min_val_per_dataset=int(args.min_val_per_dataset),
        max_val_per_dataset=int(args.max_val_per_dataset),
    )
    means, pert_mean_audit = compute_train_pert_means(args.data_dir, out_split)

    args.out_split.parent.mkdir(parents=True, exist_ok=True)
    args.pert_means_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_split.write_text(json.dumps(out_split, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(args.pert_means_file, **means)

    payload = {
        "canonical_split": str(args.canonical_split),
        "output_split": str(args.out_split),
        "pert_means_file": str(args.pert_means_file),
        "data_dir": str(args.data_dir),
        "condition_metadata": str(metadata_path),
        "seed": int(args.seed),
        "val_fraction": float(args.val_fraction),
        "min_val_per_dataset": int(args.min_val_per_dataset),
        "max_val_per_dataset": int(args.max_val_per_dataset),
        "summary": summary,
        "split_audit": split_audit,
        "pert_mean_audit": pert_mean_audit,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_md": str(args.out_md), "status": summary["gate_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
