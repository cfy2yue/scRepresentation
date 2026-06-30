#!/usr/bin/env python3
"""Construct a train-only single-gene validation split and pert-mean artifact.

The output split is for checkpoint/model selection only. It holds out a small,
deterministic subset of canonical train single-gene conditions as the split
``test`` group, leaving canonical test untouched for final reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_CANONICAL_SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_OUT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_single_val_v1.json"
DEFAULT_ARTIFACT = (
    ROOT
    / "runs/latentfm_xverse_trainonly_single_val_20260622/artifacts/"
    "xverse_trainonly_pert_means_split_seed42_singleval_v1.npz"
)
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_trainonly_single_val_split_v1_audit_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_TRAINONLY_SINGLE_VAL_SPLIT_V1_AUDIT_20260622.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_drug(entry: dict[str, Any], ds: str) -> bool:
    raw = str(entry.get("perturbation_type_raw", entry.get("perturbation_type", ""))).strip().lower()
    if raw in {"drug", "chemical", "compound", "small molecule", "small-molecule"}:
        return True
    dsl = ds.lower()
    return any(tok in dsl for tok in ("sciplex", "chempert", "chemical", "drug"))


def genes_for(metadata: dict[str, Any], ds: str, cond: str) -> list[str]:
    entry = (metadata.get(ds) or {}).get(cond) or {}
    return [str(g).strip().upper() for g in entry.get("genes") or [] if str(g).strip()]


def stable_score(seed: int, ds: str, cond: str) -> str:
    return hashlib.sha256(f"{seed}\t{ds}\t{cond}".encode("utf-8")).hexdigest()


def construct_split(
    split: dict[str, Any],
    metadata: dict[str, Any],
    *,
    seed: int,
    val_fraction: float,
    min_val_per_dataset: int,
    max_val_per_dataset: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    out: dict[str, Any] = {}
    audit_rows = []
    for ds, groups in sorted(split.items()):
        train = [str(c) for c in groups.get("train") or []]
        canonical_test = [str(c) for c in groups.get("test") or []]
        single_candidates = []
        for cond in train:
            entry = (metadata.get(ds) or {}).get(cond) or {}
            genes = genes_for(metadata, ds, cond)
            if len(genes) == 1 and not is_drug(entry, ds):
                single_candidates.append(cond)
        single_candidates = sorted(single_candidates, key=lambda c: stable_score(seed, ds, c))
        if not single_candidates:
            val = []
        else:
            n_val = int(math.ceil(len(single_candidates) * float(val_fraction)))
            n_val = max(int(min_val_per_dataset), n_val)
            n_val = min(int(max_val_per_dataset), n_val, len(single_candidates))
            val = sorted(single_candidates[:n_val])
        val_set = set(val)
        train_new = [c for c in train if c not in val_set]
        out[ds] = {
            "train": train_new,
            "test": val,
            "test_single": val,
            "internal_val_from_canonical_train_single": val,
            "canonical_test_reference": canonical_test,
        }
        audit_rows.append(
            {
                "dataset": ds,
                "canonical_train": len(train),
                "canonical_test_reference": len(canonical_test),
                "train_single_candidates": len(single_candidates),
                "internal_val_single": len(val),
                "new_train": len(train_new),
            }
        )
    return out, audit_rows


def condition_index_map(h5: h5py.File) -> dict[str, int]:
    conds = h5["conditions"][:]
    out = {}
    for idx, cond in enumerate(conds):
        if isinstance(cond, bytes):
            key = cond.decode("utf-8")
        else:
            key = str(cond)
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
                if total is None:
                    total = arr.sum(axis=0, dtype=np.float64)
                else:
                    total += arr.sum(axis=0, dtype=np.float64)
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
                    "missing_conditions": missing[:10],
                    "n_missing_conditions": len(missing),
                }
            )
    return means, audit


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Train-Only Single-Val Split v1 Audit 2026-06-22",
        "",
        "Status: `pass_train_only_selection_artifact`",
        "",
        "## Provenance",
        "",
        f"- canonical split: `{payload['canonical_split']}`",
        f"- output split: `{payload['output_split']}`",
        f"- train-only pert means: `{payload['pert_means_file']}`",
        f"- condition metadata: `{payload['condition_metadata']}`",
        f"- seed: `{payload['seed']}`",
        "",
        "Leakage guard:",
        "- internal validation conditions are selected only from canonical train single-gene conditions;",
        "- canonical test is preserved only as `canonical_test_reference` metadata and is not used for training-time selection;",
        "- `pert_means_file` is computed only from the new split's `train` conditions.",
        "",
        "## Split Counts",
        "",
        "| dataset | canonical train | single candidates | internal val single | new train | canonical test ref |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["split_audit"]:
        lines.append(
            f"| {row['dataset']} | {row['canonical_train']} | {row['train_single_candidates']} | "
            f"{row['internal_val_single']} | {row['new_train']} | {row['canonical_test_reference']} |"
        )
    lines += [
        "",
        "## Pert-Mean Artifact Counts",
        "",
        "| dataset | train conditions used | train cells used | missing conditions |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["pert_mean_audit"]:
        lines.append(
            f"| {row['dataset']} | {row['train_conditions_used']} | {row['train_cells_used']} | "
            f"{row['n_missing_conditions']} |"
        )
    lines += [""]
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
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--min-val-per-dataset", type=int, default=1)
    parser.add_argument("--max-val-per-dataset", type=int, default=32)
    args = parser.parse_args()

    metadata_path = args.data_dir / "condition_metadata.json"
    split = load_json(args.canonical_split)
    metadata = load_json(metadata_path)
    out_split, split_audit = construct_split(
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
        "split_audit": split_audit,
        "pert_mean_audit": pert_mean_audit,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_md": str(args.out_md), "pert_means_file": str(args.pert_means_file)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
