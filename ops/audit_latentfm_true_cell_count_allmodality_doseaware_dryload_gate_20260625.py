#!/usr/bin/env python3
"""Dry-load gate for dose-aware all-modality capped-H5 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
COUPLED = ROOT / "CoupledFM"
if str(COUPLED) not in sys.path:
    sys.path.insert(0, str(COUPLED))

from model.latent.dataset import CrossDatasetFMDataset  # noqa: E402


MATERIALIZER_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_materializer_gate_20260625.json"
SCHEMA_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_schema_gate_20260625.json"
BIFLOW_DIR = ROOT / "dataset/biFlow_data"
OUT_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_dryload_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_ALLMODALITY_DOSEAWARE_DRYLOAD_GATE_20260625.md"


def load_json(path: Path) -> Any:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_split_for_loader(split: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for ds, groups in split.items():
        out[str(ds)] = {
            "train": [str(c) for c in groups.get("train") or []],
            "test": [str(c) for c in groups.get("internal_val_allmodality_doseaware") or groups.get("test") or []],
        }
    return out


def dryload_row(row: dict[str, Any], *, max_batches: int) -> dict[str, Any]:
    reasons: list[str] = []
    data_dir = Path(row["data_dir"])
    split_file = Path(row["split_file"])
    for path in [data_dir / "manifest.json", data_dir / "pert_means.npz", data_dir / "ctrl_means.npz", split_file]:
        if not path.exists():
            reasons.append(f"missing:{path.name}")
    if reasons:
        return {"run_id": row["run_id"], "status": "fail", "reasons": reasons}
    raw_split = load_json(split_file)
    split = normalize_split_for_loader(raw_split)
    with np.load(data_dir / "pert_means.npz") as npz:
        means_keys = set(npz.files)
    train_datasets = {ds for ds, groups in split.items() if groups.get("train")}
    if means_keys != train_datasets:
        reasons.append(f"pert_means_keys_mismatch:{len(means_keys)}_vs_{len(train_datasets)}")
    results: dict[str, Any] = {}
    for mode, min_cells in [("train", 8), ("test", 8)]:
        ds = CrossDatasetFMDataset(
            str(data_dir),
            split,
            batch_size=8,
            seed=int(row.get("seed", 42)),
            mode=mode,
            min_cells=min_cells,
            ds_alpha=1.0,
            use_pert_condition=False,
            biflow_dir=str(BIFLOW_DIR),
            latent_backbone="xverse",
            perturbation_family_filter="all",
            silent=True,
        )
        if ds.total_conditions <= 0:
            reasons.append(f"{mode}_dataset_empty")
            results[mode] = {"total_conditions": ds.total_conditions, "epoch_steps": ds.epoch_steps, "batches_checked": 0}
            continue
        batches_checked = 0
        iterator = iter(ds)
        for _ in range(max_batches):
            batch = next(iterator)
            if len(batch) not in {4, 5}:
                reasons.append(f"{mode}_unexpected_batch_len:{len(batch)}")
                break
            src, gt, ds_name, cond_name = batch[:4]
            if tuple(src.shape) != tuple(gt.shape):
                reasons.append(f"{mode}_src_gt_shape_mismatch")
            if src.ndim != 2 or gt.ndim != 2:
                reasons.append(f"{mode}_unexpected_tensor_ndim")
            if src.shape[1] != 384 or gt.shape[1] != 384:
                reasons.append(f"{mode}_embedding_dim_not_384")
            if not ds_name or not cond_name:
                reasons.append(f"{mode}_empty_dataset_or_condition_name")
            batches_checked += 1
        results[mode] = {
            "total_conditions": ds.total_conditions,
            "epoch_steps": ds.epoch_steps,
            "batches_checked": batches_checked,
        }
    return {
        "run_id": row["run_id"],
        "status": "ok" if not reasons else "fail",
        "reasons": reasons[:20],
        "data_dir": str(data_dir),
        "split_file": str(split_file),
        "results": results,
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM All-Modality Dose-Aware Dry-Load Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only structural loader compatibility gate.",
        "- Instantiates `CrossDatasetFMDataset` with `use_pert_condition=False`.",
        "- Does not train, infer, read canonical metrics, read canonical multi, read held-out Track C query, or use GPU.",
        "",
        "## Rows",
        "",
        "| run id | status | train conds | test conds | reasons |",
        "|---|---|---:|---:|---|",
    ]
    for row in payload["rows"]:
        train = (row.get("results") or {}).get("train") or {}
        test = (row.get("results") or {}).get("test") or {}
        lines.append(
            f"| `{row['run_id']}` | `{row['status']}` | {train.get('total_conditions', 0)} | {test.get('total_conditions', 0)} | {', '.join(row.get('reasons') or []) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- GPU authorized: `{payload['gpu_authorized']}`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-run-id", default="")
    ap.add_argument("--max-batches", type=int, default=2)
    args = ap.parse_args()
    materializer = load_json(MATERIALIZER_JSON)
    schema = load_json(SCHEMA_JSON)
    rows = materializer.get("materialized_rows") or []
    if args.only_run_id:
        rows = [row for row in rows if row.get("run_id") == args.only_run_id]
        if not rows:
            raise SystemExit(f"run id is not materialized: {args.only_run_id}")
    audit_rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    if not materializer.get("materialized"):
        reasons.append("materializer_not_materialized")
    if schema.get("status") != "allmodality_doseaware_schema_pass_no_gpu":
        reasons.append(f"schema_gate_not_pass:{schema.get('status')}")
    if not rows:
        reasons.append("no_materialized_rows")
    if not reasons:
        audit_rows = [dryload_row(row, max_batches=max(1, int(args.max_batches))) for row in rows]
    if reasons:
        status = "allmodality_doseaware_dryload_not_ready_no_gpu"
        next_action = "run after materialization and schema gate pass"
    elif all(row.get("status") == "ok" for row in audit_rows):
        status = "allmodality_doseaware_dryload_pass_no_gpu"
        next_action = "run design controls and chemical-conditioning dryload before any GPU"
    else:
        status = "allmodality_doseaware_dryload_fail_no_gpu"
        next_action = "fix loader compatibility failures"
    payload = {
        "status": status,
        "reasons": reasons,
        "materializer_json": str(MATERIALIZER_JSON),
        "schema_json": str(SCHEMA_JSON),
        "rows": audit_rows,
        "gpu_authorized": False,
        "next_action": next_action,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
