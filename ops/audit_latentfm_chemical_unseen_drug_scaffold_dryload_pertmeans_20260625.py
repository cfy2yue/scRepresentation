#!/usr/bin/env python3
"""Dry-load and train-only pert-mean gate for chemical unseen-drug/scaffold splits."""

from __future__ import annotations

import importlib.util
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


DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
BIFLOW_DIR = ROOT / "dataset/biFlow_data"
SPLIT_JSON = ROOT / "reports/latentfm_chemical_unseen_drug_scaffold_loader_splits_20260625.json"
HELPER = ROOT / "ops/build_latentfm_xverse_scaling_splits_20260624.py"
OUT_DIR = ROOT / "runs/latentfm_chemical_unseen_drug_scaffold_splits_20260625/artifacts"
OUT_JSON = ROOT / "reports/latentfm_chemical_unseen_drug_scaffold_dryload_pertmeans_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_CHEMICAL_UNSEEN_DRUG_SCAFFOLD_DRYLOAD_PERTMEANS_20260625.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_helper():
    spec = importlib.util.spec_from_file_location("xverse_scaling_split_helper", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {HELPER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dryload(split: dict[str, Any], *, seed: int, max_batches: int = 3) -> dict[str, Any]:
    reasons: list[str] = []
    results: dict[str, Any] = {}
    for mode in ("train", "test"):
        ds = CrossDatasetFMDataset(
            str(DATA_DIR),
            split,
            batch_size=8,
            seed=seed,
            mode=mode,
            min_cells=8,
            ds_alpha=1.0,
            use_pert_condition=False,
            biflow_dir=str(BIFLOW_DIR),
            latent_backbone="xverse",
            perturbation_family_filter="all",
            silent=True,
        )
        checked = 0
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
                reasons.append(f"{mode}_unexpected_ndim")
            if src.shape[1] != 384 or gt.shape[1] != 384:
                reasons.append(f"{mode}_embedding_dim_not_384")
            if not ds_name or not cond_name:
                reasons.append(f"{mode}_empty_name")
            checked += 1
        results[mode] = {
            "total_conditions": int(ds.total_conditions),
            "epoch_steps": int(ds.epoch_steps),
            "batches_checked": checked,
        }
        if ds.total_conditions <= 0 or ds.epoch_steps <= 0:
            reasons.append(f"{mode}_empty_dataset")
    return {"status": "ok" if not reasons else "fail", "reasons": reasons[:20], "results": results}


def main() -> int:
    helper = load_helper()
    split_manifest = load_json(SPLIT_JSON)
    if split_manifest.get("status") != "chemical_unseen_drug_scaffold_loader_splits_ready_no_gpu":
        raise SystemExit(f"split manifest not ready: {split_manifest.get('status')}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in split_manifest["rows"]:
        mode = row["mode"]
        split_file = Path(row["split_file"])
        split = load_json(split_file)
        dry = dryload(split, seed=42)
        means, audit = helper.compute_train_pert_means(DATA_DIR, split)
        pert_file = OUT_DIR / f"{mode}_trainonly_pert_means.npz"
        np.savez_compressed(pert_file, **means)
        mean_status = "ok" if all(r.get("status") in {"ok", "empty_train_dataset"} for r in audit) else "check"
        rows.append(
            {
                "mode": mode,
                "split_file": str(split_file),
                "dryload": dry,
                "pert_means_file": str(pert_file),
                "n_datasets_with_means": len(means),
                "pert_mean_audit": audit,
                "status": "ok" if dry["status"] == "ok" and mean_status == "ok" else "fail",
            }
        )
    status = "chemical_unseen_drug_scaffold_dryload_pertmeans_ready_gpu_candidate" if all(r["status"] == "ok" for r in rows) else "chemical_unseen_drug_scaffold_dryload_pertmeans_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": status.endswith("gpu_candidate"),
        "boundary": {
            "task": "CPU-only dry-load and train-only pert means",
            "uses_training": False,
            "uses_model_outputs": False,
            "uses_canonical_multi": False,
            "uses_trackc_query": False,
            "pert_means_scope": "train split only",
        },
        "rows": rows,
        "next_action": "resource audit then launch bounded chemical unseen-drug/scaffold GPU smoke" if status.endswith("gpu_candidate") else "fix dryload/pertmeans failures",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Chemical Unseen-Drug/Scaffold Dryload + Pert Means",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only loader and train-only pert-mean gate.",
        "- No training, model outputs, canonical multi, or Track C query.",
        "",
        "| mode | status | train conds | test conds | datasets with means | pert means | reasons |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        train = row["dryload"]["results"]["train"]
        test = row["dryload"]["results"]["test"]
        reasons = ", ".join(row["dryload"].get("reasons") or []) or "none"
        lines.append(
            f"| `{row['mode']}` | `{row['status']}` | {train['total_conditions']} | {test['total_conditions']} | "
            f"{row['n_datasets_with_means']} | `{row['pert_means_file']}` | {reasons} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- `gpu_authorized`: `{payload['gpu_authorized']}`",
        f"- next action: {payload['next_action']}",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
