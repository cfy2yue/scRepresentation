#!/usr/bin/env python3
"""Validate a LatentFM HDF5 bundle without loading large arrays into RAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _as_int(v: Any) -> int:
    return int(v.item() if hasattr(v, "item") else v)


def _load_manifest(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_means(data_dir: Path, name: str) -> dict[str, np.ndarray]:
    path = data_dir / name
    if not path.is_file():
        return {}
    obj = np.load(str(path))
    return {str(k): obj[k] for k in obj.files}


def _validate_h5(path: Path, manifest_ds: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    with h5py.File(str(path), "r") as f:
        ctrl_key = "ctrl/emb" if "ctrl/emb" in f else "ir/emb" if "ir/emb" in f else ""
        if not ctrl_key:
            errors.append("missing ctrl/emb")
        if "gt/emb" not in f:
            errors.append("missing gt/emb")
        if "conditions" not in f:
            errors.append("missing conditions")
        if ctrl_key and "ctrl/offsets" not in f and "ir/offsets" not in f:
            errors.append("missing ctrl/offsets")
        if "gt/offsets" not in f:
            errors.append("missing gt/offsets")
        if errors:
            return {}, errors

        ctrl_offsets_key = "ctrl/offsets" if "ctrl/offsets" in f else "ir/offsets"
        ctrl = f[ctrl_key]
        gt = f["gt/emb"]
        ctrl_offsets = f[ctrl_offsets_key][:]
        gt_offsets = f["gt/offsets"][:]
        conds = f["conditions"].asstr()[:].tolist()

        if ctrl.ndim != 2:
            errors.append(f"{ctrl_key} must be 2D, got shape={ctrl.shape}")
        if gt.ndim != 2:
            errors.append(f"gt/emb must be 2D, got shape={gt.shape}")
        if ctrl.ndim == 2 and gt.ndim == 2 and ctrl.shape[1] != gt.shape[1]:
            errors.append(f"embedding dim mismatch ctrl={ctrl.shape[1]} gt={gt.shape[1]}")
        if len(ctrl_offsets) != len(conds) + 1:
            errors.append("ctrl offsets length does not match conditions")
        if len(gt_offsets) != len(conds) + 1:
            errors.append("gt offsets length does not match conditions")
        if len(ctrl_offsets) and _as_int(ctrl_offsets[0]) != 0:
            errors.append("ctrl offsets must start at 0")
        if len(gt_offsets) and _as_int(gt_offsets[0]) != 0:
            errors.append("gt offsets must start at 0")
        if np.any(np.diff(ctrl_offsets) < 0):
            errors.append("ctrl offsets are not monotonic")
        if np.any(np.diff(gt_offsets) < 0):
            errors.append("gt offsets are not monotonic")
        if len(ctrl_offsets) and _as_int(ctrl_offsets[-1]) != int(ctrl.shape[0]):
            errors.append("ctrl offsets last value does not equal ctrl row count")
        if len(gt_offsets) and _as_int(gt_offsets[-1]) != int(gt.shape[0]):
            errors.append("gt offsets last value does not equal gt row count")

        expected_conditions = [str(c) for c in manifest_ds.get("conditions", [])]
        if expected_conditions and conds != expected_conditions:
            errors.append("conditions differ from manifest")
        for key, observed in (
            ("n_conds", len(conds)),
            ("n_src", int(ctrl.shape[0])),
            ("n_gt", int(gt.shape[0])),
        ):
            expected = manifest_ds.get(key)
            if expected is not None and int(expected) != observed:
                errors.append(f"{key} differs from manifest: expected={expected} observed={observed}")

        return {
            "conditions": len(conds),
            "ctrl_rows": int(ctrl.shape[0]),
            "gt_rows": int(gt.shape[0]),
            "emb_dim": int(gt.shape[1]) if gt.ndim == 2 else None,
            "ctrl_key": ctrl_key,
        }, errors


def validate_bundle(data_dir: Path, *, require_metadata: bool = False) -> dict[str, Any]:
    manifest = _load_manifest(data_dir)
    datasets = manifest.get("datasets", {})
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(f"manifest has no datasets: {data_dir / 'manifest.json'}")

    ctrl_means = _load_means(data_dir, "ctrl_means.npz")
    pert_means = _load_means(data_dir, "pert_means.npz")
    metadata_path = data_dir / "condition_metadata.json"
    metadata = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    errors: list[str] = []
    dataset_reports: dict[str, Any] = {}
    total_conditions = 0
    total_ctrl_rows = 0
    total_gt_rows = 0
    emb_dim = manifest.get("emb_dim")

    for ds_name, ds_meta in sorted(datasets.items()):
        h5_path = data_dir / f"{ds_name}.h5"
        if not h5_path.is_file():
            errors.append(f"{ds_name}: missing HDF5 file {h5_path.name}")
            continue
        report, ds_errors = _validate_h5(h5_path, ds_meta)
        if ds_errors:
            errors.extend([f"{ds_name}: {e}" for e in ds_errors])
        if report:
            dataset_reports[ds_name] = report
            total_conditions += int(report["conditions"])
            total_ctrl_rows += int(report["ctrl_rows"])
            total_gt_rows += int(report["gt_rows"])
            if emb_dim is None:
                emb_dim = int(report["emb_dim"])
            elif int(emb_dim) != int(report["emb_dim"]):
                errors.append(f"{ds_name}: emb_dim {report['emb_dim']} differs from manifest/global {emb_dim}")

        if ds_name not in ctrl_means:
            errors.append(f"{ds_name}: missing ctrl_means entry")
        elif report and tuple(ctrl_means[ds_name].shape) != (int(report["emb_dim"]),):
            errors.append(f"{ds_name}: ctrl_means shape {ctrl_means[ds_name].shape} != ({report['emb_dim']},)")
        if ds_name not in pert_means:
            errors.append(f"{ds_name}: missing pert_means entry")
        elif report and tuple(pert_means[ds_name].shape) != (int(report["emb_dim"]),):
            errors.append(f"{ds_name}: pert_means shape {pert_means[ds_name].shape} != ({report['emb_dim']},)")
        if require_metadata and ds_name not in metadata:
            errors.append(f"{ds_name}: missing condition_metadata entry")

    for key, observed in (
        ("total_conditions", total_conditions),
        ("total_src_cells", total_ctrl_rows),
        ("total_gt_cells", total_gt_rows),
    ):
        expected = manifest.get(key)
        if expected is not None and int(expected) != observed:
            errors.append(f"{key} differs from manifest: expected={expected} observed={observed}")

    return {
        "data_dir": str(data_dir),
        "ok": not errors,
        "errors": errors,
        "summary": {
            "datasets": len(dataset_reports),
            "conditions": total_conditions,
            "ctrl_rows": total_ctrl_rows,
            "gt_rows": total_gt_rows,
            "emb_dim": None if emb_dim is None else int(emb_dim),
            "has_condition_metadata": metadata_path.is_file(),
            "ctrl_means": len(ctrl_means),
            "pert_means": len(pert_means),
        },
        "datasets": dataset_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--require-metadata", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = validate_bundle(args.data_dir.expanduser().resolve(), require_metadata=args.require_metadata)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
