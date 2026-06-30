#!/usr/bin/env python3
"""Schema/provenance gate for true cell-count capped-H5 artifacts."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
MATERIALIZER_JSON = ROOT / "reports/latentfm_true_cell_count_capped_h5_materializer_gate_20260624.json"
BASE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
OUT_JSON = ROOT / "reports/latentfm_true_cell_count_capped_h5_schema_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_CAPPED_H5_SCHEMA_GATE_20260624.md"
FORBIDDEN_TRAIN_KEYS = {"test", "test_single", "test_multi", "test_multi_seen", "test_multi_unseen1", "test_multi_unseen2", "canonical_test_reference"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def decode_conditions(raw: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in raw]


def array_sha256(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def audit_h5(path: Path, split_groups: dict[str, Any], budget: int) -> dict[str, Any]:
    reasons = []
    train = {str(c) for c in split_groups.get("train") or []}
    eval_set = set()
    for key, values in split_groups.items():
        if key == "train" or not isinstance(values, list) or key == "canonical_test_reference":
            continue
        eval_set.update(str(c) for c in values)
    with h5py.File(path, "r") as h5:
        required = ["conditions", "ctrl/emb", "ctrl/offsets", "gt/emb", "gt/offsets"]
        missing = [key for key in required if key not in h5]
        if missing:
            return {"dataset": path.stem, "status": "fail", "reasons": [f"missing_h5_keys:{missing}"]}
        conds = decode_conditions(h5["conditions"][:])
        ctrl_shape = tuple(h5["ctrl/emb"].shape)
        gt_shape = tuple(h5["gt/emb"].shape)
        ctrl_offsets = h5["ctrl/offsets"][:]
        gt_offsets = h5["gt/offsets"][:]
    if ctrl_shape != gt_shape:
        reasons.append("ctrl_gt_shape_mismatch")
    if len(ctrl_offsets) != len(conds) + 1 or len(gt_offsets) != len(conds) + 1:
        reasons.append("offset_length_mismatch")
    if int(ctrl_offsets[-1]) != int(ctrl_shape[0]) or int(gt_offsets[-1]) != int(gt_shape[0]):
        reasons.append("offset_final_row_mismatch")
    cond_set = set(conds)
    missing_split = sorted((train | eval_set) - cond_set)
    extra = sorted(cond_set - (train | eval_set))
    if missing_split:
        reasons.append(f"missing_split_conditions:{len(missing_split)}")
    if extra:
        reasons.append(f"extra_conditions:{len(extra)}")
    low_or_wrong_train = []
    for idx, cond in enumerate(conds):
        n = int(gt_offsets[idx + 1] - gt_offsets[idx])
        if cond in train and n != int(budget):
            low_or_wrong_train.append((cond, n))
    if low_or_wrong_train:
        reasons.append(f"train_budget_row_mismatch:{len(low_or_wrong_train)}")
    return {
        "dataset": path.stem,
        "status": "ok" if not reasons else "fail",
        "reasons": reasons,
        "n_conditions": len(conds),
        "n_train_conditions": len(train),
        "n_eval_conditions": len(eval_set),
        "rows": int(gt_shape[0]),
        "emb_dim": int(gt_shape[1]) if len(gt_shape) == 2 else None,
        "train_budget_mismatches_preview": low_or_wrong_train[:10],
    }


def audit_sample_provenance(data_dir: Path, split: dict[str, Any], budget: int) -> dict[str, Any]:
    reasons = []
    jsonl_path = data_dir / "sampled_index_manifest.jsonl"
    npz_path = data_dir / "sampled_indices.npz"
    if not jsonl_path.exists() or not npz_path.exists():
        return {
            "status": "fail",
            "reasons": [
                f"missing_sample_provenance:{p}"
                for p in [jsonl_path, npz_path]
                if not p.exists()
            ],
        }
    rows = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    expected = {
        (str(ds), str(cond))
        for ds, groups in split.items()
        for cond in (groups.get("train") or [])
    }
    observed = {(str(r.get("dataset")), str(r.get("condition"))) for r in rows}
    if expected != observed:
        reasons.append(f"sample_manifest_condition_mismatch:expected{len(expected)}_observed{len(observed)}")
    duplicate_count = len(rows) - len(observed)
    if duplicate_count:
        reasons.append(f"sample_manifest_duplicate_rows:{duplicate_count}")
    with np.load(npz_path) as npz:
        npz_keys = set(npz.files)
        for row in rows:
            for group in ("gt", "ctrl"):
                key = str(row.get(f"{group}_key"))
                if key not in npz_keys:
                    reasons.append(f"sample_indices_missing_key:{key}")
                    continue
                arr = np.asarray(npz[key], dtype=np.int64)
                if int(arr.size) != int(budget) or int(row.get(f"{group}_sample_n", -1)) != int(budget):
                    reasons.append(f"sample_indices_wrong_budget:{row.get('dataset')}:{row.get('condition')}:{group}")
                if array_sha256(arr) != str(row.get(f"{group}_rel_sha256")):
                    reasons.append(f"sample_indices_hash_mismatch:{row.get('dataset')}:{row.get('condition')}:{group}")
    return {
        "status": "ok" if not reasons else "fail",
        "reasons": reasons[:20],
        "n_train_conditions": len(expected),
        "n_manifest_rows": len(rows),
        "sampled_indices_file": str(npz_path),
        "sampled_index_manifest_file": str(jsonl_path),
    }


def audit_row(row: dict[str, Any], base_split: dict[str, Any]) -> dict[str, Any]:
    data_dir = Path(row["data_dir"])
    split_file = Path(row["split_file"])
    budget = int(row["run_id"].split("_budget")[-1].split("_seed")[0])
    reasons = []
    for path in [
        data_dir / "manifest.json",
        data_dir / "condition_metadata.json",
        data_dir / "ctrl_means.npz",
        data_dir / "pert_means.npz",
        data_dir / "sampled_indices.npz",
        data_dir / "sampled_index_manifest.jsonl",
        split_file,
    ]:
        if not path.exists():
            reasons.append(f"missing:{path}")
    if reasons:
        return {"run_id": row["run_id"], "status": "fail", "reasons": reasons}
    split = load_json(split_file)
    manifest = load_json(data_dir / "manifest.json")
    manifest_datasets = manifest.get("datasets") or {}
    if not isinstance(manifest_datasets, dict):
        reasons.append("manifest_datasets_not_dict")
    train_overlap = []
    subset_violations = []
    forbidden_key_present = []
    h5_rows = []
    for ds, groups in split.items():
        base_groups = base_split.get(ds) or {}
        base_train = {str(c) for c in base_groups.get("train") or []}
        train = {str(c) for c in groups.get("train") or []}
        if not train.issubset(base_train):
            subset_violations.append(ds)
        for key in FORBIDDEN_TRAIN_KEYS:
            if key in groups and key == "canonical_test_reference":
                forbidden_key_present.append(f"{ds}:{key}")
        eval_conditions = set()
        for key, values in base_groups.items():
            if key == "train" or not isinstance(values, list):
                continue
            eval_conditions.update(str(c) for c in values)
        overlap = sorted(train & eval_conditions)
        if overlap:
            train_overlap.append({"dataset": ds, "n": len(overlap), "preview": overlap[:10]})
        h5_path = data_dir / f"{ds}.h5"
        if h5_path.exists():
            h5_audit = audit_h5(h5_path, groups, budget)
            h5_rows.append(h5_audit)
            manifest_meta = manifest_datasets.get(ds) if isinstance(manifest_datasets, dict) else None
            manifest_conditions = manifest_meta.get("conditions") if isinstance(manifest_meta, dict) else None
            if not isinstance(manifest_conditions, list):
                reasons.append(f"manifest_conditions_not_list:{ds}")
            elif h5_audit.get("n_conditions") != len(manifest_conditions):
                reasons.append(f"manifest_conditions_count_mismatch:{ds}")
    means_keys = {}
    for name in ["ctrl_means.npz", "pert_means.npz"]:
        with np.load(data_dir / name) as npz:
            means_keys[name] = sorted(npz.files)
    split_train_datasets = sorted(ds for ds, groups in split.items() if groups.get("train"))
    if means_keys.get("pert_means.npz") != split_train_datasets:
        reasons.append("pert_means_dataset_keys_do_not_match_train_datasets")
    if means_keys.get("ctrl_means.npz") != split_train_datasets:
        reasons.append("ctrl_means_dataset_keys_do_not_match_train_datasets")
    if subset_violations:
        reasons.append(f"train_not_subset_of_base_train:{subset_violations}")
    if train_overlap:
        reasons.append(f"train_overlaps_base_eval:{len(train_overlap)}")
    if forbidden_key_present:
        reasons.append(f"forbidden_split_keys_present:{forbidden_key_present[:5]}")
    failed_h5 = [r for r in h5_rows if r.get("status") != "ok"]
    if failed_h5:
        reasons.append(f"h5_schema_failures:{len(failed_h5)}")
    sample_provenance = audit_sample_provenance(data_dir, split, budget)
    if sample_provenance.get("status") != "ok":
        reasons.append(f"sample_provenance_fail:{';'.join(sample_provenance.get('reasons') or [])}")
    return {
        "run_id": row["run_id"],
        "status": "ok" if not reasons else "fail",
        "reasons": reasons,
        "data_dir": str(data_dir),
        "split_file": str(split_file),
        "n_h5_datasets": len(h5_rows),
        "h5_rows": h5_rows,
        "means_keys": means_keys,
        "train_overlap": train_overlap,
        "sample_provenance": sample_provenance,
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM True Cell-Count Capped-H5 Schema Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only artifact schema/provenance gate.",
        "- Does not read canonical metrics, canonical multi, Track C query, train, infer, or use GPU.",
        "",
        "## Rows",
        "",
        "| run id | status | h5 datasets | reasons |",
        "|---|---|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(f"| `{row['run_id']}` | `{row['status']}` | {row.get('n_h5_datasets', 0)} | {', '.join(row.get('reasons') or []) or 'none'} |")
    lines.extend(["", "## Decision", "", f"- GPU authorized: `{payload['gpu_authorized']}`", f"- next action: `{payload['next_action']}`", "", "## JSON", "", f"`{OUT_JSON}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    materializer = load_json(MATERIALIZER_JSON)
    rows = materializer.get("materialized_rows") or []
    base_split = load_json(BASE_SPLIT)
    audit_rows = [audit_row(row, base_split) for row in rows]
    if not rows:
        status = "capped_h5_schema_gate_not_ready_no_materialized_rows"
        next_action = "wait_for_materialization"
    elif all(row.get("status") == "ok" for row in audit_rows):
        status = "capped_h5_schema_gate_pass_no_gpu"
        next_action = "prepare count-only and dataset-id controls before any GPU"
    else:
        status = "capped_h5_schema_gate_fail_no_gpu"
        next_action = "fix artifact schema/provenance failures"
    payload = {
        "status": status,
        "boundary": {
            "cpu_only": True,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "uses_gpu": False,
        },
        "materializer_json": str(MATERIALIZER_JSON),
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
