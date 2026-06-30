#!/usr/bin/env python3
"""Schema/provenance gate for dose-aware all-modality capped-H5 artifacts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
MATERIALIZER_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_materializer_gate_20260625.json"
FEASIBILITY_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_feasibility_gate_20260625.json"
OUT_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_schema_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_ALLMODALITY_DOSEAWARE_SCHEMA_GATE_20260625.md"

SCIPLEX_DATASETS = {"sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7"}


def load_json(path: Path) -> Any:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def decode_conditions(raw: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in raw]


def audit_h5(path: Path, split_groups: dict[str, Any], budget: int) -> dict[str, Any]:
    reasons: list[str] = []
    if not path.exists():
        return {"dataset": path.stem, "status": "fail", "reasons": ["h5_missing"]}
    train = {str(c) for c in split_groups.get("train") or []}
    eval_set = {str(c) for c in split_groups.get("internal_val_allmodality_doseaware") or []}
    with h5py.File(path, "r") as h5:
        required = ["conditions", "ctrl/emb", "ctrl/offsets", "gt/emb", "gt/offsets"]
        missing = [key for key in required if key not in h5]
        if missing:
            return {"dataset": path.stem, "status": "fail", "reasons": [f"missing_h5_keys:{missing}"]}
        conds = decode_conditions(h5["conditions"][:])
        gt_shape = tuple(h5["gt/emb"].shape)
        ctrl_shape = tuple(h5["ctrl/emb"].shape)
        gt_offsets = np.asarray(h5["gt/offsets"][:], dtype=np.int64)
        ctrl_offsets = np.asarray(h5["ctrl/offsets"][:], dtype=np.int64)
    if gt_shape != ctrl_shape:
        reasons.append("gt_ctrl_shape_mismatch")
    if len(gt_shape) != 2 or gt_shape[1] != 384:
        reasons.append("embedding_dim_not_384")
    if len(gt_offsets) != len(conds) + 1 or len(ctrl_offsets) != len(conds) + 1:
        reasons.append("offset_length_mismatch")
    if int(gt_offsets[-1]) != int(gt_shape[0]) or int(ctrl_offsets[-1]) != int(ctrl_shape[0]):
        reasons.append("offset_final_row_mismatch")
    if np.any(np.diff(gt_offsets) <= 0) or np.any(np.diff(ctrl_offsets) <= 0):
        reasons.append("nonpositive_condition_row_count")
    cond_set = set(conds)
    expected = train | eval_set
    missing_split = sorted(expected - cond_set)
    extra = sorted(cond_set - expected)
    if missing_split:
        reasons.append(f"missing_split_conditions:{len(missing_split)}")
    if extra:
        reasons.append(f"extra_conditions:{len(extra)}")
    train_bad = []
    eval_zero = []
    for idx, cond in enumerate(conds):
        n = int(gt_offsets[idx + 1] - gt_offsets[idx])
        if cond in train and n != int(budget):
            train_bad.append((cond, n))
        if cond in eval_set and n <= 0:
            eval_zero.append(cond)
    if train_bad:
        reasons.append(f"train_budget_mismatch:{len(train_bad)}")
    if eval_zero:
        reasons.append(f"eval_zero_rows:{len(eval_zero)}")
    return {
        "dataset": path.stem,
        "status": "ok" if not reasons else "fail",
        "reasons": reasons,
        "n_conditions": len(conds),
        "n_train_conditions": len(train),
        "n_eval_conditions": len(eval_set),
        "rows": int(gt_shape[0]) if gt_shape else 0,
        "emb_dim": int(gt_shape[1]) if len(gt_shape) == 2 else None,
        "train_budget_mismatch_preview": train_bad[:5],
    }


def audit_materialized_row(row: dict[str, Any]) -> dict[str, Any]:
    data_dir = Path(row["data_dir"])
    split_file = Path(row["split_file"])
    budget = int(row["budget"])
    reasons: list[str] = []
    required = [
        data_dir / "manifest.json",
        data_dir / "ctrl_means.npz",
        data_dir / "pert_means.npz",
        data_dir / "sampled_indices.npz",
        data_dir / "sampled_indices_summary.json.gz",
        split_file,
    ]
    for path in required:
        if not path.exists():
            reasons.append(f"missing:{path.name}")
    if reasons:
        return {"run_id": row["run_id"], "status": "fail", "reasons": reasons, "data_dir": str(data_dir)}

    split = load_json(split_file)
    manifest = load_json(data_dir / "manifest.json")
    with gzip.open(data_dir / "sampled_indices_summary.json.gz", "rt", encoding="utf-8") as handle:
        sampled_summary = json.load(handle)
    h5_rows = []
    modality_counts = {"train_gene": 0, "eval_gene": 0, "train_chemical": 0, "eval_chemical": 0}
    for ds, groups in sorted(split.items()):
        if "canonical_test_reference" in groups:
            reasons.append(f"canonical_reference_key_present:{ds}")
        is_chem = ds in SCIPLEX_DATASETS
        modality_counts["train_chemical" if is_chem else "train_gene"] += len(groups.get("train") or [])
        modality_counts["eval_chemical" if is_chem else "eval_gene"] += len(groups.get("internal_val_allmodality_doseaware") or [])
        h5_rows.append(audit_h5(data_dir / f"{ds}.h5", groups, budget))
        if ds not in sampled_summary:
            reasons.append(f"sampled_summary_missing_dataset:{ds}")
    if any(v <= 0 for v in modality_counts.values()):
        reasons.append(f"empty_modality_count:{modality_counts}")
    failed_h5 = [r for r in h5_rows if r.get("status") != "ok"]
    if failed_h5:
        reasons.append(f"h5_failures:{len(failed_h5)}")
    manifest_datasets = manifest.get("datasets") or {}
    if set(manifest_datasets) != set(split):
        reasons.append("manifest_dataset_set_mismatch")
    with np.load(data_dir / "sampled_indices.npz") as npz:
        if not npz.files:
            reasons.append("sampled_indices_npz_empty")
        sampled_key_count = len(npz.files)
    means_keys = {}
    for name in ("ctrl_means.npz", "pert_means.npz"):
        with np.load(data_dir / name) as npz:
            means_keys[name] = sorted(npz.files)
    train_datasets = sorted(ds for ds, groups in split.items() if groups.get("train"))
    if means_keys.get("ctrl_means.npz") != train_datasets:
        reasons.append("ctrl_means_keys_do_not_match_train_datasets")
    if means_keys.get("pert_means.npz") != train_datasets:
        reasons.append("pert_means_keys_do_not_match_train_datasets")
    return {
        "run_id": row["run_id"],
        "status": "ok" if not reasons else "fail",
        "reasons": reasons,
        "data_dir": str(data_dir),
        "split_file": str(split_file),
        "modality_counts": modality_counts,
        "n_h5_datasets": len(h5_rows),
        "sampled_index_arrays": sampled_key_count,
        "h5_rows": h5_rows,
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM All-Modality Dose-Aware Schema Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only schema/provenance gate for dose-aware all-modality artifacts.",
        "- Does not train, infer, read canonical metrics, read canonical multi, read held-out Track C query, or use GPU.",
        "",
        "## Rows",
        "",
        "| run id | status | modality counts | h5 datasets | sampled arrays | reasons |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['run_id']}` | `{row['status']}` | `{row.get('modality_counts', {})}` | {row.get('n_h5_datasets', 0)} | {row.get('sampled_index_arrays', 0)} | {', '.join(row.get('reasons') or []) or 'none'} |"
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
    materializer = load_json(MATERIALIZER_JSON)
    feasibility = load_json(FEASIBILITY_JSON)
    materialized_rows = materializer.get("materialized_rows") or []
    rows = [audit_materialized_row(row) for row in materialized_rows]
    if not materialized_rows:
        status = "allmodality_doseaware_schema_not_ready_no_materialized_rows"
        next_action = "wait_for_materialization"
    elif all(row.get("status") == "ok" for row in rows):
        status = "allmodality_doseaware_schema_pass_no_gpu"
        next_action = "run_dryload_and_design_controls"
    else:
        status = "allmodality_doseaware_schema_fail_no_gpu"
        next_action = "fix_schema_or_provenance_failures"
    payload = {
        "status": status,
        "boundary": {
            "cpu_only": True,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "uses_gpu": False,
        },
        "inputs": {"materializer_json": str(MATERIALIZER_JSON), "feasibility_json": str(FEASIBILITY_JSON)},
        "feasibility_status": feasibility.get("status"),
        "rows": rows,
        "gpu_authorized": False,
        "next_action": next_action,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
