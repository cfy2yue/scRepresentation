#!/usr/bin/env python3
"""Backfill deterministic sampled-row provenance for true cell-count artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
BASE_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
MATERIALIZER_JSON = ROOT / "reports/latentfm_true_cell_count_capped_h5_materializer_gate_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_true_cell_count_sample_provenance_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_SAMPLE_PROVENANCE_GATE_20260624.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def decode_conditions(raw: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in raw]


def stable_seed(*parts: object) -> int:
    raw = "\t".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2**32)


def sample_indices(n: int, k: int, *, key: str) -> np.ndarray:
    if k <= 0 or n <= k:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(stable_seed(key))
    return np.sort(rng.choice(np.arange(n, dtype=np.int64), size=int(k), replace=False))


def array_sha256(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def safe_key(dataset: str, condition: str, group: str) -> str:
    digest = hashlib.sha256(f"{dataset}\t{condition}\t{group}".encode("utf-8")).hexdigest()[:20]
    return f"{dataset}__{digest}__{group}"


def source_counts(dataset: str) -> dict[str, dict[str, int]]:
    src_path = BASE_DATA_DIR / f"{dataset}.h5"
    with h5py.File(src_path, "r") as h5:
        conds = decode_conditions(h5["conditions"][:])
        gt_offsets = h5["gt/offsets"][:]
        ctrl_offsets = h5["ctrl/offsets"][:]
    return {
        cond: {
            "gt_n": int(gt_offsets[i + 1] - gt_offsets[i]),
            "ctrl_n": int(ctrl_offsets[i + 1] - ctrl_offsets[i]),
            "gt_start": int(gt_offsets[i]),
            "ctrl_start": int(ctrl_offsets[i]),
        }
        for i, cond in enumerate(conds)
    }


def artifact_emb_dim(data_dir: Path) -> int | None:
    for h5_path in sorted(data_dir.glob("*.h5")):
        with h5py.File(h5_path, "r") as h5:
            if "gt/emb" in h5 and len(h5["gt/emb"].shape) == 2:
                return int(h5["gt/emb"].shape[1])
    return None


def h5_conditions(data_dir: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for h5_path in sorted(data_dir.glob("*.h5")):
        with h5py.File(h5_path, "r") as h5:
            if "conditions" in h5:
                out[h5_path.stem] = decode_conditions(h5["conditions"][:])
    return out


def backfill_row(row: dict[str, Any], *, write: bool) -> dict[str, Any]:
    run_id = str(row["run_id"])
    data_dir = Path(row["data_dir"])
    split_file = Path(row["split_file"])
    manifest_file = data_dir / "manifest.json"
    reasons: list[str] = []
    for path in [data_dir, split_file, manifest_file]:
        if not path.exists():
            reasons.append(f"missing:{path}")
    if reasons:
        return {"run_id": run_id, "status": "fail", "reasons": reasons}

    split = load_json(split_file)
    manifest = load_json(manifest_file)
    budget = int(manifest.get("budget", row.get("budget", 0)))
    seed = int(manifest.get("seed", row.get("seed", 0)))
    arrays: dict[str, np.ndarray] = {}
    jsonl_rows: list[dict[str, Any]] = []
    train_conditions = 0
    train_cells_gt = 0
    train_cells_ctrl = 0
    source_cache: dict[str, dict[str, dict[str, int]]] = {}

    for dataset, groups in sorted(split.items()):
        train = [str(c) for c in groups.get("train") or []]
        if not train:
            continue
        source_cache.setdefault(dataset, source_counts(dataset))
        counts = source_cache[dataset]
        for condition in train:
            info = counts.get(condition)
            if info is None:
                reasons.append(f"missing_source_condition:{dataset}:{condition}")
                continue
            gt_rel = sample_indices(info["gt_n"], budget, key=f"gt|{dataset}|{condition}|{seed}|{budget}")
            ctrl_rel = sample_indices(info["ctrl_n"], budget, key=f"ctrl|{dataset}|{condition}|{seed}|{budget}")
            gt_key = safe_key(dataset, condition, "gt")
            ctrl_key = safe_key(dataset, condition, "ctrl")
            arrays[gt_key] = gt_rel
            arrays[ctrl_key] = ctrl_rel
            train_conditions += 1
            train_cells_gt += int(gt_rel.size)
            train_cells_ctrl += int(ctrl_rel.size)
            jsonl_rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "condition": condition,
                    "budget": budget,
                    "seed": seed,
                    "gt_key": gt_key,
                    "ctrl_key": ctrl_key,
                    "gt_source_start": info["gt_start"],
                    "ctrl_source_start": info["ctrl_start"],
                    "gt_source_n": info["gt_n"],
                    "ctrl_source_n": info["ctrl_n"],
                    "gt_sample_n": int(gt_rel.size),
                    "ctrl_sample_n": int(ctrl_rel.size),
                    "gt_rel_sha256": array_sha256(gt_rel),
                    "ctrl_rel_sha256": array_sha256(ctrl_rel),
                }
            )

    if train_conditions <= 0:
        reasons.append("no_train_conditions")
    if train_cells_gt != train_conditions * budget or train_cells_ctrl != train_conditions * budget:
        reasons.append("sample_count_not_equal_budget")

    if write and not reasons:
        emb_dim = artifact_emb_dim(data_dir)
        np.savez_compressed(data_dir / "sampled_indices.npz", **arrays)
        with (data_dir / "sampled_index_manifest.jsonl").open("w", encoding="utf-8") as f:
            for item in jsonl_rows:
                f.write(json.dumps(item, sort_keys=True) + "\n")
        conditions_by_dataset = h5_conditions(data_dir)
        if emb_dim is not None:
            manifest["emb_dim"] = emb_dim
        datasets = manifest.get("datasets") or {}
        if isinstance(datasets, dict):
            for dataset, meta in datasets.items():
                if not isinstance(meta, dict):
                    continue
                conds = conditions_by_dataset.get(str(dataset))
                if conds is None:
                    continue
                meta["conditions"] = conds
                meta["n_conditions"] = len(conds)
        manifest["sampled_indices_file"] = str(data_dir / "sampled_indices.npz")
        manifest["sampled_index_manifest_file"] = str(data_dir / "sampled_index_manifest.jsonl")
        manifest["sampled_index_provenance"] = {
            "method": "deterministic_sha256_seeded_without_replacement",
            "train_conditions": train_conditions,
            "budget": budget,
            "seed": seed,
        }
        manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "status": "ok" if not reasons else "fail",
        "reasons": reasons,
        "data_dir": str(data_dir),
        "budget": budget,
        "seed": seed,
        "train_conditions": train_conditions,
        "train_cells_gt": train_cells_gt,
        "train_cells_ctrl": train_cells_ctrl,
        "write": bool(write),
        "sampled_indices_file": str(data_dir / "sampled_indices.npz"),
        "sampled_index_manifest_file": str(data_dir / "sampled_index_manifest.jsonl"),
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM True Cell-Count Sample Provenance Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only sampled-row provenance backfill/check.",
        "- Reconstructs deterministic train-condition sampled row indices from source H5 offsets, budget, seed, dataset, and condition.",
        "- Does not read canonical metrics, canonical multi, Track C query, train, infer, or use GPU.",
        "",
        "## Rows",
        "",
        "| run id | status | budget | seed | train conditions | reasons |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['run_id']}` | `{row['status']}` | {row.get('budget', '')} | {row.get('seed', '')} | {row.get('train_conditions', 0)} | {', '.join(row.get('reasons') or []) or 'none'} |"
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
    ap.add_argument("--write", action="store_true", help="write sampled_indices.npz and sampled_index_manifest.jsonl")
    ap.add_argument("--only-run-id", default="", help="optional run id")
    args = ap.parse_args()

    materializer = load_json(MATERIALIZER_JSON)
    rows = materializer.get("materialized_rows") or []
    if args.only_run_id:
        rows = [r for r in rows if r.get("run_id") == args.only_run_id]
        if not rows:
            raise SystemExit(f"run id is not materialized: {args.only_run_id}")

    audited = [backfill_row(row, write=args.write) for row in rows]
    if not rows:
        status = "sample_provenance_not_ready_no_materialized_rows"
        next_action = "wait_for_capped_h5_materialization"
    elif all(row.get("status") == "ok" for row in audited):
        status = "sample_provenance_written_no_gpu" if args.write else "sample_provenance_dryrun_pass_no_gpu"
        next_action = "run capped-H5 schema gate, then pre-GPU controls"
    else:
        status = "sample_provenance_fail_no_gpu"
        next_action = "fix sampled-row provenance failures"

    payload = {
        "status": status,
        "write": bool(args.write),
        "materializer_json": str(MATERIALIZER_JSON),
        "rows": audited,
        "gpu_authorized": False,
        "next_action": next_action,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
