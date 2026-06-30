#!/usr/bin/env python3
"""Materialize or dry-run capped latent H5 data dirs for true cell-count scaling.

Default mode is a CPU-only dry run. Use --materialize only after reviewing the
schema/provenance gate. The materialized dirs cap train conditions only and keep
internal validation/test conditions at their source row counts, so the protocol
changes training cell count rather than evaluation reference size.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
BASE_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
BASE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
PROTOCOL_JSON = ROOT / "reports/latentfm_true_cell_count_scaling_protocol_20260624.json"
OUT_DATA_ROOT = ROOT / "runs/latentfm_true_cell_count_scaling_capped_h5_20260624/artifacts"
OUT_SPLIT_ROOT = ROOT / "dataset/biFlow_data/xverse_true_cell_count_scaling_splits_20260624"
OUT_JSON = ROOT / "reports/latentfm_true_cell_count_capped_h5_materializer_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_CAPPED_H5_MATERIALIZER_GATE_20260624.md"

EXCLUDED_SPLIT_KEYS = {"canonical_test_reference"}
CHUNK_ROWS = 2048
GZIP_LEVEL = 4
ROLE_MODALITY_MINIMA = {
    "all_modality_fixed64_budget16_32_64": {"train": {"gene": 50, "chemical": 50}, "eval": {"gene": 20, "chemical": 20}},
    "gene_only_fixed256_budget64_128_256": {"train": {"gene": 100}, "eval": {"gene": 20}},
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_seed(*parts: object) -> int:
    raw = "\t".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2**32)


def decode_conditions(raw: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in raw]


def read_protocol_conditions(path: Path) -> tuple[set[tuple[str, str]], dict[tuple[str, str], str]]:
    out: set[tuple[str, str]] = set()
    modality: dict[tuple[str, str], str] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (str(row["dataset"]), str(row["condition"]))
            out.add(key)
            modality[key] = str(row.get("modality") or "unknown")
    return out, modality


def split_condition_sets(base_split: dict[str, Any], allowed: set[tuple[str, str]]) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, set[str]]]]:
    new_split: dict[str, dict[str, list[str]]] = {}
    role_sets: dict[str, dict[str, set[str]]] = {}
    for ds, groups in sorted(base_split.items()):
        out_groups: dict[str, list[str]] = {}
        train = [str(c) for c in groups.get("train") or [] if (ds, str(c)) in allowed]
        out_groups["train"] = train
        train_set = set(train)
        eval_set: set[str] = set()
        for key, values in groups.items():
            if key == "train" or key in EXCLUDED_SPLIT_KEYS or not isinstance(values, list):
                continue
            vals = [str(c) for c in values]
            out_groups[key] = vals
            eval_set.update(vals)
        new_split[ds] = out_groups
        role_sets[ds] = {"train": train_set, "eval": eval_set}
    return new_split, role_sets


def infer_modality(ds: str, cond: str, modality_by_key: dict[tuple[str, str], str], metadata: dict[str, Any]) -> str:
    direct = modality_by_key.get((ds, cond))
    if direct and direct != "unknown":
        return direct
    if "sciplex" in ds.lower():
        return "chemical"
    meta = ((metadata.get(ds) or {}).get(cond) or {})
    ptype = str(meta.get("perturbation_type_raw", meta.get("perturbation_type", ""))).lower()
    if ptype in {"drug", "chemical", "compound", "small molecule", "small-molecule"}:
        return "chemical"
    if ptype:
        return "gene"
    return "gene"


def h5_condition_counts(path: Path) -> tuple[list[str], dict[str, int], int]:
    with h5py.File(path, "r") as h5:
        conds = decode_conditions(h5["conditions"][:])
        offsets = h5["gt/offsets"][:]
        dim = int(h5["gt/emb"].shape[1])
    return conds, {c: int(offsets[i + 1] - offsets[i]) for i, c in enumerate(conds)}, dim


def sample_indices(n: int, k: int, *, key: str) -> np.ndarray:
    if k <= 0 or n <= k:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(stable_seed(key))
    return np.sort(rng.choice(np.arange(n, dtype=np.int64), size=int(k), replace=False))


def append_condition(
    out_h5: h5py.File,
    src_h5: h5py.File,
    *,
    cond_idx: int,
    cond: str,
    out_start: int,
    cap: int | None,
    seed: int,
    dataset: str,
) -> int:
    src_gt_offsets = src_h5["gt/offsets"][:]
    src_ctrl_offsets = src_h5["ctrl/offsets"][:]
    gt_lo, gt_hi = int(src_gt_offsets[cond_idx]), int(src_gt_offsets[cond_idx + 1])
    ctrl_lo, ctrl_hi = int(src_ctrl_offsets[cond_idx]), int(src_ctrl_offsets[cond_idx + 1])
    n_gt = gt_hi - gt_lo
    n_ctrl = ctrl_hi - ctrl_lo
    if cap is None:
        gt_rel = np.arange(n_gt, dtype=np.int64)
        ctrl_rel = np.arange(n_ctrl, dtype=np.int64)
        n = min(n_gt, n_ctrl)
        gt_rel = gt_rel[:n]
        ctrl_rel = ctrl_rel[:n]
    else:
        n = int(cap)
        gt_rel = sample_indices(n_gt, n, key=f"gt|{dataset}|{cond}|{seed}|{cap}")
        ctrl_rel = sample_indices(n_ctrl, n, key=f"ctrl|{dataset}|{cond}|{seed}|{cap}")
    gt_rows = np.asarray(src_h5["gt/emb"][gt_lo + gt_rel], dtype=np.float32)
    ctrl_rows = np.asarray(src_h5["ctrl/emb"][ctrl_lo + ctrl_rel], dtype=np.float32)
    n = min(gt_rows.shape[0], ctrl_rows.shape[0])
    out_h5["gt/emb"][out_start:out_start + n] = gt_rows[:n]
    out_h5["ctrl/emb"][out_start:out_start + n] = ctrl_rows[:n]
    return n


def materialize_dataset(
    *,
    src_path: Path,
    out_path: Path,
    dataset: str,
    conditions: list[str],
    train_set: set[str],
    budget: int,
    seed: int,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with h5py.File(src_path, "r") as src:
        src_conds = decode_conditions(src["conditions"][:])
        cmap = {c: i for i, c in enumerate(src_conds)}
        dim = int(src["gt/emb"].shape[1])
        counts = []
        for cond in conditions:
            idx = cmap[cond]
            offsets = src["gt/offsets"]
            n = int(offsets[idx + 1] - offsets[idx])
            counts.append(min(n, int(budget)) if cond in train_set else n)
        offsets_out = np.zeros(len(conditions) + 1, dtype=np.int64)
        offsets_out[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))
        total = int(offsets_out[-1])
        with h5py.File(tmp, "w") as out:
            out.create_dataset("conditions", data=np.asarray(conditions, dtype=object), dtype=h5py.string_dtype("utf-8"))
            out.create_dataset("gt/offsets", data=offsets_out)
            out.create_dataset("ctrl/offsets", data=offsets_out)
            chunk = (min(CHUNK_ROWS, max(1, total)), dim)
            out.create_dataset("gt/emb", shape=(total, dim), dtype="float32", chunks=chunk, compression="gzip", compression_opts=GZIP_LEVEL)
            out.create_dataset("ctrl/emb", shape=(total, dim), dtype="float32", chunks=chunk, compression="gzip", compression_opts=GZIP_LEVEL)
            for i, cond in enumerate(conditions):
                cap = int(budget) if cond in train_set else None
                append_condition(out, src, cond_idx=cmap[cond], cond=cond, out_start=int(offsets_out[i]), cap=cap, seed=seed, dataset=dataset)
    tmp.replace(out_path)
    return {
        "dataset": dataset,
        "conditions": list(conditions),
        "n_conditions": len(conditions),
        "rows": int(sum(counts)),
        "path": str(out_path),
    }


def compute_train_means(data_dir: Path, split: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    ctrl_means: dict[str, np.ndarray] = {}
    pert_means: dict[str, np.ndarray] = {}
    audit = []
    for ds, groups in sorted(split.items()):
        train = [str(c) for c in groups.get("train") or []]
        if not train:
            audit.append({"dataset": ds, "status": "empty_train_dataset", "train_conditions_used": 0, "train_cells_used": 0})
            continue
        h5_path = data_dir / f"{ds}.h5"
        with h5py.File(h5_path, "r") as h5:
            conds = decode_conditions(h5["conditions"][:])
            cmap = {c: i for i, c in enumerate(conds)}
            totals = {"gt": None, "ctrl": None}
            n_cells = 0
            missing = []
            used = 0
            for cond in train:
                idx = cmap.get(cond)
                if idx is None:
                    missing.append(cond)
                    continue
                lo = int(h5["gt/offsets"][idx])
                hi = int(h5["gt/offsets"][idx + 1])
                if hi <= lo:
                    continue
                for group in ("gt", "ctrl"):
                    arr = np.asarray(h5[f"{group}/emb"][lo:hi], dtype=np.float64)
                    summed = arr.sum(axis=0, dtype=np.float64)
                    totals[group] = summed if totals[group] is None else totals[group] + summed
                n_cells += hi - lo
                used += 1
            if n_cells <= 0 or totals["gt"] is None or totals["ctrl"] is None:
                audit.append({"dataset": ds, "status": "no_train_cells", "train_conditions_used": used, "train_cells_used": n_cells})
                continue
            pert_means[ds] = (totals["gt"] / float(n_cells)).astype(np.float32)
            ctrl_means[ds] = (totals["ctrl"] / float(n_cells)).astype(np.float32)
            audit.append({"dataset": ds, "status": "ok", "train_conditions_used": used, "train_cells_used": int(n_cells), "n_missing_conditions": len(missing), "missing_conditions": missing[:10]})
    return ctrl_means, pert_means, audit


def build_plan(protocol: dict[str, Any], base_split: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    allowed, modality_by_key = read_protocol_conditions(Path(protocol["manifest_tsv"]))
    split, role_sets = split_condition_sets(base_split, allowed)
    rows = []
    for budget in protocol["budgets"]:
        for seed in protocol["subsample_seeds"]:
            run_id = f"{protocol['name']}_budget{budget}_seed{seed}"
            data_dir = OUT_DATA_ROOT / run_id
            split_file = OUT_SPLIT_ROOT / f"split_{run_id}.json"
            dataset_rows = []
            total_rows = 0
            missing = []
            low_train = []
            train_modality_counts: dict[str, int] = defaultdict(int)
            eval_modality_counts: dict[str, int] = defaultdict(int)
            for ds, roles in sorted(role_sets.items()):
                h5_path = BASE_DATA_DIR / f"{ds}.h5"
                conds, counts, dim = h5_condition_counts(h5_path)
                cond_set = set(conds)
                train = set(roles["train"])
                eval_set = set(roles["eval"])
                for cond in train:
                    train_modality_counts[infer_modality(ds, cond, modality_by_key, metadata)] += 1
                for cond in eval_set:
                    eval_modality_counts[infer_modality(ds, cond, modality_by_key, metadata)] += 1
                keep = sorted(train | eval_set, key=conds.index)
                ds_missing = [c for c in keep if c not in cond_set]
                ds_low = [c for c in train if counts.get(c, 0) < int(budget)]
                if ds_missing:
                    missing.append({"dataset": ds, "conditions": ds_missing[:10], "n": len(ds_missing)})
                if ds_low:
                    low_train.append({"dataset": ds, "conditions": ds_low[:10], "n": len(ds_low)})
                train_rows = len(train) * int(budget)
                eval_rows = sum(counts.get(c, 0) for c in eval_set)
                total_rows += train_rows + eval_rows
                dataset_rows.append({"dataset": ds, "train_conditions": len(train), "eval_conditions": len(eval_set), "kept_conditions": len(keep), "estimated_rows": int(train_rows + eval_rows), "emb_dim": dim})
            readiness_reasons = []
            minima = ROLE_MODALITY_MINIMA.get(str(protocol["name"]), {})
            for role, required in minima.items():
                observed = train_modality_counts if role == "train" else eval_modality_counts
                for modality, min_count in required.items():
                    if int(observed.get(modality, 0)) < int(min_count):
                        readiness_reasons.append(f"{role}_{modality}_conditions_lt_{min_count}")
            rows.append({
                "run_id": run_id,
                "protocol": protocol["name"],
                "budget": int(budget),
                "seed": int(seed),
                "data_dir": str(data_dir),
                "split_file": str(split_file),
                "split": split,
                "dataset_rows": dataset_rows,
                "estimated_total_rows_ctrl_gt_each": int(total_rows),
                "missing": missing,
                "low_train": low_train,
                "train_modality_counts": dict(sorted(train_modality_counts.items())),
                "eval_modality_counts": dict(sorted(eval_modality_counts.items())),
                "launcher_readiness_reasons": readiness_reasons,
                "launcher_ready": not readiness_reasons,
            })
    return rows


def materialize_plan_row(row: dict[str, Any]) -> dict[str, Any]:
    data_dir = Path(row["data_dir"])
    split_file = Path(row["split_file"])
    split_file.parent.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    split = row["split"]
    split_file.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_metadata = load_json(BASE_DATA_DIR / "condition_metadata.json")
    out_metadata: dict[str, dict[str, Any]] = {}
    materialized = []
    for ds, groups in sorted(split.items()):
        train = set(str(c) for c in groups.get("train") or [])
        eval_set = set()
        for key, values in groups.items():
            if key == "train" or key in EXCLUDED_SPLIT_KEYS or not isinstance(values, list):
                continue
            eval_set.update(str(c) for c in values)
        with h5py.File(BASE_DATA_DIR / f"{ds}.h5", "r") as src:
            src_conds = decode_conditions(src["conditions"][:])
        keep = [c for c in src_conds if c in train or c in eval_set]
        if not keep:
            continue
        mat = materialize_dataset(src_path=BASE_DATA_DIR / f"{ds}.h5", out_path=data_dir / f"{ds}.h5", dataset=ds, conditions=keep, train_set=train, budget=int(row["budget"]), seed=int(row["seed"]))
        materialized.append(mat)
        src_meta = source_metadata.get(ds) or {}
        out_metadata[ds] = {c: src_meta.get(c, {}) for c in keep}
    ctrl_means, pert_means, audit = compute_train_means(data_dir, split)
    np.savez_compressed(data_dir / "ctrl_means.npz", **ctrl_means)
    np.savez_compressed(data_dir / "pert_means.npz", **pert_means)
    (data_dir / "condition_metadata.json").write_text(json.dumps(out_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"source": "latentfm_true_cell_count_capped_h5", "base_data_dir": str(BASE_DATA_DIR), "split_file": str(split_file), "budget": int(row["budget"]), "seed": int(row["seed"]), "datasets": {m["dataset"]: m for m in materialized}, "total_rows_ctrl_gt_each": int(sum(m["rows"] for m in materialized)), "condition_metadata_file": str(data_dir / "condition_metadata.json")}
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"run_id": row["run_id"], "data_dir": str(data_dir), "split_file": str(split_file), "n_datasets": len(materialized), "means_audit": audit, "status": "ok" if all(a.get("status") in {"ok", "empty_train_dataset"} for a in audit) else "check"}


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM True Cell-Count Capped-H5 Materializer Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only capped latent-H5 materializer gate.",
        "- Train conditions are capped by budget; internal validation/test conditions are kept at source row counts.",
        "- Excludes `canonical_test_reference`; does not read canonical metrics, canonical multi, Track C query, train, infer, or use GPU.",
        "",
        "## Mode",
        "",
        f"- materialized: `{payload['materialized']}`",
        "",
        "## Plan Rows",
        "",
        "| run id | budget | seed | train modalities | eval modalities | estimated rows | missing | low train | launcher ready | reasons | data dir |",
        "|---|---:|---:|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in payload["plan_rows"]:
        lines.append(
            f"| `{row['run_id']}` | {row['budget']} | {row['seed']} | `{row.get('train_modality_counts', {})}` | `{row.get('eval_modality_counts', {})}` | {row['estimated_total_rows_ctrl_gt_each']} | {sum(m['n'] for m in row['missing'])} | {sum(m['n'] for m in row['low_train'])} | `{row.get('launcher_ready')}` | {', '.join(row.get('launcher_readiness_reasons') or []) or 'none'} | `{row['data_dir']}` |"
        )
    lines.extend(["", "## Decision", "", f"- GPU authorized: `{payload['gpu_authorized']}`", f"- next action: `{payload['next_action']}`", "", "## JSON", "", f"`{OUT_JSON}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materialize", action="store_true", help="write capped H5 data dirs; default is dry-run only")
    ap.add_argument("--only-run-id", default="", help="optional single run id to materialize")
    ap.add_argument("--only-launcher-ready", action="store_true", help="restrict to rows that pass launcher-readiness checks")
    args = ap.parse_args()

    base_split = load_json(BASE_SPLIT)
    metadata = load_json(BASE_DATA_DIR / "condition_metadata.json")
    protocol_payload = load_json(PROTOCOL_JSON)
    plan_rows: list[dict[str, Any]] = []
    for protocol in protocol_payload["protocols"]:
        plan_rows.extend(build_plan(protocol, base_split, metadata))
    if args.only_run_id:
        plan_rows = [r for r in plan_rows if r["run_id"] == args.only_run_id]
        if not plan_rows:
            raise SystemExit(f"unknown run id: {args.only_run_id}")
    if args.only_launcher_ready:
        plan_rows = [r for r in plan_rows if r.get("launcher_ready")]
        if not plan_rows:
            raise SystemExit("no launcher-ready rows")

    bad = [r for r in plan_rows if r["missing"] or r["low_train"] or not r.get("launcher_ready")]
    materialized_rows = []
    if args.materialize:
        if bad:
            raise SystemExit("refusing to materialize plan rows with missing or low-train conditions")
        for row in plan_rows:
            materialized_rows.append(materialize_plan_row(row))

    status = "capped_h5_materializer_dryrun_pass_no_gpu" if not bad else "capped_h5_materializer_dryrun_fail_no_gpu"
    if args.materialize:
        status = "capped_h5_materialized_no_gpu" if all(r.get("status") == "ok" for r in materialized_rows) else "capped_h5_materialized_check_no_gpu"
    public_rows = [{k: v for k, v in r.items() if k != "split"} for r in plan_rows]
    payload = {
        "status": status,
        "materialized": bool(args.materialize),
        "boundary": {"cpu_only": True, "reads_canonical_metrics": False, "reads_canonical_multi": False, "reads_trackc_query": False, "uses_gpu": False, "excluded_split_keys": sorted(EXCLUDED_SPLIT_KEYS)},
        "base_data_dir": str(BASE_DATA_DIR),
        "base_split": str(BASE_SPLIT),
        "protocol_json": str(PROTOCOL_JSON),
        "plan_rows": public_rows,
        "materialized_rows": materialized_rows,
        "bad_rows": [{k: v for k, v in r.items() if k != "split"} for r in bad],
        "only_launcher_ready": bool(args.only_launcher_ready),
        "gpu_authorized": False,
        "next_action": "review dry-run then launch detached CPU materialization for selected protocol rows" if (not args.materialize and not bad) else ("prepare bounded GPU smoke only after materialized artifacts and launcher data-dir override gate" if args.materialize else "fix or select only launcher-ready rows before artifact generation"),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "materialized": bool(args.materialize), "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
