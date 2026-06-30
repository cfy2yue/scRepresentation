#!/usr/bin/env python3
"""Materialize dose-aware all-modality capped latent H5 artifacts.

CPU-only. Builds all-modality true-cell/cell-cap data dirs from:
- gene datasets: current xverse latent H5 files;
- SciPlex chemical datasets: xverse per-cell embeddings with dose-level obs.

GPU training remains blocked until schema/dryload/design gates pass.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
BASE_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
BASE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
PROTOCOL_TSV = ROOT / "reports/latentfm_true_cell_count_scaling_protocol_20260624/all_modality_fixed64_budget16_32_64.tsv"
FEASIBILITY_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_feasibility_gate_20260625.json"
XVERSE_EMB_ROOT = ROOT / "scFM_output/embeddings/xverse"
OUT_DATA_ROOT = ROOT / "runs/latentfm_true_cell_count_allmodality_doseaware_capped_h5_20260625/artifacts"
OUT_SPLIT_ROOT = ROOT / "dataset/biFlow_data/xverse_true_cell_count_allmodality_doseaware_splits_20260625"
OUT_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_materializer_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_ALLMODALITY_DOSEAWARE_MATERIALIZER_GATE_20260625.md"

SCIPLEX_DATASETS = ("sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7")
BUDGETS = (16, 32, 64)
SEEDS = (42, 43, 44)
EXCLUDED_SPLIT_KEYS = {"canonical_test_reference"}
CHUNK_ROWS = 2048
GZIP_LEVEL = 4


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_seed(*parts: object) -> int:
    raw = "\t".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2**32)


def sanitize_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value)[:180]


def decode_conditions(raw: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in raw]


def sample_indices(n: int, k: int, *, key: str, replace: bool = False) -> np.ndarray:
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    if (not replace) and n <= k:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(stable_seed(key))
    return np.sort(rng.choice(np.arange(n, dtype=np.int64), size=int(k), replace=replace))


def read_protocol_rows() -> list[dict[str, str]]:
    with PROTOCOL_TSV.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def split_roles(groups: dict[str, Any]) -> tuple[set[str], set[str]]:
    train = {str(x) for x in groups.get("train") or []}
    eval_set: set[str] = set()
    for key, values in groups.items():
        if key == "train" or key in EXCLUDED_SPLIT_KEYS or not isinstance(values, list):
            continue
        eval_set.update(str(x) for x in values)
    return train, eval_set


def background_from_dataset(dataset: str) -> str:
    return dataset.replace("sciplex3_", "", 1)


def drug_from_cov_drug(dataset: str, cov_drug: str) -> str:
    bg = background_from_dataset(dataset)
    val = str(cov_drug)
    if val.startswith(bg + "_"):
        return val[len(bg) + 1 :]
    return val


def numeric_dose(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("-inf")


def build_gene_split(protocol_rows: list[dict[str, str]], base_split: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    allowed_by_ds: dict[str, set[str]] = defaultdict(set)
    for row in protocol_rows:
        if row.get("modality") == "gene":
            allowed_by_ds[str(row.get("dataset") or "")].add(str(row.get("condition") or ""))
    out: dict[str, dict[str, list[str]]] = {}
    for ds, allowed in sorted(allowed_by_ds.items()):
        train_base, eval_base = split_roles(base_split.get(ds) or {})
        train = sorted(allowed & train_base)
        eval_set = sorted(allowed & eval_base)
        if train or eval_set:
            out[ds] = {"train": train, "internal_val_allmodality_doseaware": eval_set}
    return out


def sciplex_obs(dataset: str) -> pd.DataFrame:
    obs_path = XVERSE_EMB_ROOT / dataset / "raw" / "obs.parquet"
    obs = pd.read_parquet(obs_path)
    obs = obs.reset_index(drop=False)
    obs["_row_index"] = np.arange(len(obs), dtype=np.int64)
    return obs


def build_sciplex_splits(base_split: dict[str, Any]) -> tuple[dict[str, dict[str, list[str]]], dict[str, pd.DataFrame]]:
    splits: dict[str, dict[str, list[str]]] = {}
    obs_by_ds: dict[str, pd.DataFrame] = {}
    for ds in SCIPLEX_DATASETS:
        obs = sciplex_obs(ds)
        obs_by_ds[ds] = obs
        train_drugs, _eval_drugs = split_roles(base_split.get(ds) or {})
        pert = obs[obs["control"].astype(str).isin({"0", "False", "false"})].copy()
        pert["drug"] = pert["cov_drug"].map(lambda x: drug_from_cov_drug(ds, str(x)))
        pert["dose_condition"] = pert["cov_drug_dose_name"].astype(str)
        pert = pert[pert["drug"].isin(train_drugs)].copy()
        grouped = (
            pert.groupby(["drug", "dose_condition"], observed=True)
            .agg(n_cells=("dose_condition", "size"), dose=("dose", "first"))
            .reset_index()
        )
        eligible = grouped[grouped["n_cells"] >= 64].copy()
        train_conditions: set[str] = set()
        eval_conditions: set[str] = set()
        for _drug, g in eligible.groupby("drug", sort=True):
            g = g.copy()
            g["_dose_num"] = g["dose"].map(numeric_dose)
            g = g.sort_values(["_dose_num", "dose_condition"])
            conds = list(g["dose_condition"])
            if len(conds) >= 2:
                eval_conditions.add(conds[-1])
                train_conditions.update(conds[:-1])
            elif conds:
                train_conditions.add(conds[0])
        splits[ds] = {
            "train": sorted(train_conditions),
            "internal_val_allmodality_doseaware": sorted(eval_conditions),
        }
    return splits, obs_by_ds


def build_split() -> tuple[dict[str, dict[str, list[str]]], dict[str, pd.DataFrame]]:
    base_split = load_json(BASE_SPLIT)
    protocol_rows = read_protocol_rows()
    split = build_gene_split(protocol_rows, base_split)
    sciplex_split, obs_by_ds = build_sciplex_splits(base_split)
    split.update(sciplex_split)
    return split, obs_by_ds


def materialize_gene_dataset(
    *,
    dataset: str,
    groups: dict[str, list[str]],
    data_dir: Path,
    budget: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    src_path = BASE_DATA_DIR / f"{dataset}.h5"
    out_path = data_dir / f"{dataset}.h5"
    train_set = set(groups.get("train") or [])
    eval_set = set(groups.get("internal_val_allmodality_doseaware") or [])
    with h5py.File(src_path, "r") as src:
        src_conds = decode_conditions(src["conditions"][:])
        cmap = {c: i for i, c in enumerate(src_conds)}
        keep = [c for c in src_conds if c in train_set or c in eval_set]
        dim = int(src["gt/emb"].shape[1])
        counts = []
        for cond in keep:
            idx = cmap[cond]
            n_src = int(src["gt/offsets"][idx + 1] - src["gt/offsets"][idx])
            counts.append(min(n_src, budget) if cond in train_set else n_src)
        offsets = np.zeros(len(keep) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))
        total = int(offsets[-1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as out:
            out.create_dataset("conditions", data=np.asarray(keep, dtype=object), dtype=h5py.string_dtype("utf-8"))
            out.create_dataset("gt/offsets", data=offsets)
            out.create_dataset("ctrl/offsets", data=offsets)
            chunk = (min(CHUNK_ROWS, max(1, total)), dim)
            out.create_dataset("gt/emb", shape=(total, dim), dtype="float32", chunks=chunk, compression="gzip", compression_opts=GZIP_LEVEL)
            out.create_dataset("ctrl/emb", shape=(total, dim), dtype="float32", chunks=chunk, compression="gzip", compression_opts=GZIP_LEVEL)
            sampled: dict[str, np.ndarray] = {}
            summary = {}
            for i, cond in enumerate(keep):
                idx = cmap[cond]
                gt_lo, gt_hi = int(src["gt/offsets"][idx]), int(src["gt/offsets"][idx + 1])
                ctrl_lo, ctrl_hi = int(src["ctrl/offsets"][idx]), int(src["ctrl/offsets"][idx + 1])
                n_gt = gt_hi - gt_lo
                n_ctrl = ctrl_hi - ctrl_lo
                n = int(counts[i])
                cap = cond in train_set
                gt_rel = sample_indices(n_gt, n, key=f"gene|gt|{dataset}|{cond}|{seed}|{budget}") if cap else np.arange(n_gt, dtype=np.int64)[:n]
                ctrl_rel = sample_indices(n_ctrl, n, key=f"gene|ctrl|{dataset}|{cond}|{seed}|{budget}") if cap else np.arange(n_ctrl, dtype=np.int64)[:n]
                out["gt/emb"][int(offsets[i]) : int(offsets[i + 1])] = np.asarray(src["gt/emb"][gt_lo + gt_rel], dtype=np.float32)
                out["ctrl/emb"][int(offsets[i]) : int(offsets[i + 1])] = np.asarray(src["ctrl/emb"][ctrl_lo + ctrl_rel], dtype=np.float32)
                key = sanitize_key(f"{dataset}__{cond}")
                sampled[f"{key}__gt"] = gt_rel.astype(np.int64)
                sampled[f"{key}__ctrl"] = ctrl_rel.astype(np.int64)
                summary[cond] = {"role": "train" if cap else "eval", "n_rows": n}
    return {"dataset": dataset, "path": str(out_path), "n_conditions": len(keep), "rows": total, "kind": "gene"}, sampled, summary


def materialize_sciplex_dataset(
    *,
    dataset: str,
    groups: dict[str, list[str]],
    obs: pd.DataFrame,
    data_dir: Path,
    budget: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    latent_path = XVERSE_EMB_ROOT / dataset / "raw" / "latent.npy"
    latent = np.load(latent_path, mmap_mode="r")
    out_path = data_dir / f"{dataset}.h5"
    train_set = set(groups.get("train") or [])
    eval_set = set(groups.get("internal_val_allmodality_doseaware") or [])
    keep = sorted(train_set | eval_set)
    control_idx = np.asarray(obs.loc[obs["control"].astype(str).isin({"1", "True", "true"}), "_row_index"], dtype=np.int64)
    if control_idx.size == 0:
        raise RuntimeError(f"no control rows for {dataset}")
    pert = obs[obs["control"].astype(str).isin({"0", "False", "false"})].copy()
    pert["dose_condition"] = pert["cov_drug_dose_name"].astype(str)
    idx_by_cond = {
        cond: np.asarray(frame["_row_index"], dtype=np.int64)
        for cond, frame in pert.groupby("dose_condition", observed=True)
    }
    counts = []
    for cond in keep:
        n_src = int(idx_by_cond.get(cond, np.zeros(0, dtype=np.int64)).size)
        if n_src <= 0:
            raise RuntimeError(f"missing pert rows for {dataset}:{cond}")
        counts.append(min(n_src, budget) if cond in train_set else n_src)
    offsets = np.zeros(len(keep) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))
    total = int(offsets[-1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {}
    with h5py.File(out_path, "w") as out:
        out.create_dataset("conditions", data=np.asarray(keep, dtype=object), dtype=h5py.string_dtype("utf-8"))
        out.create_dataset("gt/offsets", data=offsets)
        out.create_dataset("ctrl/offsets", data=offsets)
        chunk = (min(CHUNK_ROWS, max(1, total)), int(latent.shape[1]))
        out.create_dataset("gt/emb", shape=(total, int(latent.shape[1])), dtype="float32", chunks=chunk, compression="gzip", compression_opts=GZIP_LEVEL)
        out.create_dataset("ctrl/emb", shape=(total, int(latent.shape[1])), dtype="float32", chunks=chunk, compression="gzip", compression_opts=GZIP_LEVEL)
        for i, cond in enumerate(keep):
            src_idx = idx_by_cond[cond]
            n = int(counts[i])
            cap = cond in train_set
            gt_rel = sample_indices(src_idx.size, n, key=f"chem|gt|{dataset}|{cond}|{seed}|{budget}") if cap else np.arange(src_idx.size, dtype=np.int64)[:n]
            ctrl_rel = sample_indices(control_idx.size, n, key=f"chem|ctrl|{dataset}|{cond}|{seed}|{budget}", replace=control_idx.size < n)
            gt_idx = src_idx[gt_rel]
            ctrl_idx = control_idx[ctrl_rel]
            out["gt/emb"][int(offsets[i]) : int(offsets[i + 1])] = np.asarray(latent[gt_idx], dtype=np.float32)
            out["ctrl/emb"][int(offsets[i]) : int(offsets[i + 1])] = np.asarray(latent[ctrl_idx], dtype=np.float32)
            key = sanitize_key(f"{dataset}__{cond}")
            sampled[f"{key}__gt_source_rows"] = gt_idx.astype(np.int64)
            sampled[f"{key}__ctrl_source_rows"] = ctrl_idx.astype(np.int64)
            first = pert.loc[pert["dose_condition"] == cond].iloc[0]
            summary[cond] = {
                "role": "train" if cap else "eval",
                "n_rows": n,
                "cov_drug": str(first.get("cov_drug", "")),
                "dose": str(first.get("dose", "")),
                "pathway": str(first.get("pathway", "")),
                "target": str(first.get("target", "")),
            }
    return {"dataset": dataset, "path": str(out_path), "n_conditions": len(keep), "rows": total, "kind": "chemical"}, sampled, summary


def compute_train_means(data_dir: Path, split: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    ctrl_means: dict[str, np.ndarray] = {}
    pert_means: dict[str, np.ndarray] = {}
    audit: list[dict[str, Any]] = []
    for ds, groups in sorted(split.items()):
        train = [str(c) for c in groups.get("train") or []]
        if not train:
            audit.append({"dataset": ds, "status": "empty_train_dataset", "train_conditions_used": 0, "train_cells_used": 0})
            continue
        with h5py.File(data_dir / f"{ds}.h5", "r") as h5:
            conds = decode_conditions(h5["conditions"][:])
            cmap = {c: i for i, c in enumerate(conds)}
            totals = {"gt": None, "ctrl": None}
            n_cells = 0
            used = 0
            missing = []
            for cond in train:
                idx = cmap.get(cond)
                if idx is None:
                    missing.append(cond)
                    continue
                lo, hi = int(h5["gt/offsets"][idx]), int(h5["gt/offsets"][idx + 1])
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


def materialize_run(run_id: str, split: dict[str, Any], obs_by_ds: dict[str, pd.DataFrame], budget: int, seed: int) -> dict[str, Any]:
    data_dir = OUT_DATA_ROOT / run_id
    data_dir.mkdir(parents=True, exist_ok=True)
    split_file = OUT_SPLIT_ROOT / f"split_{run_id}.json"
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sampled_arrays: dict[str, np.ndarray] = {}
    condition_summary: dict[str, Any] = {}
    dataset_rows = []
    for ds, groups in sorted(split.items()):
        if ds in SCIPLEX_DATASETS:
            row, sampled, summary = materialize_sciplex_dataset(dataset=ds, groups=groups, obs=obs_by_ds[ds], data_dir=data_dir, budget=budget, seed=seed)
        else:
            row, sampled, summary = materialize_gene_dataset(dataset=ds, groups=groups, data_dir=data_dir, budget=budget, seed=seed)
        dataset_rows.append(row)
        condition_summary[ds] = summary
        for key, arr in sampled.items():
            sampled_arrays[sanitize_key(ds) + "__" + key] = arr

    ctrl_means, pert_means, means_audit = compute_train_means(data_dir, split)
    np.savez_compressed(data_dir / "ctrl_means.npz", **ctrl_means)
    np.savez_compressed(data_dir / "pert_means.npz", **pert_means)
    np.savez_compressed(data_dir / "sampled_indices.npz", **sampled_arrays)
    with gzip.open(data_dir / "sampled_indices_summary.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(condition_summary, handle, indent=2, sort_keys=True)
    manifest = {
        "source": "latentfm_true_cell_count_allmodality_doseaware_capped_h5",
        "budget": int(budget),
        "seed": int(seed),
        "split_file": str(split_file),
        "data_dir": str(data_dir),
        "base_gene_data_dir": str(BASE_DATA_DIR),
        "xverse_embedding_root": str(XVERSE_EMB_ROOT),
        "datasets": {row["dataset"]: row for row in dataset_rows},
        "total_rows_ctrl_gt_each": int(sum(row["rows"] for row in dataset_rows)),
        "sampled_indices_npz": str(data_dir / "sampled_indices.npz"),
        "sampled_indices_summary": str(data_dir / "sampled_indices_summary.json.gz"),
        "means_audit": means_audit,
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"run_id": run_id, "data_dir": str(data_dir), "split_file": str(split_file), "budget": int(budget), "seed": int(seed), "n_datasets": len(dataset_rows), "total_rows_ctrl_gt_each": manifest["total_rows_ctrl_gt_each"], "status": "ok"}


def plan_rows(split: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for budget in BUDGETS:
        for seed in SEEDS:
            run_id = f"all_modality_doseaware_fixed64_budget16_32_64_budget{budget}_seed{seed}"
            modality_counts = {"train_gene": 0, "eval_gene": 0, "train_chemical": 0, "eval_chemical": 0}
            for ds, groups in split.items():
                is_chem = ds in SCIPLEX_DATASETS
                modality_counts[("train_chemical" if is_chem else "train_gene")] += len(groups.get("train") or [])
                modality_counts[("eval_chemical" if is_chem else "eval_gene")] += len(groups.get("internal_val_allmodality_doseaware") or [])
            rows.append({"run_id": run_id, "budget": int(budget), "seed": int(seed), "data_dir": str(OUT_DATA_ROOT / run_id), "split_file": str(OUT_SPLIT_ROOT / f"split_{run_id}.json"), "modality_counts": modality_counts, "launcher_ready": all(v > 0 for v in modality_counts.values())})
    return rows


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM All-Modality Dose-Aware Materializer Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only dose-aware capped-H5 materializer.",
        "- Chemical SciPlex conditions use xverse per-cell embeddings grouped by `cov_drug_dose_name`.",
        "- Canonical reference drugs are excluded from split construction.",
        "- Does not train, infer, read canonical metrics, read canonical multi, read held-out Track C query, or use GPU.",
        "",
        f"- materialized: `{payload['materialized']}`",
        "",
        "## Plan Rows",
        "",
        "| run id | budget | seed | modality counts | launcher ready | data dir |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in payload["plan_rows"]:
        lines.append(f"| `{row['run_id']}` | {row['budget']} | {row['seed']} | `{row['modality_counts']}` | `{row['launcher_ready']}` | `{row['data_dir']}` |")
    if payload["materialized_rows"]:
        lines.extend(["", "## Materialized Rows", "", "| run id | rows each | datasets | data dir |", "|---|---:|---:|---|"])
        for row in payload["materialized_rows"]:
            lines.append(f"| `{row['run_id']}` | {row['total_rows_ctrl_gt_each']} | {row['n_datasets']} | `{row['data_dir']}` |")
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
    ap.add_argument("--materialize", action="store_true")
    ap.add_argument("--only-run-id", default="")
    args = ap.parse_args()

    feasibility = load_json(FEASIBILITY_JSON)
    if not feasibility.get("cpu_materializer_authorized_next"):
        raise SystemExit("dose-aware feasibility gate has not authorized materializer")
    split, obs_by_ds = build_split()
    planned = plan_rows(split)
    selected = planned
    if args.only_run_id:
        selected = [r for r in planned if r["run_id"] == args.only_run_id]
        if not selected:
            raise SystemExit(f"unknown run id: {args.only_run_id}")
    bad = [r for r in selected if not r["launcher_ready"]]
    materialized = []
    if args.materialize:
        if bad:
            raise SystemExit("refusing to materialize non-launcher-ready rows")
        for row in selected:
            materialized.append(materialize_run(row["run_id"], split, obs_by_ds, int(row["budget"]), int(row["seed"])))
    status = "allmodality_doseaware_materializer_dryrun_pass_no_gpu" if not bad else "allmodality_doseaware_materializer_dryrun_fail_no_gpu"
    if args.materialize:
        status = "allmodality_doseaware_materialized_no_gpu" if all(r["status"] == "ok" for r in materialized) else "allmodality_doseaware_materialized_check_no_gpu"
    payload = {
        "status": status,
        "materialized": bool(args.materialize),
        "boundary": {
            "cpu_only": True,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "uses_gpu": False,
            "excluded_split_keys": sorted(EXCLUDED_SPLIT_KEYS),
        },
        "inputs": {"feasibility_json": str(FEASIBILITY_JSON), "protocol_tsv": str(PROTOCOL_TSV), "base_split": str(BASE_SPLIT), "xverse_embedding_root": str(XVERSE_EMB_ROOT)},
        "plan_rows": planned,
        "selected_rows": selected,
        "bad_rows": bad,
        "materialized_rows": materialized,
        "gpu_authorized": False,
        "next_action": "run_schema_dryload_design_gates" if args.materialize and not bad else ("launch_detached_cpu_materialization" if not args.materialize and not bad else "fix_bad_rows"),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "materialized": bool(args.materialize), "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
