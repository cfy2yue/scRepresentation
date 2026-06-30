#!/usr/bin/env python3
"""Prepare LatentFM HDF5 files from scFMBench embedding exports.

Input layout:

    <scfm-output>/embeddings/<model>/<dataset>/raw/
      latent.npy
      obs.parquet | obs.csv.gz | obs.csv
      meta.json

Output layout matches ``prepare_fm_data.py``:

    <out-dir>/<dataset>.h5
      ctrl/emb, ctrl/offsets
      gt/emb, gt/offsets
      conditions
    <out-dir>/manifest.json
    <out-dir>/ctrl_means.npz
    <out-dir>/pert_means.npz

For each perturbation condition, the source pool is sampled from all control
cells with replacement to match the number of GT cells for that condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable

for var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(var, "1")

import h5py
import numpy as np
import pandas as pd


CHUNK_ROWS = 256
GZIP_LEVEL = 1


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _stable_seed(seed: int, *parts: object) -> int:
    text = "::".join([str(seed), *(str(p) for p in parts)])
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _load_obs(raw_dir: Path) -> pd.DataFrame:
    meta_path = raw_dir / "meta.json"
    candidates: list[Path] = []
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            art = str(meta.get("obs_artifact", "") or "").strip()
            if art:
                candidates.append(raw_dir / art)
        except Exception:
            pass
    candidates.extend([raw_dir / "obs.parquet", raw_dir / "obs.csv.gz", raw_dir / "obs.csv"])
    for p in candidates:
        if not p.is_file():
            continue
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        return pd.read_csv(p)
    raise FileNotFoundError(f"missing obs sidecar in {raw_dir}")


def _pick_condition_col(obs: pd.DataFrame, explicit: str = "") -> str:
    if explicit:
        if explicit not in obs.columns:
            raise KeyError(f"--condition-col {explicit!r} not found in obs")
        return explicit
    for c in ("condition", "perturbation", "cov_drug", "gene", "target"):
        if c in obs.columns:
            return c
    raise KeyError("could not infer condition column; pass --condition-col")


def _control_mask(obs: pd.DataFrame, control_col: str = "") -> np.ndarray:
    if control_col:
        if control_col not in obs.columns:
            raise KeyError(f"--control-col {control_col!r} not found in obs")
        vals = obs[control_col]
        if pd.api.types.is_bool_dtype(vals):
            return vals.to_numpy(dtype=bool)
        if pd.api.types.is_numeric_dtype(vals):
            return vals.fillna(0).to_numpy(dtype=float) > 0
        s = vals.astype(str).str.lower().str.strip()
        return s.isin({"1", "true", "yes", "control", "ctrl"}).to_numpy()

    for c in ("is_control", "control"):
        if c in obs.columns:
            return _control_mask(obs, c)

    for c in ("condition", "perturbation", "cov_drug", "gene", "target"):
        if c in obs.columns:
            s = obs[c].astype(str).str.lower().str.strip()
            return s.isin({"control", "ctrl", "vehicle", "dmso", "non-targeting", "non_targeting"}).to_numpy()

    raise KeyError("could not infer control cells; pass --control-col")


def _clean_str(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    s = str(value).strip()
    return "" if s.lower() in {"", "nan", "none", "<na>"} else s


def _first_obs_value(obs: pd.DataFrame, row_idx: int, candidates: tuple[str, ...]) -> str:
    for c in candidates:
        if c not in obs.columns:
            continue
        val = _clean_str(obs.iloc[int(row_idx)][c])
        if val:
            return val
    return ""


def _looks_like_drug_dataset(dataset_id: str, cond_col: str) -> bool:
    d = dataset_id.lower()
    c = cond_col.lower()
    if any(tok in d for tok in ("sciplex", "chempert", "chemical", "drug")):
        return True
    return c in {"drug", "cov_drug", "compound", "chemical", "drugname_drugconc", "smiles"}


def _condition_metadata_for_export(
    *,
    obs: pd.DataFrame,
    dataset_id: str,
    cond: str,
    first_gt_idx: int,
    cond_col: str,
    perturbation_type: str,
    chem_obs_column: str,
) -> dict:
    ptype = _clean_str(perturbation_type)
    if not ptype:
        ptype = _first_obs_value(
            obs,
            first_gt_idx,
            ("perturbation_type", "pert_type", "perturbation_kind", "modality"),
        )
    if not ptype and _looks_like_drug_dataset(dataset_id, cond_col):
        ptype = "drug"

    chem_obs_value = ""
    ccol = _clean_str(chem_obs_column)
    if ccol:
        if ccol not in obs.columns:
            raise KeyError(f"--chem-obs-column {ccol!r} not found in obs")
        chem_obs_value = _clean_str(obs.iloc[int(first_gt_idx)][ccol])
    if not chem_obs_value and str(ptype).strip().lower() in {"drug", "chemical", "compound", "small molecule"}:
        chem_obs_value = str(cond)

    genes: list[str] = []
    if str(ptype).strip().lower() not in {"drug", "chemical", "compound", "small molecule"}:
        gene_val = _first_obs_value(obs, first_gt_idx, ("gene", "target", "perturbation"))
        if gene_val:
            genes = [g.strip().upper() for part in gene_val.replace(",", "+").replace("|", "+").split("+") for g in part.split() if g.strip()]

    meta = {
        "perturbation_type_raw": ptype or None,
        "genes": genes,
        "chem_obs_value": chem_obs_value or None,
        "chem_source": (f"drug={chem_obs_value}" if chem_obs_value else None),
        "condition_col": cond_col,
    }
    return {k: v for k, v in meta.items() if v is not None}


def _iter_chunks(indices: np.ndarray, chunk_rows: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), chunk_rows):
        yield indices[start : start + chunk_rows]


def _write_indexed_rows(out_ds, latent: np.ndarray, indices: np.ndarray, out_start: int) -> None:
    pos = int(out_start)
    for idx in _iter_chunks(indices, CHUNK_ROWS):
        # h5py/numpy advanced indexing works best with monotonic indices. Restore
        # the requested sampled order after reading the sorted block.
        order = np.argsort(idx, kind="mergesort")
        sorted_idx = idx[order]
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        block = np.asarray(latent[sorted_idx], dtype=np.float32)[inv]
        out_ds[pos : pos + len(idx)] = block
        pos += len(idx)


def _dataset_mean(h5_path: Path, key: str, chunk_rows: int = 20_000) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        ds = f[key]
        n = int(ds.shape[0])
        dim = int(ds.shape[1])
        if n == 0:
            return np.zeros((dim,), dtype=np.float32)
        acc = np.zeros((dim,), dtype=np.float64)
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            acc += np.asarray(ds[start:end], dtype=np.float32).sum(axis=0, dtype=np.float64)
    return (acc / max(1, n)).astype(np.float32)


def process_one(
    *,
    raw_dir: Path,
    out_path: Path,
    dataset_id: str,
    condition_col: str,
    control_col: str,
    perturbation_type: str,
    chem_obs_column: str,
    seed: int,
    min_gt_cells: int,
    max_conditions: int,
    max_cells_per_condition: int,
    force: bool,
) -> dict:
    if out_path.exists() and not force:
        with h5py.File(out_path, "r") as f:
            conds = f["conditions"].asstr()[:].tolist()
            c_off = f["ctrl/offsets"][:]
            g_off = f["gt/offsets"][:]
            dim = int(f["gt/emb"].shape[1])
        return {
            "dataset_id": dataset_id,
            "status": "already_done",
            "n_conds": len(conds),
            "n_src": int(c_off[-1]),
            "n_gt": int(g_off[-1]),
            "src_per_cond": int(c_off[1] - c_off[0]) if len(conds) else 0,
            "emb_dim": dim,
            "conditions": conds,
            "out_path": str(out_path),
        }

    latent_path = raw_dir / "latent.npy"
    if not latent_path.is_file():
        raise FileNotFoundError(f"missing latent.npy: {latent_path}")
    obs = _load_obs(raw_dir)
    latent = np.load(latent_path, mmap_mode="r")
    if latent.ndim != 2:
        raise ValueError(f"latent must be 2D, got {latent.shape} in {latent_path}")
    if len(obs) != latent.shape[0]:
        raise ValueError(f"obs/latent row mismatch in {raw_dir}: obs={len(obs)} latent={latent.shape[0]}")

    cond_col = _pick_condition_col(obs, condition_col)
    is_ctrl = _control_mask(obs, control_col)
    ctrl_idx = np.flatnonzero(is_ctrl)
    if len(ctrl_idx) == 0:
        raise ValueError(f"no control cells found in {raw_dir}")

    cond_series = obs[cond_col].astype(str).fillna("")
    conds = []
    gt_indices = []
    for cond in sorted(c for c in cond_series.unique().tolist() if c and c.lower() != "nan"):
        mask = (cond_series.to_numpy() == cond) & (~is_ctrl)
        idx = np.flatnonzero(mask)
        if len(idx) < min_gt_cells:
            continue
        if max_cells_per_condition > 0 and len(idx) > max_cells_per_condition:
            rng_sub = np.random.default_rng(_stable_seed(seed, dataset_id, cond, "gt"))
            idx = np.sort(rng_sub.choice(idx, size=max_cells_per_condition, replace=False))
        conds.append(cond)
        gt_indices.append(idx.astype(np.int64))
        if max_conditions > 0 and len(conds) >= max_conditions:
            break
    if not conds:
        raise ValueError(f"no non-control conditions with >= {min_gt_cells} GT cells in {raw_dir}")

    gt_counts = np.array([len(x) for x in gt_indices], dtype=np.int64)
    gt_offsets = np.zeros(len(conds) + 1, dtype=np.int64)
    gt_offsets[1:] = np.cumsum(gt_counts)
    ctrl_offsets = gt_offsets.copy()
    total = int(gt_offsets[-1])
    dim = int(latent.shape[1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    rng = np.random.default_rng(_stable_seed(seed, dataset_id, "ctrl"))
    with h5py.File(tmp, "w") as f:
        ctrl_ds = f.create_dataset(
            "ctrl/emb",
            shape=(total, dim),
            dtype="float32",
            chunks=(min(CHUNK_ROWS, max(1, total)), dim),
            compression="gzip",
            compression_opts=GZIP_LEVEL,
        )
        gt_ds = f.create_dataset(
            "gt/emb",
            shape=(total, dim),
            dtype="float32",
            chunks=(min(CHUNK_ROWS, max(1, total)), dim),
            compression="gzip",
            compression_opts=GZIP_LEVEL,
        )
        for i, (cond, gt_idx) in enumerate(zip(conds, gt_indices)):
            start = int(gt_offsets[i])
            n = len(gt_idx)
            sampled_ctrl = rng.choice(ctrl_idx, size=n, replace=True).astype(np.int64)
            _write_indexed_rows(ctrl_ds, latent, sampled_ctrl, start)
            _write_indexed_rows(gt_ds, latent, gt_idx, start)
            if (i + 1) % 50 == 0 or i + 1 == len(conds):
                print(f"    [{_ts()}] {dataset_id}: {i + 1}/{len(conds)} conds rows={start + n:,}", flush=True)
        f.create_dataset("ctrl/offsets", data=ctrl_offsets)
        f.create_dataset("gt/offsets", data=gt_offsets)
        f.create_dataset("conditions", data=np.array(conds, dtype=object), dtype=h5py.string_dtype())
        f.attrs["source_raw_dir"] = str(raw_dir)
        f.attrs["condition_col"] = cond_col
        f.attrs["n_control_pool"] = int(len(ctrl_idx))

    if out_path.exists():
        out_path.unlink()
    tmp.rename(out_path)
    condition_metadata = {
        cond: _condition_metadata_for_export(
            obs=obs,
            dataset_id=dataset_id,
            cond=cond,
            first_gt_idx=int(gt_idx[0]),
            cond_col=cond_col,
            perturbation_type=perturbation_type,
            chem_obs_column=chem_obs_column,
        )
        for cond, gt_idx in zip(conds, gt_indices)
    }
    return {
        "dataset_id": dataset_id,
        "status": "ok",
        "n_conds": len(conds),
        "n_src": total,
        "n_gt": total,
        "src_per_cond": int(gt_counts[0]) if len(gt_counts) else 0,
        "emb_dim": dim,
        "conditions": conds,
        "out_path": str(out_path),
        "condition_col": cond_col,
        "n_control_pool": int(len(ctrl_idx)),
        "condition_metadata": condition_metadata,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings-root", type=Path, default=Path("/data/cyx/1030/scLatent/scFM_output/embeddings"))
    ap.add_argument("--model", required=True, help="scFMBench model name, e.g. stack")
    ap.add_argument("--datasets", nargs="*", default=None, help="Dataset IDs; default: all under model")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--condition-col", default="")
    ap.add_argument("--control-col", default="")
    ap.add_argument(
        "--perturbation-type",
        default="",
        help="Optional fixed perturbation type for all exported conditions, e.g. drug or CRISPRi.",
    )
    ap.add_argument(
        "--chem-obs-column",
        default="",
        help="Optional obs column to use as the chemical cache key; defaults to condition name for drug datasets.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-gt-cells", type=int, default=16)
    ap.add_argument("--max-conditions", type=int, default=0)
    ap.add_argument("--max-cells-per-condition", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    model_root = args.embeddings_root / args.model
    if args.datasets:
        datasets = list(args.datasets)
    else:
        datasets = sorted(p.name for p in model_root.iterdir() if (p / "raw").is_dir())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Prepare scFMBench LatentFM data")
    print(f"  model: {args.model}")
    print(f"  input: {model_root}")
    print(f"  out:   {args.out_dir}")
    print(f"  N:     {len(datasets)}")
    print("=" * 72, flush=True)

    manifest = {
        "source": "scFMBench",
        "model": args.model,
        "embeddings_root": str(args.embeddings_root),
        "emb_dim": None,
        "datasets": {},
        "total_conditions": 0,
        "total_src_cells": 0,
        "total_gt_cells": 0,
        "seed": int(args.seed),
        "min_gt_cells": int(args.min_gt_cells),
        "max_conditions": int(args.max_conditions),
        "max_cells_per_condition": int(args.max_cells_per_condition),
        "perturbation_type": str(args.perturbation_type or ""),
        "chem_obs_column": str(args.chem_obs_column or ""),
    }
    condition_metadata: dict[str, dict[str, dict]] = {}
    ok = skip = fail = 0
    for ds in datasets:
        raw_dir = model_root / ds / "raw"
        out_path = args.out_dir / f"{ds}.h5"
        try:
            info = process_one(
                raw_dir=raw_dir,
                out_path=out_path,
                dataset_id=ds,
                condition_col=args.condition_col,
                control_col=args.control_col,
                perturbation_type=args.perturbation_type,
                chem_obs_column=args.chem_obs_column,
                seed=args.seed,
                min_gt_cells=args.min_gt_cells,
                max_conditions=args.max_conditions,
                max_cells_per_condition=args.max_cells_per_condition,
                force=args.force,
            )
            if info["status"] == "already_done":
                skip += 1
            else:
                ok += 1
            manifest["emb_dim"] = info["emb_dim"]
            manifest["datasets"][ds] = {
                "n_conds": info["n_conds"],
                "n_src": info["n_src"],
                "n_gt": info["n_gt"],
                "src_per_cond": info["src_per_cond"],
                "conditions": info["conditions"],
                "out_path": info["out_path"],
            }
            condition_metadata[ds] = info.get("condition_metadata", {})
            manifest["total_conditions"] += int(info["n_conds"])
            manifest["total_src_cells"] += int(info["n_src"])
            manifest["total_gt_cells"] += int(info["n_gt"])
            print(f"[{ds}] {info['status']} conds={info['n_conds']} rows={info['n_gt']:,} dim={info['emb_dim']}", flush=True)
        except Exception as exc:
            fail += 1
            print(f"[{ds}] ERROR: {type(exc).__name__}: {exc}", flush=True)

    ctrl_means = {}
    pert_means = {}
    for ds in sorted(manifest["datasets"]):
        h5_path = args.out_dir / f"{ds}.h5"
        if h5_path.is_file():
            ctrl_means[ds] = _dataset_mean(h5_path, "ctrl/emb")
            pert_means[ds] = _dataset_mean(h5_path, "gt/emb")
    if ctrl_means:
        np.savez_compressed(args.out_dir / "ctrl_means.npz", **ctrl_means)
    if pert_means:
        np.savez_compressed(args.out_dir / "pert_means.npz", **pert_means)
    (args.out_dir / "condition_metadata.json").write_text(
        json.dumps(condition_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest["condition_metadata_file"] = str(args.out_dir / "condition_metadata.json")
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Done ok={ok} skip={skip} fail={fail} manifest={args.out_dir / 'manifest.json'}", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
