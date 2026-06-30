#!/usr/bin/env python3
"""Evaluate a LatentFM checkpoint by condition and cell background.

This eval-only CLI is intentionally separate from the canonical split/family
evaluators. It is meant for Jiang xverse diagnostics where the LatentFM HDF5s
were converted from scFMBench raw sidecars that still contain `cell_type`.

No training, checkpoint selection, or split mutation is performed.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

from model.latent.config import Config
from model.latent.dataset import CrossDatasetFMDataset
from model.latent.eval_split_groups import (
    _group_as_test_split,
    _json_default,
    _load_cfg,
    _load_manifest,
    _load_means_file,
    _load_split,
    _resolve_means_file,
)
from model.latent.fm_ot import median_sigmas, mmd2_biased, mmd2_unbiased
from model.latent.train import (
    _cross_dataset_kw,
    _pearson_np,
    _pert_chunk,
    _pert_for_eval_batch,
    _pert_to_device,
    build_model,
    checkpoint_ema_is_active,
    load_model_weights_only,
    ode_integrate,
)
from model.utils.train.ema import ModelEMA


DEFAULT_GROUPS = ("test_single",)
DEFAULT_DATASETS = ("Jiang_IFNB", "Jiang_IFNG", "Jiang_INS", "Jiang_TGFB", "Jiang_TNFA")


def _stable_seed(seed: int, *parts: object) -> int:
    text = "::".join([str(seed), *(str(p) for p in parts)])
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _eval_rank(seed: int, *parts: object) -> int:
    text = "background_eval:" + ":".join(str(p) for p in (seed, *parts))
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def _clean_str(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def _control_mask(obs: pd.DataFrame) -> np.ndarray:
    for col in ("is_control", "control"):
        if col not in obs.columns:
            continue
        vals = obs[col]
        if pd.api.types.is_bool_dtype(vals):
            return vals.to_numpy(dtype=bool)
        if pd.api.types.is_numeric_dtype(vals):
            return vals.fillna(0).to_numpy(dtype=float) > 0
        s = vals.astype(str).str.lower().str.strip()
        return s.isin({"1", "true", "yes", "control", "ctrl"}).to_numpy()
    for col in ("condition", "perturbation", "cov_drug", "gene", "target"):
        if col in obs.columns:
            s = obs[col].astype(str).str.lower().str.strip()
            return s.isin({"control", "ctrl", "vehicle", "dmso", "non-targeting", "non_targeting"}).to_numpy()
    raise KeyError("could not infer control cells from obs sidecar")


def _read_obs(raw_dir: Path) -> pd.DataFrame:
    meta_path = raw_dir / "meta.json"
    candidates: list[Path] = []
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            artifact = str(meta.get("obs_artifact", "") or "").strip()
            if artifact:
                candidates.append(raw_dir / artifact)
        except Exception:
            pass
    candidates.extend([raw_dir / "obs.parquet", raw_dir / "obs.csv.gz", raw_dir / "obs.csv"])
    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    raise FileNotFoundError(f"missing obs sidecar in {raw_dir}")


def _available_conditions(split: dict[str, dict[str, list[str]]], datasets: list[str], groups: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for ds_name in datasets:
        ds_split = split.get(ds_name) or {}
        conds = []
        for group in groups:
            conds.extend(str(c) for c in ds_split.get(group, []) if str(c))
        if conds:
            out[ds_name] = sorted(dict.fromkeys(conds))
    return out


class BackgroundIndex:
    """Reconstruct per-condition source/GT background row indices."""

    def __init__(
        self,
        *,
        h5_path: Path,
        manifest_seed: int,
        max_cells_per_condition: int,
        background_column: str,
    ) -> None:
        self.h5_path = h5_path
        self.manifest_seed = int(manifest_seed)
        self.max_cells_per_condition = int(max_cells_per_condition)
        self.background_column = str(background_column)
        self.by_condition: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        self._build()

    def _build(self) -> None:
        with h5py.File(self.h5_path, "r") as handle:
            conds = [str(c) for c in handle["conditions"].asstr()[:].tolist()]
            raw_dir_s = str(handle.attrs.get("source_raw_dir", "") or "")
            cond_col = str(handle.attrs.get("condition_col", "") or "perturbation")
            dataset_id = self.h5_path.stem
        if not raw_dir_s:
            raise ValueError(f"{self.h5_path} lacks source_raw_dir attr")
        obs = _read_obs(Path(raw_dir_s))
        if self.background_column not in obs.columns:
            raise KeyError(f"{self.background_column!r} not in obs sidecar for {dataset_id}")
        if cond_col not in obs.columns:
            raise KeyError(f"{cond_col!r} not in obs sidecar for {dataset_id}")
        cond_series = obs[cond_col].astype(str).fillna("").to_numpy()
        bg_series = obs[self.background_column].map(_clean_str).to_numpy()
        is_ctrl = _control_mask(obs)
        ctrl_idx = np.flatnonzero(is_ctrl)
        rng_ctrl = np.random.default_rng(_stable_seed(self.manifest_seed, dataset_id, "ctrl"))
        for cond in conds:
            mask = (cond_series == cond) & (~is_ctrl)
            gt_idx = np.flatnonzero(mask)
            if self.max_cells_per_condition > 0 and len(gt_idx) > self.max_cells_per_condition:
                rng_sub = np.random.default_rng(_stable_seed(self.manifest_seed, dataset_id, cond, "gt"))
                gt_idx = np.sort(rng_sub.choice(gt_idx, size=self.max_cells_per_condition, replace=False))
            sampled_ctrl = rng_ctrl.choice(ctrl_idx, size=len(gt_idx), replace=True).astype(np.int64)
            per_bg: dict[str, dict[str, np.ndarray]] = {}
            backgrounds = sorted({str(v) for v in np.concatenate([bg_series[gt_idx], bg_series[sampled_ctrl]]) if str(v)})
            for bg in backgrounds:
                gt_rel = np.flatnonzero(bg_series[gt_idx] == bg).astype(np.int64)
                src_rel = np.flatnonzero(bg_series[sampled_ctrl] == bg).astype(np.int64)
                if gt_rel.size or src_rel.size:
                    per_bg[bg] = {"gt": gt_rel, "src": src_rel}
            self.by_condition[cond] = per_bg

    def backgrounds(self, cond: str) -> list[str]:
        return sorted(self.by_condition.get(cond, {}).keys())

    def rel_indices(self, cond: str, background: str) -> tuple[np.ndarray, np.ndarray]:
        rec = self.by_condition.get(cond, {}).get(background)
        if rec is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return rec["src"], rec["gt"]


def _mean_dict(path: Path) -> dict[str, np.ndarray] | None:
    return _load_means_file(path)


def _evaluate_rows(
    *,
    model: torch.nn.Module,
    dataset: CrossDatasetFMDataset,
    bg_indices: dict[str, BackgroundIndex],
    tasks: list[tuple[str, str, str]],
    cfg: Config,
    device: torch.device,
    ctrl_means: dict[str, np.ndarray] | None,
    pert_means: dict[str, np.ndarray] | None,
    ode_steps: int,
    max_chunk: int,
    eval_max_mse_cells: int,
    eval_max_mmd_cells: int,
    min_bg_cells: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    use_pert = bool(getattr(cfg, "use_pert_condition", False))
    model.eval()
    for ds_name, cond, bg in tasks:
        handle = dataset.handles.get(ds_name)
        if handle is None:
            skipped.append({"dataset": ds_name, "condition": cond, "cell_background": bg, "reason": "missing_dataset_handle"})
            continue
        src_rel, gt_rel = bg_indices[ds_name].rel_indices(cond, bg)
        if src_rel.size < min_bg_cells or gt_rel.size < min_bg_cells:
            skipped.append(
                {
                    "dataset": ds_name,
                    "condition": cond,
                    "cell_background": bg,
                    "reason": "background_cell_count_below_min",
                    "n_src_bg": int(src_rel.size),
                    "n_gt_bg": int(gt_rel.size),
                }
            )
            continue
        seed = int(getattr(cfg, "seed", 0) or 0)
        rng_mse = np.random.RandomState(_eval_rank(seed, "mse", ds_name, cond, bg) % (2**32 - 1))
        rng_mmd = np.random.RandomState(_eval_rank(seed, "mmd", ds_name, cond, bg) % (2**32 - 1))

        n_mse = min(int(src_rel.size), int(gt_rel.size))
        if eval_max_mse_cells > 0:
            n_mse = min(n_mse, int(eval_max_mse_cells))
        src_mse_rel = src_rel[rng_mse.permutation(src_rel.size)[:n_mse]]
        gt_mse_rel = gt_rel[rng_mse.permutation(gt_rel.size)[:n_mse]]
        src_mse = torch.from_numpy(handle.read_src_rows(cond, src_mse_rel)).float()
        gt_mse = torch.from_numpy(handle.read_gt_rows(cond, gt_mse_rel)).float()

        mse_sum = 0.0
        mae_sum = 0.0
        for start in range(0, n_mse, max_chunk):
            end = min(start + max_chunk, n_mse)
            src_c = src_mse[start:end].to(device)
            gt_c = gt_mse[start:end].to(device)
            batch = int(src_c.size(0))
            t = torch.rand(batch, device=device)
            x_t = (1.0 - t[:, None]) * src_c + t[:, None] * gt_c
            dx_t = gt_c - src_c
            pb_dev = None
            if use_pert:
                pb_cpu = _pert_for_eval_batch(dataset, ds_name, cond, batch)
                pb_dev = _pert_to_device(pb_cpu, device)
            from model.latent.train import _model_latent_velocity  # local import keeps CLI surface narrow

            v_pred = _model_latent_velocity(model, x_t, t, src_c, pb_dev)
            mse_sum += float(torch.nn.functional.mse_loss(v_pred, dx_t, reduction="sum").detach().cpu())
            mae_sum += float(torch.nn.functional.l1_loss(v_pred, dx_t, reduction="sum").detach().cpu())
        denom = max(1, n_mse * int(getattr(cfg, "emb_dim", src_mse.shape[1])))

        n_src_eval = min(int(src_rel.size), int(eval_max_mmd_cells) if eval_max_mmd_cells > 0 else int(src_rel.size))
        n_gt_eval = min(int(gt_rel.size), int(eval_max_mmd_cells) if eval_max_mmd_cells > 0 else int(gt_rel.size))
        src_eval_rel = src_rel[rng_mmd.permutation(src_rel.size)[:n_src_eval]]
        gt_eval_rel = gt_rel[rng_mmd.permutation(gt_rel.size)[:n_gt_eval]]
        src_eval = torch.from_numpy(handle.read_src_rows(cond, src_eval_rel)).float()
        gt_eval = torch.from_numpy(handle.read_gt_rows(cond, gt_eval_rel)).float()

        pred_parts = []
        pb_cpu_full = _pert_for_eval_batch(dataset, ds_name, cond, n_src_eval) if use_pert else None
        for start in range(0, n_src_eval, max_chunk):
            end = min(start + max_chunk, n_src_eval)
            src_c = src_eval[start:end].to(device)
            pb_dev = None
            if use_pert and pb_cpu_full is not None:
                pb_dev = _pert_to_device(_pert_chunk(pb_cpu_full, start, end), device)
            pred_parts.append(
                ode_integrate(
                    model,
                    src_c,
                    src_c,
                    cfg,
                    n_steps=int(ode_steps),
                    perturbation_batch=pb_dev if use_pert else None,
                ).detach().cpu()
            )
        pred = torch.cat(pred_parts, dim=0).to(device)
        gt_dev = gt_eval.to(device)
        sigmas, dyy = median_sigmas(gt_dev, return_D2=True)
        mmd_raw = float(mmd2_unbiased(pred, gt_dev, sigmas, Dyy=dyy).item())
        mmd_biased = float(mmd2_biased(pred, gt_dev, sigmas, Dyy=dyy).item())
        pred_mean = pred.mean(dim=0).detach().cpu().numpy()
        gt_mean = gt_eval.mean(dim=0).numpy()
        direct = _pearson_np(pred_mean, gt_mean)
        ctrl_mean = None if ctrl_means is None else ctrl_means.get(ds_name)
        pert_mean = None if pert_means is None else pert_means.get(ds_name)
        p_ctrl = None if ctrl_mean is None else _pearson_np(pred_mean - ctrl_mean, gt_mean - ctrl_mean)
        p_pert = None if pert_mean is None else _pearson_np(pred_mean - pert_mean, gt_mean - pert_mean)
        rows.append(
            {
                "dataset": ds_name,
                "condition": cond,
                "cell_background": bg,
                "test_mmd": mmd_raw,
                "test_mmd_biased": mmd_biased,
                "test_mmd_clamped": max(mmd_raw, 0.0),
                "direct_pearson": direct,
                "pearson_ctrl": p_ctrl,
                "pearson_pert": p_pert,
                "eval_mse": mse_sum / float(denom),
                "eval_mae": mae_sum / float(denom),
                "n_src_bg": int(src_rel.size),
                "n_gt_bg": int(gt_rel.size),
                "n_src_eval": int(n_src_eval),
                "n_gt_eval": int(n_gt_eval),
                "source_pool_mode": "converted_h5_sampled_controls_filtered_by_obs_background",
            }
        )
    model.train()
    return rows, skipped


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_rows": 0}
    metrics = ("pearson_pert", "pearson_ctrl", "direct_pearson", "test_mmd_clamped")
    summary: dict[str, Any] = {
        "n_rows": len(rows),
        "datasets": len({r["dataset"] for r in rows}),
        "conditions": len({(r["dataset"], r["condition"]) for r in rows}),
        "backgrounds": len({r["cell_background"] for r in rows}),
    }
    for metric in metrics:
        values = [float(r[metric]) for r in rows if r.get(metric) is not None and math.isfinite(float(r[metric]))]
        if values:
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_median"] = float(np.median(values))
            by_ds = defaultdict(list)
            by_bg = defaultdict(list)
            for row in rows:
                val = row.get(metric)
                if val is None:
                    continue
                by_ds[str(row["dataset"])].append(float(val))
                by_bg[str(row["cell_background"])].append(float(val))
            summary[f"{metric}_dataset_means"] = {k: float(np.mean(v)) for k, v in sorted(by_ds.items())}
            summary[f"{metric}_background_means"] = {k: float(np.mean(v)) for k, v in sorted(by_bg.items())}
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-file", type=Path, required=True)
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--biflow-dir", type=str, required=True)
    ap.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    ap.add_argument("--groups", nargs="*", default=list(DEFAULT_GROUPS))
    ap.add_argument("--background-column", default="cell_type")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--device", default="")
    ap.add_argument("--ode-steps", type=int, default=20)
    ap.add_argument("--max-chunk", type=int, default=512)
    ap.add_argument("--eval-max-mse-cells", type=int, default=512)
    ap.add_argument("--eval-max-mmd-cells", type=int, default=512)
    ap.add_argument("--min-bg-cells", type=int, default=16)
    ap.add_argument("--max-background-rows", type=int, default=0, help="Debug cap after deterministic sorting; 0=all")
    ap.add_argument("--eval-seed", type=int, default=None)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--pert-means-file", default="")
    args = ap.parse_args()

    ckpt_path = args.checkpoint.expanduser().resolve()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"checkpoint must contain a model key: {ckpt_path}")
    cfg = _load_cfg(ckpt, data_dir=args.data_dir, biflow_dir=args.biflow_dir)
    cfg.gpu = int(args.gpu)
    if args.eval_seed is not None:
        cfg.seed = int(args.eval_seed)
    cfg.eval_max_mse_cells = int(args.eval_max_mse_cells)
    cfg.eval_max_mmd_cells = int(args.eval_max_mmd_cells)

    device_s = str(args.device).strip() or (f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)
    data_dir = Path(args.data_dir).expanduser().resolve()
    split_path = args.split_file.expanduser().resolve()
    manifest = _load_manifest(data_dir, cfg.manifest)
    split = _load_split(split_path)
    datasets = [str(ds) for ds in args.datasets]
    groups = [str(g) for g in args.groups]
    conds_by_ds = _available_conditions(split, datasets, groups)
    group_split = {
        ds_name: {"train": [], "test": [c for c in conds if c in set(map(str, manifest["datasets"].get(ds_name, {}).get("conditions", [])))]}
        for ds_name, conds in conds_by_ds.items()
        if ds_name in manifest.get("datasets", {})
    }
    cd_kw = _cross_dataset_kw(cfg)
    dataset = CrossDatasetFMDataset(
        str(data_dir),
        group_split,
        cfg.batch_size,
        cfg.seed,
        mode="test",
        min_cells=1,
        ds_alpha=1.0,
        silent=True,
        **cd_kw,
    )
    manifest_seed = int(manifest.get("seed", cfg.seed))
    max_cells = int(manifest.get("max_cells_per_condition", 0) or 0)
    bg_indices = {
        ds_name: BackgroundIndex(
            h5_path=data_dir / f"{ds_name}.h5",
            manifest_seed=manifest_seed,
            max_cells_per_condition=max_cells,
            background_column=str(args.background_column),
        )
        for ds_name in dataset.ds_names
    }
    tasks = []
    for ds_name in sorted(dataset.ds_names):
        for cond in sorted(dataset.ds_conds.get(ds_name, [])):
            for bg in bg_indices[ds_name].backgrounds(cond):
                tasks.append((ds_name, cond, bg))
    tasks = sorted(tasks)
    if int(args.max_background_rows) > 0:
        tasks = tasks[: int(args.max_background_rows)]

    ctrl_means = _mean_dict(data_dir / "ctrl_means.npz")
    pert_means = _mean_dict(
        _resolve_means_file(str(args.pert_means_file or ""), data_dir=data_dir, default_name="pert_means.npz")
    )

    model = build_model(cfg, device)
    missing_keys, unexpected_keys, skipped_shape_mismatch = load_model_weights_only(
        ckpt_path, model, device, strict=False
    )
    ema = None
    if not args.no_ema and checkpoint_ema_is_active(ckpt, cfg):
        ema = ModelEMA(
            model,
            decay=float(getattr(cfg, "ema_decay", 0.999)),
            update_after=int(getattr(cfg, "ema_update_after", 0)),
            update_every=int(getattr(cfg, "ema_update_every", 1)),
            device=device,
        )
        ema.load_state_dict(ckpt["ema"], strict=False)

    if ema is not None:
        with ema.apply_to(model):
            rows, skipped = _evaluate_rows(
                model=model,
                dataset=dataset,
                bg_indices=bg_indices,
                tasks=tasks,
                cfg=cfg,
                device=device,
                ctrl_means=ctrl_means,
                pert_means=pert_means,
                ode_steps=int(args.ode_steps),
                max_chunk=int(args.max_chunk),
                eval_max_mse_cells=int(args.eval_max_mse_cells),
                eval_max_mmd_cells=int(args.eval_max_mmd_cells),
                min_bg_cells=int(args.min_bg_cells),
            )
    else:
        rows, skipped = _evaluate_rows(
            model=model,
            dataset=dataset,
            bg_indices=bg_indices,
            tasks=tasks,
            cfg=cfg,
            device=device,
            ctrl_means=ctrl_means,
            pert_means=pert_means,
            ode_steps=int(args.ode_steps),
            max_chunk=int(args.max_chunk),
            eval_max_mse_cells=int(args.eval_max_mse_cells),
            eval_max_mmd_cells=int(args.eval_max_mmd_cells),
            min_bg_cells=int(args.min_bg_cells),
        )

    payload = {
        "checkpoint": str(ckpt_path),
        "checkpoint_step": ckpt.get("step"),
        "data_dir": str(data_dir),
        "split_file": str(split_path),
        "datasets": datasets,
        "groups": groups,
        "background_column": str(args.background_column),
        "manifest_seed": manifest_seed,
        "max_cells_per_condition": max_cells,
        "eval_caps": {
            "max_background_rows": int(args.max_background_rows),
            "max_mse_cells": int(args.eval_max_mse_cells),
            "max_mmd_cells": int(args.eval_max_mmd_cells),
            "min_bg_cells": int(args.min_bg_cells),
            "ode_steps": int(args.ode_steps),
        },
        "used_ema": ema is not None,
        "load_state": {
            "strict": False,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "skipped_shape_mismatch": skipped_shape_mismatch,
        },
        "condition_background_metrics": rows,
        "skipped_backgrounds": skipped,
        "summary": _summarize(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "rows": len(rows), "skipped": len(skipped)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
