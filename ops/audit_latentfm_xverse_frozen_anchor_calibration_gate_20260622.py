#!/usr/bin/env python3
"""Frozen-anchor prediction-space calibration gate for xverse LatentFM.

This read-only gate runs the frozen xverse anchor on train-only v2 rows, fits
simple prediction-space calibrators on train-single conditions, and evaluates
them on train-only internal proxy groups. It does not train the flow and does
not read canonical test outcomes.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model.latent.dataset import CrossDatasetFMDataset
from model.latent.eval_split_groups import _load_cfg, _load_manifest, _load_split
from model.latent.fm_ot import CondOTPath
from model.latent.train import (
    _cross_dataset_kw,
    _pearson_np,
    _pert_for_eval_batch,
    _pert_to_device,
    build_model,
    checkpoint_ema_is_active,
    load_model_weights_only,
    ode_integrate,
)
from model.utils.train.ema import ModelEMA


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_CHECKPOINT = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    "xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
)
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
DEFAULT_PERT_MEANS = (
    ROOT
    / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/"
    "xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz"
)
DEFAULT_RUN_ROOT = ROOT / "runs/latentfm_xverse_frozen_anchor_calibration_20260622"
DEFAULT_GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
MODES = ("identity", "global_scalar", "diag_affine", "diag_affine_shuffled_target")


def stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def genes_for(metadata: dict[str, Any], ds: str, cond: str) -> list[str]:
    entry = (metadata.get(ds) or {}).get(cond) or {}
    return [str(g).strip() for g in entry.get("genes") or [] if str(g).strip()]


def select_train_single_split(
    split: dict[str, dict[str, list[str]]],
    metadata: dict[str, Any],
    manifest: dict[str, Any],
    *,
    max_per_dataset: int,
    seed: int,
) -> dict[str, dict[str, list[str]]]:
    out = {}
    for ds, obj in sorted(split.items()):
        allowed = set(map(str, (manifest.get("datasets", {}).get(ds, {}) or {}).get("conditions", [])))
        conds = [str(c) for c in obj.get("train", []) if str(c) in allowed and len(genes_for(metadata, ds, str(c))) == 1]
        conds = sorted(conds, key=lambda c: hashlib.sha256(f"caltrain|{seed}|{ds}|{c}".encode()).hexdigest())
        if max_per_dataset > 0:
            conds = conds[:max_per_dataset]
        if conds:
            out[ds] = {"train": [], "test": conds}
    return out


def select_group_split(
    split: dict[str, dict[str, list[str]]],
    manifest: dict[str, Any],
    group: str,
    *,
    max_per_dataset: int,
    seed: int,
) -> dict[str, dict[str, list[str]]]:
    out = {}
    for ds, obj in sorted(split.items()):
        allowed = set(map(str, (manifest.get("datasets", {}).get(ds, {}) or {}).get("conditions", [])))
        conds = [str(c) for c in obj.get(group, []) if str(c) in allowed]
        conds = sorted(conds, key=lambda c: hashlib.sha256(f"calval|{seed}|{group}|{ds}|{c}".encode()).hexdigest())
        if max_per_dataset > 0:
            conds = conds[:max_per_dataset]
        if conds:
            out[ds] = {"train": [], "test": conds}
    return out


@torch.no_grad()
def predict_rows(
    *,
    model: torch.nn.Module,
    dataset: CrossDatasetFMDataset,
    cfg: Any,
    device: torch.device,
    group_split: dict[str, dict[str, list[str]]],
    group: str,
    max_cells: int,
    max_chunk: int,
    ode_steps: int,
) -> list[dict[str, Any]]:
    rows = []
    model.eval()
    for ds_name, parts in sorted(group_split.items()):
        handle = dataset.handles[ds_name]
        for cond in sorted(parts.get("test", [])):
            src_full = torch.from_numpy(handle.read_src(cond)).float()
            gt_full = torch.from_numpy(handle.read_gt(cond)).float()
            seed = stable_int(f"{getattr(cfg, 'seed', 0)}|{group}|{ds_name}|{cond}|cal")
            rng = np.random.default_rng(seed)
            n_src = min(int(src_full.size(0)), int(max_cells))
            n_gt = min(int(gt_full.size(0)), int(max_cells))
            src_eval = src_full[rng.permutation(src_full.size(0))[:n_src]]
            gt_eval = gt_full[rng.permutation(gt_full.size(0))[:n_gt]]
            ctrl_mean = src_eval.mean(dim=0).numpy().astype(np.float32)
            gt_mean = gt_eval.mean(dim=0).numpy().astype(np.float32)
            pb_cpu = _pert_for_eval_batch(dataset, ds_name, cond, int(n_src))
            pb_dev_full = _pert_to_device(pb_cpu, device)
            pred_parts = []
            for st in range(0, int(n_src), int(max_chunk)):
                en = min(st + int(max_chunk), int(n_src))
                src = src_eval[st:en].to(device)
                pb_use = tuple(None if x is None else x[st:en] for x in pb_dev_full)
                pred = ode_integrate(
                    model,
                    src,
                    src,
                    cfg,
                    n_steps=int(ode_steps),
                    perturbation_batch=pb_use,
                )
                pred_parts.append(pred.cpu())
            pred_mean = torch.cat(pred_parts, dim=0).mean(dim=0).numpy().astype(np.float32)
            rows.append(
                {
                    "dataset": ds_name,
                    "condition": cond,
                    "group": group,
                    "n_src_eval": int(n_src),
                    "n_gt_eval": int(n_gt),
                    "ctrl_mean": ctrl_mean,
                    "gt_mean": gt_mean,
                    "pred_mean": pred_mean,
                    "true_resid": (gt_mean - ctrl_mean).astype(np.float32),
                    "pred_resid": (pred_mean - ctrl_mean).astype(np.float32),
                }
            )
    return rows


def fit_scalar(pred: np.ndarray, true: np.ndarray) -> float:
    num = float(np.sum(pred * true))
    den = float(np.sum(pred * pred)) + 1e-8
    return num / den


def fit_diag_affine(pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    px = pred.astype(np.float64)
    ty = true.astype(np.float64)
    pm = px.mean(axis=0)
    tm = ty.mean(axis=0)
    cov = np.mean((px - pm) * (ty - tm), axis=0)
    var = np.mean((px - pm) ** 2, axis=0)
    scale = cov / np.maximum(var, 1e-8)
    bias = tm - scale * pm
    return scale.astype(np.float32), bias.astype(np.float32)


def calibrate(
    pred: np.ndarray,
    *,
    mode: str,
    scalar: float | None = None,
    diag: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    if mode == "identity":
        return pred.astype(np.float32)
    if mode == "global_scalar":
        assert scalar is not None
        return (float(scalar) * pred).astype(np.float32)
    if mode in {"diag_affine", "diag_affine_shuffled_target"}:
        assert diag is not None
        scale, bias = diag
        return (pred * scale + bias).astype(np.float32)
    raise ValueError(mode)


def score_pp(row: dict[str, Any], pred_resid: np.ndarray, pert_means: dict[str, np.ndarray]) -> float | None:
    pert = pert_means.get(str(row["dataset"]))
    if pert is None:
        return None
    pred_endpoint = row["ctrl_mean"] + pred_resid
    return float(_pearson_np(pred_endpoint - pert, row["gt_mean"] - pert))


def fit_eval_lodo(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    *,
    pert_means: dict[str, np.ndarray],
    seed: int,
) -> list[dict[str, Any]]:
    out = []
    datasets = sorted({str(r["dataset"]) for r in val_rows})
    for heldout in datasets:
        train = [r for r in train_rows if str(r["dataset"]) != heldout]
        vals = [r for r in val_rows if str(r["dataset"]) == heldout]
        if len(train) < 16 or not vals:
            continue
        pred_train = np.stack([r["pred_resid"] for r in train]).astype(np.float32)
        true_train = np.stack([r["true_resid"] for r in train]).astype(np.float32)
        scalar = fit_scalar(pred_train, true_train)
        diag = fit_diag_affine(pred_train, true_train)
        rng = np.random.default_rng(stable_int(f"calshuf|{seed}|{heldout}"))
        diag_shuf = fit_diag_affine(pred_train, true_train[rng.permutation(len(true_train))])
        for row in vals:
            for mode in MODES:
                pred = calibrate(
                    row["pred_resid"],
                    mode=mode,
                    scalar=scalar,
                    diag=diag_shuf if mode == "diag_affine_shuffled_target" else diag,
                )
                pp = score_pp(row, pred, pert_means)
                out.append(
                    {
                        "dataset": row["dataset"],
                        "condition": row["condition"],
                        "group": row["group"],
                        "mode": mode,
                        "pearson_pert_proxy": pp,
                    }
                )
    return out


def paired_bootstrap(rows: list[dict[str, Any]], group: str, candidate: str, baseline: str, *, n_boot: int, seed: int) -> dict[str, Any]:
    paired: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["group"] != group or row["pearson_pert_proxy"] is None:
            continue
        paired[(str(row["dataset"]), str(row["condition"]))][str(row["mode"])] = float(row["pearson_pert_proxy"])
    by_ds: dict[str, list[float]] = defaultdict(list)
    for (ds, _cond), vals in paired.items():
        if candidate in vals and baseline in vals:
            by_ds[ds].append(vals[candidate] - vals[baseline])
    datasets = sorted(ds for ds, vals in by_ds.items() if vals)
    if not datasets:
        return {"status": "missing", "group": group, "candidate": candidate, "baseline": baseline}
    point = float(np.mean([np.mean(by_ds[ds]) for ds in datasets]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(int(n_boot)):
        sample_ds = rng.choice(datasets, size=len(datasets), replace=True)
        means = []
        for ds in sample_ds:
            vals = np.asarray(by_ds[str(ds)], dtype=np.float64)
            means.append(float(np.mean(rng.choice(vals, size=len(vals), replace=True))))
        boot.append(float(np.mean(means)))
    arr = np.asarray(boot, dtype=np.float64)
    leave = {}
    for ds in datasets:
        rest = [d for d in datasets if d != ds]
        if rest:
            leave[ds] = float(np.mean([np.mean(by_ds[d]) for d in rest]))
    return {
        "status": "ok",
        "group": group,
        "candidate": candidate,
        "baseline": baseline,
        "n_conditions": int(sum(len(by_ds[d]) for d in datasets)),
        "n_datasets": int(len(datasets)),
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": float(np.mean(arr > 0.0)),
        "p_harm": float(np.mean(arr < 0.0)),
        "leave_one_min": min(leave.values()) if leave else None,
    }


def absolute_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for group in DEFAULT_GROUPS:
        for mode in MODES:
            by_ds: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                if row["group"] == group and row["mode"] == mode and row["pearson_pert_proxy"] is not None:
                    by_ds[str(row["dataset"])].append(float(row["pearson_pert_proxy"]))
            vals = [float(np.mean(v)) for v in by_ds.values() if v]
            out.append(
                {
                    "group": group,
                    "mode": mode,
                    "mean": None if not vals else float(np.mean(vals)),
                    "n_datasets": len(vals),
                    "n_conditions": sum(len(v) for v in by_ds.values()),
                }
            )
    return out


def decide(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    by = {(d["group"], d["candidate"], d["baseline"]): d for d in deltas}
    reasons = []
    cross = by.get((DEFAULT_GROUPS[0], "diag_affine", "identity")) or {}
    fam = by.get((DEFAULT_GROUPS[1], "diag_affine", "identity")) or {}
    cross_shuf = by.get((DEFAULT_GROUPS[0], "diag_affine", "diag_affine_shuffled_target")) or {}
    if cross.get("status") != "ok" or not (
        float(cross.get("p_improve") or 0.0) >= 0.90 or float((cross.get("ci95") or [0.0])[0]) > 0.0
    ):
        reasons.append("cross_background_diag_affine_not_supported")
    if fam.get("status") != "ok" or float(fam.get("p_harm") if fam.get("p_harm") is not None else 1.0) > 0.20:
        reasons.append("family_diag_affine_harm_risk")
    if cross_shuf.get("status") != "ok" or float(cross_shuf.get("delta_mean") or 0.0) <= 0.0:
        reasons.append("not_better_than_shuffled_target_control")
    if cross.get("leave_one_min") is None or float(cross["leave_one_min"]) <= 0.0:
        reasons.append("cross_background_leave_one_dataset_flips_or_nonpositive")
    status = "cpu_gate_pass_eval_only_calibrator_candidate" if not reasons else "cpu_gate_fail_do_not_launch_canonical_posthoc"
    return {
        "status": status,
        "action": "run_eval_only_canonical_calibrated_posthoc" if not reasons else "keep_calibration_diagnostic_only",
        "reasons": reasons,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Frozen-Anchor Calibration Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
        "",
        "## Provenance",
        "",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- pert_means_file: `{payload['pert_means_file']}`",
        f"- leakage_status: `{payload['leakage_status']}`",
        f"- train prediction rows: `{payload['n_train_prediction_rows']}`",
        f"- validation prediction rows: `{payload['n_val_prediction_rows']}`",
        f"- max_train_conditions_per_dataset: `{payload['max_train_conditions_per_dataset']}`",
        f"- max_val_conditions_per_dataset: `{payload['max_val_conditions_per_dataset']}`",
        f"- max_cells: `{payload['max_cells']}`",
        f"- ode_steps: `{payload['ode_steps']}`",
        "",
        "## Absolute Scores",
        "",
        "| group | mode | n cond | n ds | pp proxy |",
        "|---|---|---:|---:|---:|",
    ]
    for row in payload["absolute_scores"]:
        lines.append(
            f"| {row['group']} | `{row['mode']}` | {row['n_conditions']} | {row['n_datasets']} | {fmt(row['mean'])} |"
        )
    lines += [
        "",
        "## Paired Deltas",
        "",
        "| group | candidate | baseline | n cond | n ds | delta | 95% CI | p improve | p harm | leave-one min | status |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in payload["paired_deltas"]:
        ci = row.get("ci95") or [None, None]
        lines.append(
            f"| {row['group']} | {row['candidate']} | {row['baseline']} | "
            f"{row.get('n_conditions', 0)} | {row.get('n_datasets', 0)} | {fmt(row.get('delta_mean'))} | "
            f"[{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(row.get('p_improve'))} | "
            f"{fmt(row.get('p_harm'))} | {fmt(row.get('leave_one_min'))} | {row.get('status')} |"
        )
    lines += ["", "## Gate Reasons", ""]
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{r}`" for r in reasons] if reasons else ["- none"])
    lines += [
        "",
        "## Interpretation",
        "",
        "- Passing this gate would justify an eval-only canonical posthoc with the frozen calibrator, not flow training.",
        "- Failing this gate means prediction-space calibration is diagnostic only.",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--data-dir", default="")
    ap.add_argument("--biflow-dir", default="")
    ap.add_argument("--pert-means-file", type=Path, default=DEFAULT_PERT_MEANS)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT / "xverse_anchor_calibration_light_ode10_cell128")
    ap.add_argument("--device", default="")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ode-steps", type=int, default=10)
    ap.add_argument("--max-cells", type=int, default=128)
    ap.add_argument("--max-chunk", type=int, default=128)
    ap.add_argument("--max-train-conditions-per-dataset", type=int, default=32)
    ap.add_argument("--max-val-conditions-per-dataset", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args()

    ckpt_path = args.checkpoint.expanduser().resolve()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = _load_cfg(ckpt, data_dir=args.data_dir, biflow_dir=args.biflow_dir)
    cfg.gpu = int(args.gpu)
    cfg.eval_max_conditions = 0
    cfg.eval_max_conditions_per_dataset = 0
    cfg.eval_max_mse_cells = int(args.max_cells)
    cfg.eval_max_mmd_cells = int(args.max_cells)
    data_dir = Path(cfg.data_dir).expanduser().resolve()
    split_path = args.split_file.expanduser().resolve()
    device_s = args.device.strip() or (f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)

    manifest = _load_manifest(data_dir, cfg.manifest)
    metadata = load_json(Path(manifest["condition_metadata_file"]))
    split = _load_split(split_path)
    pert_means = {k: v.astype(np.float32) for k, v in np.load(str(args.pert_means_file.expanduser().resolve())).items()}

    model = build_model(cfg, device)
    load_model_weights_only(ckpt_path, model, device, strict=False)
    if not args.no_ema and checkpoint_ema_is_active(ckpt, cfg):
        ema = ModelEMA(
            model,
            decay=float(getattr(cfg, "ema_decay", 0.999)),
            update_after=int(getattr(cfg, "ema_update_after", 0)),
            update_every=int(getattr(cfg, "ema_update_every", 1)),
            device=device,
        )
        ema.load_state_dict(ckpt["ema"], strict=False)
        ema.copy_to(model)

    cd_kw = _cross_dataset_kw(cfg)
    train_split = select_train_single_split(
        split,
        metadata,
        manifest,
        max_per_dataset=int(args.max_train_conditions_per_dataset),
        seed=int(getattr(cfg, "seed", 0) or 0),
    )
    train_ds = CrossDatasetFMDataset(
        str(data_dir),
        train_split,
        cfg.batch_size,
        cfg.seed,
        mode="test",
        min_cells=16,
        ds_alpha=1.0,
        silent=True,
        **cd_kw,
    )
    train_rows = predict_rows(
        model=model,
        dataset=train_ds,
        cfg=cfg,
        device=device,
        group_split=train_split,
        group="train_single_calibration",
        max_cells=int(args.max_cells),
        max_chunk=int(args.max_chunk),
        ode_steps=int(args.ode_steps),
    )

    val_rows = []
    for group in DEFAULT_GROUPS:
        group_split = select_group_split(
            split,
            manifest,
            group,
            max_per_dataset=int(args.max_val_conditions_per_dataset),
            seed=int(getattr(cfg, "seed", 0) or 0),
        )
        ds = CrossDatasetFMDataset(
            str(data_dir),
            group_split,
            cfg.batch_size,
            cfg.seed,
            mode="test",
            min_cells=16,
            ds_alpha=1.0,
            silent=True,
            **cd_kw,
        )
        val_rows.extend(
            predict_rows(
                model=model,
                dataset=ds,
                cfg=cfg,
                device=device,
                group_split=group_split,
                group=group,
                max_cells=int(args.max_cells),
                max_chunk=int(args.max_chunk),
                ode_steps=int(args.ode_steps),
            )
        )

    eval_rows = fit_eval_lodo(train_rows, val_rows, pert_means=pert_means, seed=int(getattr(cfg, "seed", 0) or 0))
    deltas = []
    for group in DEFAULT_GROUPS:
        for mode in ("global_scalar", "diag_affine"):
            deltas.append(
                paired_bootstrap(
                    eval_rows,
                    group,
                    mode,
                    "identity",
                    n_boot=int(args.bootstrap),
                    seed=stable_int(f"{group}|{mode}|identity"),
                )
            )
        deltas.append(
            paired_bootstrap(
                eval_rows,
                group,
                "diag_affine",
                "diag_affine_shuffled_target",
                n_boot=int(args.bootstrap),
                seed=stable_int(f"{group}|diag|shuf"),
            )
        )

    payload = {
        "checkpoint": str(ckpt_path),
        "checkpoint_step": ckpt.get("step"),
        "split_file": str(split_path),
        "data_dir": str(data_dir),
        "pert_means_file": str(args.pert_means_file.expanduser().resolve()),
        "leakage_status": "train_only_v2_train_single_fit_to_internal_proxy_no_canonical_test_no_posthoc_no_flow_training",
        "max_train_conditions_per_dataset": int(args.max_train_conditions_per_dataset),
        "max_val_conditions_per_dataset": int(args.max_val_conditions_per_dataset),
        "max_cells": int(args.max_cells),
        "ode_steps": int(args.ode_steps),
        "n_train_prediction_rows": len(train_rows),
        "n_val_prediction_rows": len(val_rows),
        "absolute_scores": absolute_scores(eval_rows),
        "paired_deltas": deltas,
        "decision": decide(deltas),
        "config": dataclasses.asdict(cfg),
    }
    args.run_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.run_root / "prediction_calibration_arrays.npz",
        train_pred=np.stack([r["pred_resid"] for r in train_rows]).astype(np.float32),
        train_true=np.stack([r["true_resid"] for r in train_rows]).astype(np.float32),
        val_pred=np.stack([r["pred_resid"] for r in val_rows]).astype(np.float32),
        val_true=np.stack([r["true_resid"] for r in val_rows]).astype(np.float32),
        train_dataset=np.asarray([r["dataset"] for r in train_rows]),
        train_condition=np.asarray([r["condition"] for r in train_rows]),
        val_dataset=np.asarray([r["dataset"] for r in val_rows]),
        val_condition=np.asarray([r["condition"] for r in val_rows]),
        val_group=np.asarray([r["group"] for r in val_rows]),
    )
    out_json = args.run_root / "frozen_anchor_calibration_gate.json"
    out_md = args.run_root / "FROZEN_ANCHOR_CALIBRATION_GATE.md"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
