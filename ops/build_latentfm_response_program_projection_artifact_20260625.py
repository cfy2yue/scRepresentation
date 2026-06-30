#!/usr/bin/env python3
"""Build a train-only response-program projection artifact for true-cell repair.

This is a bounded, query-blind artifact builder. It evaluates anchor and
candidate checkpoints on an internal/train-only split, saves row-level
candidate-minus-anchor residual vectors, builds response-program PCA axes from
train conditions only, and applies a fail-closed CPU gate before any GPU smoke
can be considered.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch


ROOT = Path("/data/cyx/1030/scLatent")
COUPLEDFM = ROOT / "CoupledFM"
EVAL_RESIDUALS = COUPLEDFM / "model/latent/eval_condition_residuals.py"
DEFAULT_ANCHOR_JSON = ROOT / (
    "runs/latentfm_true_cell_count_budget128_tail_stability_6k_20260625/"
    "xverse_truecell_nested_budget128_tailstable_seed42_6000/posthoc_eval_internal/"
    "split_group_eval_anchor_internal_ode20.json"
)
DEFAULT_CANDIDATE_JSON = ROOT / (
    "runs/latentfm_true_cell_count_budget128_tail_stability_6k_20260625/"
    "xverse_truecell_nested_budget128_tailstable_seed42_6000/posthoc_eval_internal/"
    "split_group_eval_candidate_internal_ode20.json"
)
OUT_DIR = ROOT / "reports/response_program_projection_20260625"
OUT_JSON = ROOT / "reports/latentfm_response_program_projection_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_RESPONSE_PROGRAM_PROJECTION_GATE_20260625.md"
OUT_CSV = ROOT / "reports/latentfm_response_program_projection_rows_20260625.csv"
OUT_NPZ = OUT_DIR / "latentfm_response_program_projection_vectors_20260625.npz"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_eval_module() -> Any:
    sys.path.insert(0, str(COUPLEDFM))
    spec = importlib.util.spec_from_file_location("latent_eval_condition_residuals", EVAL_RESIDUALS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {EVAL_RESIDUALS}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def metric_map(payload: dict[str, Any], group: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = (payload.get("groups", {}).get(group, {}) or {}).get("condition_metrics") or []
    return {(str(r["dataset"]), str(r["condition"])): r for r in rows}


def stable_seed(*parts: object) -> int:
    import hashlib

    text = "|".join(str(p) for p in parts)
    return int(hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest(), 16) % (2**32)


def sample_mean(arr: np.ndarray, max_cells: int, seed: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if max_cells > 0 and arr.shape[0] > max_cells:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(arr.shape[0], size=max_cells, replace=False))
        arr = arr[idx]
    return arr.mean(axis=0).astype(np.float32)


def load_means(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path) as obj:
        return {str(k): np.asarray(v, dtype=np.float32) for k, v in obj.items()}


def build_train_axes(
    mod: Any,
    *,
    data_dir: Path,
    split_file: Path,
    max_train_conditions: int,
    max_cells: int,
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = mod._load_manifest(data_dir, "manifest.json")
    split = mod._load_split(split_file)
    cfg = mod.Config(data_dir=str(data_dir), split_file=str(split_file), seed=seed)
    ctrl_means = load_means(data_dir / "ctrl_means.npz")
    pert_means = load_means(data_dir / "pert_means.npz")
    train_split = {
        ds: {"train": sorted(set(parts.get("train", []))), "test": sorted(set(parts.get("train", [])))}
        for ds, parts in split.items()
        if parts.get("train")
    }
    pairs = [(ds, cond) for ds, parts in train_split.items() for cond in parts["test"]]
    pairs = sorted(pairs, key=lambda p: (stable_seed("train_axes", seed, p[0], p[1]), p[0], p[1]))
    if max_train_conditions > 0:
        pairs = pairs[:max_train_conditions]
    selected = defaultdict(lambda: {"train": [], "test": []})
    for ds, cond in pairs:
        selected[ds]["test"].append(cond)
    selected = {ds: dict(v) for ds, v in selected.items()}
    ds_obj = mod.CrossDatasetFMDataset(
        str(data_dir),
        selected,
        64,
        seed,
        mode="test",
        min_cells=16,
        ds_alpha=1.0,
        silent=True,
    )
    residuals: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    for ds in ds_obj.ds_names:
        handle = ds_obj.handles[ds]
        ref = pert_means.get(ds)
        if ref is None:
            ref = ctrl_means.get(ds)
        for cond in ds_obj.ds_conds[ds]:
            gt = handle.read_gt(cond)
            gt_mean = sample_mean(gt, max_cells, stable_seed("train_gt", seed, ds, cond))
            if ref is None:
                ref_vec = np.zeros_like(gt_mean, dtype=np.float32)
            else:
                ref_vec = np.asarray(ref, dtype=np.float32)
            residuals.append((gt_mean - ref_vec).astype(np.float32))
            meta_rows.append({"dataset": ds, "condition": cond})
    mat = np.stack(residuals, axis=0).astype(np.float32)
    centered = mat - mat.mean(axis=0, keepdims=True)
    _, s, vh = np.linalg.svd(centered.astype(np.float64), full_matrices=False)
    k = max(1, min(int(n_components), vh.shape[0], vh.shape[1]))
    axes = vh[:k].astype(np.float32)
    var = s**2
    explained = (var[:k] / max(float(var.sum()), 1e-12)).astype(float).tolist()
    meta = {
        "n_train_axis_rows": len(meta_rows),
        "n_axis_datasets": len({r["dataset"] for r in meta_rows}),
        "n_components": int(k),
        "explained_variance_ratio": explained,
        "max_train_conditions": int(max_train_conditions),
        "max_cells": int(max_cells),
    }
    return axes, meta


def evaluate_checkpoint(
    mod: Any,
    *,
    checkpoint: Path,
    data_dir: Path,
    split_file: Path,
    groups: list[str],
    device: torch.device,
    max_conditions: int,
    max_conditions_per_group: int,
    eval_max_cells: int,
    ode_steps: int,
    max_chunk: int,
    seed: int,
    selected_pairs: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg = mod._load_cfg(ckpt, data_dir=str(data_dir), biflow_dir=str(ROOT / "dataset/biFlow_data"))
    cfg.gpu = 0
    cfg.seed = int(seed)
    manifest = mod._load_manifest(data_dir, cfg.manifest)
    split = mod._load_split(split_file)
    condition_metadata = mod._load_condition_metadata(data_dir)
    if selected_pairs is not None:
        pairs = sorted(set((str(ds), str(cond)) for ds, cond in selected_pairs))
        if max_conditions > 0 and len(pairs) > max_conditions:
            pairs = sorted(
                pairs,
                key=lambda p: (stable_seed("vector_eval", seed, p[0], p[1]), p[0], p[1]),
            )[:max_conditions]
        cond_groups = {pair: list(groups) for pair in pairs}
        selected_dd: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"train": [], "test": []})
        for ds_name, cond in pairs:
            selected_dd[ds_name]["test"].append(cond)
        selected = {ds_name: dict(parts) for ds_name, parts in selected_dd.items()}
    else:
        family_splits = mod.build_family_group_splits(
            manifest=manifest,
            split=split,
            condition_metadata=condition_metadata,
        )
        split_splits = mod._split_group_splits(manifest=manifest, split=split, groups=groups)
        all_group_splits = {**family_splits, **split_splits}
        wanted = {g: all_group_splits[g] for g in groups if g in all_group_splits}
        cond_groups = mod._condition_group_index(wanted)
        selected = mod._selected_split(
            cond_groups,
            max_conditions=max_conditions if max_conditions > 0 else None,
            max_conditions_per_group=max_conditions_per_group if max_conditions_per_group > 0 else None,
            seed=int(seed),
        )
    if not selected:
        raise RuntimeError("no selected conditions")
    ctrl_means = mod._load_means(data_dir, "ctrl_means.npz")
    pert_means = mod._load_means(data_dir, "pert_means.npz")
    model = mod.build_model(cfg, device)
    from model.latent.train import checkpoint_ema_is_active, load_model_weights_only

    load_model_weights_only(checkpoint, model, device, strict=False)
    ema = None
    if "ema" in ckpt and bool(getattr(cfg, "use_ema", False)) and checkpoint_ema_is_active(ckpt, cfg):
        ema = mod.ModelEMA(
            model,
            decay=float(getattr(cfg, "ema_decay", 0.999)),
            update_after=int(getattr(cfg, "ema_update_after", 0)),
            update_every=int(getattr(cfg, "ema_update_every", 1)),
            device=device,
        )
        ema.load_state_dict(ckpt["ema"], strict=False)
    ds = mod.CrossDatasetFMDataset(
        str(data_dir),
        selected,
        cfg.batch_size,
        int(seed),
        mode="test",
        min_cells=16,
        ds_alpha=1.0,
        silent=True,
        **mod._cross_dataset_kw(cfg),
    )
    kwargs = dict(
        model=model,
        dataset=ds,
        cfg=cfg,
        device=device,
        cond_groups=cond_groups,
        condition_metadata=condition_metadata,
        ctrl_means=ctrl_means,
        pert_means=pert_means,
        ode_steps=int(ode_steps),
        max_chunk=int(max_chunk),
        eval_max_cells=int(eval_max_cells),
        skip_mmd=True,
    )
    if ema is not None:
        with ema.apply_to(model):
            return mod._evaluate_condition_rows(**kwargs)
    return mod._evaluate_condition_rows(**kwargs)


def bootstrap(values: list[float], n_boot: int = 1000, seed: int = 20260625) -> dict[str, float]:
    if not values:
        return {"low": float("nan"), "high": float("nan"), "p_le_zero": 1.0}
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = np.sort([float(arr[rng.integers(0, len(arr), size=len(arr))].mean()) for _ in range(n_boot)])
    return {
        "low": float(means[int(0.025 * (n_boot - 1))]),
        "high": float(means[int(0.975 * (n_boot - 1))]),
        "p_le_zero": float((1 + np.sum(means <= 0.0)) / (1 + len(means))),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pp = [float(r["pp_delta"]) for r in rows]
    mmd = [float(r["mmd_delta"]) for r in rows]
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_ds[row["dataset"]].append(float(row["pp_delta"]))
    ci = bootstrap(pp)
    return {
        "n": len(rows),
        "pp_mean": mean(pp) if pp else None,
        "pp_ci95": [ci["low"], ci["high"]],
        "p_le_zero": ci["p_le_zero"],
        "dataset_min_pp": min((mean(v) for v in by_ds.values()), default=None),
        "hard_harm_frac": mean([1.0 if x < -0.020 else 0.0 for x in pp]) if pp else None,
        "mmd_max": max(mmd) if mmd else None,
        "mmd_mean": mean(mmd) if mmd else None,
    }


def missing_as(value: Any, default: float) -> float:
    out = finite(value)
    return default if out is None else out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor-json", type=Path, default=DEFAULT_ANCHOR_JSON)
    ap.add_argument("--candidate-json", type=Path, default=DEFAULT_CANDIDATE_JSON)
    ap.add_argument(
        "--groups",
        nargs="*",
        default=["internal_val_cross_background_seen_gene_proxy", "internal_val_family_gene_proxy"],
    )
    ap.add_argument("--metric-group", default="internal_val_family_gene_proxy")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-conditions", type=int, default=0)
    ap.add_argument("--max-conditions-per-group", type=int, default=0)
    ap.add_argument("--eval-max-cells", type=int, default=256)
    ap.add_argument("--train-axis-max-conditions", type=int, default=768)
    ap.add_argument("--train-axis-max-cells", type=int, default=256)
    ap.add_argument("--n-components", type=int, default=16)
    ap.add_argument("--ode-steps", type=int, default=20)
    ap.add_argument("--max-chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260625)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--out-npz", type=Path, default=OUT_NPZ)
    args = ap.parse_args()

    anchor_payload = load_json(args.anchor_json)
    cand_payload = load_json(args.candidate_json)
    data_dir = Path(cand_payload["data_dir"]).expanduser().resolve()
    split_file = Path(cand_payload["split_file"]).expanduser().resolve()
    anchor_ckpt = Path(anchor_payload["checkpoint"]).expanduser().resolve()
    cand_ckpt = Path(cand_payload["checkpoint"]).expanduser().resolve()
    scalar_anchor = metric_map(anchor_payload, args.metric_group)
    scalar_cand = metric_map(cand_payload, args.metric_group)
    scalar_pairs = sorted(set(scalar_anchor) & set(scalar_cand))

    mod = load_eval_module()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    axes, axes_meta = build_train_axes(
        mod,
        data_dir=data_dir,
        split_file=split_file,
        max_train_conditions=args.train_axis_max_conditions,
        max_cells=args.train_axis_max_cells,
        n_components=args.n_components,
        seed=args.seed,
    )
    anchor_rows = evaluate_checkpoint(
        mod,
        checkpoint=anchor_ckpt,
        data_dir=data_dir,
        split_file=split_file,
        groups=list(args.groups),
        device=device,
        max_conditions=args.max_conditions,
        max_conditions_per_group=args.max_conditions_per_group,
        eval_max_cells=args.eval_max_cells,
        ode_steps=args.ode_steps,
        max_chunk=args.max_chunk,
        seed=args.seed,
        selected_pairs=scalar_pairs,
    )
    cand_rows = evaluate_checkpoint(
        mod,
        checkpoint=cand_ckpt,
        data_dir=data_dir,
        split_file=split_file,
        groups=list(args.groups),
        device=device,
        max_conditions=args.max_conditions,
        max_conditions_per_group=args.max_conditions_per_group,
        eval_max_cells=args.eval_max_cells,
        ode_steps=args.ode_steps,
        max_chunk=args.max_chunk,
        seed=args.seed,
        selected_pairs=scalar_pairs,
    )
    anchor_by_key = {(r["dataset"], r["condition"]): r for r in anchor_rows}
    cand_by_key = {(r["dataset"], r["condition"]): r for r in cand_rows}
    common = sorted(set(anchor_by_key) & set(cand_by_key) & set(scalar_anchor) & set(scalar_cand))
    if not common:
        raise RuntimeError("no common rows between vector and scalar artifacts")

    rows: list[dict[str, Any]] = []
    anchor_pred = []
    cand_pred = []
    target = []
    delta = []
    for ds, cond in common:
        ar = anchor_by_key[(ds, cond)]
        cr = cand_by_key[(ds, cond)]
        avec = np.asarray(ar["_pred_residual"], dtype=np.float32)
        cvec = np.asarray(cr["_pred_residual"], dtype=np.float32)
        tvec = np.asarray(cr["_target_residual"], dtype=np.float32)
        dvec = (cvec - avec).astype(np.float32)
        proj = axes.T @ (axes @ dvec)
        dnorm = float(np.linalg.norm(dvec))
        supported_ratio = 0.0 if dnorm <= 1e-12 else float(np.linalg.norm(proj) / dnorm)
        unsupported_ratio = 0.0 if dnorm <= 1e-12 else float(np.linalg.norm(dvec - proj) / dnorm)
        pp_delta = finite(scalar_cand[(ds, cond)].get("pearson_pert"))
        app = finite(scalar_anchor[(ds, cond)].get("pearson_pert"))
        cmmd = finite(scalar_cand[(ds, cond)].get("test_mmd_clamped"))
        ammd = finite(scalar_anchor[(ds, cond)].get("test_mmd_clamped"))
        if pp_delta is None or app is None or cmmd is None or ammd is None:
            continue
        rows.append(
            {
                "dataset": ds,
                "condition": cond,
                "groups": cr.get("groups", ""),
                "pp_delta": float(pp_delta - app),
                "mmd_delta": float(cmmd - ammd),
                "delta_norm": dnorm,
                "supported_ratio": supported_ratio,
                "unsupported_ratio": unsupported_ratio,
                "anchor_pred_target_cosine": ar.get("pred_target_cosine"),
                "candidate_pred_target_cosine": cr.get("pred_target_cosine"),
            }
        )
        anchor_pred.append(avec)
        cand_pred.append(cvec)
        target.append(tvec)
        delta.append(dvec)

    if not rows:
        raise RuntimeError("no finite rows after joining vector and scalar artifacts")

    scores = sorted(float(r["supported_ratio"]) for r in rows)
    threshold = scores[int(0.75 * (len(scores) - 1))]
    high = [r for r in rows if float(r["supported_ratio"]) >= threshold]
    low = [r for r in rows if float(r["supported_ratio"]) < threshold]
    high_summary = summarize(high)
    low_summary = summarize(low)
    margin = float(high_summary["pp_mean"] or 0.0) - float(low_summary["pp_mean"] or 0.0)

    rng = np.random.default_rng(args.seed)
    random_means = []
    dim = axes.shape[1]
    k = axes.shape[0]
    deltas = np.stack(delta, axis=0).astype(np.float64)
    pp_vals = np.asarray([r["pp_delta"] for r in rows], dtype=np.float64)
    for _ in range(200):
        q, _ = np.linalg.qr(rng.normal(size=(dim, k)))
        raxes = q.T.astype(np.float64)
        ratios = []
        for dvec in deltas:
            proj = raxes.T @ (raxes @ dvec)
            dnorm = float(np.linalg.norm(dvec))
            ratios.append(0.0 if dnorm <= 1e-12 else float(np.linalg.norm(proj) / dnorm))
        rthreshold = sorted(ratios)[int(0.75 * (len(ratios) - 1))]
        mask = np.asarray(ratios) >= rthreshold
        random_means.append(float(pp_vals[mask].mean()))
    random_p95 = sorted(random_means)[int(0.95 * (len(random_means) - 1))]
    random_margin = float(high_summary["pp_mean"] or 0.0) - float(mean(random_means))

    reasons: list[str] = []
    if len(rows) < 50:
        reasons.append("too_few_rows")
    if len({r["dataset"] for r in rows}) < 3:
        reasons.append("too_few_datasets")
    if missing_as(high_summary["pp_mean"], -999.0) < 0.025:
        reasons.append("supported_pp_below_0p025")
    ci95 = high_summary.get("pp_ci95") or [-999.0, -999.0]
    if missing_as(ci95[0], -999.0) <= 0.0:
        reasons.append("supported_ci_lower_not_positive")
    if margin < 0.020:
        reasons.append("supported_minus_unsupported_gap_below_0p020")
    if missing_as(high_summary["dataset_min_pp"], -999.0) < -0.010:
        reasons.append("dataset_min_below_minus_0p010")
    if missing_as(high_summary["hard_harm_frac"], 1.0) > 0.15:
        reasons.append("hard_harm_frac_above_0p15")
    if missing_as(high_summary["mmd_max"], 999.0) > 0.001:
        reasons.append("mmd_max_above_0p001")
    if random_margin < 0.005 or missing_as(high_summary["pp_mean"], -999.0) <= random_p95:
        reasons.append("random_axis_control_not_collapsed")

    status = "response_program_projection_gate_pass_gpu_candidate" if not reasons else "response_program_projection_gate_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": not reasons,
        "boundary": {
            "train_only_internal_split": True,
            "reads_canonical_performance": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training": False,
            "inference_for_artifact_only": True,
        },
        "inputs": {
            "anchor_json": str(args.anchor_json),
            "candidate_json": str(args.candidate_json),
            "anchor_checkpoint": str(anchor_ckpt),
            "candidate_checkpoint": str(cand_ckpt),
            "data_dir": str(data_dir),
            "split_file": str(split_file),
            "metric_group": args.metric_group,
            "groups": list(args.groups),
        },
        "axes": axes_meta,
        "summary": {
            "n_rows": len(rows),
            "n_datasets": len({r["dataset"] for r in rows}),
            "supported_threshold_q75": threshold,
            "high_supported": high_summary,
            "low_supported": low_summary,
            "supported_minus_low_pp_margin": margin,
            "random_axis_mean": mean(random_means),
            "random_axis_p95": random_p95,
            "supported_minus_random_mean_margin": random_margin,
        },
        "reasons": reasons,
        "next_action": (
            "launch a bounded GPU smoke only after external audit"
            if not reasons
            else "do not launch response-program GPU; projection support fails strict gate"
        ),
        "outputs": {
            "json": str(args.out_json),
            "md": str(args.out_md),
            "csv": str(args.out_csv),
            "npz": str(args.out_npz),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        args.out_npz,
        anchor_pred_residual=np.stack(anchor_pred, axis=0).astype(np.float32),
        candidate_pred_residual=np.stack(cand_pred, axis=0).astype(np.float32),
        target_residual=np.stack(target, axis=0).astype(np.float32),
        candidate_minus_anchor_residual=np.stack(delta, axis=0).astype(np.float32),
        axes=axes.astype(np.float32),
        dataset=np.asarray([r["dataset"] for r in rows]),
        condition=np.asarray([r["condition"] for r in rows]),
    )
    lines = [
        "# LatentFM Response-Program Projection Gate",
        "",
        f"Status: `{status}`",
        f"GPU authorized: `{not reasons}`",
        "",
        "## Boundary",
        "",
        "- Uses only train-only/internal true-cell split artifacts.",
        "- Builds PCA response-program axes from train conditions only.",
        "- Does not read canonical performance, canonical multi, Track C query, or train.",
        "",
        "## Summary",
        "",
        f"- rows: `{len(rows)}`",
        f"- datasets: `{len({r['dataset'] for r in rows})}`",
        f"- high-supported pp mean: `{high_summary['pp_mean']}`",
        f"- high-supported pp CI95: `{high_summary['pp_ci95']}`",
        f"- high-supported dataset min pp: `{high_summary['dataset_min_pp']}`",
        f"- high-supported hard-harm fraction: `{high_summary['hard_harm_frac']}`",
        f"- high-supported MMD max: `{high_summary['mmd_max']}`",
        f"- supported-minus-low pp margin: `{margin}`",
        f"- supported-minus-random-axis mean margin: `{random_margin}`",
        "",
        "## Decision",
        "",
        f"- reasons: `{reasons}`",
        f"- next action: `{payload['next_action']}`",
        "",
        "## Outputs",
        "",
        f"- JSON: `{args.out_json}`",
        f"- CSV: `{args.out_csv}`",
        f"- NPZ: `{args.out_npz}`",
    ]
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "rows": len(rows), "gpu_authorized": not reasons, "out": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
