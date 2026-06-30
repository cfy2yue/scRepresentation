#!/usr/bin/env python3
"""Evaluate KNN/additive prior correction on an existing LatentFM checkpoint.

This is a short diagnostic evaluator, not training. It runs capped ODE inference
for selected multi-perturbation conditions, builds a train-single KNN/additive
gene residual prior, then interpolates checkpoint predictions with that prior:

    corrected_mean = (1 - alpha) * model_mean + alpha * prior_mean

Metrics use the same dataset-level `ctrl_means.npz` / `pert_means.npz` frame as
LatentFM training evaluation (`pc` and `pp`).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(os.environ.get("SCFM_WORKSPACE_ROOT", "/data/cyx/1030/scLatent")).expanduser()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from model.latent.dataset import CrossDatasetFMDataset  # noqa: E402
from model.latent.eval_split_groups import _load_cfg, _load_manifest, _load_split  # noqa: E402
from model.latent.train import (  # noqa: E402
    _cross_dataset_kw,
    _model_uses_pert,
    _pearson_np,
    _pert_chunk,
    _pert_for_eval_batch,
    _pert_to_device,
    build_model,
    ode_integrate,
)
from model.utils.train.ema import ModelEMA  # noqa: E402


DEFAULT_CHECKPOINT = (
    WORKSPACE_ROOT
    / "CoupledFM/output/latentfm_runs/full_scfoundation/"
    "20260617_scfoundation_comp006_delta_w5_12k/best.pt"
)
DEFAULT_DATA_DIR = WORKSPACE_ROOT / "dataset/latentfm_full/scfoundation"
DEFAULT_BIFLOW_DIR = WORKSPACE_ROOT / "dataset/biFlow_data"
DEFAULT_GENE_CACHE = WORKSPACE_ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene"
OUT_MD = WORKSPACE_ROOT / "reports/LATENTFM_PRIOR_CORRECTION_EVAL.md"
OUT_CSV = WORKSPACE_ROOT / "reports/latentfm_prior_correction_eval.csv"
OUT_JSON = WORKSPACE_ROOT / "reports/latentfm_prior_correction_eval.json"


def safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    out = float(value)
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def stable_int_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def components_from_meta(meta: dict[str, Any] | None, cond: str) -> list[str]:
    if isinstance(meta, dict):
        genes = meta.get("genes")
        if isinstance(genes, list):
            return [str(g).strip().upper() for g in genes if str(g).strip()]
    return [x.strip().upper() for x in str(cond).split("+") if x.strip()]


def load_gene_cache(cache_dir: Path) -> tuple[dict[str, int], np.ndarray]:
    index: dict[str, int] = {}
    for line in (cache_dir / "gene_index.tsv").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0].isdigit():
            symbol, idx = parts[1], parts[2]
        elif len(parts) >= 2:
            symbol, idx = parts[0], parts[-1]
        else:
            continue
        if str(symbol).strip().lower() in {"gene_symbol", "symbol", "gene"}:
            continue
        try:
            index[str(symbol).strip().upper()] = int(str(idx).strip())
        except ValueError:
            continue
    emb = np.load(str(cache_dir / "gene_embeddings.npy"), mmap_mode="r")
    return index, emb


def gene_vec(gene: str, gene_index: dict[str, int], gene_emb: np.ndarray) -> np.ndarray | None:
    idx = int(gene_index.get(str(gene).upper(), 1))
    if idx <= 1 or idx >= int(gene_emb.shape[0]):
        return None
    vec = np.asarray(gene_emb[idx], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return None
    return (vec / norm).astype(np.float32)


def sample_mean(ds: h5py.Dataset, start: int, end: int, *, max_cells: int, seed: int) -> np.ndarray:
    n = int(end - start)
    if n <= 0:
        raise ValueError("empty condition slice")
    if max_cells > 0 and n > max_cells:
        rng = np.random.default_rng(seed)
        rel = np.sort(rng.choice(n, size=int(max_cells), replace=False))
        arr = ds[start + rel]
    else:
        arr = ds[start:end]
    return np.asarray(arr, dtype=np.float32).mean(axis=0)


def condition_residual_means(
    data_dir: Path,
    ds_name: str,
    conds: Iterable[str],
    *,
    max_cells: int,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    h5_path = data_dir / f"{ds_name}.h5"
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    with h5py.File(h5_path, "r") as h5:
        names = h5["conditions"].asstr()[:].tolist()
        c2i = {str(c): i for i, c in enumerate(names)}
        ctrl = h5["ctrl/emb"]
        gt = h5["gt/emb"]
        ctrl_off = h5["ctrl/offsets"][:]
        gt_off = h5["gt/offsets"][:]
        for cond in conds:
            if cond not in c2i:
                continue
            i = int(c2i[cond])
            ctrl_mean = sample_mean(ctrl, int(ctrl_off[i]), int(ctrl_off[i + 1]), max_cells=max_cells, seed=seed + i * 17)
            gt_mean = sample_mean(gt, int(gt_off[i]), int(gt_off[i + 1]), max_cells=max_cells, seed=seed + i * 31)
            out[cond] = (ctrl_mean, gt_mean, (gt_mean - ctrl_mean).astype(np.float32))
    return out


def build_prior_bank(
    *,
    data_dir: Path,
    ds_name: str,
    split_train: list[str],
    condition_metadata: dict[str, dict[str, Any]],
    gene_index: dict[str, int],
    gene_emb: np.ndarray,
    max_cells: int,
    seed: int,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    train_single = [c for c in split_train if "+" not in str(c)]
    means = condition_residual_means(data_dir, ds_name, train_single, max_cells=max_cells, seed=seed)
    genes: list[str] = []
    vecs: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    direct: dict[str, np.ndarray] = {}
    meta_ds = condition_metadata.get(ds_name, {})
    for cond in train_single:
        comps = components_from_meta(meta_ds.get(cond), cond)
        if len(comps) != 1 or cond not in means:
            continue
        vec = gene_vec(comps[0], gene_index, gene_emb)
        if vec is None:
            continue
        resid = means[cond][2].astype(np.float32)
        genes.append(comps[0])
        vecs.append(vec)
        residuals.append(resid)
        direct[comps[0]] = resid
    if not vecs:
        return [], np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32), {}
    return genes, np.stack(vecs).astype(np.float32), np.stack(residuals).astype(np.float32), direct


def knn_component_residual(
    gene: str,
    *,
    train_vecs: np.ndarray,
    train_residuals: np.ndarray,
    gene_index: dict[str, int],
    gene_emb: np.ndarray,
    k: int,
) -> tuple[np.ndarray | None, float | None]:
    q = gene_vec(gene, gene_index, gene_emb)
    if q is None or train_vecs.size == 0:
        return None, None
    sims = train_vecs @ q
    kk = min(max(1, int(k)), int(train_vecs.shape[0]))
    idx = np.argsort(-sims, kind="mergesort")[:kk]
    sim = sims[idx].astype(np.float32)
    weights = sim - float(sim.min())
    if float(weights.sum()) <= 1e-8:
        weights = np.ones_like(sim, dtype=np.float32)
    weights = weights / float(weights.sum())
    pred = np.sum(train_residuals[idx] * weights[:, None], axis=0).astype(np.float32)
    return pred, float(np.median(sim))


def prior_delta_for_condition(
    *,
    comps: list[str],
    direct: dict[str, np.ndarray],
    train_vecs: np.ndarray,
    train_residuals: np.ndarray,
    gene_index: dict[str, int],
    gene_emb: np.ndarray,
    k: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    parts: list[np.ndarray] = []
    sims: list[float] = []
    n_seen = 0
    n_knn = 0
    n_missing = 0
    for comp in comps:
        if comp in direct:
            parts.append(direct[comp])
            n_seen += 1
            continue
        pred, sim = knn_component_residual(
            comp,
            train_vecs=train_vecs,
            train_residuals=train_residuals,
            gene_index=gene_index,
            gene_emb=gene_emb,
            k=k,
        )
        if pred is None:
            n_missing += 1
            continue
        parts.append(pred)
        n_knn += 1
        if sim is not None:
            sims.append(float(sim))
    if not parts:
        return None, {"n_seen": n_seen, "n_knn": n_knn, "n_missing": n_missing, "median_knn_similarity": None}
    return np.sum(np.stack(parts), axis=0).astype(np.float32), {
        "n_seen": n_seen,
        "n_knn": n_knn,
        "n_missing": n_missing,
        "median_knn_similarity": float(np.median(sims)) if sims else None,
    }


def group_conditions(split_ds: dict[str, Any], groups: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for group in groups:
        out[group] = list(split_ds.get(group, []))
    return out


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(r["label"], r["dataset"], r["group"], r["alpha"], r["k"]) for r in rows})
    out: list[dict[str, Any]] = []
    for label, ds, group, alpha, k in keys:
        vals = [r for r in rows if (r["label"], r["dataset"], r["group"], r["alpha"], r["k"]) == (label, ds, group, alpha, k)]
        def avg(name: str) -> float | None:
            xs = [float(v[name]) for v in vals if v.get(name) is not None]
            return float(np.mean(xs)) if xs else None

        out.append(
            {
                "label": label,
                "dataset": ds,
                "group": group,
                "alpha": alpha,
                "k": k,
                "n_conditions": len(vals),
                "direct": avg("direct"),
                "pc": avg("pc"),
                "pp": avg("pp"),
                "prior_available_rate": avg("prior_available"),
                "mean_missing_components": avg("n_missing"),
            }
        )
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    cfg = _load_cfg(ckpt, data_dir=str(args.data_dir), biflow_dir=str(args.biflow_dir))
    cfg.gpu = int(args.gpu)
    cfg.batch_size = int(args.batch_size)

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() and not args.cpu else "cpu")
    manifest = _load_manifest(Path(args.data_dir), cfg.manifest)
    split = _load_split(Path(args.biflow_dir) / f"split_seed{cfg.split_seed}.json")
    condition_metadata = json.loads((Path(args.data_dir) / "condition_metadata.json").read_text(encoding="utf-8"))
    ctrl_means = np.load(str(Path(args.data_dir) / "ctrl_means.npz"))
    pert_means = np.load(str(Path(args.data_dir) / "pert_means.npz"))
    gene_index, gene_emb = load_gene_cache(Path(args.gene_cache))

    selected: dict[str, dict[str, list[str]]] = {}
    group_by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    for ds_name, sp in split.items():
        if args.datasets and ds_name not in args.datasets:
            continue
        if ds_name not in manifest.get("datasets", {}):
            continue
        conds: list[str] = []
        for group, vals in group_conditions(sp, args.groups).items():
            for cond in vals:
                conds.append(cond)
                group_by_pair[(ds_name, cond)].append(group)
        conds = sorted(set(conds))
        if conds:
            selected[ds_name] = {"train": [], "test": conds}
    if not selected:
        raise ValueError("no selected conditions")

    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"], strict=True)
    ema = None
    if "ema" in ckpt and bool(getattr(cfg, "use_ema", False)) and not args.no_ema:
        ema = ModelEMA(model, decay=float(getattr(cfg, "ema_decay", 0.999)), device=device)
        ema.load_state_dict(ckpt["ema"], strict=False)

    dataset = CrossDatasetFMDataset(
        str(args.data_dir),
        selected,
        int(args.batch_size),
        int(cfg.seed),
        mode="test",
        min_cells=16,
        ds_alpha=1.0,
        silent=False,
        **_cross_dataset_kw(cfg),
    )
    use_pert = _model_uses_pert(model)

    rows: list[dict[str, Any]] = []
    alpha_values = [float(x) for x in args.alphas]
    k_values = [int(x) for x in args.k_values]
    model.eval()
    ctx = ema.apply_to(model) if ema is not None else nullcontext()
    with ctx:
        for ds_name in dataset.ds_names:
            sp = split[ds_name]
            genes, train_vecs, train_resids, direct = build_prior_bank(
                data_dir=Path(args.data_dir),
                ds_name=ds_name,
                split_train=list(sp.get("train", [])),
                condition_metadata=condition_metadata,
                gene_index=gene_index,
                gene_emb=gene_emb,
                max_cells=int(args.prior_max_cells),
                seed=int(cfg.seed) + 23000,
            )
            del genes
            meta_ds = condition_metadata.get(ds_name, {})
            ds_ctrl_mean = np.asarray(ctrl_means[ds_name], dtype=np.float32) if ds_name in ctrl_means else None
            ds_pert_mean = np.asarray(pert_means[ds_name], dtype=np.float32) if ds_name in pert_means else None
            if ds_ctrl_mean is None or ds_pert_mean is None:
                continue
            handle = dataset.handles[ds_name]
            for cond in dataset.ds_conds[ds_name]:
                src_np = handle.read_src(cond)
                gt_np = handle.read_gt(cond)
                rng = np.random.RandomState(
                    int(cfg.seed) + stable_int_hash(f"{ds_name}\t{cond}") % 100000
                )
                n_src = min(int(src_np.shape[0]), int(args.eval_max_cells))
                n_gt = min(int(gt_np.shape[0]), int(args.eval_max_cells))
                src_eval = torch.from_numpy(np.asarray(src_np[rng.permutation(src_np.shape[0])[:n_src]], dtype=np.float32))
                gt_eval_np = np.asarray(gt_np[rng.permutation(gt_np.shape[0])[:n_gt]], dtype=np.float32)
                gt_mean = gt_eval_np.mean(axis=0)
                pred_parts: list[torch.Tensor] = []
                pb_dev_full = None
                if use_pert:
                    pb_cpu = _pert_for_eval_batch(dataset, ds_name, cond, int(src_eval.size(0)))
                    pb_dev_full = _pert_to_device(pb_cpu, device)
                for start in range(0, int(src_eval.size(0)), int(args.max_chunk)):
                    end = min(start + int(args.max_chunk), int(src_eval.size(0)))
                    src_c = src_eval[start:end].to(device, non_blocking=True)
                    pb_use = None if pb_dev_full is None else _pert_chunk(pb_dev_full, start, end)
                    pred = ode_integrate(
                        model,
                        src_c,
                        src_c,
                        cfg,
                        n_steps=int(args.ode_steps),
                        perturbation_batch=pb_use if use_pert else None,
                    )
                    pred_parts.append(pred.detach().cpu())
                model_mean = torch.cat(pred_parts, dim=0).mean(dim=0).numpy().astype(np.float32)

                comps = components_from_meta(meta_ds.get(cond), cond)
                for k in k_values:
                    prior_delta, prior_info = prior_delta_for_condition(
                        comps=comps,
                        direct=direct,
                        train_vecs=train_vecs,
                        train_residuals=train_resids,
                        gene_index=gene_index,
                        gene_emb=gene_emb,
                        k=k,
                    )
                    prior_mean = None if prior_delta is None else (ds_ctrl_mean + prior_delta).astype(np.float32)
                    for alpha in alpha_values:
                        if prior_mean is None and alpha > 0:
                            continue
                        corrected = model_mean if alpha == 0 else ((1.0 - alpha) * model_mean + alpha * prior_mean).astype(np.float32)
                        for group in group_by_pair.get((ds_name, cond), ["ungrouped"]):
                            rows.append(
                                {
                                    "label": args.label,
                                    "dataset": ds_name,
                                    "condition": cond,
                                    "group": group,
                                    "alpha": float(alpha),
                                    "k": int(k),
                                    "direct": safe_float(_pearson_np(corrected, gt_mean)),
                                    "pc": safe_float(_pearson_np(corrected - ds_ctrl_mean, gt_mean - ds_ctrl_mean)),
                                    "pp": safe_float(_pearson_np(corrected - ds_pert_mean, gt_mean - ds_pert_mean)),
                                    "prior_available": 1.0 if prior_mean is not None else 0.0,
                                    "n_components": len(comps),
                                    "n_seen": prior_info.get("n_seen", 0),
                                    "n_knn": prior_info.get("n_knn", 0),
                                    "n_missing": prior_info.get("n_missing", len(comps)),
                                    "median_knn_similarity": prior_info.get("median_knn_similarity"),
                                    "components": "+".join(comps),
                                }
                            )
    summary = aggregate(rows)
    meta = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": ckpt.get("step"),
        "data_dir": str(args.data_dir),
        "groups": args.groups,
        "datasets": args.datasets,
        "alphas": alpha_values,
        "k_values": k_values,
        "ode_steps": int(args.ode_steps),
        "eval_max_cells": int(args.eval_max_cells),
        "device": str(device),
        "used_ema": ema is not None,
    }
    return rows, summary, meta


def write_outputs(rows: list[dict[str, Any]], summary: list[dict[str, Any]], meta: dict[str, Any], args: argparse.Namespace) -> None:
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    Path(args.out_json).write_text(json.dumps({"meta": meta, "summary": summary, "rows": rows}, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM Prior-Correction Evaluation 2026-06-19",
        "",
        "This is a capped evaluator, not training. It interpolates an existing",
        "LatentFM checkpoint prediction with a train-single KNN/additive residual",
        "prior and reports LatentFM-frame direct/pc/pp metrics.",
        "",
        "## Metadata",
        "",
        f"- Label: `{args.label}`",
        f"- Checkpoint: `{meta['checkpoint']}`",
        f"- Step: `{meta.get('checkpoint_step')}`",
        f"- Data dir: `{meta['data_dir']}`",
        f"- Device: `{meta['device']}`",
        f"- ODE steps: `{meta['ode_steps']}`",
        f"- Eval max cells: `{meta['eval_max_cells']}`",
        f"- CSV: `{args.out_csv}`",
        f"- JSON: `{args.out_json}`",
        "",
        "## Summary",
        "",
        "| Dataset | Group | alpha | k | n | direct | pc | pp | prior rate | missing comps |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    priority = [
        r
        for r in summary
        if r["group"] in {"test_multi_seen", "test_multi_unseen1", "test_multi_unseen2"}
    ]
    priority.sort(key=lambda r: (r["dataset"], r["group"], float(r["alpha"]), int(r["k"])))
    for row in priority:
        lines.append(
            "| {dataset} | `{group}` | {alpha} | {k} | {n_conditions} | {direct} | {pc} | {pp} | {prior_available_rate} | {mean_missing_components} |".format(
                dataset=row["dataset"],
                group=row["group"],
                alpha=fmt(row["alpha"]),
                k=row["k"],
                n_conditions=row["n_conditions"],
                direct=fmt(row["direct"]),
                pc=fmt(row["pc"]),
                pp=fmt(row["pp"]),
                prior_available_rate=fmt(row["prior_available_rate"]),
                mean_missing_components=fmt(row["mean_missing_components"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "A useful alpha should improve `pp` on multi-unseen groups without",
            "collapsing `pc`. This evaluator does not report MMD because it combines",
            "condition means rather than full predicted distributions.",
            "",
        ]
    )
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--label", default="primary_scfoundation_prior_correction")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--biflow-dir", type=Path, default=DEFAULT_BIFLOW_DIR)
    ap.add_argument("--gene-cache", type=Path, default=DEFAULT_GENE_CACHE)
    ap.add_argument("--groups", nargs="*", default=["test_multi_seen", "test_multi_unseen1", "test_multi_unseen2"])
    ap.add_argument("--datasets", nargs="*", default=["NormanWeissman2019_filtered", "Wessels"])
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--k-values", nargs="*", type=int, default=[5, 10])
    ap.add_argument("--ode-steps", type=int, default=20)
    ap.add_argument("--eval-max-cells", type=int, default=128)
    ap.add_argument("--prior-max-cells", type=int, default=512)
    ap.add_argument("--max-chunk", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    rows, summary, meta = evaluate(args)
    write_outputs(rows, summary, meta, args)
    print(json.dumps({"out_md": str(args.out_md), "rows": len(rows), "summary_rows": len(summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
