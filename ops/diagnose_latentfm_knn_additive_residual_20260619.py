#!/usr/bin/env python3
"""KNN/additive residual baseline for zero-shot multi-perturbation splits.

This diagnostic is read-only and CPU-only. It asks whether scGPT gene embedding
geometry plus observed train-single residuals can explain held-out multi-gene
latent deltas, especially `test_multi_unseen1/2`.

It deliberately limits scope to datasets that actually have multi-perturbation
visibility groups in the canonical split. It reads condition slices from HDF5
with a per-condition cell cap and never loads full matrices into RAM.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DATA_DIR = ROOT / "dataset/latentfm_full/scfoundation"
SPLIT_FILE = ROOT / "dataset/biFlow_data/split_seed42.json"
GENE_CACHE = ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene"
OUT_MD = ROOT / "reports/LATENTFM_KNN_ADDITIVE_RESIDUAL_DIAGNOSTIC_20260619.md"
OUT_CSV = ROOT / "reports/latentfm_knn_additive_residual_diagnostic_20260619.csv"
OUT_JSON = ROOT / "reports/latentfm_knn_additive_residual_diagnostic_20260619.json"


@dataclass
class Row:
    dataset: str
    condition: str
    group: str
    n_components: int
    n_seen_components: int
    n_knn_components: int
    n_missing_embedding_components: int
    k: int
    direct_pearson: float | None
    pearson_ctrl: float | None
    pearson_pert: float | None
    pred_target_cosine: float | None
    pred_target_pearson: float | None
    pred_norm: float
    target_norm: float
    median_knn_similarity: float | None
    components: str


def _components(cond: str) -> list[str]:
    return [x.strip().upper() for x in str(cond).split("+") if x.strip()]


def _safe_float(v: float) -> float | None:
    if math.isnan(v) or math.isinf(v):
        return None
    return float(v)


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return None
    return _safe_float(float(np.dot(a, b) / denom))


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return None
    return _safe_float(float(np.dot(aa, bb) / denom))


def _sample_mean(ds: h5py.Dataset, start: int, end: int, *, max_cells: int, seed: int) -> np.ndarray:
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
    v = np.asarray(gene_emb[idx], dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        return None
    return v / norm


def condition_means(
    h5_path: Path,
    conds: list[str],
    *,
    max_cells: int,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    with h5py.File(h5_path, "r") as h5:
        all_conds = h5["conditions"].asstr()[:].tolist()
        c2i = {str(c): i for i, c in enumerate(all_conds)}
        ctrl = h5["ctrl/emb"]
        gt = h5["gt/emb"]
        ctrl_offsets = h5["ctrl/offsets"][:]
        gt_offsets = h5["gt/offsets"][:]
        for cond in conds:
            if cond not in c2i:
                continue
            i = int(c2i[cond])
            cm = _sample_mean(
                ctrl,
                int(ctrl_offsets[i]),
                int(ctrl_offsets[i + 1]),
                max_cells=max_cells,
                seed=seed + i * 17,
            )
            gm = _sample_mean(
                gt,
                int(gt_offsets[i]),
                int(gt_offsets[i + 1]),
                max_cells=max_cells,
                seed=seed + i * 31,
            )
            out[cond] = (cm.astype(np.float32), gm.astype(np.float32), (gm - cm).astype(np.float32))
    return out


def knn_residual(
    gene: str,
    *,
    train_genes: list[str],
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
    # Shift to positive weights; fall back to uniform if all neighbors are tied.
    weights = sim - float(sim.min())
    if float(weights.sum()) <= 1e-8:
        weights = np.ones_like(sim, dtype=np.float32)
    weights = weights / float(weights.sum())
    pred = np.sum(train_residuals[idx] * weights[:, None], axis=0).astype(np.float32)
    return pred, float(np.median(sim))


def evaluate_dataset(
    ds_name: str,
    split: dict[str, Any],
    *,
    gene_index: dict[str, int],
    gene_emb: np.ndarray,
    ks: list[int],
    max_cells: int,
    seed: int,
) -> list[Row]:
    sp = split[ds_name]
    groups = {
        "test_multi_seen": list(sp.get("test_multi_seen", [])),
        "test_multi_unseen1": list(sp.get("test_multi_unseen1", [])),
        "test_multi_unseen2": list(sp.get("test_multi_unseen2", [])),
    }
    multi_conds = sorted({c for vals in groups.values() for c in vals})
    train_single = [c for c in sp.get("train", []) if "+" not in str(c)]
    if not multi_conds or not train_single:
        return []

    h5_path = DATA_DIR / f"{ds_name}.h5"
    wanted = sorted(set(train_single) | set(multi_conds))
    means = condition_means(h5_path, wanted, max_cells=max_cells, seed=seed)
    pert_means_npz = np.load(str(DATA_DIR / "pert_means.npz"))
    pert_mean = np.asarray(pert_means_npz[ds_name], dtype=np.float32) if ds_name in pert_means_npz else None
    train_genes: list[str] = []
    train_vec_rows: list[np.ndarray] = []
    train_resid_rows: list[np.ndarray] = []
    for cond in train_single:
        comps = _components(cond)
        if len(comps) != 1 or cond not in means:
            continue
        vec = gene_vec(comps[0], gene_index, gene_emb)
        if vec is None:
            continue
        train_genes.append(comps[0])
        train_vec_rows.append(vec)
        train_resid_rows.append(means[cond][2])
    if not train_vec_rows:
        return []
    train_vecs = np.stack(train_vec_rows, axis=0).astype(np.float32)
    train_resids = np.stack(train_resid_rows, axis=0).astype(np.float32)
    direct_resid = {g: r for g, r in zip(train_genes, train_resid_rows)}

    rows: list[Row] = []
    for group, conds in groups.items():
        for cond in conds:
            if cond not in means:
                continue
            comps = _components(cond)
            ctrl_mean, gt_mean, target = means[cond]
            for k in ks:
                pred_parts: list[np.ndarray] = []
                knn_sims: list[float] = []
                seen = 0
                knn = 0
                missing_emb = 0
                for comp in comps:
                    if comp in direct_resid:
                        pred_parts.append(direct_resid[comp])
                        seen += 1
                        continue
                    pred, sim = knn_residual(
                        comp,
                        train_genes=train_genes,
                        train_vecs=train_vecs,
                        train_residuals=train_resids,
                        gene_index=gene_index,
                        gene_emb=gene_emb,
                        k=k,
                    )
                    if pred is None:
                        missing_emb += 1
                        continue
                    pred_parts.append(pred)
                    knn += 1
                    if sim is not None:
                        knn_sims.append(float(sim))
                if not pred_parts:
                    continue
                pred_sum = np.sum(np.stack(pred_parts, axis=0), axis=0).astype(np.float32)
                pred_mean = (ctrl_mean + pred_sum).astype(np.float32)
                rows.append(
                    Row(
                        dataset=ds_name,
                        condition=cond,
                        group=group,
                        n_components=len(comps),
                        n_seen_components=seen,
                        n_knn_components=knn,
                        n_missing_embedding_components=missing_emb,
                        k=int(k),
                        direct_pearson=_pearson(pred_mean, gt_mean),
                        pearson_ctrl=_pearson(pred_mean - ctrl_mean, gt_mean - ctrl_mean),
                        pearson_pert=(
                            _pearson(pred_mean - pert_mean, gt_mean - pert_mean)
                            if pert_mean is not None
                            else None
                        ),
                        pred_target_cosine=_cosine(pred_sum, target),
                        pred_target_pearson=_pearson(pred_sum, target),
                        pred_norm=float(np.linalg.norm(pred_sum)),
                        target_norm=float(np.linalg.norm(target)),
                        median_knn_similarity=float(np.median(knn_sims)) if knn_sims else None,
                        components="+".join(comps),
                    )
                )
    return rows


def _mean(vals: list[float | None]) -> float | None:
    clean = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    return float(np.mean(clean)) if clean else None


def _median(vals: list[float | None]) -> float | None:
    clean = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    return float(np.median(clean)) if clean else None


def summarize(rows: list[Row]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(r.dataset, r.group, r.k) for r in rows})
    for ds, group, k in keys:
        vals = [r for r in rows if r.dataset == ds and r.group == group and r.k == k]
        out.append(
            {
                "dataset": ds,
                "group": group,
                "k": k,
                "n_conditions": len(vals),
                "mean_cosine": _mean([r.pred_target_cosine for r in vals]),
                "median_cosine": _median([r.pred_target_cosine for r in vals]),
                "mean_pc": _mean([r.pearson_ctrl for r in vals]),
                "median_pc": _median([r.pearson_ctrl for r in vals]),
                "mean_pp": _mean([r.pearson_pert for r in vals]),
                "median_pp": _median([r.pearson_pert for r in vals]),
                "mean_direct": _mean([r.direct_pearson for r in vals]),
                "mean_seen_components": _mean([float(r.n_seen_components) for r in vals]),
                "mean_knn_components": _mean([float(r.n_knn_components) for r in vals]),
                "median_knn_similarity": _median([r.median_knn_similarity for r in vals]),
            }
        )
    return out


def fmt(v: Any) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_outputs(rows: list[Row], summary: list[dict[str, Any]], *, max_cells: int, ks: list[int]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        fields = list(asdict(rows[0]).keys()) if rows else list(Row.__dataclass_fields__)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    payload = {
        "data_dir": str(DATA_DIR),
        "split_file": str(SPLIT_FILE),
        "gene_cache": str(GENE_CACHE),
        "max_cells_per_condition": max_cells,
        "k_values": ks,
        "rows": [asdict(r) for r in rows],
        "summary": summary,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM KNN Additive Residual Diagnostic 2026-06-19",
        "",
        "This CPU-only diagnostic asks whether scGPT gene embedding neighbors plus",
        "observed train-single latent residuals can reconstruct held-out multi-gene",
        "response deltas. It is a baseline/diagnostic, not a trained model.",
        "",
        "## Scope",
        "",
        f"- Data dir: `{DATA_DIR}`",
        f"- Split: `{SPLIT_FILE}`",
        f"- Gene cache: `{GENE_CACHE}`",
        f"- Max cells per condition mean: `{max_cells}`",
        f"- K values: `{', '.join(map(str, ks))}`",
        f"- Condition rows: `{len(rows)}`",
        f"- CSV: `{OUT_CSV}`",
        f"- JSON: `{OUT_JSON}`",
        "",
        "## Summary",
        "",
        "| Dataset | Group | k | n | mean cosine | mean pc | mean pp | mean direct | mean seen comps | mean KNN comps | median KNN sim |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {dataset} | `{group}` | {k} | {n_conditions} | {mean_cosine} | "
            "{mean_pc} | {mean_pp} | {mean_direct} | {mean_seen_components} | "
            "{mean_knn_components} | {median_knn_similarity} |".format(
                dataset=row["dataset"],
                group=row["group"],
                k=row["k"],
                n_conditions=row["n_conditions"],
                mean_cosine=fmt(row["mean_cosine"]),
                mean_pc=fmt(row["mean_pc"]),
                mean_pp=fmt(row["mean_pp"]),
                mean_direct=fmt(row["mean_direct"]),
                mean_seen_components=fmt(row["mean_seen_components"]),
                mean_knn_components=fmt(row["mean_knn_components"]),
                median_knn_similarity=fmt(row["median_knn_similarity"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "If this simple KNN/additive baseline is weak on `test_multi_unseen2`,",
            "then the main bottleneck is likely missing condition-response geometry",
            "for genes absent from train, not only the LatentFM velocity architecture.",
            "If it is strong where LatentFM is weak, then a useful next branch should",
            "inject or distill this neighbor/additive prior into the condition path.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    gene_index, gene_emb = load_gene_cache(GENE_CACHE)
    ks = [1, 5, 10]
    max_cells = 512
    all_rows: list[Row] = []
    for ds_name, sp in sorted(split.items()):
        if not any(sp.get(g) for g in ("test_multi_seen", "test_multi_unseen1", "test_multi_unseen2")):
            continue
        h5_path = DATA_DIR / f"{ds_name}.h5"
        if not h5_path.is_file():
            continue
        all_rows.extend(
            evaluate_dataset(
                ds_name,
                split,
                gene_index=gene_index,
                gene_emb=gene_emb,
                ks=ks,
                max_cells=max_cells,
                seed=42019,
            )
        )
    summary = summarize(all_rows)
    write_outputs(all_rows, summary, max_cells=max_cells, ks=ks)
    print(json.dumps({"report": str(OUT_MD), "rows": len(all_rows), "summary_rows": len(summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
