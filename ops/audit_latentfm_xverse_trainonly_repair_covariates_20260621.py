#!/usr/bin/env python3
"""Screen train-only covariates for xverse response-repair effects.

This is a CPU-only diagnostic. Route/covariate selection here is posthoc and
therefore cannot be used as a promotion claim. Inputs used as covariates are
restricted to canonical train metadata/residuals and pretrained gene embeddings.
Held-out condition outcomes are used only to evaluate candidate covariates.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
DEFAULT_DELTA_JSON = ROOT / "reports/latentfm_xverse_response_repair_aux025_delta_audit_20260621.json"
DEFAULT_SCGPT = ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene"
DEFAULT_CELLNAVI = ROOT / "pretrainckpt/genepert_cache/cellnavi_embed_gene"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_trainonly_repair_covariates_20260621.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_TRAINONLY_REPAIR_COVARIATES_20260621.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decode(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def condition_mean(handle: h5py.File, group: str, idx: int, max_cells: int) -> np.ndarray | None:
    offsets = np.asarray(handle[f"{group}/offsets"])
    start, end = int(offsets[idx]), int(offsets[idx + 1])
    if end <= start:
        return None
    arr = handle[f"{group}/emb"]
    if max_cells > 0 and end - start > max_cells:
        # Deterministic head sample is enough for covariate screening and avoids
        # importing a condition-outcome keyed sampler here.
        end = start + max_cells
    return np.asarray(arr[start:end], dtype=np.float32).mean(axis=0)


def train_single_residual_bank(
    data_dir: Path,
    split: dict[str, Any],
    metadata: dict[str, Any],
    max_cells: int,
) -> tuple[dict[str, dict[str, float]], dict[str, list[float]], dict[str, int]]:
    same: dict[str, dict[str, float]] = defaultdict(dict)
    global_vals: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, int] = {}
    for ds, obj in sorted(split.items()):
        train = [str(x) for x in obj.get("train", [])]
        counts[str(ds)] = 0
        path = data_dir / f"{ds}.h5"
        if not path.is_file():
            continue
        with h5py.File(path, "r") as handle:
            conditions = decode(np.asarray(handle["conditions"]))
            by_cond = {c: i for i, c in enumerate(conditions)}
            for cond in train:
                meta = (metadata.get(ds) or {}).get(cond) or {}
                genes = [str(g) for g in meta.get("genes") or []]
                if len(genes) != 1:
                    continue
                i = by_cond.get(cond)
                if i is None:
                    continue
                ctrl = condition_mean(handle, "ctrl", i, max_cells)
                gt = condition_mean(handle, "gt", i, max_cells)
                if ctrl is None or gt is None:
                    continue
                norm = float(np.linalg.norm(gt - ctrl))
                gene = genes[0]
                same[str(ds)][gene] = norm
                global_vals[gene].append(norm)
                counts[str(ds)] += 1
    return same, global_vals, counts


def load_gene_embeddings(cache_dir: Path) -> tuple[dict[str, int], np.ndarray]:
    index: dict[str, int] = {}
    with (cache_dir / "gene_index.tsv").open(encoding="utf-8") as handle:
        reader = csv.DictReader((line for line in handle if not line.startswith("#")), fieldnames=["gene_symbol", "index"], delimiter="\t")
        for row in reader:
            gene = str(row.get("gene_symbol") or "").strip()
            if not gene or gene in {"PAD", "UNK"}:
                continue
            try:
                index[gene] = int(row["index"])
            except Exception:
                continue
    emb = np.load(cache_dir / "gene_embeddings.npy")
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norm, 1e-8)
    return index, emb.astype(np.float32)


def pair_cos(genes: list[str], index: dict[str, int], emb: np.ndarray) -> float | None:
    if len(genes) < 2:
        return None
    ids = [index[g] for g in genes if g in index]
    if len(ids) < 2:
        return None
    vals = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            vals.append(float(np.dot(emb[ids[i]], emb[ids[j]])))
    return float(np.mean(vals)) if vals else None


def mean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def median(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return float(np.median(vals)) if vals else None


def q(values: list[float | None], pct: float) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return float(np.percentile(vals, pct)) if vals else None


def enrich_rows(
    delta_payload: dict[str, Any],
    metadata: dict[str, Any],
    same_bank: dict[str, dict[str, float]],
    global_bank: dict[str, list[float]],
    train_counts: dict[str, int],
    scgpt_index: dict[str, int],
    scgpt_emb: np.ndarray,
    cellnavi_index: dict[str, int],
    cellnavi_emb: np.ndarray,
) -> list[dict[str, Any]]:
    out = []
    groups = {g["group"]: g for g in delta_payload.get("groups") or []}
    for group_name in ("test_multi", "test_multi_unseen2"):
        for row in groups.get(group_name, {}).get("rows") or []:
            ds = str(row["dataset"])
            cond = str(row["condition"])
            meta = (metadata.get(ds) or {}).get(cond) or {}
            genes = [str(g) for g in meta.get("genes") or []]
            same_hits = sum(1 for g in genes if g in same_bank.get(ds, {}))
            global_hits = sum(1 for g in genes if g in global_bank)
            same_norms = [same_bank.get(ds, {}).get(g) for g in genes]
            global_norms = [mean(global_bank[g]) if g in global_bank else None for g in genes]
            rr = dict(row)
            rr.update(
                {
                    "n_genes": len(genes),
                    "genes": genes,
                    "dataset_train_single_count": int(train_counts.get(ds, 0)),
                    "same_hits": int(same_hits),
                    "global_hits": int(global_hits),
                    "same_full": same_hits == len(genes) and len(genes) > 0,
                    "global_full": global_hits == len(genes) and len(genes) > 0,
                    "same_hit_frac": same_hits / max(len(genes), 1),
                    "global_hit_frac": global_hits / max(len(genes), 1),
                    "same_mean_resid_norm": mean(same_norms),
                    "same_max_resid_norm": max([v for v in same_norms if v is not None], default=None),
                    "global_mean_resid_norm": mean(global_norms),
                    "global_max_resid_norm": max([v for v in global_norms if v is not None], default=None),
                    "scgpt_pair_cos": pair_cos(genes, scgpt_index, scgpt_emb),
                    "cellnavi_pair_cos": pair_cos(genes, cellnavi_index, cellnavi_emb),
                }
            )
            out.append(rr)
    return out


def summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "mean_pp_delta": mean([r.get("pp_delta") for r in rows]),
        "mean_mmd_delta": mean([r.get("mmd_delta") for r in rows]),
        "pp_improve_frac": sum((r.get("pp_delta") or 0.0) > 0 for r in rows) / len(rows),
        "mmd_improve_frac": sum((r.get("mmd_delta") or 0.0) < 0 for r in rows) / len(rows),
        "pp_positive_frac_delta": (
            sum((r.get("candidate_pp") or 0.0) > 0 for r in rows)
            - sum((r.get("anchor_pp") or 0.0) > 0 for r in rows)
        )
        / len(rows),
        "mmd_gt_005_frac_delta": (
            sum((r.get("candidate_mmd") or 0.0) > 0.05 for r in rows)
            - sum((r.get("anchor_mmd") or 0.0) > 0.05 for r in rows)
        )
        / len(rows),
    }


def route_screen(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feature_values = {
        name: [r.get(name) for r in rows if r.get(name) is not None]
        for name in ["same_hit_frac", "dataset_train_single_count", "same_mean_resid_norm", "global_mean_resid_norm", "scgpt_pair_cos", "cellnavi_pair_cos"]
    }
    med = {k: median(v) for k, v in feature_values.items()}
    q75 = {k: q(v, 75) for k, v in feature_values.items()}
    q25 = {k: q(v, 25) for k, v in feature_values.items()}
    routes: list[tuple[str, Any]] = [
        ("global_full", lambda r: bool(r.get("global_full"))),
        ("same_full", lambda r: bool(r.get("same_full"))),
        ("same_not_full", lambda r: not bool(r.get("same_full"))),
        ("dataset_train_single_ge_50", lambda r: int(r.get("dataset_train_single_count") or 0) >= 50),
        ("dataset_train_single_lt_50", lambda r: int(r.get("dataset_train_single_count") or 0) < 50),
    ]
    for name in ["same_mean_resid_norm", "global_mean_resid_norm", "scgpt_pair_cos", "cellnavi_pair_cos"]:
        if med.get(name) is not None:
            routes.append((f"{name}_ge_median", lambda r, n=name, t=med[name]: r.get(n) is not None and float(r[n]) >= float(t)))
            routes.append((f"{name}_lt_median", lambda r, n=name, t=med[name]: r.get(n) is not None and float(r[n]) < float(t)))
        if q75.get(name) is not None:
            routes.append((f"{name}_ge_q75", lambda r, n=name, t=q75[name]: r.get(n) is not None and float(r[n]) >= float(t)))
        if q25.get(name) is not None:
            routes.append((f"{name}_le_q25", lambda r, n=name, t=q25[name]: r.get(n) is not None and float(r[n]) <= float(t)))
    out = []
    for route_name, pred in routes:
        selected = [r for r in rows if pred(r)]
        if not selected:
            continue
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            by_group[str(row["group"])].append(row)
        out.append(
            {
                "route": route_name,
                "selected": len(selected),
                "groups": {group: summarize_subset(vals) for group, vals in sorted(by_group.items())},
                "leakage_note": "route definition uses train-only/deployable covariates, but route was selected posthoc in this audit",
            }
        )
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.4f}"
    return str(value)


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Train-Only Repair Covariate Audit 2026-06-21",
        "",
        "This is a CPU-only posthoc screen. Covariates are train-only/deployable, but route selection is outcome-informed and cannot be a promotion claim.",
        "",
        "## Provenance",
        "",
        f"- data_dir: `{payload['data_dir']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- delta_json: `{payload['delta_json']}`",
        f"- leakage status: `{payload['leakage_status']}`",
        "",
        "## Feature Screen",
        "",
        "| route | selected | group | mean pp delta | mean MMD delta | pp improve frac | MMD improve frac | pp-positive frac delta | MMD>0.05 frac delta |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for route in payload["routes"]:
        for group, s in route["groups"].items():
            lines.append(
                "| {route} | {selected} | {group} | {pp} | {mmd} | {pif} | {mif} | {ppf} | {mmdf} |".format(
                    route=route["route"],
                    selected=route["selected"],
                    group=group,
                    pp=fmt(s.get("mean_pp_delta")),
                    mmd=fmt(s.get("mean_mmd_delta")),
                    pif=fmt(s.get("pp_improve_frac")),
                    mif=fmt(s.get("mmd_improve_frac")),
                    ppf=fmt(s.get("pp_positive_frac_delta")),
                    mmdf=fmt(s.get("mmd_gt_005_frac_delta")),
                )
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Treat any promising route as a hypothesis only; this report used held-out outcomes to compare routes.",
        "- Dataset-count routes may act as dataset proxies and need leave-dataset/dataset-resampled validation before GPU use.",
        "- If no route has broad test_multi and unseen2 support without MMD burden, do not launch another response-weight sweep.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--delta-json", type=Path, default=DEFAULT_DELTA_JSON)
    parser.add_argument("--scgpt-cache", type=Path, default=DEFAULT_SCGPT)
    parser.add_argument("--cellnavi-cache", type=Path, default=DEFAULT_CELLNAVI)
    parser.add_argument("--max-cells-per-condition", type=int, default=512)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    split = load_json(args.split_file)
    metadata = load_json(args.data_dir / "condition_metadata.json")
    same_bank, global_bank, train_counts = train_single_residual_bank(
        args.data_dir, split, metadata, args.max_cells_per_condition
    )
    scgpt_index, scgpt_emb = load_gene_embeddings(args.scgpt_cache)
    cellnavi_index, cellnavi_emb = load_gene_embeddings(args.cellnavi_cache)
    rows = enrich_rows(
        load_json(args.delta_json),
        metadata,
        same_bank,
        global_bank,
        train_counts,
        scgpt_index,
        scgpt_emb,
        cellnavi_index,
        cellnavi_emb,
    )
    payload = {
        "data_dir": str(args.data_dir),
        "split_file": str(args.split_file),
        "delta_json": str(args.delta_json),
        "scgpt_cache": str(args.scgpt_cache),
        "cellnavi_cache": str(args.cellnavi_cache),
        "max_cells_per_condition": int(args.max_cells_per_condition),
        "leakage_status": "covariates_train_only_or_pretrained; heldout outcomes used only for posthoc screening",
        "n_rows": len(rows),
        "train_single_counts_by_dataset": train_counts,
        "rows": rows,
        "routes": route_screen(rows),
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
