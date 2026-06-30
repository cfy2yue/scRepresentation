#!/usr/bin/env python3
"""CPU-only target-gene embedding-cluster balance gate for xverse cap120."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
SPLIT_DIR = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624"
CAP120_SPLIT = SPLIT_DIR / "split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
TYPE_BALANCED_SPLIT = SPLIT_DIR / "split_seed42_xverse_trainonly_scaling_type_balanced_cap120_v2.json"
GENE_CACHE = ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene"
OUT_SPLIT = SPLIT_DIR / "split_seed42_xverse_trainonly_scaling_target_cluster_balanced_cap120_v2.json"
OUT_JSON = ROOT / "reports/latentfm_xverse_target_cluster_balance_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_TARGET_CLUSTER_BALANCE_GATE_20260624.md"

PRESERVE_DATASETS = {"NormanWeissman2019_filtered", "Wessels"}
TARGET_CRISPRI_COUNT = 550
N_CLUSTERS = 8
RARE_CLUSTER_SHARE = 0.05


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_key(seed: int, ds: str, cond: str) -> str:
    return hashlib.sha256(f"{seed}\t{ds}\t{cond}".encode("utf-8")).hexdigest()


def normalize_symbol(sym: str) -> str:
    return str(sym).strip().upper()


def load_gene_index(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].lower() in {"symbol", "gene", "gene_symbol"}:
            continue
        if len(parts) < 2:
            parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        out[normalize_symbol(parts[0])] = int(parts[-1].strip().split()[0])
    return out


def perturbation_type(metadata: dict[str, Any], ds: str, cond: str) -> str:
    entry = ((metadata.get(ds) or {}).get(cond) or {})
    raw = str(entry.get("perturbation_type_raw", entry.get("perturbation_type", ""))).strip()
    if raw:
        return raw
    if "sciplex" in ds.lower():
        return "drug"
    return "unknown"


def is_drug(ptype: str) -> bool:
    return ptype.lower() in {"drug", "chemical", "compound", "small molecule", "small-molecule"}


def condition_genes(metadata: dict[str, Any], ds: str, cond: str) -> list[str]:
    entry = ((metadata.get(ds) or {}).get(cond) or {})
    genes = entry.get("genes")
    if isinstance(genes, list):
        return [normalize_symbol(g) for g in genes if str(g).strip()]
    return [normalize_symbol(x) for x in str(cond).split("+") if x.strip()]


def load_rows(split: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for ds, groups in sorted(split.items()):
        for cond in groups.get("train") or []:
            cond_s = str(cond)
            ptype = perturbation_type(metadata, str(ds), cond_s)
            genes = condition_genes(metadata, str(ds), cond_s)
            rows.append(
                {
                    "dataset": str(ds),
                    "condition": cond_s,
                    "ptype": ptype,
                    "genes": genes,
                    "is_drug": is_drug(ptype),
                }
            )
    return rows


def gene_embedding(row: dict[str, Any], symbol_to_idx: dict[str, int], emb: np.ndarray) -> np.ndarray | None:
    idxs = [symbol_to_idx.get(g, 1) for g in row["genes"]]
    idxs = [i for i in idxs if i > 1 and i < emb.shape[0]]
    if not idxs:
        return None
    vec = np.asarray(emb[idxs], dtype=np.float32).mean(axis=0)
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return None
    return vec / norm


def kmeans_unit(x: np.ndarray, *, k: int, seed: int, n_iter: int = 60) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    first = int(rng.integers(0, n))
    centers = [x[first]]
    min_dist = np.sum((x - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        idx = int(np.argmax(min_dist))
        centers.append(x[idx])
        dist = np.sum((x - centers[-1]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
    c = np.stack(centers).astype(np.float32)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        sims = x @ c.T
        new_labels = np.argmax(sims, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                c[j] = x[int(rng.integers(0, n))]
                continue
            v = x[mask].mean(axis=0)
            norm = float(np.linalg.norm(v))
            c[j] = v / max(norm, 1e-12)
    return labels


def copy_split_with_train(base: dict[str, Any], selected: set[tuple[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ds, groups in sorted(base.items()):
        new_groups: dict[str, Any] = {}
        for key, val in groups.items():
            new_groups[key] = [str(x) for x in val] if isinstance(val, list) else val
        new_groups["train"] = sorted(cond for d, cond in selected if d == str(ds))
        out[str(ds)] = new_groups
    return out


def condition_sizes(data_dir: Path, ds: str) -> dict[str, int]:
    with h5py.File(data_dir / f"{ds}.h5", "r") as h5:
        conds = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in h5["conditions"][:]]
        offsets = h5["gt/offsets"][:]
        return {cond: int(offsets[i + 1] - offsets[i]) for i, cond in enumerate(conds)}


def n_eff(n: int, ds_alpha: float = 0.7, min_selected: int = 0) -> int:
    if n <= 0:
        return 0
    base = n if ds_alpha >= 1.0 else int(math.ceil(n**ds_alpha))
    return max(1, min(n, max(base, min_selected)))


def exposure_summary(split: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    rows = []
    total_steps = 0.0
    for ds, groups in sorted(split.items()):
        train = [str(c) for c in groups.get("train") or []]
        sizes = condition_sizes(DATA_DIR, str(ds))
        valid = [c for c in train if c in sizes]
        selected = n_eff(len(valid))
        visits = [max(1, math.ceil(sizes[c] / 64)) for c in valid]
        avg_visits = float(sum(visits) / max(1, len(visits)))
        steps = selected * avg_visits
        total_steps += steps
        ptypes = Counter(perturbation_type(metadata, str(ds), c) for c in valid)
        rows.append(
            {
                "dataset": str(ds),
                "train_conditions": len(valid),
                "selected_conditions_per_epoch": selected,
                "epoch_steps_est": steps,
                "perturbation_type_counts": dict(sorted(ptypes.items())),
            }
        )
    for row in rows:
        row["epoch_step_share"] = float(row["epoch_steps_est"] / max(total_steps, 1.0))
    jiang_share = sum(row["epoch_step_share"] for row in rows if row["dataset"].startswith("Jiang_"))
    max_share = max((row["epoch_step_share"] for row in rows), default=0.0)
    entropy = 0.0
    for row in rows:
        p = float(row["epoch_step_share"])
        if p > 0:
            entropy -= p * math.log(p)
    norm_entropy = entropy / math.log(max(2, len(rows)))
    return {
        "rows": rows,
        "epoch_steps_est": total_steps,
        "jiang_epoch_step_share": jiang_share,
        "max_dataset_epoch_step_share": max_share,
        "normalized_dataset_step_entropy": norm_entropy,
    }


def cluster_summary(rows: list[dict[str, Any]], selected: set[tuple[str, str]]) -> dict[str, Any]:
    cluster_counts: Counter[int] = Counter()
    selected_rows = [r for r in rows if (r["dataset"], r["condition"]) in selected and not r["is_drug"] and r.get("cluster") is not None]
    for row in selected_rows:
        cluster_counts[int(row["cluster"])] += 1
    total = sum(cluster_counts.values())
    max_share = max(cluster_counts.values() or [0]) / max(1, total)
    return {
        "n_target_conditions": total,
        "cluster_counts": {str(k): int(v) for k, v in sorted(cluster_counts.items())},
        "max_cluster_share": float(max_share),
    }


def ptype_counts(rows: list[dict[str, Any]], selected: set[tuple[str, str]]) -> dict[str, int]:
    c = Counter()
    for row in rows:
        if (row["dataset"], row["condition"]) in selected:
            c[row["ptype"]] += 1
    return dict(sorted(c.items()))


def build_candidate(rows: list[dict[str, Any]], type_selected: set[tuple[str, str]], seed: int) -> set[tuple[str, str]]:
    selected = {
        (row["dataset"], row["condition"])
        for row in rows
        if (row["dataset"], row["condition"]) in type_selected and row["ptype"] != "CRISPRi"
    }
    for row in rows:
        if row["dataset"] in PRESERVE_DATASETS and not row["is_drug"]:
            selected.add((row["dataset"], row["condition"]))

    crispri_rows = [
        row
        for row in rows
        if row["ptype"] == "CRISPRi" and (row["dataset"], row["condition"]) not in selected
    ]
    cluster_cap120 = Counter(int(row["cluster"]) for row in rows if not row["is_drug"] and row.get("cluster") is not None)
    total_target_goal = (
        sum(1 for row in rows if row["ptype"] != "CRISPRi" and not row["is_drug"])
        + TARGET_CRISPRI_COUNT
    )
    max_per_cluster = int(math.floor(0.20 * total_target_goal))
    selected_cluster = Counter(
        int(row["cluster"])
        for row in rows
        if (row["dataset"], row["condition"]) in selected and not row["is_drug"] and row.get("cluster") is not None
    )
    rare_clusters = {cl for cl, n in cluster_cap120.items() if n / max(1, sum(cluster_cap120.values())) <= RARE_CLUSTER_SHARE}
    rare_min = {cl: int(math.ceil(0.80 * cluster_cap120[cl])) for cl in rare_clusters}

    by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in crispri_rows:
        if row.get("cluster") is None:
            continue
        by_cluster[int(row["cluster"])].append(row)
    for cl in by_cluster:
        by_cluster[cl].sort(key=lambda r: stable_key(seed, r["dataset"], r["condition"]))

    def add_row(row: dict[str, Any]) -> bool:
        key = (row["dataset"], row["condition"])
        if key in selected:
            return False
        cl = int(row["cluster"])
        if selected_cluster[cl] >= max_per_cluster:
            return False
        selected.add(key)
        selected_cluster[cl] += 1
        return True

    for cl in sorted(rare_clusters):
        while selected_cluster[cl] < rare_min[cl] and by_cluster.get(cl):
            row = by_cluster[cl].pop(0)
            add_row(row)

    while sum(1 for row in rows if row["ptype"] == "CRISPRi" and (row["dataset"], row["condition"]) in selected) < TARGET_CRISPRI_COUNT:
        candidates = []
        for cl, bucket in by_cluster.items():
            if not bucket or selected_cluster[cl] >= max_per_cluster:
                continue
            candidates.append((selected_cluster[cl], cl, bucket[0]))
        if not candidates:
            break
        _, cl, row = sorted(candidates, key=lambda x: (x[0], x[1], stable_key(seed, x[2]["dataset"], x[2]["condition"])))[0]
        by_cluster[cl].pop(0)
        add_row(row)
    return selected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    ap.add_argument("--out-split", type=Path, default=OUT_SPLIT)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()

    cap_split = read_json(CAP120_SPLIT)
    type_split = read_json(TYPE_BALANCED_SPLIT)
    metadata = read_json(DATA_DIR / "condition_metadata.json")
    symbol_to_idx = load_gene_index(GENE_CACHE / "gene_index.tsv")
    emb = np.load(GENE_CACHE / "gene_embeddings.npy", mmap_mode="r")

    rows = load_rows(cap_split, metadata)
    vectors = []
    vector_rows = []
    missing_embedding = []
    for row in rows:
        if row["is_drug"]:
            row["cluster"] = None
            continue
        vec = gene_embedding(row, symbol_to_idx, emb)
        if vec is None:
            row["cluster"] = None
            missing_embedding.append(f"{row['dataset']}:{row['condition']}")
            continue
        vectors.append(vec)
        vector_rows.append(row)
    labels = kmeans_unit(np.stack(vectors), k=int(args.n_clusters), seed=int(args.seed))
    for row, label in zip(vector_rows, labels):
        row["cluster"] = int(label)

    cap_selected = {(r["dataset"], r["condition"]) for r in rows}
    type_selected = {
        (str(ds), str(cond))
        for ds, groups in type_split.items()
        for cond in groups.get("train") or []
    }
    candidate_selected = build_candidate(rows, type_selected, int(args.seed) + 101)
    candidate_split = copy_split_with_train(cap_split, candidate_selected)

    cap_cluster = cluster_summary(rows, cap_selected)
    type_cluster = cluster_summary(rows, type_selected)
    cand_cluster = cluster_summary(rows, candidate_selected)
    cap_exposure = exposure_summary(cap_split, metadata)
    type_exposure = exposure_summary(type_split, metadata)
    cand_exposure = exposure_summary(candidate_split, metadata)
    cap_ptypes = ptype_counts(rows, cap_selected)
    type_ptypes = ptype_counts(rows, type_selected)
    cand_ptypes = ptype_counts(rows, candidate_selected)

    rare_failures = []
    cap_counts = Counter(int(r["cluster"]) for r in rows if not r["is_drug"] and r.get("cluster") is not None)
    cand_counts = Counter(
        int(r["cluster"])
        for r in rows
        if (r["dataset"], r["condition"]) in candidate_selected and not r["is_drug"] and r.get("cluster") is not None
    )
    total_cap_target = sum(cap_counts.values())
    for cl, count in sorted(cap_counts.items()):
        if count / max(1, total_cap_target) <= RARE_CLUSTER_SHARE:
            retain = cand_counts[cl] / max(1, count)
            if retain < 0.80:
                rare_failures.append({"cluster": cl, "cap120": count, "candidate": cand_counts[cl], "retain_fraction": retain})

    candidate_train_conditions = len(candidate_selected)
    datasets_with_train = sum(1 for groups in candidate_split.values() if groups.get("train"))
    preserve_missing = [
        f"{r['dataset']}:{r['condition']}"
        for r in rows
        if r["dataset"] in PRESERVE_DATASETS
        and not r["is_drug"]
        and (r["dataset"], r["condition"]) not in candidate_selected
    ]
    ptype_failures = [
        ptype
        for ptype, baseline in type_ptypes.items()
        if cand_ptypes.get(ptype, 0) < baseline
    ]
    reasons = []
    if candidate_train_conditions < 900:
        reasons.append("candidate_train_conditions_lt_900")
    if datasets_with_train < 20:
        reasons.append("datasets_with_train_lt_20")
    if cand_cluster["max_cluster_share"] > 0.20:
        reasons.append("target_cluster_max_share_gt_0p20")
    if rare_failures:
        reasons.append("rare_cluster_retention_lt_0p80")
    if preserve_missing:
        reasons.append("preserve_norman_wessels_missing")
    if ptype_failures:
        reasons.append("perturbation_type_coverage_worse_than_type_balanced")
    if cand_ptypes.get("Cas13", 0) < type_ptypes.get("Cas13", 0):
        reasons.append("cas13_not_retained")
    if cand_exposure["jiang_epoch_step_share"] > 0.50:
        reasons.append("jiang_epoch_step_share_gt_0p50")
    if cand_exposure["normalized_dataset_step_entropy"] <= cap_exposure["normalized_dataset_step_entropy"]:
        reasons.append("dataset_exposure_entropy_not_above_cap120")

    status = "target_cluster_balance_gate_pass" if not reasons else "target_cluster_balance_gate_fail"
    if status == "target_cluster_balance_gate_pass":
        args.out_split.parent.mkdir(parents=True, exist_ok=True)
        args.out_split.write_text(json.dumps(candidate_split, indent=2, ensure_ascii=False), encoding="utf-8")

    payload = {
        "status": status,
        "reasons": reasons,
        "boundary": {
            "uses": ["train-only split train groups", "condition_metadata genes/perturbation_type", "scGPT gene embeddings", "h5 gt offsets for exposure"],
            "forbidden": ["canonical outcomes", "Track C query", "active logs", "posthoc predictions", "residual/error features", "expression matrices"],
        },
        "files": {
            "cap120_split": str(CAP120_SPLIT),
            "type_balanced_split": str(TYPE_BALANCED_SPLIT),
            "candidate_split": str(args.out_split) if status == "target_cluster_balance_gate_pass" else "",
            "gene_cache": str(GENE_CACHE),
        },
        "candidate": {
            "train_conditions": candidate_train_conditions,
            "datasets_with_train": datasets_with_train,
            "perturbation_type_counts": cand_ptypes,
            "cluster": cand_cluster,
            "exposure": {k: v for k, v in cand_exposure.items() if k != "rows"},
        },
        "baselines": {
            "cap120_all": {
                "train_conditions": len(cap_selected),
                "perturbation_type_counts": cap_ptypes,
                "cluster": cap_cluster,
                "exposure": {k: v for k, v in cap_exposure.items() if k != "rows"},
            },
            "type_balanced_cap120": {
                "train_conditions": len(type_selected),
                "perturbation_type_counts": type_ptypes,
                "cluster": type_cluster,
                "exposure": {k: v for k, v in type_exposure.items() if k != "rows"},
            },
        },
        "rare_cluster_failures": rare_failures,
        "preserve_missing": preserve_missing[:30],
        "ptype_failures": ptype_failures,
        "missing_embedding_examples": missing_embedding[:30],
        "out_json": str(args.out_json),
        "out_md": str(args.out_md),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(args.out_md)}, indent=2))
    return 0


def render_md(payload: dict[str, Any]) -> str:
    def f(x: Any) -> str:
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    lines = [
        "# LatentFM xverse Target-Cluster Balance Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only gate over train-only split metadata and frozen gene embeddings.",
        "- Uses scGPT gene embeddings for target clustering and h5 GT offsets for exposure estimates.",
        "- Does not read canonical outcomes, Track C query, active logs, posthoc predictions, residual/error features, or expression matrices.",
        "",
        "## Summary",
        "",
        "| arm | train conds | max target-cluster share | Jiang step share | max dataset step share | dataset entropy | ptypes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    rows = [
        ("cap120_all", payload["baselines"]["cap120_all"]),
        ("type_balanced_cap120", payload["baselines"]["type_balanced_cap120"]),
        ("target_cluster_candidate", payload["candidate"]),
    ]
    for name, row in rows:
        ptypes = ", ".join(f"{k}:{v}" for k, v in row["perturbation_type_counts"].items())
        lines.append(
            f"| `{name}` | {row['train_conditions']} | {f(row['cluster']['max_cluster_share'])} | "
            f"{f(row['exposure']['jiang_epoch_step_share'])} | {f(row['exposure']['max_dataset_epoch_step_share'])} | "
            f"{f(row['exposure']['normalized_dataset_step_entropy'])} | {ptypes} |"
        )
    if payload["reasons"]:
        lines += ["", "Gate reasons:"]
        lines.extend(f"- `{r}`" for r in payload["reasons"])
    lines += [
        "",
        "## Candidate Split",
        "",
        f"`{payload['files']['candidate_split'] or 'not written because gate failed'}`",
        "",
        "## JSON",
        "",
        f"`{payload['out_json']}`",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
