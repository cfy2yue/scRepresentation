#!/usr/bin/env python3
"""Audit xverse train-single synthetic-composition feasibility.

Train-only inputs:
- canonical train single-gene xverse residuals;
- condition metadata / split.

Held-out multi GT is used only in a clearly marked diagnostic section to
measure whether additive train-single residual priors point toward true multi
responses. It must not be used directly as a GPU training target for canonical
zero-shot claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
DEFAULT_ANCHOR_SPLIT = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/"
    / "posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
)
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_synthetic_composition_feasibility_20260621.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_SYNTHETIC_COMPOSITION_FEASIBILITY_20260621.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decode(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def stable_limit(items: list[str], k: int, key: str) -> list[str]:
    if k <= 0 or len(items) <= k:
        return list(items)
    return sorted(items, key=lambda x: hashlib.sha256(f"{key}|{x}".encode()).hexdigest())[:k]


def condition_mean(handle: h5py.File, group: str, idx: int, max_cells: int) -> np.ndarray | None:
    offsets = np.asarray(handle[f"{group}/offsets"])
    start, end = int(offsets[idx]), int(offsets[idx + 1])
    if end <= start:
        return None
    if max_cells > 0 and end - start > max_cells:
        end = start + max_cells
    return np.asarray(handle[f"{group}/emb"][start:end], dtype=np.float32).mean(axis=0)


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return None
    return float(np.dot(a, b) / denom)


def train_single_bank(
    data_dir: Path,
    split: dict[str, Any],
    metadata: dict[str, Any],
    max_cells: int,
    max_train_single_per_dataset: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[np.ndarray]], dict[str, int]]:
    same: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    global_bank: dict[str, list[np.ndarray]] = defaultdict(list)
    counts: dict[str, int] = {}
    for ds, obj in sorted(split.items()):
        train = [str(x) for x in obj.get("train", [])]
        train = stable_limit(train, max_train_single_per_dataset, f"synthetic_comp|train|{ds}")
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
                resid = (gt - ctrl).astype(np.float32)
                gene = genes[0]
                same[str(ds)][gene] = resid
                global_bank[gene].append(resid)
                counts[str(ds)] += 1
    return same, global_bank, counts


def anchor_metrics(anchor_payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for group, obj in (anchor_payload.get("groups") or {}).items():
        if group not in {"test_multi", "test_multi_seen", "test_multi_unseen1", "test_multi_unseen2"}:
            continue
        for row in obj.get("condition_metrics") or []:
            if isinstance(row, dict):
                key = (str(row.get("dataset")), str(row.get("condition")))
                out.setdefault(key, {}).update(
                    {
                        "anchor_group": group,
                        "anchor_pp": row.get("pearson_pert"),
                        "anchor_pc": row.get("pearson_ctrl"),
                        "anchor_mmd": row.get("test_mmd_clamped"),
                    }
                )
    return out


def mean_vec(vectors: list[np.ndarray]) -> np.ndarray | None:
    if not vectors:
        return None
    return np.mean(np.stack(vectors).astype(np.float32), axis=0)


def additive_prior(
    genes: list[str],
    ds: str,
    same: dict[str, dict[str, np.ndarray]],
    global_bank: dict[str, list[np.ndarray]],
    scope: str,
) -> tuple[np.ndarray | None, int]:
    vecs: list[np.ndarray] = []
    for gene in genes:
        if scope == "same":
            v = same.get(ds, {}).get(gene)
        else:
            v = mean_vec(global_bank.get(gene, []))
        if v is None:
            continue
        vecs.append(v)
    if len(vecs) != len(genes):
        return None, len(vecs)
    return np.sum(np.stack(vecs).astype(np.float32), axis=0), len(vecs)


def pairwise_prior_cos(genes: list[str], ds: str, same: dict[str, dict[str, np.ndarray]], global_bank: dict[str, list[np.ndarray]], scope: str) -> float | None:
    vecs = []
    for gene in genes:
        v = same.get(ds, {}).get(gene) if scope == "same" else mean_vec(global_bank.get(gene, []))
        if v is not None:
            vecs.append(v)
    if len(vecs) < 2:
        return None
    vals = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            c = cosine(vecs[i], vecs[j])
            if c is not None:
                vals.append(c)
    return float(np.mean(vals)) if vals else None


def collect_test_multi_rows(
    data_dir: Path,
    split: dict[str, Any],
    metadata: dict[str, Any],
    same: dict[str, dict[str, np.ndarray]],
    global_bank: dict[str, list[np.ndarray]],
    anchor: dict[tuple[str, str], dict[str, Any]],
    max_cells: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ds, obj in sorted(split.items()):
        conds = []
        for group in ("test_multi_seen", "test_multi_unseen1", "test_multi_unseen2"):
            conds.extend(str(x) for x in obj.get(group, []))
        path = data_dir / f"{ds}.h5"
        if not path.is_file() or not conds:
            continue
        with h5py.File(path, "r") as handle:
            conditions = decode(np.asarray(handle["conditions"]))
            by_cond = {c: i for i, c in enumerate(conditions)}
            for cond in sorted(set(conds)):
                meta = (metadata.get(ds) or {}).get(cond) or {}
                genes = [str(g) for g in meta.get("genes") or []]
                if len(genes) < 2:
                    continue
                idx = by_cond.get(cond)
                if idx is None:
                    continue
                ctrl = condition_mean(handle, "ctrl", idx, max_cells)
                gt = condition_mean(handle, "gt", idx, max_cells)
                gt_delta = None if ctrl is None or gt is None else (gt - ctrl).astype(np.float32)
                same_prior, same_hits = additive_prior(genes, str(ds), same, global_bank, "same")
                global_prior, global_hits = additive_prior(genes, str(ds), same, global_bank, "global")
                am = anchor.get((str(ds), cond), {})
                rows.append(
                    {
                        "dataset": str(ds),
                        "condition": cond,
                        "split_group": am.get("anchor_group"),
                        "genes": genes,
                        "n_genes": len(genes),
                        "same_hits": same_hits,
                        "global_hits": global_hits,
                        "same_full": same_prior is not None,
                        "global_full": global_prior is not None,
                        "same_prior_norm": None if same_prior is None else float(np.linalg.norm(same_prior)),
                        "global_prior_norm": None if global_prior is None else float(np.linalg.norm(global_prior)),
                        "gt_delta_norm": None if gt_delta is None else float(np.linalg.norm(gt_delta)),
                        "same_prior_gt_cos": cosine(same_prior, gt_delta),
                        "global_prior_gt_cos": cosine(global_prior, gt_delta),
                        "same_pair_resid_cos": pairwise_prior_cos(genes, str(ds), same, global_bank, "same"),
                        "global_pair_resid_cos": pairwise_prior_cos(genes, str(ds), same, global_bank, "global"),
                        "anchor_pp": am.get("anchor_pp"),
                        "anchor_pc": am.get("anchor_pc"),
                        "anchor_mmd": am.get("anchor_mmd"),
                    }
                )
    return rows


def fnum(value: Any) -> float | None:
    try:
        if value is None:
            return None
        val = float(value)
        if val != val:
            return None
        return val
    except Exception:
        return None


def summarize(rows: list[dict[str, Any]], key: str | None = None) -> list[dict[str, Any]] | dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if key is None:
        return summarize_group(rows)
    for row in rows:
        groups[str(row.get(key))].append(row)
    return [
        {"group": name, **summarize_group(vals)}
        for name, vals in sorted(groups.items())
    ]


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    def vals(name: str) -> list[float]:
        return [float(v) for r in rows if (v := fnum(r.get(name))) is not None]
    out = {
        "n": len(rows),
        "same_full_frac": sum(bool(r.get("same_full")) for r in rows) / len(rows),
        "global_full_frac": sum(bool(r.get("global_full")) for r in rows) / len(rows),
    }
    for name in ("same_prior_gt_cos", "global_prior_gt_cos", "same_pair_resid_cos", "global_pair_resid_cos", "anchor_pp", "anchor_mmd", "global_prior_norm", "gt_delta_norm"):
        xs = vals(name)
        out[f"mean_{name}"] = float(np.mean(xs)) if xs else None
        out[f"median_{name}"] = float(np.median(xs)) if xs else None
    return out


def correlation_rows(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    def corr(a_name: str, b_name: str) -> float | None:
        pairs = [(fnum(r.get(a_name)), fnum(r.get(b_name))) for r in rows]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        if len(pairs) < 3:
            return None
        a = np.asarray([x for x, _ in pairs], dtype=np.float64)
        b = np.asarray([y for _, y in pairs], dtype=np.float64)
        a -= a.mean()
        b -= b.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return None if denom <= 1e-12 else float(np.dot(a, b) / denom)

    return {
        "global_prior_gt_cos_vs_anchor_pp": corr("global_prior_gt_cos", "anchor_pp"),
        "global_prior_gt_cos_vs_anchor_mmd": corr("global_prior_gt_cos", "anchor_mmd"),
        "global_pair_resid_cos_vs_anchor_pp": corr("global_pair_resid_cos", "anchor_pp"),
        "global_prior_norm_vs_anchor_pp": corr("global_prior_norm", "anchor_pp"),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.4f}"
    return str(value)


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Synthetic-Composition Feasibility Audit 2026-06-21",
        "",
        "This CPU audit builds additive priors from canonical train single-gene xverse residuals.",
        "Held-out multi GT is used only for feasibility diagnostics, not as a training target.",
        "",
        "## Provenance",
        "",
        f"- data_dir: `{payload['data_dir']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- anchor_split_json: `{payload['anchor_split_json']}`",
        f"- leakage status: `{payload['leakage_status']}`",
        "",
        "## Overall Summary",
        "",
    ]
    overall = payload["overall"]
    for key, value in overall.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## By Split Group",
        "",
        "| group | n | same full frac | global full frac | median global prior/GT cos | median anchor pp | median anchor MMD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["by_split_group"]:
        lines.append(
            "| {group} | {n} | {same} | {glob} | {cos} | {pp} | {mmd} |".format(
                group=row["group"],
                n=row["n"],
                same=fmt(row["same_full_frac"]),
                glob=fmt(row["global_full_frac"]),
                cos=fmt(row["median_global_prior_gt_cos"]),
                pp=fmt(row["median_anchor_pp"]),
                mmd=fmt(row["median_anchor_mmd"]),
            )
        )
    lines.extend([
        "",
        "## Focus Dataset Summary",
        "",
        "| dataset | n | same full frac | global full frac | median global prior/GT cos | median anchor pp | median anchor MMD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["by_dataset"]:
        if row["group"] not in {"Wessels", "NormanWeissman2019_filtered", "GasperiniShendure2019_lowMOI"}:
            continue
        lines.append(
            "| {group} | {n} | {same} | {glob} | {cos} | {pp} | {mmd} |".format(
                group=row["group"],
                n=row["n"],
                same=fmt(row["same_full_frac"]),
                glob=fmt(row["global_full_frac"]),
                cos=fmt(row["median_global_prior_gt_cos"]),
                pp=fmt(row["median_anchor_pp"]),
                mmd=fmt(row["median_anchor_mmd"]),
            )
        )
    lines.extend([
        "",
        "## Correlations",
        "",
    ])
    for key, value in payload["correlations"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Worst Anchor / Additive Prior Diagnostic Cases",
        "",
        "| dataset | condition | split | anchor pp | anchor MMD | global prior/GT cos | global pair cos | global norm |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in payload["worst_anchor_rows"]:
        lines.append(
            "| {dataset} | {condition} | {split} | {pp} | {mmd} | {cos} | {pair} | {norm} |".format(
                dataset=row["dataset"],
                condition=row["condition"],
                split=row.get("split_group"),
                pp=fmt(fnum(row.get("anchor_pp"))),
                mmd=fmt(fnum(row.get("anchor_mmd"))),
                cos=fmt(fnum(row.get("global_prior_gt_cos"))),
                pair=fmt(fnum(row.get("global_pair_resid_cos"))),
                norm=fmt(fnum(row.get("global_prior_norm"))),
            )
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- High global coverage alone is not enough; use the prior/GT cosine diagnostics only as feasibility evidence.",
        "- A GPU branch is only justified if train-only prior features suggest a stable route or target that can be constrained by anchor replay.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--anchor-split-json", type=Path, default=DEFAULT_ANCHOR_SPLIT)
    parser.add_argument("--max-cells-per-condition", type=int, default=512)
    parser.add_argument("--max-train-single-per-dataset", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    split = load_json(args.split_file)
    metadata = load_json(args.data_dir / "condition_metadata.json")
    same, global_bank, train_counts = train_single_bank(
        args.data_dir,
        split,
        metadata,
        args.max_cells_per_condition,
        args.max_train_single_per_dataset,
    )
    rows = collect_test_multi_rows(
        args.data_dir,
        split,
        metadata,
        same,
        global_bank,
        anchor_metrics(load_json(args.anchor_split_json)),
        args.max_cells_per_condition,
    )
    worst = sorted(
        [r for r in rows if fnum(r.get("anchor_pp")) is not None],
        key=lambda r: float(r["anchor_pp"]),
    )[:16]
    payload = {
        "data_dir": str(args.data_dir),
        "split_file": str(args.split_file),
        "anchor_split_json": str(args.anchor_split_json),
        "max_cells_per_condition": int(args.max_cells_per_condition),
        "max_train_single_per_dataset": int(args.max_train_single_per_dataset),
        "leakage_status": "train_single_prior_features_only; heldout_multi_gt_used_for_diagnostic_alignment_only",
        "train_single_counts_by_dataset": train_counts,
        "overall": summarize(rows),
        "by_split_group": summarize(rows, "split_group"),
        "by_dataset": summarize(rows, "dataset"),
        "correlations": correlation_rows(rows),
        "worst_anchor_rows": worst,
        "rows": rows,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
