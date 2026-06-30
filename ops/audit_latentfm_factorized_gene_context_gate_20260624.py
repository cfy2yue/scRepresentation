#!/usr/bin/env python3
"""CPU gate for factorized gene x context delta surrogate."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
SPLIT_FILE = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
METADATA_FILE = DATA_DIR / "condition_metadata.json"
ANCHOR_MEANS = ROOT / "reports/latentfm_xverse_nuisance_condition_means_20260624/split_group_eval_anchor_internal_means_ode20.json"
OUT_JSON = ROOT / "reports/latentfm_factorized_gene_context_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_FACTORIZED_GENE_CONTEXT_GATE_20260624.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
MAX_CELLS_PER_CONDITION = 256
N_CONTEXT_PC = 12
RIDGE = 5.0
BOOT_N = 1000
SEED = 42


@dataclass(frozen=True)
class TrainRow:
    dataset: str
    condition: str
    genes: tuple[str, ...]
    ctrl: np.ndarray
    delta: np.ndarray


@dataclass(frozen=True)
class EvalRow:
    group: str
    dataset: str
    condition: str
    genes: tuple[str, ...]
    ctrl: np.ndarray
    pert: np.ndarray
    gt: np.ndarray
    anchor_pred: np.ndarray
    anchor_pp: float


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_seed(key: str) -> int:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def decode(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def sample_slice(arr: h5py.Dataset, lo: int, hi: int, *, key: str) -> np.ndarray:
    n = int(hi - lo)
    if n <= 0:
        raise ValueError(f"empty slice for {key}")
    if n > MAX_CELLS_PER_CONDITION:
        rng = np.random.default_rng(stable_seed(key))
        idx = np.sort(rng.choice(n, size=MAX_CELLS_PER_CONDITION, replace=False))
        return np.asarray(arr[lo + idx], dtype=np.float64)
    return np.asarray(arr[lo:hi], dtype=np.float64)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a - float(np.mean(a))
    bb = b - float(np.mean(b))
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(aa, bb) / den)


def metadata_genes(metadata: dict[str, dict[str, dict[str, Any]]], ds: str, cond: str) -> tuple[str, ...]:
    genes = metadata.get(ds, {}).get(cond, {}).get("genes") or []
    return tuple(sorted(str(g).upper() for g in genes if str(g)))


def load_train_rows(control: str = "main") -> list[TrainRow]:
    split = load_json(SPLIT_FILE)
    metadata = load_json(METADATA_FILE)
    rows: list[TrainRow] = []
    for ds, groups in split.items():
        h5_path = DATA_DIR / f"{ds}.h5"
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as h5:
            cidx = {c: i for i, c in enumerate(decode(np.asarray(h5["conditions"])))}
            ctrl = h5["ctrl/emb"] if "ctrl/emb" in h5 else h5["ir/emb"]
            gt = h5["gt/emb"]
            ctrl_offsets = np.asarray(h5["ctrl/offsets"] if "ctrl/offsets" in h5 else h5["ir/offsets"])
            gt_offsets = np.asarray(h5["gt/offsets"])
            for cond in groups.get("train", []):
                cond = str(cond)
                genes = metadata_genes(metadata, ds, cond)
                if not genes or cond not in cidx:
                    continue
                i = cidx[cond]
                c0, c1 = int(ctrl_offsets[i]), int(ctrl_offsets[i + 1])
                g0, g1 = int(gt_offsets[i]), int(gt_offsets[i + 1])
                ctrl_arr = sample_slice(ctrl, c0, c1, key=f"ctrl|{ds}|{cond}")
                gt_arr = sample_slice(gt, g0, g1, key=f"gt|{ds}|{cond}")
                ctrl_mean = np.mean(ctrl_arr, axis=0)
                delta = np.mean(gt_arr, axis=0) - ctrl_mean
                rows.append(TrainRow(ds, cond, genes, ctrl_mean, delta))
    if control == "gene_shuffle":
        rng = random.Random(SEED + 611)
        genes = [r.genes for r in rows]
        rng.shuffle(genes)
        rows = [TrainRow(r.dataset, r.condition, genes[i], r.ctrl, r.delta) for i, r in enumerate(rows)]
    elif control == "context_shuffle":
        rng = random.Random(SEED + 612)
        ctrls = [r.ctrl for r in rows]
        rng.shuffle(ctrls)
        rows = [TrainRow(r.dataset, r.condition, r.genes, ctrls[i], r.delta) for i, r in enumerate(rows)]
    elif control == "delta_shuffle":
        rng = random.Random(SEED + 613)
        deltas = [r.delta for r in rows]
        rng.shuffle(deltas)
        rows = [TrainRow(r.dataset, r.condition, r.genes, r.ctrl, deltas[i]) for i, r in enumerate(rows)]
    return rows


def load_eval_rows() -> list[EvalRow]:
    metadata = load_json(METADATA_FILE)
    obj = load_json(ANCHOR_MEANS)
    rows = []
    for group in GROUPS:
        for r in obj["groups"][group]["condition_metrics"]:
            ds = str(r["dataset"])
            cond = str(r["condition"])
            genes = metadata_genes(metadata, ds, cond)
            if not genes:
                continue
            rows.append(
                EvalRow(
                    group=group,
                    dataset=ds,
                    condition=cond,
                    genes=genes,
                    ctrl=np.asarray(r["ctrl_mean"], dtype=np.float64),
                    pert=np.asarray(r["pert_mean"], dtype=np.float64),
                    gt=np.asarray(r["gt_mean"], dtype=np.float64),
                    anchor_pred=np.asarray(r["pred_mean"], dtype=np.float64),
                    anchor_pp=float(r["pearson_pert"]),
                )
            )
    return rows


def gene_mean_map(rows: list[TrainRow]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    by_gene: dict[str, list[np.ndarray]] = {}
    for row in rows:
        for gene in row.genes:
            by_gene.setdefault(gene, []).append(row.delta)
    global_mean = np.mean(np.stack([r.delta for r in rows], axis=0), axis=0)
    return {gene: np.mean(np.stack(vals, axis=0), axis=0) for gene, vals in by_gene.items()}, global_mean


def gene_mean_for(genes: tuple[str, ...], gmap: dict[str, np.ndarray], global_mean: np.ndarray) -> tuple[np.ndarray, int]:
    vals = [gmap[g] for g in genes if g in gmap]
    if not vals:
        return global_mean, 0
    return np.mean(np.stack(vals, axis=0), axis=0), len(vals)


def fit_context_residual_model(rows: list[TrainRow], mode: str) -> dict[str, Any]:
    gmap, global_mean = gene_mean_map(rows)
    ctrl = np.stack([r.ctrl for r in rows], axis=0)
    delta = np.stack([r.delta for r in rows], axis=0)
    gene_base = np.stack([gene_mean_for(r.genes, gmap, global_mean)[0] for r in rows], axis=0)
    ctrl_center = np.mean(ctrl, axis=0)
    xc = ctrl - ctrl_center
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[: min(N_CONTEXT_PC, vt.shape[0])].T
    x = xc @ basis
    if mode == "context_only":
        y = delta
    else:
        y = delta - gene_base
    xtx = x.T @ x
    ridge = RIDGE * np.eye(xtx.shape[0], dtype=np.float64)
    coef = np.linalg.solve(xtx + ridge, x.T @ y)
    return {"gmap": gmap, "global_mean": global_mean, "ctrl_center": ctrl_center, "basis": basis, "coef": coef}


def predict_delta(row: EvalRow, model: dict[str, Any], mode: str) -> tuple[np.ndarray, int]:
    gene_base, support_n = gene_mean_for(row.genes, model["gmap"], model["global_mean"])
    x = (row.ctrl - model["ctrl_center"]) @ model["basis"]
    context = x @ model["coef"]
    if mode == "gene_only":
        return gene_base, support_n
    if mode == "context_only":
        return context, support_n
    if mode == "gene_plus_context":
        return gene_base + context, support_n
    raise ValueError(mode)


def score_rows(train_rows: list[TrainRow], evals: list[EvalRow], mode: str, control: str) -> list[dict[str, Any]]:
    out = []
    datasets = sorted({r.dataset for r in evals})
    for heldout in datasets:
        train = [r for r in train_rows if r.dataset != heldout]
        if not train:
            continue
        model = fit_context_residual_model(train, mode)
        for row in [r for r in evals if r.dataset == heldout]:
            pred_delta, support_n = predict_delta(row, model, mode)
            pred = row.ctrl + pred_delta
            pp = corr(pred - row.ctrl, row.pert - row.ctrl)
            anchor_resid = float(np.linalg.norm(row.anchor_pred - row.gt) / math.sqrt(row.gt.size))
            pred_resid = float(np.linalg.norm(pred - row.gt) / math.sqrt(row.gt.size))
            out.append(
                {
                    "group": row.group,
                    "dataset": row.dataset,
                    "condition": row.condition,
                    "mode": mode,
                    "control": control,
                    "support_n": support_n,
                    "delta_pp": float(pp - row.anchor_pp),
                    "residual_delta": float(pred_resid - anchor_resid),
                }
            )
    return out


def bootstrap(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(SEED)
    arr = np.asarray(values, dtype=np.float64)
    means = []
    for _ in range(BOOT_N):
        idx = [rng.randrange(len(arr)) for _ in arr]
        means.append(float(np.mean(arr[idx])))
    means_arr = np.asarray(means, dtype=np.float64)
    return float(np.quantile(means_arr, 0.025)), float(np.quantile(means_arr, 0.975)), float(np.mean(means_arr < 0.0))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(r["delta_pp"]) for r in rows]
    lo, hi, p_harm = bootstrap(vals)
    by_ds: dict[str, list[float]] = {}
    for r in rows:
        by_ds.setdefault(str(r["dataset"]), []).append(float(r["delta_pp"]))
    return {
        "n": len(rows),
        "mean_pp_delta": float(np.mean(vals)),
        "ci95_low": lo,
        "ci95_high": hi,
        "bootstrap_p_harm": p_harm,
        "dataset_min_pp_delta": float(min(sum(v) / len(v) for v in by_ds.values())),
        "mean_residual_delta": float(np.mean([float(r["residual_delta"]) for r in rows])),
        "mean_support_n": float(np.mean([float(r["support_n"]) for r in rows])),
    }


def decide(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(r["group"], r["mode"], r["control"]): r["summary"] for r in results}
    mode = "gene_plus_context"
    cross = by_key[(GROUPS[0], mode, "main")]
    family = by_key[(GROUPS[1], mode, "main")]
    gene_cross = by_key[(GROUPS[0], "gene_only", "main")]
    context_cross = by_key[(GROUPS[0], "context_only", "main")]
    reasons = []
    if cross["mean_pp_delta"] < 0.010:
        reasons.append("cross_pp_delta_below_0.010")
    if family["mean_pp_delta"] < 0.010:
        reasons.append("family_pp_delta_below_0.010")
    if cross["dataset_min_pp_delta"] < -0.020:
        reasons.append("cross_dataset_min_below_minus_0.020")
    if family["dataset_min_pp_delta"] < -0.020:
        reasons.append("family_dataset_min_below_minus_0.020")
    if family["mean_residual_delta"] > 0.0:
        reasons.append("family_residual_mean_worse")
    if cross["mean_pp_delta"] <= max(gene_cross["mean_pp_delta"], context_cross["mean_pp_delta"]) + 0.002:
        reasons.append("interaction_not_better_than_gene_or_context_baseline")
    for control in ("gene_shuffle", "context_shuffle", "delta_shuffle"):
        csum = by_key[(GROUPS[0], mode, control)]
        if csum["mean_pp_delta"] >= 0.005:
            reasons.append(f"{control}_cross_not_collapsed")
    passed = not reasons
    return {
        "status": "factorized_gene_context_gate_pass_cpu_reopen_only" if passed else "factorized_gene_context_gate_fail_no_gpu",
        "gpu_authorized": False,
        "cpu_reopen_authorized": passed,
        "reasons": reasons,
        "cross_mean_pp_delta": cross["mean_pp_delta"],
        "family_mean_pp_delta": family["mean_pp_delta"],
        "cross_dataset_min": cross["dataset_min_pp_delta"],
        "family_dataset_min": family["dataset_min_pp_delta"],
    }


def render_md(payload: dict[str, Any]) -> str:
    d = payload["decision"]
    lines = [
        "# LatentFM Factorized Gene x Context Gate",
        "",
        f"Status: `{d['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only conditional-reopen gate.",
        "- Uses train split H5 deltas/control means, condition metadata, and completed anchor internal condition means.",
        "- Nested leave-one-dataset-out; eval dataset is excluded when fitting gene/context models.",
        "- Does not read canonical outcomes, canonical multi, Track C query, active logs, new GPU artifacts, or use GPU.",
        "- This surrogate evaluates Pearson/residual mean behavior only, so a pass would authorize only a deeper CPU gate/launcher design, not immediate GPU.",
        "",
        "## Decision",
        "",
        f"- GPU authorized: `{d['gpu_authorized']}`",
        f"- CPU reopen authorized: `{d['cpu_reopen_authorized']}`",
        f"- reasons: `{d['reasons']}`",
        f"- factorized cross pp delta: `{d['cross_mean_pp_delta']:.6f}`",
        f"- factorized family pp delta: `{d['family_mean_pp_delta']:.6f}`",
        f"- cross dataset-min: `{d['cross_dataset_min']:.6f}`",
        f"- family dataset-min: `{d['family_dataset_min']:.6f}`",
        "",
        "## Summaries",
        "",
        "| group | mode | control | n | mean pp delta | 95% CI | p_harm | dataset min | mean residual delta | support n |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in payload["results"]:
        s = r["summary"]
        lines.append(
            f"| `{r['group']}` | `{r['mode']}` | `{r['control']}` | {s['n']} | {s['mean_pp_delta']:.6f} | [{s['ci95_low']:.6f}, {s['ci95_high']:.6f}] | {s['bootstrap_p_harm']:.3f} | {s['dataset_min_pp_delta']:.6f} | {s['mean_residual_delta']:.6f} | {s['mean_support_n']:.2f} |"
        )
    lines.extend(["", "## JSON", "", f"`{OUT_JSON}`", ""])
    return "\n".join(lines)


def main() -> None:
    evals = load_eval_rows()
    results = []
    for control in ("main", "gene_shuffle", "context_shuffle", "delta_shuffle"):
        trows = load_train_rows(control)
        modes = ("gene_only", "context_only", "gene_plus_context") if control == "main" else ("gene_plus_context",)
        for mode in modes:
            scored = score_rows(trows, evals, mode, control)
            for group in GROUPS:
                rows = [r for r in scored if r["group"] == group]
                results.append({"group": group, "mode": mode, "control": control, "summary": summarize(rows)})
    payload = {
        "boundary": {
            "split_file": str(SPLIT_FILE),
            "metadata_file": str(METADATA_FILE),
            "anchor_means": str(ANCHOR_MEANS),
            "max_cells_per_condition": MAX_CELLS_PER_CONDITION,
            "n_context_pc": N_CONTEXT_PC,
            "ridge": RIDGE,
            "seed": SEED,
        },
        "results": results,
        "decision": decide(results),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    print(OUT_MD)


if __name__ == "__main__":
    main()
