#!/usr/bin/env python3
"""CPU gate for target-gene observability as a scaling mechanism.

This audit uses train/internal artifacts and control h5ad files only. It does
not read canonical multi, Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_xverse_scaling_count_smokes_20260624"
CONTROL_DIR = ROOT / "dataset/Training_data/scfoundation/control_scfoundation"
METADATA = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
OUT_JSON = ROOT / "reports/latentfm_scaling_target_activity_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_SCALING_TARGET_ACTIVITY_GATE_20260624.md"

GROUP = "internal_val_cross_background_seen_gene_proxy"
CAP30_RUN = RUN_ROOT / "xverse_scaling_cap30_all_3k_seed42"
CAP120_RUN = RUN_ROOT / "xverse_scaling_cap120_all_3k_seed42"
SEED = 42


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_categorical(group: h5py.Group) -> list[str]:
    codes = group["codes"][:]
    cats_obj = group["categories"]
    if isinstance(cats_obj, h5py.Group):
        cats = cats_obj["values"][:]
    else:
        cats = cats_obj[:]
    cats_s = [x.decode() if isinstance(x, bytes) else str(x) for x in cats]
    return [cats_s[int(c)] if int(c) >= 0 else "" for c in codes]


def read_string_array(obj: h5py.Dataset | h5py.Group) -> list[str]:
    if isinstance(obj, h5py.Group):
        return read_categorical(obj)
    arr = obj[:]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def normalize_gene(name: str) -> str:
    return str(name).strip().upper()


def metric_rows(path: Path, group: str) -> dict[tuple[str, str], dict[str, Any]]:
    payload = load_json(path)
    rows = ((payload.get("groups") or {}).get(group) or {}).get("condition_metrics") or []
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        ds = str(row.get("dataset") or "")
        cond = str(row.get("condition") or "")
        if ds and cond:
            out[(ds, cond)] = row
    return out


def load_internal_pair_rows() -> list[dict[str, Any]]:
    cap30_c = metric_rows(CAP30_RUN / "posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json", GROUP)
    cap30_a = metric_rows(CAP30_RUN / "posthoc_eval_internal/split_group_eval_anchor_internal_ode20.json", GROUP)
    cap120_c = metric_rows(CAP120_RUN / "posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json", GROUP)
    cap120_a = metric_rows(CAP120_RUN / "posthoc_eval_internal/split_group_eval_anchor_internal_ode20.json", GROUP)
    rows = []
    for key in sorted(set(cap30_c) & set(cap30_a) & set(cap120_c) & set(cap120_a)):
        vals = {
            "cap30_c_pp": cap30_c[key].get("pearson_pert"),
            "cap30_a_pp": cap30_a[key].get("pearson_pert"),
            "cap120_c_pp": cap120_c[key].get("pearson_pert"),
            "cap120_a_pp": cap120_a[key].get("pearson_pert"),
            "cap30_c_mmd": cap30_c[key].get("test_mmd_clamped"),
            "cap30_a_mmd": cap30_a[key].get("test_mmd_clamped"),
            "cap120_c_mmd": cap120_c[key].get("test_mmd_clamped"),
            "cap120_a_mmd": cap120_a[key].get("test_mmd_clamped"),
        }
        if any(v is None for v in vals.values()):
            continue
        ds, cond = key
        cap30_pp = float(vals["cap30_c_pp"]) - float(vals["cap30_a_pp"])
        cap120_pp = float(vals["cap120_c_pp"]) - float(vals["cap120_a_pp"])
        cap30_mmd = float(vals["cap30_c_mmd"]) - float(vals["cap30_a_mmd"])
        cap120_mmd = float(vals["cap120_c_mmd"]) - float(vals["cap120_a_mmd"])
        rows.append(
            {
                "dataset": ds,
                "condition": cond,
                "pp_delta_cap120_minus_cap30": cap120_pp - cap30_pp,
                "mmd_delta_cap120_minus_cap30": cap120_mmd - cap30_mmd,
            }
        )
    return rows


def find_control_h5ad(dataset: str) -> Path | None:
    path = CONTROL_DIR / f"{dataset}.h5ad"
    return path if path.exists() else None


def compute_activity(metadata: dict[str, Any], needed: set[tuple[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset, conditions in sorted(metadata.items()):
        relevant = {cond for ds, cond in needed if ds == dataset and cond in conditions and conditions[cond].get("genes")}
        if not relevant:
            continue
        h5ad = find_control_h5ad(dataset)
        if h5ad is None:
            for cond in relevant:
                out[(dataset, cond)] = {"status": "missing_control_h5ad"}
            continue
        with h5py.File(h5ad, "r") as handle:
            obs = handle["obs"]
            condition_col = str(next(iter(conditions.values())).get("condition_col") or "perturbation")
            if condition_col not in obs:
                condition_col = "perturbation" if "perturbation" in obs else condition_col
            if condition_col not in obs:
                for cond in relevant:
                    out[(dataset, cond)] = {"status": "missing_condition_col", "condition_col": condition_col}
                continue
            row_conditions = read_string_array(obs[condition_col])
            gene_symbols = read_string_array(handle["var"]["Gene_symbol"])
            gene_to_idx = {normalize_gene(g): i for i, g in enumerate(gene_symbols)}
            cond_to_gene_idxs: dict[str, list[int]] = {}
            for cond in relevant:
                idxs = [gene_to_idx[normalize_gene(g)] for g in conditions[cond].get("genes", []) if normalize_gene(g) in gene_to_idx]
                cond_to_gene_idxs[cond] = sorted(set(idxs))
                if not idxs:
                    out[(dataset, cond)] = {
                        "status": "target_gene_not_in_matrix",
                        "target_genes": conditions[cond].get("genes", []),
                    }
            x = handle["X"]
            if not isinstance(x, h5py.Group) or "indptr" not in x or "indices" not in x or "data" not in x:
                for cond in relevant:
                    out.setdefault((dataset, cond), {"status": "unsupported_x_format"})
                continue
            indptr = x["indptr"]
            indices = x["indices"]
            data = x["data"]
            # control_scfoundation files contain a dataset/background-level
            # control pool; their perturbation column is often all "control".
            # Therefore activity is a background observability feature, not a
            # condition-row pairing feature.
            all_target_idxs = sorted({idx for idxs in cond_to_gene_idxs.values() for idx in idxs})
            idx_acc = {idx: {"sum": 0.0, "nonzero": 0} for idx in all_target_idxs}
            n_cells = len(row_conditions)
            for row_i in range(n_cells):
                start = int(indptr[row_i])
                end = int(indptr[row_i + 1])
                row_idx = indices[start:end]
                row_data = data[start:end]
                for target_idx in all_target_idxs:
                    pos = np.searchsorted(row_idx, target_idx)
                    if pos < len(row_idx) and int(row_idx[pos]) == target_idx:
                        val = float(row_data[pos])
                        idx_acc[target_idx]["sum"] += val
                        if val > 0:
                            idx_acc[target_idx]["nonzero"] += 1
            for cond in relevant:
                if (dataset, cond) in out and out[(dataset, cond)].get("status") == "target_gene_not_in_matrix":
                    continue
                idxs = cond_to_gene_idxs.get(cond) or []
                if not idxs or n_cells <= 0:
                    out[(dataset, cond)] = {"status": "no_control_rows_for_dataset"}
                    continue
                denom = float(n_cells * len(idxs))
                total_sum = sum(idx_acc[idx]["sum"] for idx in idxs)
                total_nonzero = sum(idx_acc[idx]["nonzero"] for idx in idxs)
                out[(dataset, cond)] = {
                    "status": "ok",
                    "n_control_cells": int(n_cells),
                    "n_target_genes": int(len(idxs)),
                    "target_expr_mean": float(total_sum) / denom,
                    "target_expr_nonzero_fraction": float(total_nonzero) / denom,
                }
    return out


def mean(vals: list[float]) -> float | None:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return sum(xs) / len(xs) if xs else None


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return float("nan")
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float:
    return corr(rank(xs), rank(ys))


def permutation_p_abs(xs: list[float], ys: list[float], *, n_perm: int = 2000) -> float:
    obs = abs(spearman(xs, ys))
    if not math.isfinite(obs):
        return 1.0
    rng = random.Random(SEED)
    y = list(ys)
    hit = 0
    for _ in range(n_perm):
        rng.shuffle(y)
        val = abs(spearman(xs, y))
        if math.isfinite(val) and val >= obs:
            hit += 1
    return (hit + 1.0) / (n_perm + 1.0)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("activity_status") == "ok"]
    expr = [float(r["target_expr_mean"]) for r in ok]
    frac = [float(r["target_expr_nonzero_fraction"]) for r in ok]
    pp = [float(r["pp_delta_cap120_minus_cap30"]) for r in ok]
    mmd = [float(r["mmd_delta_cap120_minus_cap30"]) for r in ok]
    if ok:
        q66 = sorted(frac)[max(0, int(0.66 * (len(frac) - 1)))]
        high = [r for r in ok if float(r["target_expr_nonzero_fraction"]) >= q66]
        low = [r for r in ok if float(r["target_expr_nonzero_fraction"]) < q66]
    else:
        q66, high, low = 0.0, [], []
    by_dataset = defaultdict(list)
    for row in high:
        by_dataset[row["dataset"]].append(float(row["pp_delta_cap120_minus_cap30"]))
    dataset_means = {ds: mean(vals) for ds, vals in by_dataset.items()}
    return {
        "n_rows_total": len(rows),
        "n_rows_activity_ok": len(ok),
        "activity_ok_fraction": len(ok) / max(1, len(rows)),
        "spearman_pp_vs_expr_mean": spearman(expr, pp),
        "spearman_pp_vs_expr_mean_perm_p_abs": permutation_p_abs(expr, pp),
        "spearman_pp_vs_nonzero_fraction": spearman(frac, pp),
        "spearman_pp_vs_nonzero_fraction_perm_p_abs": permutation_p_abs(frac, pp),
        "spearman_mmd_vs_nonzero_fraction": spearman(frac, mmd),
        "high_activity_threshold_nonzero_fraction": q66,
        "high_activity_n": len(high),
        "high_activity_pp_mean": mean([float(r["pp_delta_cap120_minus_cap30"]) for r in high]),
        "high_activity_mmd_mean": mean([float(r["mmd_delta_cap120_minus_cap30"]) for r in high]),
        "low_activity_n": len(low),
        "low_activity_pp_mean": mean([float(r["pp_delta_cap120_minus_cap30"]) for r in low]),
        "low_activity_mmd_mean": mean([float(r["mmd_delta_cap120_minus_cap30"]) for r in low]),
        "high_activity_dataset_min_pp": min((v for v in dataset_means.values() if v is not None), default=None),
        "high_activity_negative_dataset_tails_lt_minus_0p02": sum(1 for v in dataset_means.values() if v is not None and v < -0.02),
    }


def fmt(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):+.6f}"
    except Exception:
        return str(v)


def main() -> int:
    metadata = load_json(METADATA)
    pair_rows = load_internal_pair_rows()
    needed = {(r["dataset"], r["condition"]) for r in pair_rows}
    activity = compute_activity(metadata, needed)
    joined = []
    for row in pair_rows:
        act = activity.get((row["dataset"], row["condition"]), {"status": "not_gene_or_missing"})
        out = dict(row)
        out["activity_status"] = act.get("status")
        if act.get("status") == "ok":
            out.update(
                {
                    "n_control_cells": act["n_control_cells"],
                    "n_target_genes": act["n_target_genes"],
                    "target_expr_mean": act["target_expr_mean"],
                    "target_expr_nonzero_fraction": act["target_expr_nonzero_fraction"],
                }
            )
        joined.append(out)
    summary = summarize(joined)
    reasons = []
    if summary["n_rows_activity_ok"] < 50:
        reasons.append("too_few_activity_covered_rows")
    if float(summary["spearman_pp_vs_nonzero_fraction"] or 0.0) <= 0.20:
        reasons.append("activity_pp_spearman_not_material")
    if float(summary["spearman_pp_vs_nonzero_fraction_perm_p_abs"] or 1.0) > 0.10:
        reasons.append("activity_pp_permutation_not_significant")
    if float(summary["high_activity_pp_mean"] or -1.0) < 0.020:
        reasons.append("high_activity_pp_mean_below_0p020")
    if float(summary["high_activity_dataset_min_pp"] if summary["high_activity_dataset_min_pp"] is not None else -1.0) < -0.020:
        reasons.append("high_activity_dataset_tail_below_minus_0p020")
    status = "target_activity_gate_fail_no_gpu" if reasons else "target_activity_gate_pass_external_review_next"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "train_internal_only": True,
            "control_h5ad_only": True,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "inputs": {
            "metadata": str(METADATA),
            "control_dir": str(CONTROL_DIR),
            "cap30_run": str(CAP30_RUN),
            "cap120_run": str(CAP120_RUN),
            "group": GROUP,
        },
        "summary": summary,
        "reasons": reasons,
        "rows": joined,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Scaling Target-Activity Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only train/internal gate using control h5ad target-gene observability.",
        "- Does not read canonical multi, held-out Track C query, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- total paired cap120-cap30 rows: `{summary['n_rows_total']}`",
        f"- activity-covered rows: `{summary['n_rows_activity_ok']}`",
        f"- Spearman pp vs target expression mean: `{fmt(summary['spearman_pp_vs_expr_mean'])}`; perm p `{fmt(summary['spearman_pp_vs_expr_mean_perm_p_abs'])}`",
        f"- Spearman pp vs target nonzero fraction: `{fmt(summary['spearman_pp_vs_nonzero_fraction'])}`; perm p `{fmt(summary['spearman_pp_vs_nonzero_fraction_perm_p_abs'])}`",
        f"- high-activity pp/MMD mean: `{fmt(summary['high_activity_pp_mean'])}` / `{fmt(summary['high_activity_mmd_mean'])}`",
        f"- high-activity dataset min pp: `{fmt(summary['high_activity_dataset_min_pp'])}`",
        f"- high-activity negative dataset tails `< -0.02`: `{summary['high_activity_negative_dataset_tails_lt_minus_0p02']}`",
        "",
        "## Decision",
        "",
        f"- reasons: `{reasons}`",
        "- GPU authorized: `False`",
        "- A pass would require external review before one bounded target-activity adapter smoke.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "summary": summary, "out_md": str(OUT_MD)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
