#!/usr/bin/env python3
"""Train-only reliability preflight for robust-loss fallback.

This CPU-only preflight computes train-condition reliability summaries from
HDF5 embeddings and tests whether dataset-level reliability covaries with
internal candidate-vs-anchor deltas. It is weaker than the full protocol in
LATENTFM_TRAINONLY_RELIABILITY_WEIGHTED_LOSS_GATE_PROTOCOL_20260624.md and
does not authorize GPU by itself.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

import h5py
import numpy as np

ROOT = Path("/data/cyx/1030/scLatent")
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
SPLIT_FILE = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
OUT_JSON = ROOT / "reports/latentfm_trainonly_reliability_preflight_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRAINONLY_RELIABILITY_PREFLIGHT_20260624.md"

RUNS = {
    "cap120": {
        "anchor": ROOT / "runs/latentfm_xverse_scaling_count_smokes_20260624/xverse_scaling_cap120_all_3k_seed42/posthoc_eval_internal/split_group_eval_anchor_internal_ode20.json",
        "candidate": ROOT / "runs/latentfm_xverse_scaling_count_smokes_20260624/xverse_scaling_cap120_all_3k_seed42/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json",
    },
    "cap60_protocol": {
        "anchor": ROOT / "runs/latentfm_scaling_protocol_matrix_20260624/xverse_scaling_protocol_cap60_primary19_3k_seed42/posthoc_eval_internal/split_group_eval_anchor_internal_ode20.json",
        "candidate": ROOT / "runs/latentfm_scaling_protocol_matrix_20260624/xverse_scaling_protocol_cap60_primary19_3k_seed42/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json",
    },
}

GROUPS = ("internal_val_cross_background_seen_gene_proxy", "internal_val_family_gene_proxy")
MAX_CELLS_PER_CONDITION = 512
SEED = 42
N_PERM = 2000


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def decode_conditions(values: np.ndarray) -> list[str]:
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def sample_slice(arr: h5py.Dataset, lo: int, hi: int, *, max_cells: int, key: str) -> np.ndarray:
    n = int(hi - lo)
    if n <= 0:
        raise ValueError(f"empty slice for {key}")
    if max_cells > 0 and n > max_cells:
        rng = np.random.default_rng(abs(hash(key)) % (2**32))
        idx = np.sort(rng.choice(n, size=max_cells, replace=False))
        return np.asarray(arr[lo + idx], dtype=np.float64)
    return np.asarray(arr[lo:hi], dtype=np.float64)


def condition_reliability() -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    split = load_json(SPLIT_FILE)
    rows: list[dict[str, Any]] = []
    for ds, groups in split.items():
        train = [str(c) for c in groups.get("train", [])]
        if not train:
            continue
        h5_path = DATA_DIR / f"{ds}.h5"
        with h5py.File(h5_path, "r") as h5:
            conds = decode_conditions(np.asarray(h5["conditions"]))
            cidx = {c: i for i, c in enumerate(conds)}
            ctrl = h5["ctrl/emb"] if "ctrl/emb" in h5 else h5["ir/emb"]
            gt = h5["gt/emb"]
            ctrl_offsets = np.asarray(h5["ctrl/offsets"] if "ctrl/offsets" in h5 else h5["ir/offsets"])
            gt_offsets = np.asarray(h5["gt/offsets"])
            for cond in train:
                if cond not in cidx:
                    continue
                i = cidx[cond]
                c0, c1 = int(ctrl_offsets[i]), int(ctrl_offsets[i + 1])
                g0, g1 = int(gt_offsets[i]), int(gt_offsets[i + 1])
                ctrl_arr = sample_slice(ctrl, c0, c1, max_cells=MAX_CELLS_PER_CONDITION, key=f"ctrl|{ds}|{cond}")
                gt_arr = sample_slice(gt, g0, g1, max_cells=MAX_CELLS_PER_CONDITION, key=f"gt|{ds}|{cond}")
                ctrl_mean = ctrl_arr.mean(axis=0)
                gt_mean = gt_arr.mean(axis=0)
                delta = gt_mean - ctrl_mean
                ctrl_var = float(np.mean(np.var(ctrl_arr, axis=0)))
                gt_var = float(np.mean(np.var(gt_arr, axis=0)))
                sem = math.sqrt(ctrl_var / max(1, len(ctrl_arr)) + gt_var / max(1, len(gt_arr)))
                norm = float(np.linalg.norm(delta))
                rows.append(
                    {
                        "dataset": ds,
                        "condition": cond,
                        "n_ctrl": int(c1 - c0),
                        "n_gt": int(g1 - g0),
                        "response_norm": norm,
                        "mean_var": float((ctrl_var + gt_var) / 2.0),
                        "sem_proxy": sem,
                        "snr_proxy": float(norm / (sem + 1e-8)),
                        "log_n_gt": float(math.log1p(g1 - g0)),
                    }
                )
    by_ds: dict[str, dict[str, float]] = {}
    for ds in sorted({r["dataset"] for r in rows}):
        ds_rows = [r for r in rows if r["dataset"] == ds]
        by_ds[ds] = {
            "n_train_conditions": float(len(ds_rows)),
            "mean_log_n_gt": float(mean([r["log_n_gt"] for r in ds_rows])),
            "median_response_norm": float(np.median([r["response_norm"] for r in ds_rows])),
            "median_sem_proxy": float(np.median([r["sem_proxy"] for r in ds_rows])),
            "median_snr_proxy": float(np.median([r["snr_proxy"] for r in ds_rows])),
            "high_sem_frac": float(np.mean([r["sem_proxy"] > np.median([x["sem_proxy"] for x in rows]) for r in ds_rows])),
        }
    return rows, by_ds


def condition_metric_map(path: Path, group: str) -> dict[tuple[str, str], dict[str, float]]:
    obj = load_json(path)
    metrics = obj["groups"][group]["condition_metrics"]
    out = {}
    for row in metrics:
        out[(str(row["dataset"]), str(row["condition"]))] = {
            "pearson_pert": float(row["pearson_pert"]),
            "test_mmd_clamped": float(row["test_mmd_clamped"]),
        }
    return out


def delta_by_dataset() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_name, paths in RUNS.items():
        for group in GROUPS:
            anchor = condition_metric_map(paths["anchor"], group)
            cand = condition_metric_map(paths["candidate"], group)
            common = sorted(set(anchor) & set(cand))
            for ds in sorted({k[0] for k in common}):
                keys = [k for k in common if k[0] == ds]
                pp = [cand[k]["pearson_pert"] - anchor[k]["pearson_pert"] for k in keys]
                mmd = [cand[k]["test_mmd_clamped"] - anchor[k]["test_mmd_clamped"] for k in keys]
                rows.append(
                    {
                        "run": run_name,
                        "group": group,
                        "dataset": ds,
                        "n_eval_conditions": len(keys),
                        "delta_pp": float(mean(pp)),
                        "delta_mmd": float(mean(mmd)),
                    }
                )
    return rows


def rankdata(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and vals[order[j]] == vals[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    mx, my = mean(x), mean(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def perm_p_abs(x: list[float], y: list[float], seed: int) -> float:
    obs = abs(spearman(x, y))
    rng = random.Random(seed)
    yp = list(y)
    count = 1
    for _ in range(N_PERM):
        rng.shuffle(yp)
        if abs(spearman(x, yp)) >= obs:
            count += 1
    return count / (N_PERM + 1)


def correlation_rows(ds_rel: dict[str, dict[str, float]], ds_delta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = [
        "mean_log_n_gt",
        "median_response_norm",
        "median_sem_proxy",
        "median_snr_proxy",
        "high_sem_frac",
    ]
    rows: list[dict[str, Any]] = []
    seed = SEED
    for run in sorted({r["run"] for r in ds_delta}):
        for group in GROUPS:
            subset = [r for r in ds_delta if r["run"] == run and r["group"] == group and r["dataset"] in ds_rel]
            for feature in features:
                x = [float(ds_rel[r["dataset"]][feature]) for r in subset]
                y = [float(r["delta_pp"]) for r in subset]
                rho = spearman(x, y)
                rows.append(
                    {
                        "run": run,
                        "group": group,
                        "feature": feature,
                        "target": "delta_pp",
                        "n": len(subset),
                        "spearman": rho,
                        "perm_p_abs": perm_p_abs(x, y, seed),
                        "material": bool(abs(rho) >= 0.45 and perm_p_abs(x, y, seed + 999) <= 0.10),
                    }
                )
                seed += 1
    return rows


def decide(corrs: list[dict[str, Any]]) -> dict[str, Any]:
    material = [r for r in corrs if r["material"]]
    reasons = []
    if len(material) < 2:
        reasons.append("fewer_than_two_material_reliability_correlations")
    if not any(r["feature"] in {"median_sem_proxy", "median_snr_proxy", "high_sem_frac"} for r in material):
        reasons.append("no_target_stability_feature_material")
    status = "trainonly_reliability_preflight_pass_full_cpu_gate_next_no_gpu"
    if reasons:
        status = "trainonly_reliability_preflight_fail_no_gpu"
    return {
        "status": status,
        "gpu_authorized": False,
        "material_correlations": len(material),
        "reasons": reasons,
        "next_action": "Implement full nested LODO condition-level reliability CPU gate." if not reasons else "Keep reliability weighting as protocol only unless a stronger gate is implemented.",
    }


def render_md(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# LatentFM Train-Only Reliability Preflight",
        "",
        f"Status: `{decision['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only preflight; no GPU authorization.",
        "- Reads train split H5 embeddings plus completed train-only internal eval JSONs.",
        "- Does not read canonical outcomes, canonical multi, Track C query, or active posthoc logs.",
        "",
        "## Decision",
        "",
        f"- material correlations: `{decision['material_correlations']}`",
        f"- GPU authorized: `{decision['gpu_authorized']}`",
        "",
        "Reasons:",
    ]
    lines.extend([f"- `{r}`" for r in decision["reasons"]] or ["- none"])
    lines.extend(
        [
            "",
            "## Top Correlations",
            "",
            "| run | group | feature | n | spearman | perm_p_abs | material |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    top = sorted(payload["correlations"], key=lambda r: abs(r["spearman"]), reverse=True)[:20]
    for row in top:
        lines.append(
            f"| `{row['run']}` | `{row['group']}` | `{row['feature']}` | {row['n']} | {row['spearman']:+.4f} | {row['perm_p_abs']:.4f} | {row['material']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is only a dataset-level preflight. A pass would not launch GPU; it would justify implementing the full nested condition-level reliability CPU gate from the protocol report.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rel_rows, by_ds = condition_reliability()
    deltas = delta_by_dataset()
    corrs = correlation_rows(by_ds, deltas)
    payload = {
        "boundary": {
            "split_file": str(SPLIT_FILE),
            "data_dir": str(DATA_DIR),
            "max_cells_per_condition": MAX_CELLS_PER_CONDITION,
            "no_canonical_or_query": True,
            "no_gpu": True,
            "runs": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in RUNS.items()},
        },
        "summary": {
            "n_train_condition_rows": len(rel_rows),
            "n_datasets_with_reliability": len(by_ds),
            "n_delta_rows": len(deltas),
        },
        "dataset_reliability": by_ds,
        "dataset_deltas": deltas,
        "correlations": corrs,
        "decision": decide(corrs),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    OUT_MD.write_text(render_md(payload))
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
