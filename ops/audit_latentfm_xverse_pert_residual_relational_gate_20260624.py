#!/usr/bin/env python3
"""CPU-only gate for a Track A perturbation-residual relational loss.

The training hook already exists and is default-off. This audit checks whether
train/internal residual targets contain enough nontrivial same-gene structure to
justify a GPU smoke. It reads only train/internal condition-mean artifacts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
MEAN_DIR = ROOT / "reports/latentfm_xverse_nuisance_condition_means_20260624"
ANCHOR_JSON = MEAN_DIR / "split_group_eval_anchor_internal_means_ode20.json"
CAP120_JSON = MEAN_DIR / "split_group_eval_cap120_internal_means_ode20.json"
OUT_JSON = ROOT / "reports/latentfm_xverse_pert_residual_relational_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_PERT_RESIDUAL_RELATIONAL_GATE_20260624.md"
GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
K = 5
N_SHUFFLE = 200
SEED = 20260624


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def vec(row: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(row[key], dtype=np.float64)


def normalize(mat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return mat / norm


def corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a - float(np.mean(a))
    bb = b - float(np.mean(b))
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(aa, bb) / den)


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    xr = rankdata_average(x[mask])
    yr = rankdata_average(y[mask])
    if float(np.std(xr)) <= 1e-12 or float(np.std(yr)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def rows_by_group(obj: dict[str, Any], group: str) -> list[dict[str, Any]]:
    return list(obj["groups"][group]["condition_metrics"])


def align_rows(
    anchor_rows: list[dict[str, Any]],
    cap_rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    cap = {(str(r["dataset"]), str(r["condition"])): r for r in cap_rows}
    out = []
    for row in anchor_rows:
        key = (str(row["dataset"]), str(row["condition"]))
        if key in cap:
            out.append((row, cap[key]))
    return out


def topk_indices(sim: np.ndarray, k: int) -> np.ndarray:
    work = sim.copy()
    np.fill_diagonal(work, -np.inf)
    kk = min(k, max(0, work.shape[1] - 1))
    if kk <= 0:
        return np.empty((work.shape[0], 0), dtype=int)
    return np.argsort(-work, axis=1, kind="mergesort")[:, :kk]


def topk_fraction(labels: list[str], nn: np.ndarray) -> float:
    if nn.size == 0:
        return float("nan")
    vals = []
    for i in range(nn.shape[0]):
        vals.extend([labels[j] == labels[i] for j in nn[i]])
    return float(np.mean(vals)) if vals else float("nan")


def duplicate_support(labels: list[str], datasets: list[str]) -> dict[str, Any]:
    by_label: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for label, ds in zip(labels, datasets):
        by_label.setdefault(label, set()).add(ds)
        counts[label] = counts.get(label, 0) + 1
    duplicate_labels = {lab: counts[lab] for lab, n in counts.items() if n > 1}
    cross_dataset_duplicate_labels = {
        lab: counts[lab] for lab, dss in by_label.items() if counts[lab] > 1 and len(dss) > 1
    }
    return {
        "n_unique_conditions": len(counts),
        "n_duplicate_conditions": len(duplicate_labels),
        "n_duplicate_rows": int(sum(duplicate_labels.values())),
        "n_cross_dataset_duplicate_conditions": len(cross_dataset_duplicate_labels),
        "n_cross_dataset_duplicate_rows": int(sum(cross_dataset_duplicate_labels.values())),
        "top_duplicates": sorted(cross_dataset_duplicate_labels.items(), key=lambda item: (-item[1], item[0]))[:10],
    }


def residual_metrics(anchor_row: dict[str, Any], cap_row: dict[str, Any]) -> dict[str, float]:
    target = vec(anchor_row, "gt_mean") - vec(anchor_row, "pert_mean")
    anchor_pred = vec(anchor_row, "pred_mean") - vec(anchor_row, "pert_mean")
    cap_pred = vec(cap_row, "pred_mean") - vec(cap_row, "pert_mean")
    return {
        "anchor_resid_cos": corr(anchor_pred, target),
        "cap120_resid_cos": corr(cap_pred, target),
        "anchor_pp": float(anchor_row["pearson_pert"]),
        "cap120_pp": float(cap_row["pearson_pert"]),
        "cap120_mmd_minus_anchor": float(cap_row["test_mmd_clamped"]) - float(anchor_row["test_mmd_clamped"]),
    }


def summarize_group(
    anchor_rows: list[dict[str, Any]],
    cap_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    pairs = align_rows(anchor_rows, cap_rows)
    labels = [str(a["condition"]) for a, _ in pairs]
    datasets = [str(a["dataset"]) for a, _ in pairs]
    target_resid = np.vstack([vec(a, "gt_mean") - vec(a, "pert_mean") for a, _ in pairs])
    sim = normalize(target_resid) @ normalize(target_resid).T
    nn = topk_indices(sim, K)
    same_condition = topk_fraction(labels, nn)
    same_dataset = topk_fraction(datasets, nn)
    rng = np.random.default_rng(seed)
    shuf_condition = []
    shuf_dataset = []
    for _ in range(N_SHUFFLE):
        perm = rng.permutation(len(pairs))
        shuf_condition.append(topk_fraction([labels[i] for i in perm], nn))
        shuf_dataset.append(topk_fraction([datasets[i] for i in perm], nn))
    per_row = [residual_metrics(a, c) for a, c in pairs]
    anchor_cos = np.asarray([r["anchor_resid_cos"] for r in per_row], dtype=float)
    cap_cos = np.asarray([r["cap120_resid_cos"] for r in per_row], dtype=float)
    pp_delta = np.asarray([r["cap120_pp"] - r["anchor_pp"] for r in per_row], dtype=float)
    cos_delta = cap_cos - anchor_cos
    mmd_delta = np.asarray([r["cap120_mmd_minus_anchor"] for r in per_row], dtype=float)
    support = duplicate_support(labels, datasets)
    return {
        "n_rows": len(pairs),
        "n_datasets": len(set(datasets)),
        "duplicate_support": support,
        "target_residual_neighborhood": {
            "topk": K,
            "same_condition_fraction": same_condition,
            "same_condition_shuffle_mean": float(np.mean(shuf_condition)),
            "same_condition_enrichment": same_condition - float(np.mean(shuf_condition)),
            "same_dataset_fraction": same_dataset,
            "same_dataset_shuffle_mean": float(np.mean(shuf_dataset)),
            "same_dataset_enrichment": same_dataset - float(np.mean(shuf_dataset)),
        },
        "model_residual_alignment": {
            "anchor_resid_cos_mean": float(np.nanmean(anchor_cos)),
            "cap120_resid_cos_mean": float(np.nanmean(cap_cos)),
            "cap120_minus_anchor_resid_cos_mean": float(np.nanmean(cos_delta)),
            "spearman_cos_delta_vs_pp_delta": spearman(cos_delta, pp_delta),
            "spearman_cos_delta_vs_mmd_delta": spearman(cos_delta, mmd_delta),
        },
        "dataset_rows": dataset_rows(datasets, cos_delta, pp_delta, mmd_delta),
    }


def dataset_rows(
    datasets: list[str],
    cos_delta: np.ndarray,
    pp_delta: np.ndarray,
    mmd_delta: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for ds in sorted(set(datasets)):
        idx = np.asarray([i for i, item in enumerate(datasets) if item == ds], dtype=int)
        rows.append(
            {
                "dataset": ds,
                "n": int(len(idx)),
                "cos_delta_mean": float(np.nanmean(cos_delta[idx])),
                "pp_delta_mean": float(np.nanmean(pp_delta[idx])),
                "mmd_delta_mean": float(np.nanmean(mmd_delta[idx])),
            }
        )
    return rows


def decide(groups: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    for group, item in groups.items():
        support = item["duplicate_support"]
        neigh = item["target_residual_neighborhood"]
        align = item["model_residual_alignment"]
        if support["n_cross_dataset_duplicate_rows"] < 20:
            reasons.append(f"{group}_cross_dataset_same_condition_support_lt_20")
        if neigh["same_condition_enrichment"] < 0.05:
            reasons.append(f"{group}_same_condition_residual_enrichment_lt_0p05")
        if neigh["same_dataset_fraction"] > 0.70:
            reasons.append(f"{group}_target_residual_neighbors_dataset_dominated")
        if not math.isfinite(align["cap120_minus_anchor_resid_cos_mean"]) or align["cap120_minus_anchor_resid_cos_mean"] < 0.005:
            reasons.append(f"{group}_cap120_not_improving_residual_cosine_enough")
        if not math.isfinite(align["spearman_cos_delta_vs_pp_delta"]) or align["spearman_cos_delta_vs_pp_delta"] < 0.25:
            reasons.append(f"{group}_residual_cos_delta_not_predictive_of_pp_delta")
    status = (
        "pert_residual_relational_gate_pass_design_one_gpu_smoke"
        if not reasons
        else "pert_residual_relational_gate_fail_no_gpu"
    )
    action = (
        "prepare_default_off_pert_residual_relational_smoke"
        if not reasons
        else "do_not_launch_pert_residual_relational_without_new_signal"
    )
    return {"status": status, "recommended_action": action, "reasons": reasons}


def fmt(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:+.6f}"


def write_md(payload: dict[str, Any]) -> None:
    lines = [
        "# LatentFM xverse Pert-Residual Relational Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['recommended_action']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only audit over train/internal condition-mean artifacts.",
        "- No canonical test, canonical multi, Track C query, active logs, or held-out posthoc outcomes are read.",
        "- Tests whether the existing default-off `pert_residual_relational_loss` hook has enough residual-neighborhood signal to justify a GPU smoke.",
        "",
        "## Gate",
        "",
        "- cross-dataset same-condition duplicate rows `>= 20` in each group;",
        "- target-residual top-k same-condition enrichment over shuffled labels `>= +0.05`;",
        "- target-residual top-k same-dataset fraction `<= 0.70`;",
        "- cap120 improves residual cosine over anchor by `>= +0.005`;",
        "- residual-cosine delta Spearman with pp delta `>= +0.25`.",
        "",
        "## Group Summary",
        "",
        "| group | n | cross-dataset duplicate rows | same-cond enrich | same-dataset frac | cap120-anchor resid cos | Spearman cosΔ/ppΔ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        item = payload["groups"][group]
        support = item["duplicate_support"]
        neigh = item["target_residual_neighborhood"]
        align = item["model_residual_alignment"]
        lines.append(
            "| {group} | {n} | {dup} | {sce} | {sdf} | {cd} | {sp} |".format(
                group=group,
                n=item["n_rows"],
                dup=support["n_cross_dataset_duplicate_rows"],
                sce=fmt(neigh["same_condition_enrichment"]),
                sdf=fmt(neigh["same_dataset_fraction"]),
                cd=fmt(align["cap120_minus_anchor_resid_cos_mean"]),
                sp=fmt(align["spearman_cos_delta_vs_pp_delta"]),
            )
        )
    lines.extend(["", "## Gate Reasons", ""])
    if payload["decision"]["reasons"]:
        lines.extend([f"- `{reason}`" for reason in payload["decision"]["reasons"]])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A fail means the residual-relational hook should stay closed for Track A until a new residual target/source produces stronger train-only structure.",
            "- This does not evaluate or select on canonical held-out metrics.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    anchor = load_json(ANCHOR_JSON)
    cap120 = load_json(CAP120_JSON)
    groups = {
        group: summarize_group(
            rows_by_group(anchor, group),
            rows_by_group(cap120, group),
            seed=SEED + i * 1000,
        )
        for i, group in enumerate(GROUPS)
    }
    payload = {
        "inputs": {"anchor": str(ANCHOR_JSON), "cap120": str(CAP120_JSON)},
        "leakage_status": "train_internal_condition_means_only_no_canonical_no_multi_no_query",
        "groups": groups,
    }
    payload["decision"] = decide(groups)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_md(payload)
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
