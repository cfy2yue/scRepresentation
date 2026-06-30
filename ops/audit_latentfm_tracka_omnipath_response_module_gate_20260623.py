#!/usr/bin/env python3
"""Track A OmniPath regulatory response-module CPU gate.

The earlier OmniPath Track A gate used static TF/target degree and sign
features.  This gate is materially different: for each leave-one-dataset-out
fold it uses only training-fold proxy outcomes from regulatory neighbors to
estimate whether the xverse anchor should beat the train-only gene baseline.

It does not read canonical test, canonical multi, held-out query artifacts,
active logs, or GPU artifacts.  Passing would authorize only a later code and
provenance gate, not immediate GPU.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
PRIOR_DIR = ROOT / "dataset/external_priors/omnipath_tf_20260623"
PRIOR_SUMMARY = PRIOR_DIR / "omnipath_tf_prior_summary.json"
PRIOR_EDGES = PRIOR_DIR / "omnipath_tf_target_edges.tsv"
XVERSE_ROWS = REPORTS / "latentfm_xverse_tracka_residual_forensics_20260622.json"
OUT_JSON = REPORTS / "latentfm_tracka_omnipath_response_module_gate_20260623.json"
OUT_MD = REPORTS / "LATENTFM_TRACKA_OMNIPATH_RESPONSE_MODULE_GATE_20260623.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
BOOT_N = 2000
SEED = 20260623


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}"


def load_edges(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            tf = str(row.get("tf") or "").strip().upper()
            target = str(row.get("target") or "").strip().upper()
            if not tf or not target:
                continue
            rows.append(
                {
                    "tf": tf,
                    "target": target,
                    "sign": int(float(row.get("sign") or 0)),
                    "n_raw_rows": int(float(row.get("n_raw_rows") or 1)),
                }
            )
    return rows


def graph_maps(edges: list[dict[str, Any]]) -> dict[str, dict[str, list[tuple[str, int, float]]]]:
    downstream: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    upstream: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    undirected: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    for row in edges:
        tf = row["tf"]
        target = row["target"]
        sign = int(row["sign"])
        weight = float(np.log1p(max(int(row["n_raw_rows"]), 1)))
        downstream[tf].append((target, sign, weight))
        upstream[target].append((tf, sign, weight))
        undirected[tf].append((target, sign, weight))
        undirected[target].append((tf, sign, weight))
    return {"downstream": downstream, "upstream": upstream, "undirected": undirected}


def shuffled_edges(edges: list[dict[str, Any]], row_genes: list[str], seed: int) -> list[dict[str, Any]]:
    genes = sorted({g for g in row_genes if g})
    rng = np.random.default_rng(seed)
    perm = list(genes)
    rng.shuffle(perm)
    gene_map = dict(zip(genes, perm))
    out = []
    for row in edges:
        item = dict(row)
        item["tf"] = gene_map.get(str(row["tf"]).upper(), str(row["tf"]).upper())
        item["target"] = gene_map.get(str(row["target"]).upper(), str(row["target"]).upper())
        out.append(item)
    return out


def paired_bootstrap(rows: list[dict[str, Any]], candidate: str, baseline: str, *, seed: int) -> dict[str, Any]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        cand = as_float(row.get(candidate))
        base = as_float(row.get(baseline))
        if np.isfinite(cand) and np.isfinite(base):
            by_ds[str(row["dataset"])].append(float(cand - base))
    keys = sorted(ds for ds, vals in by_ds.items() if vals)
    point_by_ds = {ds: float(np.mean(by_ds[ds])) for ds in keys}
    point = float(np.mean(list(point_by_ds.values()))) if point_by_ds else float("nan")
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(BOOT_N):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        vals = []
        for ds in sampled:
            arr = np.asarray(by_ds[str(ds)], dtype=float)
            vals.append(float(np.mean(rng.choice(arr, size=len(arr), replace=True))))
        boot.append(float(np.mean(vals)))
    arr = np.asarray(boot, dtype=float)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": float(np.mean(arr > 0.0)),
        "p_harm": float(np.mean(arr < 0.0)),
        "dataset_deltas": point_by_ds,
        "dataset_min": float(min(point_by_ds.values())) if point_by_ds else float("nan"),
    }


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


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None
    xr = rankdata_average(x[mask])
    yr = rankdata_average(y[mask])
    if float(np.std(xr)) <= 1e-12 or float(np.std(yr)) <= 1e-12:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def r2_score(y: np.ndarray, pred: np.ndarray) -> float | None:
    mask = np.isfinite(y) & np.isfinite(pred)
    if int(mask.sum()) < 3:
        return None
    yy = y[mask]
    pp = pred[mask]
    denom = float(np.sum((yy - yy.mean()) ** 2))
    if denom <= 1e-12:
        return None
    return float(1.0 - np.sum((yy - pp) ** 2) / denom)


def aggregate_neighbor_response(
    gene: str,
    train_by_gene: dict[str, list[float]],
    graph: dict[str, dict[str, list[tuple[str, int, float]]]],
    mode: str,
) -> tuple[float, dict[str, Any]]:
    gene = gene.upper()
    if mode not in graph:
        raise ValueError(mode)
    vals = []
    signed_vals = []
    weights = []
    n_signed = 0
    for neighbor, sign, weight in graph[mode].get(gene, []):
        ys = train_by_gene.get(neighbor)
        if not ys:
            continue
        value = float(np.mean(ys))
        vals.append(value)
        weights.append(weight)
        if int(sign) != 0:
            signed_vals.append(float(sign) * value)
            n_signed += 1
    if not vals:
        return float("nan"), {"n_neighbors": 0, "n_signed_neighbors": 0, "coverage": 0.0}
    arr = np.asarray(vals, dtype=float)
    w = np.asarray(weights, dtype=float)
    weighted = float(np.sum(arr * w) / max(np.sum(w), 1e-12))
    signed = float(np.mean(signed_vals)) if signed_vals else weighted
    pred = 0.5 * weighted + 0.5 * signed
    return pred, {
        "n_neighbors": int(len(vals)),
        "n_signed_neighbors": int(n_signed),
        "coverage": 1.0,
        "weighted_neighbor_mean": weighted,
        "signed_neighbor_mean": signed,
    }


def evaluate_group(all_rows: list[dict[str, Any]], graph: dict[str, dict[str, list[tuple[str, int, float]]]], group: str, *, mode: str) -> dict[str, Any]:
    rows = [dict(row) for row in all_rows if row.get("group") == group]
    scored = []
    pred = np.full(len(rows), np.nan, dtype=float)
    y = np.asarray([as_float(row["anchor_minus_gene_raw_mean"]) for row in rows], dtype=float)
    folds = []
    for ds in sorted({str(row["dataset"]) for row in rows}):
        train = [row for row in rows if str(row["dataset"]) != ds]
        test_idx = [i for i, row in enumerate(rows) if str(row["dataset"]) == ds]
        train_by_gene: dict[str, list[float]] = defaultdict(list)
        for row in train:
            val = as_float(row.get("anchor_minus_gene_raw_mean"))
            if np.isfinite(val):
                train_by_gene[str(row["gene"]).upper()].append(float(val))
        fold_cov = 0
        for i in test_idx:
            row = rows[i]
            p, meta = aggregate_neighbor_response(str(row["gene"]), train_by_gene, graph, mode)
            if not np.isfinite(p):
                # Conservative fallback: if no regulatory response evidence is
                # available, stay with the train-only gene baseline.
                p = -1e-9
            else:
                fold_cov += 1
            pred[i] = p
            item = dict(row)
            item["response_module_pred_anchor_minus_gene"] = float(p)
            item["response_module_route_anchor"] = bool(p > 0.0)
            item["response_module_routed_xverse_or_gene"] = (
                as_float(row["anchor_pearson_pert"]) if p > 0.0 else as_float(row["gene_raw_mean"])
            )
            item.update({f"response_module_{k}": v for k, v in meta.items()})
            scored.append(item)
        folds.append({"dataset": ds, "n_train": len(train), "n_test": len(test_idx), "n_neighbor_covered": int(fold_cov)})
    paired = [
        paired_bootstrap(scored, "response_module_routed_xverse_or_gene", baseline, seed=SEED + i)
        for i, baseline in enumerate(("gene_raw_mean", "dataset_mean", "global_mean", "anchor_pearson_pert"))
    ]
    return {
        "group": group,
        "mode": mode,
        "coverage_fraction": float(np.mean([np.isfinite(v) and v != -1e-9 for v in pred])) if len(pred) else 0.0,
        "lodo": {"folds": folds, "spearman": spearman(y, pred), "r2": r2_score(y, pred)},
        "paired_deltas": paired,
        "scored_rows": scored,
    }


def paired_row(result: dict[str, Any], baseline: str) -> dict[str, Any]:
    return next(row for row in result["paired_deltas"] if row["baseline"] == baseline)


def choose_mode(results_by_mode: dict[str, list[dict[str, Any]]]) -> str:
    # Fixed train-only choice: prefer the mode with the best worst-group
    # internal LODO delta vs gene, using only proxy folds.
    ranked = []
    for mode, results in results_by_mode.items():
        deltas = [paired_row(result, "gene_raw_mean")["delta_mean"] for result in results]
        harms = [paired_row(result, "gene_raw_mean")["p_harm"] for result in results]
        ranked.append((float(min(deltas)), float(-max(harms)), float(np.mean(deltas)), mode))
    ranked.sort(reverse=True)
    return ranked[0][3]


def decide(results: list[dict[str, Any]], shuffled_results: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    for result, shuf in zip(results, shuffled_results, strict=True):
        group = result["group"]
        if float(result["coverage_fraction"]) < 0.30:
            reasons.append(f"{group}_neighbor_response_coverage_below_0p30")
        if (result["lodo"].get("spearman") or 0.0) < 0.10 and (result["lodo"].get("r2") or -999.0) < 0.00:
            reasons.append(f"{group}_predictive_signal_below_gate")
        vs_gene = paired_row(result, "gene_raw_mean")
        vs_shuf = paired_bootstrap(
            [
                {
                    **row,
                    "shuffled_response_module_routed_xverse_or_gene": shuf["scored_rows"][i]["response_module_routed_xverse_or_gene"],
                }
                for i, row in enumerate(result["scored_rows"])
            ],
            "response_module_routed_xverse_or_gene",
            "shuffled_response_module_routed_xverse_or_gene",
            seed=SEED + 400,
        )
        if float(vs_gene["delta_mean"]) < 0.02:
            reasons.append(f"{group}_delta_vs_gene_below_0p02")
        if float(vs_gene["p_harm"]) > 0.20:
            reasons.append(f"{group}_harm_vs_gene_above_0p20")
        if float(vs_gene["dataset_min"]) < -0.02:
            reasons.append(f"{group}_dataset_min_vs_gene_below_minus_0p02")
        if float(vs_shuf["delta_mean"]) < 0.02:
            reasons.append(f"{group}_shuffled_graph_control_not_separated")
    status = "tracka_omnipath_response_module_gate_pass_code_gate_next_no_gpu" if not reasons else "tracka_omnipath_response_module_gate_fail_no_gpu"
    return {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "code_gate_only_if_pass_else_none",
        "reasons": reasons,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track A OmniPath Response-Module Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- Uses only xverse train-only/internal proxy rows and frozen OmniPath TF-target edges.",
        "- Leave-one-dataset-out folds estimate regulatory-neighbor response from other datasets only.",
        "- Does not read canonical test, canonical multi, held-out query, active logs, or GPU artifacts.",
        "- Passing would authorize only a later code/provenance gate, not immediate GPU.",
        "",
        "## Prior",
        "",
        f"- edges TSV: `{payload['prior']['edges_tsv']}`",
        f"- raw TSV SHA256: `{payload['prior']['raw_tsv_sha256']}`",
        f"- edge TSV SHA256: `{payload['prior']['edge_tsv_sha256']}`",
        f"- selected mode: `{payload['selected_mode']}`",
        "",
        "## Group Results",
        "",
        "| group | coverage | LODO Spearman | LODO R2 | delta vs gene | p harm | dataset min | delta vs anchor |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        vs_gene = paired_row(result, "gene_raw_mean")
        vs_anchor = paired_row(result, "anchor_pearson_pert")
        lines.append(
            f"| {result['group']} | {result['coverage_fraction']:.3f} | {fmt(result['lodo'].get('spearman'))} | "
            f"{fmt(result['lodo'].get('r2'))} | {fmt(vs_gene['delta_mean'])} | {fmt(vs_gene['p_harm'])} | "
            f"{fmt(vs_gene['dataset_min'])} | {fmt(vs_anchor['delta_mean'])} |"
        )
    lines.extend(["", "## Shuffled-Graph Control", "", "| group | coverage | delta vs gene | p harm |", "|---|---:|---:|---:|"])
    for result in payload["shuffled_results"]:
        vs_gene = paired_row(result, "gene_raw_mean")
        lines.append(
            f"| {result['group']} | {result['coverage_fraction']:.3f} | {fmt(vs_gene['delta_mean'])} | {fmt(vs_gene['p_harm'])} |"
        )
    lines.extend(["", "## Gate Reasons", ""])
    reasons = payload["decision"].get("reasons") or []
    lines.extend(f"- `{reason}`" for reason in reasons) if reasons else lines.append("- none")
    lines.extend(["", "## Decision", "", payload["decision_text"], ""])
    return "\n".join(lines)


def main() -> int:
    prior_summary = load_json(PRIOR_SUMMARY)
    rows_payload = load_json(XVERSE_ROWS)
    all_rows = rows_payload["condition_rows"]
    edges = load_edges(PRIOR_EDGES)
    row_genes = [str(row["gene"]).upper() for row in all_rows]
    graph = graph_maps(edges)
    modes = ("upstream", "downstream", "undirected")
    results_by_mode = {mode: [evaluate_group(all_rows, graph, group, mode=mode) for group in GROUPS] for mode in modes}
    selected_mode = choose_mode(results_by_mode)
    results = results_by_mode[selected_mode]
    shuf_graph = graph_maps(shuffled_edges(edges, row_genes, SEED + 101))
    shuffled_results = [evaluate_group(all_rows, shuf_graph, group, mode=selected_mode) for group in GROUPS]
    decision = decide(results, shuffled_results)
    payload = {
        "status": decision["status"],
        "timestamp": "2026-06-23 15:00 CST",
        "boundary": {
            "query_free": True,
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "heldout_query_read": False,
            "active_log_read": False,
            "gpu_authorization": "none",
        },
        "prior": {
            "edges_tsv": str(PRIOR_EDGES),
            "raw_tsv_sha256": prior_summary["hashes"]["raw_tsv"],
            "edge_tsv_sha256": prior_summary["hashes"]["edges_tsv"],
            "deduplicated_edges": prior_summary["deduplicated_edges"],
        },
        "xverse_rows": str(XVERSE_ROWS),
        "selected_mode": selected_mode,
        "results_by_mode_summary": {
            mode: {
                result["group"]: {
                    "coverage_fraction": result["coverage_fraction"],
                    "delta_vs_gene": paired_row(result, "gene_raw_mean")["delta_mean"],
                    "p_harm_vs_gene": paired_row(result, "gene_raw_mean")["p_harm"],
                    "spearman": result["lodo"].get("spearman"),
                    "r2": result["lodo"].get("r2"),
                }
                for result in results
            }
            for mode, results in results_by_mode.items()
        },
        "results": results,
        "shuffled_results": shuffled_results,
        "decision": decision,
        "decision_text": (
            "The regulatory-neighbor response module is not promoted to GPU. "
            "It is a materially different OmniPath use than static degree/sign routing, "
            "but it must improve both Track A proxy groups with low harm and separate "
            "from a shuffled graph control before any model work."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "gpu_authorization": "none", "selected_mode": selected_mode, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
