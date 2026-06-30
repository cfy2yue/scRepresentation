#!/usr/bin/env python3
"""Track A residual-forensics map for xverse anchor internal-val failures.

This CPU audit joins the frozen anchor internal-val eval with train-only
baseline predictions and target residual geometry. It asks whether anchor
failures are explained by a non-closed covariate, rather than the already
closed gene/dataset shrink/router family.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
GENE_GATE_SCRIPT = ROOT / "ops/audit_latentfm_xverse_gene_reliability_router_gate_20260622.py"
DEFAULT_ANCHOR_JSON = ROOT / "reports/latentfm_xverse_tracka_anchor_internal_val_error_map_20260622.json"
DEFAULT_BASELINE_JSON = ROOT / "reports/latentfm_xverse_gene_reliability_router_gate_20260622.json"
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
DEFAULT_PERT_MEANS = (
    ROOT
    / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/"
    "xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz"
)
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_tracka_residual_forensics_20260622.json"
DEFAULT_OUT_CSV = ROOT / "reports/latentfm_xverse_tracka_residual_forensics_conditions_20260622.csv"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_TRACKA_RESIDUAL_FORENSICS_20260622.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
COVARIATES = (
    "target_residual_norm",
    "gene_pred_norm",
    "dataset_pred_norm",
    "global_pred_norm",
    "gene_dataset_cosine",
    "gene_target_cosine",
    "dataset_target_cosine",
    "gene_minus_dataset_score",
    "gene_train_count",
    "anchor_mmd_clamped",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_gene_module() -> Any:
    spec = importlib.util.spec_from_file_location("gene_reliability_gate", GENE_GATE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {GENE_GATE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 3 or x.size != y.size:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return None
    return float(np.dot(x, y) / denom)


def cosine(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return None
    return float(np.dot(x, y) / denom)


def rankdata(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float64)
    i = 0
    while i < arr.size:
        j = i + 1
        while j < arr.size and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 5:
        return None
    rx = rankdata([p[0] for p in pairs])
    ry = rankdata([p[1] for p in pairs])
    return pearson(rx, ry)


def linear_r2(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 5:
        return None
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return None if ss_tot <= 1e-12 else float(max(0.0, 1.0 - ss_res / ss_tot))


def equal_dataset_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = row.get(key)
        if val is not None:
            by_ds[str(row["dataset"])].append(float(val))
    vals = [float(np.mean(v)) for v in by_ds.values() if v]
    return None if not vals else float(np.mean(vals))


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    gene = load_gene_module()
    anchor = load_json(args.anchor_json)
    data_dir = args.data_dir.resolve()
    split = gene.load_json(args.split_file)
    manifest = gene.load_json(data_dir / "manifest.json")
    metadata = gene.load_json(Path(manifest["condition_metadata_file"]))
    train_rows, val_rows = gene.collect_rows(
        data_dir,
        split,
        metadata,
        max_train_per_dataset=args.max_train_per_dataset,
        max_cells=args.max_cells_per_condition,
    )
    sums = gene.build_sums(train_rows)
    val_by_key = {
        (str(row["group"]), str(row["dataset"]), str(row["condition"])): row
        for row in val_rows
    }
    anchor_by_key = {
        (str(row["group"]), str(row["dataset"]), str(row["condition"])): row
        for row in anchor.get("condition_rows", [])
    }
    out = []
    for key, row in sorted(val_by_key.items()):
        anchor_row = anchor_by_key.get(key)
        if anchor_row is None:
            continue
        dataset_mean, gene_mean, global_mean, gene_count = gene.component_means(sums, row, exclude_row=False)
        target = np.asarray(row["residual"], dtype=np.float32)
        item = {
            "group": key[0],
            "dataset": key[1],
            "condition": key[2],
            "gene": str(row["gene"]),
            "gene_train_count": int(gene_count),
            "anchor_pearson_pert": anchor_row.get("anchor_pearson_pert"),
            "anchor_mmd_clamped": anchor_row.get("anchor_mmd_clamped"),
            "gene_raw_mean": anchor_row.get("gene_raw_mean"),
            "dataset_mean": anchor_row.get("dataset_mean"),
            "global_mean": anchor_row.get("global_mean"),
            "shrink_k8": anchor_row.get("shrink_k8"),
            "anchor_minus_gene_raw_mean": None,
            "anchor_minus_dataset_mean": None,
            "gene_minus_dataset_score": None,
            "target_residual_norm": float(np.linalg.norm(target)),
            "gene_pred_norm": float(np.linalg.norm(gene_mean)),
            "dataset_pred_norm": float(np.linalg.norm(dataset_mean)),
            "global_pred_norm": float(np.linalg.norm(global_mean)),
            "gene_dataset_cosine": cosine(gene_mean, dataset_mean),
            "gene_target_cosine": cosine(gene_mean, target),
            "dataset_target_cosine": cosine(dataset_mean, target),
        }
        if item["anchor_pearson_pert"] is not None and item["gene_raw_mean"] is not None:
            item["anchor_minus_gene_raw_mean"] = float(item["anchor_pearson_pert"]) - float(item["gene_raw_mean"])
        if item["anchor_pearson_pert"] is not None and item["dataset_mean"] is not None:
            item["anchor_minus_dataset_mean"] = float(item["anchor_pearson_pert"]) - float(item["dataset_mean"])
        if item["gene_raw_mean"] is not None and item["dataset_mean"] is not None:
            item["gene_minus_dataset_score"] = float(item["gene_raw_mean"]) - float(item["dataset_mean"])
        out.append(item)
    return out


def covariate_tests(rows: list[dict[str, Any]], target: str) -> list[dict[str, Any]]:
    out = []
    for group in GROUPS:
        group_rows = [r for r in rows if r["group"] == group and r.get(target) is not None]
        datasets = sorted({r["dataset"] for r in group_rows})
        for cov in COVARIATES:
            xs = [r.get(cov) for r in group_rows]
            ys = [r.get(target) for r in group_rows]
            rho = spearman(xs, ys)
            r2 = linear_r2(xs, ys)
            loo = []
            for ds in datasets:
                sub = [r for r in group_rows if r["dataset"] != ds]
                val = spearman([r.get(cov) for r in sub], [r.get(target) for r in sub])
                if val is not None:
                    loo.append(float(val))
            out.append(
                {
                    "group": group,
                    "target": target,
                    "covariate": cov,
                    "n_conditions": len(group_rows),
                    "n_datasets": len(datasets),
                    "spearman": rho,
                    "linear_r2": r2,
                    "loo_abs_spearman_min": None if not loo else float(min(abs(v) for v in loo)),
                    "loo_spearman_min": None if not loo else float(min(loo)),
                    "loo_spearman_max": None if not loo else float(max(loo)),
                    "closed_family": cov in {"gene_train_count", "gene_minus_dataset_score"},
                }
            )
    return out


def dataset_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for group in GROUPS:
        for ds in sorted({r["dataset"] for r in rows if r["group"] == group}):
            sub = [r for r in rows if r["group"] == group and r["dataset"] == ds]
            out.append(
                {
                    "group": group,
                    "dataset": ds,
                    "n_conditions": len(sub),
                    "anchor_minus_gene_raw_mean": float(np.mean([r["anchor_minus_gene_raw_mean"] for r in sub])),
                    "anchor_minus_dataset_mean": float(np.mean([r["anchor_minus_dataset_mean"] for r in sub])),
                    "target_residual_norm": float(np.mean([r["target_residual_norm"] for r in sub])),
                    "gene_target_cosine": float(np.mean([r["gene_target_cosine"] for r in sub if r["gene_target_cosine"] is not None])),
                    "dataset_target_cosine": float(np.mean([r["dataset_target_cosine"] for r in sub if r["dataset_target_cosine"] is not None])),
                }
            )
    return out


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    candidates = []
    for row in payload["covariate_tests"]:
        if row["target"] != "anchor_minus_gene_raw_mean":
            continue
        if row.get("closed_family"):
            continue
        rho = row.get("spearman")
        r2 = row.get("linear_r2")
        loo = row.get("loo_abs_spearman_min")
        if rho is None or r2 is None or loo is None:
            continue
        if abs(float(rho)) >= 0.35 or float(r2) >= 0.15:
            if float(loo) >= 0.20:
                candidates.append(row)
    groups_with_candidates = sorted({r["group"] for r in candidates})
    if set(groups_with_candidates) != set(GROUPS):
        reasons.append("no_nonclosed_covariate_stable_in_both_tracka_proxy_groups")
    if not candidates:
        reasons.append("residual_forensics_explained_by_closed_or_unstable_covariates_only")
    status = "cpu_forensics_candidate_for_followup_gate" if not reasons else "cpu_forensics_no_gpu_mechanism"
    action = "design_cpu_predictor_gate_before_gpu" if not reasons else "do_not_launch_gpu_from_residual_forensics"
    return {
        "status": status,
        "action": action,
        "reasons": reasons,
        "candidate_covariates": candidates,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Track A Residual Forensics",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
        "",
        "## Provenance",
        "",
        f"- anchor map JSON: `{payload['anchor_json']}`",
        f"- baseline gate JSON: `{payload['baseline_json']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- data_dir: `{payload['data_dir']}`",
        f"- condition CSV: `{payload['condition_csv']}`",
        f"- leakage status: `{payload['leakage_status']}`",
        f"- condition rows: `{payload['n_rows']}`",
        "",
        "## Covariate Tests",
        "",
        "| group | target | covariate | n | Spearman | R2 | LOO abs rho min | closed family |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in sorted(payload["covariate_tests"], key=lambda r: (r["group"], r["target"], -(abs(r["spearman"]) if r.get("spearman") is not None else -1))):
        lines.append(
            f"| {row['group']} | `{row['target']}` | `{row['covariate']}` | {row['n_conditions']} | "
            f"{fmt(row.get('spearman'))} | {fmt(row.get('linear_r2'))} | "
            f"{fmt(row.get('loo_abs_spearman_min'))} | `{row.get('closed_family')}` |"
        )
    lines += [
        "",
        "## Dataset Summary",
        "",
        "| group | dataset | n | anchor-gene | anchor-dataset | target norm | gene-target cos | dataset-target cos |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["dataset_summary"]:
        lines.append(
            f"| {row['group']} | {row['dataset']} | {row['n_conditions']} | "
            f"{fmt(row['anchor_minus_gene_raw_mean'])} | {fmt(row['anchor_minus_dataset_mean'])} | "
            f"{fmt(row['target_residual_norm'])} | {fmt(row['gene_target_cosine'])} | "
            f"{fmt(row['dataset_target_cosine'])} |"
        )
    lines += ["", "## Gate Reasons", ""]
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{r}`" for r in reasons] if reasons else ["- none"])
    lines += [
        "",
        "## Interpretation",
        "",
        "- Passing this audit does not authorize GPU; it only authorizes a separate CPU predictor gate.",
        "- Closed covariate families include gene count and gene/dataset score-winner effects.",
        "- No canonical test, canonical multi, or Track C query evidence is used.",
    ]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "group", "dataset", "condition", "gene", "gene_train_count",
        "anchor_pearson_pert", "anchor_mmd_clamped", "gene_raw_mean", "dataset_mean",
        "global_mean", "shrink_k8", "anchor_minus_gene_raw_mean",
        "anchor_minus_dataset_mean", "gene_minus_dataset_score", "target_residual_norm",
        "gene_pred_norm", "dataset_pred_norm", "global_pred_norm", "gene_dataset_cosine",
        "gene_target_cosine", "dataset_target_cosine",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-json", type=Path, default=DEFAULT_ANCHOR_JSON)
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--pert-means-file", type=Path, default=DEFAULT_PERT_MEANS)
    parser.add_argument("--max-train-per-dataset", type=int, default=768)
    parser.add_argument("--max-cells-per-condition", type=int, default=256)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    rows = build_rows(args)
    payload = {
        "anchor_json": str(args.anchor_json),
        "baseline_json": str(args.baseline_json),
        "data_dir": str(args.data_dir),
        "split_file": str(args.split_file),
        "pert_means_file": str(args.pert_means_file),
        "condition_csv": str(args.out_csv),
        "leakage_status": "train_only_internal_val_anchor_eval_plus_train_only_residual_geometry_no_canonical_no_query",
        "n_rows": len(rows),
        "condition_rows": rows,
        "dataset_summary": dataset_summary(rows),
        "covariate_tests": covariate_tests(rows, "anchor_minus_gene_raw_mean") + covariate_tests(rows, "anchor_minus_dataset_mean"),
    }
    payload["decision"] = decide(payload)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_csv, rows)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
