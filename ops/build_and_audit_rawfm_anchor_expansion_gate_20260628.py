#!/usr/bin/env python3
"""Build and audit RawFM anchor-expansion gene-budget manifests.

This gate is intentionally conservative: it only prepares response-neutral
abundance/random ballast budgets. GPU promotion depends on final fixed-step
smokes, not on this CPU score table.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
if str(ROOT / "ops") not in sys.path:
    sys.path.insert(0, str(ROOT / "ops"))

import audit_rawfm_structural_gene_budget_gate_20260628 as structural_gate  # noqa: E402


OUT_DIR = ROOT / "reports/rawfm_anchor_expansion_gate_20260628"
MANIFEST_DIR = ROOT / "reports/rawfm_anchor_expansion_manifest_20260628"
DATASET = "Wessels"
K_VALUES = (512, 1024)
SEED = 42


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def topk(score: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), int(score.shape[0])))
    idx = np.argpartition(-score, kth=k - 1)[:k]
    return idx[np.argsort(-score[idx])].astype(int)


def random_k(n_genes: int, k: int, rng: np.random.Generator) -> np.ndarray:
    k = max(1, min(int(k), int(n_genes)))
    return np.sort(rng.choice(n_genes, size=k, replace=False)).astype(int)


def abundance_matched_random(
    selected_keep: np.ndarray,
    abundance_score: np.ndarray,
    rng: np.random.Generator,
    bins: int = 20,
) -> np.ndarray:
    n_genes = int(abundance_score.shape[0])
    if n_genes <= len(selected_keep):
        return np.arange(n_genes, dtype=int)
    q = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    edges = np.quantile(abundance_score, q)
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_id = np.digitize(abundance_score, edges[1:-1], right=True)
    selected_bins = bin_id[np.asarray(selected_keep, dtype=int)]
    selected_set = set(map(int, selected_keep))
    chosen: list[int] = []
    used: set[int] = set()
    for bin_value in sorted(set(map(int, selected_bins))):
        need = int(np.sum(selected_bins == bin_value))
        pool = np.where(bin_id == bin_value)[0]
        pool = np.asarray(
            [int(i) for i in pool if int(i) not in selected_set and int(i) not in used],
            dtype=int,
        )
        if pool.size < need:
            pool = np.asarray([int(i) for i in np.where(bin_id == bin_value)[0] if int(i) not in used], dtype=int)
        if pool.size == 0:
            continue
        take = rng.choice(pool, size=min(need, pool.size), replace=False)
        chosen.extend(map(int, take))
        used.update(map(int, take))
    if len(chosen) < len(selected_keep):
        remaining = np.asarray([i for i in range(n_genes) if i not in used and i not in selected_set], dtype=int)
        if remaining.size < len(selected_keep) - len(chosen):
            remaining = np.asarray([i for i in range(n_genes) if i not in used], dtype=int)
        extra = rng.choice(remaining, size=len(selected_keep) - len(chosen), replace=False)
        chosen.extend(map(int, extra))
    return np.asarray(sorted(chosen[: len(selected_keep)]), dtype=int)


def write_manifest(path: Path, label: str, keep: np.ndarray, n_genes: int, source_detail: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "version": 1,
        "label": label,
        "source": {
            "split_file": str(structural_gate.DEFAULT_SPLIT),
            "dataset": DATASET,
            "seed": SEED,
            "k": int(len(keep)),
            "train_only": True,
            "source_detail": source_detail,
        },
        "datasets": {
            DATASET: {
                "keep_indices": [int(x) for x in keep],
                "n_genes": int(n_genes),
                "n_train_conditions": 8,
            }
        },
    }
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metric_row(label: str, keep: np.ndarray, scores: dict[str, Any], refs: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "label": label,
        "n_genes": int(len(keep)),
        "response_mean": float(np.mean(scores["response"][keep])),
        "residual_mean": float(np.mean(scores["residual"][keep])),
        "diversity_mean": float(np.mean(scores["diversity"][keep])),
        "abundance_mean": float(np.mean(scores["abundance"][keep])),
        "variance_mean": float(np.mean(scores["variance"][keep])),
        "detection_mean": float(np.mean(scores["detection"][keep])),
        "overlap_k256_random": structural_gate.overlap(keep, refs["k256_random"]),
        "overlap_k256_residual": structural_gate.overlap(keep, refs["k256_residual"]),
        "overlap_same_k_response": structural_gate.overlap(keep, refs["same_k_response"]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    scores = structural_gate.load_scores(
        DATASET,
        structural_gate.DEFAULT_SPLIT,
        structural_gate.DEFAULT_BIFLOW,
    )
    n_genes = int(scores["response"].shape[0])
    k256_random = structural_gate.read_keep(
        ROOT / "reports/rawfm_structural_gene_budget_manifest_20260628/random_gene_set_k256_seed42.json",
        DATASET,
    )
    k256_residual = structural_gate.read_keep(
        ROOT / "reports/rawfm_structural_gene_budget_manifest_20260628/response_abundance_residual_topk_k256_seed42.json",
        DATASET,
    )

    manifests: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    loader_rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        rng = np.random.default_rng(SEED + k)
        response_keep = topk(scores["response"], k)
        refs = {
            "k256_random": k256_random,
            "k256_residual": k256_residual,
            "same_k_response": response_keep,
        }
        specs = [
            (
                f"abundance_matched_random_k{k}_seed{SEED}",
                abundance_matched_random(response_keep, scores["abundance"], rng, bins=20),
                "abundance-matched random control to same-k response top genes",
            ),
            (
                f"random_gene_set_k{k}_seed{SEED}",
                random_k(n_genes, k, rng),
                "uniform random response-neutral ballast genes",
            ),
        ]
        for label, keep, source_detail in specs:
            path = MANIFEST_DIR / f"{label}.json"
            write_manifest(path, label, keep, n_genes, source_detail)
            manifests.append({"label": label, "manifest": str(path), "k": k})
            metric_rows.append(metric_row(label, keep, scores, refs))
            checks = structural_gate.loader_dryrun(
                path,
                label,
                DATASET,
                structural_gate.DEFAULT_SPLIT,
                structural_gate.DEFAULT_BIFLOW,
            )
            loader_rows.append({"label": label, "manifest": str(path), **checks})

    metric_df = pd.DataFrame(metric_rows)
    loader_df = pd.DataFrame(loader_rows)
    manifest_df = pd.DataFrame(manifests)
    loader_gate = bool(loader_df["pass"].all())
    response_overlap_gate = bool((metric_df["overlap_same_k_response"] <= 0.15).all())
    gpu_packet_authorized = bool(loader_gate and response_overlap_gate)
    status = "rawfm_anchor_expansion_gate_gpu_packet_ready" if gpu_packet_authorized else "rawfm_anchor_expansion_gate_fail_no_gpu"

    metric_csv = OUT_DIR / "rawfm_anchor_expansion_metric_rows.csv"
    loader_csv = OUT_DIR / "rawfm_anchor_expansion_loader_rows.csv"
    manifest_csv = OUT_DIR / "rawfm_anchor_expansion_manifest_rows.csv"
    metric_df.to_csv(metric_csv, index=False)
    loader_df.to_csv(loader_csv, index=False)
    manifest_df.to_csv(manifest_csv, index=False)
    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_packet_authorized": gpu_packet_authorized,
        "loader_gate": loader_gate,
        "response_overlap_gate": response_overlap_gate,
        "manifest_dir": str(MANIFEST_DIR),
        "metric_csv": str(metric_csv),
        "loader_csv": str(loader_csv),
        "manifest_csv": str(manifest_csv),
    }
    json_path = OUT_DIR / "rawfm_anchor_expansion_gate_20260628.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# RawFM Anchor Expansion Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        f"GPU packet authorized: `{gpu_packet_authorized}`",
        "",
        "## Boundary",
        "",
        "- CPU-only manifest and loader audit for Wessels k=512/k=1024 anchor expansion.",
        "- Candidate budgets are response-neutral random or abundance-matched random ballast.",
        "- No training, no inference, no GPU, no canonical multi, no Track C query.",
        "",
        "## Metric Rows",
        "",
        "| label | k | residual | response | abundance | overlap k256 random | overlap k256 residual | overlap same-k response |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metric_df.iterrows():
        lines.append(
            f"| {row['label']} | {int(row['n_genes'])} | {fmt(row['residual_mean'])} | "
            f"{fmt(row['response_mean'])} | {fmt(row['abundance_mean'])} | "
            f"{fmt(row['overlap_k256_random'])} | {fmt(row['overlap_k256_residual'])} | "
            f"{fmt(row['overlap_same_k_response'])} |"
        )
    lines.extend(
        [
            "",
            "## Loader Checks",
            "",
            "| label | kept | pass |",
            "|---|---:|---:|",
        ]
    )
    for _, row in loader_df.iterrows():
        lines.append(f"| {row['label']} | {int(row['kept_genes'])} | {bool(row['pass'])} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- loader gate: `{loader_gate}`",
            f"- response-overlap gate: `{response_overlap_gate}`",
        ]
    )
    if gpu_packet_authorized:
        lines.append(
            "- GPU packet: launch fixed-step/no-selection MMD-off smokes for all four anchor manifests when the active MMD/LR branch has freed GPUs or failed its gate."
        )
    else:
        lines.append("- No GPU launch: loader or response-overlap gate failed.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- manifests: `{manifest_csv}`",
            f"- metrics: `{metric_csv}`",
            f"- loader: `{loader_csv}`",
            f"- JSON: `{json_path}`",
            "",
        ]
    )
    (OUT_DIR / "LATENTFM_RAWFM_ANCHOR_EXPANSION_GATE_20260628.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
