#!/usr/bin/env python3
"""Build and audit hybrid residual+abundance RawFM gene-budget manifests."""

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


STRUCTURAL_MANIFEST_DIR = ROOT / "reports/rawfm_structural_gene_budget_manifest_20260628"
OUT_DIR = ROOT / "reports/rawfm_hybrid_gene_budget_gate_20260628"
MANIFEST_DIR = ROOT / "reports/rawfm_hybrid_gene_budget_manifest_20260628"
DATASET = "Wessels"
K = 256
HALF = 128


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


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def keep_from(label: str) -> np.ndarray:
    path = STRUCTURAL_MANIFEST_DIR / f"{label}_k256_seed42.json"
    obj = load_manifest(path)
    return np.asarray(obj["datasets"][DATASET]["keep_indices"], dtype=int)


def hybrid_keep(primary: np.ndarray, anchor: np.ndarray, k: int, n_genes: int) -> np.ndarray:
    chosen: list[int] = []
    seen: set[int] = set()
    for arr, limit in [(primary, HALF), (anchor, k)]:
        taken = 0
        for value in arr:
            idx = int(value)
            if idx in seen:
                continue
            chosen.append(idx)
            seen.add(idx)
            taken += 1
            if arr is primary and taken >= limit:
                break
            if len(chosen) >= k:
                break
        if len(chosen) >= k:
            break
    if len(chosen) < k:
        for idx in range(n_genes):
            if idx in seen:
                continue
            chosen.append(idx)
            seen.add(idx)
            if len(chosen) >= k:
                break
    return np.asarray(chosen[:k], dtype=int)


def write_manifest(path: Path, label: str, keep: np.ndarray, n_genes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "version": 1,
        "label": label,
        "source": {
            "split_file": str(structural_gate.DEFAULT_SPLIT),
            "source_manifest_dir": str(STRUCTURAL_MANIFEST_DIR),
            "dataset": DATASET,
            "k": K,
            "residual_or_control_half": HALF,
            "abundance_anchor": K - HALF,
            "train_only": True,
        },
        "datasets": {
            DATASET: {
                "keep_indices": keep.tolist(),
                "n_genes": int(n_genes),
                "n_train_conditions": 8,
            }
        },
    }
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metric_row(label: str, keep: np.ndarray, scores: dict[str, Any], reference: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "label": label,
        "n_genes": int(len(keep)),
        "response_mean": float(np.mean(scores["response"][keep])),
        "residual_mean": float(np.mean(scores["residual"][keep])),
        "diversity_mean": float(np.mean(scores["diversity"][keep])),
        "abundance_mean": float(np.mean(scores["abundance"][keep])),
        "variance_mean": float(np.mean(scores["variance"][keep])),
        "detection_mean": float(np.mean(scores["detection"][keep])),
        "overlap_residual_topk": structural_gate.overlap(keep, reference["residual"]),
        "overlap_abundance_topk": structural_gate.overlap(keep, reference["abundance"]),
        "overlap_control_hybrid": structural_gate.overlap(keep, reference["control_hybrid"]),
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
    residual = keep_from("response_abundance_residual_topk")
    abundance = keep_from("abundance_topk")
    residual_control = keep_from("residual_confound_matched_random")

    candidate = hybrid_keep(residual, abundance, K, n_genes)
    control = hybrid_keep(residual_control, abundance, K, n_genes)
    candidate_label = "residual128_abundance128_hybrid"
    control_label = "confound128_abundance128_hybrid_control"
    candidate_manifest = MANIFEST_DIR / f"{candidate_label}_k256_seed42.json"
    control_manifest = MANIFEST_DIR / f"{control_label}_k256_seed42.json"
    write_manifest(candidate_manifest, f"{candidate_label}_k256_seed42", candidate, n_genes)
    write_manifest(control_manifest, f"{control_label}_k256_seed42", control, n_genes)

    refs = {"residual": residual, "abundance": abundance, "control_hybrid": control}
    metric_df = pd.DataFrame(
        [
            metric_row(candidate_label, candidate, scores, refs),
            metric_row(control_label, control, scores, refs),
            metric_row("residual_topk_full", residual, scores, refs),
            metric_row("abundance_topk_full", abundance, scores, refs),
        ]
    )
    confound_df = pd.DataFrame(
        [
            {
                "comparison": "candidate_vs_control_hybrid",
                "residual_delta": float(np.mean(scores["residual"][candidate]) - np.mean(scores["residual"][control])),
                "response_delta": float(np.mean(scores["response"][candidate]) - np.mean(scores["response"][control])),
                "abundance_smd": structural_gate.smd(scores["abundance"][candidate], scores["abundance"][control]),
                "variance_smd": structural_gate.smd(scores["variance"][candidate], scores["variance"][control]),
                "detection_smd": structural_gate.smd(scores["detection"][candidate], scores["detection"][control]),
            }
        ]
    )
    loader_rows = []
    for label, path in [(candidate_label, candidate_manifest), (control_label, control_manifest)]:
        checks = structural_gate.loader_dryrun(
            path,
            label,
            DATASET,
            structural_gate.DEFAULT_SPLIT,
            structural_gate.DEFAULT_BIFLOW,
        )
        loader_rows.append({"label": label, "manifest": str(path), **checks})
    loader_df = pd.DataFrame(loader_rows)

    row = confound_df.iloc[0]
    gate = bool(
        float(row["residual_delta"]) > 0.25
        and abs(float(row["abundance_smd"])) <= 0.75
        and abs(float(row["variance_smd"])) <= 0.75
        and abs(float(row["detection_smd"])) <= 0.75
        and bool(loader_df["pass"].all())
    )
    status = "rawfm_hybrid_gene_budget_gate_gpu_packet_ready" if gate else "rawfm_hybrid_gene_budget_gate_fail_no_gpu"

    metric_csv = OUT_DIR / "rawfm_hybrid_gene_budget_metric_rows.csv"
    confound_csv = OUT_DIR / "rawfm_hybrid_gene_budget_confound_rows.csv"
    loader_csv = OUT_DIR / "rawfm_hybrid_gene_budget_loader_rows.csv"
    metric_df.to_csv(metric_csv, index=False)
    confound_df.to_csv(confound_csv, index=False)
    loader_df.to_csv(loader_csv, index=False)
    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_packet_authorized": gate,
        "candidate_manifest": str(candidate_manifest),
        "control_manifest": str(control_manifest),
        "metric_csv": str(metric_csv),
        "confound_csv": str(confound_csv),
        "loader_csv": str(loader_csv),
        "residual_delta": float(row["residual_delta"]),
        "abundance_smd": float(row["abundance_smd"]),
        "variance_smd": float(row["variance_smd"]),
        "detection_smd": float(row["detection_smd"]),
    }
    json_path = OUT_DIR / "rawfm_hybrid_gene_budget_gate_20260628.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# RawFM Hybrid Gene-Budget Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        f"GPU packet authorized: `{gate}`",
        "",
        "## Boundary",
        "",
        "- CPU-only manifest and loader gate.",
        "- Candidate: 128 residualized-response genes plus 128 abundance-anchor genes.",
        "- Matched control: 128 residual-confound-random genes plus the same abundance-anchor design.",
        "- No training, no inference, no GPU, no checkpoint selection.",
        "",
        "## Metrics",
        "",
        "| label | residual | response | abundance | overlap residual | overlap abundance | overlap control |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, m in metric_df.iterrows():
        lines.append(
            f"| {m['label']} | {fmt(m['residual_mean'])} | {fmt(m['response_mean'])} | "
            f"{fmt(m['abundance_mean'])} | {fmt(m['overlap_residual_topk'])} | "
            f"{fmt(m['overlap_abundance_topk'])} | {fmt(m['overlap_control_hybrid'])} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- residual delta candidate-control: `{fmt(payload['residual_delta'])}`",
            f"- abundance/variance/detection SMD: `{fmt(payload['abundance_smd'])}` / `{fmt(payload['variance_smd'])}` / `{fmt(payload['detection_smd'])}`",
            f"- loader pass: `{bool(loader_df['pass'].all())}`",
        ]
    )
    if gate:
        lines.append("- Decision: launch a two-run fixed-step/no-selection GPU smoke, candidate versus matched hybrid control.")
    else:
        lines.append("- Decision: no GPU launch from this hybrid.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- candidate manifest: `{candidate_manifest}`",
            f"- control manifest: `{control_manifest}`",
            f"- metric rows: `{metric_csv}`",
            f"- confound rows: `{confound_csv}`",
            f"- loader rows: `{loader_csv}`",
            f"- JSON: `{json_path}`",
            "",
        ]
    )
    (OUT_DIR / "LATENTFM_RAWFM_HYBRID_GENE_BUDGET_GATE_20260628.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
