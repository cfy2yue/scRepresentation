#!/usr/bin/env python3
"""Build and audit low-dose residual add-back RawFM gene-budget manifests."""

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


OUT_DIR = ROOT / "reports/rawfm_lowdose_residual_gate_20260628"
MANIFEST_DIR = ROOT / "reports/rawfm_lowdose_residual_manifest_20260628"
DATASET = "Wessels"
SEED = 42
SPECS = (
    {"residual": 32, "abundance": 96, "random": 128},
    {"residual": 64, "abundance": 96, "random": 96},
)


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


def quantile_bins(score: np.ndarray, bins: int) -> np.ndarray:
    q = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    edges = np.quantile(np.asarray(score, dtype=float), q)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.digitize(score, edges[1:-1], right=True).astype(int)


def multi_confound_control(
    selected_keep: np.ndarray,
    confound_scores: list[np.ndarray],
    rng: np.random.Generator,
    bins: int = 8,
) -> np.ndarray:
    n_genes = int(confound_scores[0].shape[0])
    bin_arrays = [quantile_bins(score, bins) for score in confound_scores]
    selected_set = set(map(int, selected_keep))
    selected_keys = [tuple(int(arr[i]) for arr in bin_arrays) for i in selected_keep]
    all_keys: dict[tuple[int, ...], list[int]] = {}
    for i in range(n_genes):
        all_keys.setdefault(tuple(int(arr[i]) for arr in bin_arrays), []).append(i)
    chosen: list[int] = []
    used: set[int] = set()
    for key in sorted(set(selected_keys)):
        need = int(sum(k == key for k in selected_keys))
        pool = [i for i in all_keys.get(key, []) if i not in selected_set and i not in used]
        if len(pool) < need:
            pool = [i for i in all_keys.get(key, []) if i not in used]
        if not pool:
            continue
        take = rng.choice(np.asarray(pool, dtype=int), size=min(need, len(pool)), replace=False)
        chosen.extend(map(int, take))
        used.update(map(int, take))
    if len(chosen) < len(selected_keep):
        remaining = np.asarray([i for i in range(n_genes) if i not in used and i not in selected_set], dtype=int)
        if remaining.size < len(selected_keep) - len(chosen):
            remaining = np.asarray([i for i in range(n_genes) if i not in used], dtype=int)
        extra = rng.choice(remaining, size=len(selected_keep) - len(chosen), replace=False)
        chosen.extend(map(int, extra))
    return np.asarray(chosen[: len(selected_keep)], dtype=int)


def compose_budget(
    primary: np.ndarray,
    abundance_rank: np.ndarray,
    random_rank: np.ndarray,
    n_primary: int,
    n_abundance: int,
    n_random: int,
) -> np.ndarray:
    chosen: list[int] = []
    seen: set[int] = set()

    def add_from(arr: np.ndarray, limit: int) -> None:
        for value in arr:
            if len(chosen) >= n_primary + n_abundance + n_random:
                return
            if sum(1 for _ in ()) < 0:
                return
            idx = int(value)
            if idx in seen:
                continue
            chosen.append(idx)
            seen.add(idx)
            if len(chosen) >= limit:
                return

    add_from(primary, n_primary)
    target_after_abundance = n_primary + n_abundance
    for value in abundance_rank:
        if len(chosen) >= target_after_abundance:
            break
        idx = int(value)
        if idx in seen:
            continue
        chosen.append(idx)
        seen.add(idx)
    target_total = n_primary + n_abundance + n_random
    for value in random_rank:
        if len(chosen) >= target_total:
            break
        idx = int(value)
        if idx in seen:
            continue
        chosen.append(idx)
        seen.add(idx)
    if len(chosen) < target_total:
        for idx in range(max(max(chosen, default=0) + 1, target_total * 2)):
            if len(chosen) >= target_total:
                break
            if idx in seen:
                continue
            chosen.append(idx)
            seen.add(idx)
    return np.asarray(chosen[:target_total], dtype=int)


def write_manifest(path: Path, label: str, keep: np.ndarray, n_genes: int, source: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "version": 1,
        "label": label,
        "source": {
            "split_file": str(structural_gate.DEFAULT_SPLIT),
            "dataset": DATASET,
            "seed": SEED,
            "train_only": True,
            **source,
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
        "overlap_k256_residual": structural_gate.overlap(keep, refs["k256_residual"]),
        "overlap_k256_random": structural_gate.overlap(keep, refs["k256_random"]),
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
    residual_rank = topk(scores["residual"], 512)
    abundance_rank = topk(scores["abundance"], 512)
    rng = np.random.default_rng(SEED)
    random_rank = rng.permutation(n_genes).astype(int)
    k256_residual = structural_gate.read_keep(
        ROOT / "reports/rawfm_structural_gene_budget_manifest_20260628/response_abundance_residual_topk_k256_seed42.json",
        DATASET,
    )
    k256_random = structural_gate.read_keep(
        ROOT / "reports/rawfm_structural_gene_budget_manifest_20260628/random_gene_set_k256_seed42.json",
        DATASET,
    )
    refs = {"k256_residual": k256_residual, "k256_random": k256_random}

    metric_rows: list[dict[str, Any]] = []
    confound_rows: list[dict[str, Any]] = []
    loader_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for spec in SPECS:
        n_res = int(spec["residual"])
        n_abund = int(spec["abundance"])
        n_rand = int(spec["random"])
        primary = residual_rank[:n_res]
        control_primary = multi_confound_control(
            primary,
            [scores["abundance"], scores["variance"], scores["detection"]],
            np.random.default_rng(SEED + n_res),
            bins=8,
        )
        candidate = compose_budget(primary, abundance_rank, random_rank, n_res, n_abund, n_rand)
        control = compose_budget(control_primary, abundance_rank, random_rank, n_res, n_abund, n_rand)
        base = f"residual{n_res}_abundance{n_abund}_random{n_rand}"
        ctrl_base = f"confound{n_res}_abundance{n_abund}_random{n_rand}_control"
        for label, keep, role in [(base, candidate, "candidate"), (ctrl_base, control, "control")]:
            manifest_label = f"{label}_k256_seed{SEED}"
            path = MANIFEST_DIR / f"{manifest_label}.json"
            write_manifest(
                path,
                manifest_label,
                keep,
                n_genes,
                {
                    "residual_dose": n_res,
                    "abundance_anchor": n_abund,
                    "random_ballast": n_rand,
                    "role": role,
                    "control_primary": "multi_confound_matched_random" if role == "control" else "",
                },
            )
            metric_rows.append(metric_row(label, keep, scores, refs))
            manifest_rows.append({"label": label, "role": role, "manifest": str(path)})
            checks = structural_gate.loader_dryrun(
                path,
                manifest_label,
                DATASET,
                structural_gate.DEFAULT_SPLIT,
                structural_gate.DEFAULT_BIFLOW,
            )
            loader_rows.append({"label": label, "manifest": str(path), **checks})
        confound_rows.append(
            {
                "comparison": f"{base}_vs_control",
                "residual_dose": n_res,
                "residual_delta": float(np.mean(scores["residual"][candidate]) - np.mean(scores["residual"][control])),
                "response_delta": float(np.mean(scores["response"][candidate]) - np.mean(scores["response"][control])),
                "abundance_smd": structural_gate.smd(scores["abundance"][candidate], scores["abundance"][control]),
                "variance_smd": structural_gate.smd(scores["variance"][candidate], scores["variance"][control]),
                "detection_smd": structural_gate.smd(scores["detection"][candidate], scores["detection"][control]),
            }
        )

    metric_df = pd.DataFrame(metric_rows)
    confound_df = pd.DataFrame(confound_rows)
    loader_df = pd.DataFrame(loader_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    loader_gate = bool(loader_df["pass"].all())
    confound_gate = bool(
        (
            (confound_df["residual_delta"] > 0.05)
            & (confound_df["abundance_smd"].abs() <= 0.75)
            & (confound_df["variance_smd"].abs() <= 0.75)
            & (confound_df["detection_smd"].abs() <= 0.75)
        ).any()
    )
    gpu_packet_authorized = bool(loader_gate and confound_gate)
    status = "rawfm_lowdose_residual_gate_gpu_packet_ready" if gpu_packet_authorized else "rawfm_lowdose_residual_gate_fail_no_gpu"

    metric_csv = OUT_DIR / "rawfm_lowdose_residual_metric_rows.csv"
    confound_csv = OUT_DIR / "rawfm_lowdose_residual_confound_rows.csv"
    loader_csv = OUT_DIR / "rawfm_lowdose_residual_loader_rows.csv"
    manifest_csv = OUT_DIR / "rawfm_lowdose_residual_manifest_rows.csv"
    metric_df.to_csv(metric_csv, index=False)
    confound_df.to_csv(confound_csv, index=False)
    loader_df.to_csv(loader_csv, index=False)
    manifest_df.to_csv(manifest_csv, index=False)
    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_packet_authorized": gpu_packet_authorized,
        "loader_gate": loader_gate,
        "confound_gate": confound_gate,
        "manifest_dir": str(MANIFEST_DIR),
        "metric_csv": str(metric_csv),
        "confound_csv": str(confound_csv),
        "loader_csv": str(loader_csv),
        "manifest_csv": str(manifest_csv),
    }
    json_path = OUT_DIR / "rawfm_lowdose_residual_gate_20260628.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# RawFM Low-Dose Residual Add-Back Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        f"GPU packet authorized: `{gpu_packet_authorized}`",
        "",
        "## Boundary",
        "",
        "- CPU-only manifest and loader audit for Wessels k=256 low-dose residual add-back.",
        "- Candidate/control share abundance and random ballast; only residual primary genes differ.",
        "- No training, no inference, no GPU, no canonical multi, no Track C query.",
        "",
        "## Metric Rows",
        "",
        "| label | residual | response | abundance | overlap k256 residual | overlap k256 random |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in metric_df.iterrows():
        lines.append(
            f"| {row['label']} | {fmt(row['residual_mean'])} | {fmt(row['response_mean'])} | "
            f"{fmt(row['abundance_mean'])} | {fmt(row['overlap_k256_residual'])} | "
            f"{fmt(row['overlap_k256_random'])} |"
        )
    lines.extend(
        [
            "",
            "## Confound Checks",
            "",
            "| comparison | residual delta | response delta | abundance SMD | variance SMD | detection SMD |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in confound_df.iterrows():
        lines.append(
            f"| {row['comparison']} | {fmt(row['residual_delta'])} | {fmt(row['response_delta'])} | "
            f"{fmt(row['abundance_smd'])} | {fmt(row['variance_smd'])} | {fmt(row['detection_smd'])} |"
        )
    lines.extend(["", "## Loader Checks", "", "| label | kept | pass |", "|---|---:|---:|"])
    for _, row in loader_df.iterrows():
        lines.append(f"| {row['label']} | {int(row['kept_genes'])} | {bool(row['pass'])} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- loader gate: `{loader_gate}`",
            f"- confound gate: `{confound_gate}`",
        ]
    )
    if gpu_packet_authorized:
        lines.append(
            "- GPU packet: launch fixed-step/no-selection MMD-off candidate/control smokes for the rows whose confound gate passes."
        )
    else:
        lines.append("- No GPU launch: low-dose residual add-back did not clear the CPU gate.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- manifests: `{manifest_csv}`",
            f"- metrics: `{metric_csv}`",
            f"- confounds: `{confound_csv}`",
            f"- loader: `{loader_csv}`",
            f"- JSON: `{json_path}`",
            "",
        ]
    )
    (OUT_DIR / "LATENTFM_RAWFM_LOWDOSE_RESIDUAL_GATE_20260628.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
