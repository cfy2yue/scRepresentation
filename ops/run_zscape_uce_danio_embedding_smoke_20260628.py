#!/usr/bin/env python3
"""Run a bounded UCE Danio embedding smoke for frozen ZSCAPE cells."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_INPUT = ROOT / "reports/zscape_uce_danio_latent_gate_20260628/zscape_uce_danio_128cell_smoke_input.h5ad"
DEFAULT_UCE = ROOT / "scFM_pretrained/uce/model_files"
DEFAULT_UCE_SRC = ROOT / "scFM_third_party/uce"
DEFAULT_OUT = ROOT / "runs/zscape_uce_danio_embedding_smoke_20260628/zscape_uce_danio_embedding_smoke_20260628"
PRIMARY_ROWS = ["periderm__noto__24p0h", "periderm__smo__24p0h"]


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def bootstrap_delta_distance(
    emb: np.ndarray,
    ctrl_idx: np.ndarray,
    pert_idx: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        c = rng.choice(ctrl_idx, size=len(ctrl_idx), replace=True)
        p = rng.choice(pert_idx, size=len(pert_idx), replace=True)
        vals.append(float(np.linalg.norm(emb[p].mean(axis=0) - emb[c].mean(axis=0))))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "delta_l2_boot_mean": float(arr.mean()),
        "delta_l2_ci_low": float(np.quantile(arr, 0.025)),
        "delta_l2_ci_high": float(np.quantile(arr, 0.975)),
    }


def compute_latent_posthoc(adata: ad.AnnData, emb_key: str, n_boot: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    emb = np.asarray(adata.obsm[emb_key], dtype=np.float32)
    obs = adata.obs.copy().reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    deltas: dict[str, np.ndarray] = {}

    for i, row_id in enumerate(PRIMARY_ROWS):
        mask_row = obs["row_id"].astype(str).to_numpy() == row_id
        ctrl_idx = np.where(mask_row & (obs["selection_role"].astype(str).to_numpy() == "control"))[0]
        pert_idx = np.where(mask_row & (obs["selection_role"].astype(str).to_numpy() == "perturb"))[0]
        ctrl_mean = emb[ctrl_idx].mean(axis=0)
        pert_mean = emb[pert_idx].mean(axis=0)
        delta = pert_mean - ctrl_mean
        deltas[row_id] = delta
        boot = bootstrap_delta_distance(emb, ctrl_idx, pert_idx, n_boot=n_boot, seed=20260628 + i)
        rows.append(
            {
                "row_id": row_id,
                "n_control": int(len(ctrl_idx)),
                "n_perturb": int(len(pert_idx)),
                "delta_l2": float(np.linalg.norm(delta)),
                "control_perturb_cosine": safe_cosine(ctrl_mean, pert_mean),
                "mean_control_norm": float(np.linalg.norm(ctrl_mean)),
                "mean_perturb_norm": float(np.linalg.norm(pert_mean)),
                **boot,
            }
        )

    summary = {
        "n_cells": int(adata.n_obs),
        "embedding_dim": int(emb.shape[1]),
        "embedding_mean_norm": float(np.mean(np.linalg.norm(emb, axis=1))),
        "embedding_std_mean": float(np.mean(np.std(emb, axis=0))),
        "noto_smo_delta_cosine": safe_cosine(deltas[PRIMARY_ROWS[0]], deltas[PRIMARY_ROWS[1]]),
        "n_boot": int(n_boot),
    }
    return rows, summary


def write_report(out_dir: Path, payload: dict[str, Any], posthoc_rows: list[dict[str, Any]]) -> None:
    report = out_dir / "LATENTFM_ZSCAPE_UCE_DANIO_EMBEDDING_SMOKE_20260628.md"
    lines = [
        "# LatentFM ZSCAPE UCE Danio Embedding Smoke",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `True` for this bounded embedding smoke only.",
        "",
        "## Boundary",
        "",
        "- Frozen 128-cell ZSCAPE smoke input from the UCE Danio CPU gate.",
        "- UCE inference only; no LatentFM training, checkpoint selection, canonical multi selection, or Track C query use.",
        "- Expression preprocessing is inherited from the gate: raw counts -> size factor to 1e4 -> exactly one `log1p`; `layers['counts']` preserved.",
        "- Perturbation token forcing is disabled; latent reflects expression state.",
        "",
        "## Outputs",
        "",
        f"- input h5ad: `{payload['input_h5ad']}`",
        f"- output h5ad: `{payload['output_h5ad']}`",
        f"- embeddings npy: `{payload['embedding_npy']}`",
        f"- posthoc rows: `{payload['posthoc_csv']}`",
        f"- JSON: `{payload['json']}`",
        "",
        "## Embedding Summary",
        "",
    ]
    for key, val in payload["summary"].items():
        lines.append(f"- {key}: `{val}`")
    lines.extend(
        [
            "",
            "## Periderm Latent Delta",
            "",
            "| row | n control | n perturb | delta L2 | 95% bootstrap CI | control/pert cosine |",
            "|---|---:|---:|---:|---|---:|",
        ]
    )
    for row in posthoc_rows:
        lines.append(
            "| {row_id} | {n_control} | {n_perturb} | {delta_l2:.6f} | [{lo:.6f}, {hi:.6f}] | {cos:.6f} |".format(
                row_id=row["row_id"],
                n_control=row["n_control"],
                n_perturb=row["n_perturb"],
                delta_l2=row["delta_l2"],
                lo=row["delta_l2_ci_low"],
                hi=row["delta_l2_ci_high"],
                cos=row["control_perturb_cosine"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- If the smoke completed, run a CPU latent-continuity/posthoc gate comparing these UCE deltas with the expression-space module/trajectory constraints before any larger UCE extraction.",
            "- This smoke is not evidence that LatentFM improves; it only unlocks a species-safe zebrafish latent diagnostic route.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5ad", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--uce-root", type=Path, default=DEFAULT_UCE)
    parser.add_argument("--uce-src", type=Path, default=DEFAULT_UCE_SRC)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--posthoc-only", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("COUPLEDFM_UCE_ROOT", str(args.uce_root.parent))
    os.environ.setdefault("COUPLEDFM_UCE_SRC", str(args.uce_src))
    sys.path.insert(0, str(args.uce_src / "exp_emb"))

    from uce_inference import UCEInference

    emb_key = "uce_danio_emb"
    output_h5ad = args.out_dir / "zscape_uce_danio_128cell_embedded.h5ad"
    embedding_npy = args.out_dir / "zscape_uce_danio_128cell_embeddings.npy"
    posthoc_csv = args.out_dir / "zscape_uce_danio_latent_delta_posthoc.csv"
    json_path = args.out_dir / "zscape_uce_danio_embedding_smoke_20260628.json"

    if not args.posthoc_only:
        inf = UCEInference(
            species="zebrafish",
            device=args.device,
            model_ckpt=args.uce_root / "33layer_model.torch",
            token_file=args.uce_root / "all_tokens.torch",
            spec_chrom_csv=args.uce_root / "species_chrom.csv",
            species_offsets_pkl=args.uce_root / "species_offsets.pkl",
            pe_dir=args.uce_root / "protein_embeddings",
        )
        emb = inf.encode_adata(
            str(args.input_h5ad),
            output_adata_path=str(output_h5ad),
            emb_key=emb_key,
            dataset_name="ZSCAPE_Danio_128cell",
            batch_size=args.batch_size,
            n_collate_workers=args.workers,
            show_progress=True,
        )
        np.save(embedding_npy, emb.astype(np.float32))
    elif not output_h5ad.is_file():
        raise FileNotFoundError(f"--posthoc-only requested but embedded h5ad is missing: {output_h5ad}")

    adata = ad.read_h5ad(output_h5ad)
    posthoc_rows, summary = compute_latent_posthoc(adata, emb_key=emb_key, n_boot=args.n_boot)
    pd.DataFrame(posthoc_rows).to_csv(posthoc_csv, index=False)

    payload = {
        "timestamp": now_cst(),
        "status": "zscape_uce_danio_embedding_smoke_done",
        "input_h5ad": str(args.input_h5ad),
        "output_h5ad": str(output_h5ad),
        "embedding_npy": str(embedding_npy),
        "posthoc_csv": str(posthoc_csv),
        "json": str(json_path),
        "emb_key": emb_key,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "summary": summary,
        "posthoc_rows": posthoc_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(args.out_dir, payload, posthoc_rows)
    print(json.dumps({"status": payload["status"], "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
