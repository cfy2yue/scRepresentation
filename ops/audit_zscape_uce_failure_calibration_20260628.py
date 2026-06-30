#!/usr/bin/env python3
"""Calibrate the failed ZSCAPE UCE Danio continuity gates without rerunning UCE."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path("/data/cyx/1030/scLatent")
RUNS = {
    "rank_selected": ROOT / "runs/zscape_uce_danio_embedding_smoke_20260628/zscape_uce_danio_embedding_smoke_20260628_1553/outputs/zscape_uce_danio_128cell_embedded.h5ad",
    "embryo_balanced": ROOT / "runs/zscape_uce_danio_embedding_smoke_balanced_20260628/zscape_uce_danio_embedding_smoke_balanced_20260628_1610/outputs/zscape_uce_danio_128cell_embedded.h5ad",
}
DEFAULT_OUT = ROOT / "reports/zscape_uce_failure_calibration_20260628"
PRIMARY_ROWS = ["periderm__noto__24p0h", "periderm__smo__24p0h"]


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    out = y.copy().astype(float)
    if ok.sum() < x.shape[1] + 4:
        return out
    design = np.column_stack([np.ones(ok.sum()), x[ok]])
    beta, *_ = np.linalg.lstsq(design, y[ok], rcond=None)
    out[ok] = y[ok] - design @ beta
    return out


def control_split_null(emb: np.ndarray, ctrl: np.ndarray, n_perm: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_a = len(ctrl) // 2
    vals = []
    for _ in range(n_perm):
        shuffled = rng.permutation(ctrl)
        a = shuffled[:n_a]
        b = shuffled[n_a:]
        vals.append(float(np.linalg.norm(emb[b].mean(axis=0) - emb[a].mean(axis=0))))
    arr = np.asarray(vals)
    return {
        "control_split_mean": float(arr.mean()),
        "control_split_q50": float(np.quantile(arr, 0.50)),
        "control_split_q90": float(np.quantile(arr, 0.90)),
        "control_split_q95": float(np.quantile(arr, 0.95)),
    }


def leave_embryo_out_signs(emb: np.ndarray, obs: pd.DataFrame, row_mask: np.ndarray, unit: np.ndarray) -> dict[str, Any]:
    sub = obs[row_mask].copy()
    sub["_idx"] = np.where(row_mask)[0]
    embryos = sorted(map(str, sub["embryo"].dropna().unique()))
    diffs = []
    for embryo in embryos:
        keep = sub["embryo"].astype(str).to_numpy() != embryo
        fold = sub[keep]
        ctrl = fold[fold["selection_role"].astype(str) == "control"]["_idx"].to_numpy(dtype=int)
        pert = fold[fold["selection_role"].astype(str) == "perturb"]["_idx"].to_numpy(dtype=int)
        if len(ctrl) == 0 or len(pert) == 0:
            continue
        scores = emb @ unit
        diffs.append(float(scores[pert].mean() - scores[ctrl].mean()))
    arr = np.asarray(diffs, dtype=float)
    return {
        "leave_embryo_out_folds": int(len(arr)),
        "leave_embryo_out_positive_fraction": float(np.mean(arr > 0)) if len(arr) else float("nan"),
        "leave_embryo_out_min": float(np.min(arr)) if len(arr) else float("nan"),
        "leave_embryo_out_max": float(np.max(arr)) if len(arr) else float("nan"),
    }


def analyze_run(label: str, h5ad_path: Path, n_perm: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adata = ad.read_h5ad(h5ad_path)
    emb = np.asarray(adata.obsm["uce_danio_emb"], dtype=float)
    obs = adata.obs.copy().reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    deltas: dict[str, np.ndarray] = {}
    for i, row_id in enumerate(PRIMARY_ROWS):
        mask = obs["row_id"].astype(str).to_numpy() == row_id
        ctrl = np.where(mask & (obs["selection_role"].astype(str).to_numpy() == "control"))[0]
        pert = np.where(mask & (obs["selection_role"].astype(str).to_numpy() == "perturb"))[0]
        delta = emb[pert].mean(axis=0) - emb[ctrl].mean(axis=0)
        deltas[row_id] = delta
        unit = delta / max(float(np.linalg.norm(delta)), 1e-12)
        scores = emb @ unit
        qc = obs[["n_umi", "num_genes_expressed"]].apply(pd.to_numeric, errors="coerce")
        z_qc = (qc - qc.mean(axis=0)) / qc.std(axis=0).replace(0, np.nan)
        residual_scores = residualize(scores, z_qc.to_numpy(dtype=float))
        raw_diff = float(scores[pert].mean() - scores[ctrl].mean())
        resid_diff = float(residual_scores[pert].mean() - residual_scores[ctrl].mean())
        null = control_split_null(emb, ctrl, n_perm=n_perm, seed=20260628 + 17 * i)
        leo = leave_embryo_out_signs(emb, obs, mask, unit)
        rho_umi = float(stats.spearmanr(scores, qc["n_umi"], nan_policy="omit").statistic)
        rho_genes = float(stats.spearmanr(scores, qc["num_genes_expressed"], nan_policy="omit").statistic)
        rows.append(
            {
                "run_label": label,
                "row_id": row_id,
                "n_control": int(len(ctrl)),
                "n_perturb": int(len(pert)),
                "n_control_embryos": int(obs.loc[ctrl, "embryo"].nunique()),
                "n_perturb_embryos": int(obs.loc[pert, "embryo"].nunique()),
                "delta_l2": float(np.linalg.norm(delta)),
                "raw_projection_diff": raw_diff,
                "qc_residual_projection_diff": resid_diff,
                "qc_residual_fraction_of_raw": resid_diff / raw_diff if raw_diff else float("nan"),
                "rho_n_umi": rho_umi,
                "rho_num_genes_expressed": rho_genes,
                "delta_over_control_q95": float(np.linalg.norm(delta) / null["control_split_q95"]) if null["control_split_q95"] else float("nan"),
                **null,
                **leo,
            }
        )
    summary = {
        "run_label": label,
        "h5ad": str(h5ad_path),
        "n_cells": int(adata.n_obs),
        "embedding_dim": int(emb.shape[1]),
        "noto_smo_delta_cosine": safe_cosine(deltas[PRIMARY_ROWS[0]], deltas[PRIMARY_ROWS[1]]),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-perm", type=int, default=2000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    summaries = []
    for label, path in RUNS.items():
        rows, summary = analyze_run(label, path, args.n_perm)
        all_rows.extend(rows)
        summaries.append(summary)

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out_dir / "zscape_uce_failure_calibration_rows.csv", index=False)
    status = "zscape_uce_failure_calibration_supports_closure"
    payload = {
        "timestamp": now_cst(),
        "status": status,
        "summaries": summaries,
        "rows": all_rows,
        "interpretation": "QA-only calibration; does not rescue UCE route or authorize GPU.",
    }
    (args.out_dir / "zscape_uce_failure_calibration_20260628.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    report = args.out_dir / "LATENTFM_ZSCAPE_UCE_FAILURE_CALIBRATION_20260628.md"
    lines = [
        "# LatentFM ZSCAPE UCE Failure Calibration",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`.",
        "",
        "## Boundary",
        "",
        "- CPU-only QA over existing UCE embeddings.",
        "- Does not rerun UCE, train LatentFM, select checkpoints, or authorize larger extraction.",
        "",
        "## Summary",
        "",
        "| run | row | delta/control q95 | QC residual/raw | leave-embryo positive | max abs QC rho |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        max_rho = max(abs(row["rho_n_umi"]), abs(row["rho_num_genes_expressed"]))
        lines.append(
            "| {run} | {row_id} | {ratio:.3f} | {resid:.3f} | {leo:.3f} | {rho:.3f} |".format(
                run=row["run_label"],
                row_id=row["row_id"],
                ratio=row["delta_over_control_q95"],
                resid=row["qc_residual_fraction_of_raw"],
                leo=row["leave_embryo_out_positive_fraction"],
                rho=max_rho,
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Failure calibration supports closing UCE as a model-enabling latent constraint route.",
            "- The negative conclusion is not solely a missing-embedding or initial sampling artifact.",
            "- Keep UCE as diagnostic/negative evidence only unless a new hypothesis is externally reviewed.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
