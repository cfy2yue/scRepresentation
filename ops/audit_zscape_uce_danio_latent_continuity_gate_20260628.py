#!/usr/bin/env python3
"""Posthoc gate for the ZSCAPE UCE Danio latent smoke."""

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
DEFAULT_RUN = ROOT / "runs/zscape_uce_danio_embedding_smoke_20260628/zscape_uce_danio_embedding_smoke_20260628_1553/outputs"
DEFAULT_OUT = ROOT / "reports/zscape_uce_danio_latent_continuity_gate_20260628"
PRIMARY_ROWS = ["periderm__noto__24p0h", "periderm__smo__24p0h"]


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def bootstrap_projection_diff(scores: np.ndarray, ctrl: np.ndarray, pert: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        c = rng.choice(ctrl, size=len(ctrl), replace=True)
        p = rng.choice(pert, size=len(pert), replace=True)
        vals.append(float(scores[p].mean() - scores[c].mean()))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "projection_diff_boot_mean": float(arr.mean()),
        "projection_diff_ci_low": float(np.quantile(arr, 0.025)),
        "projection_diff_ci_high": float(np.quantile(arr, 0.975)),
    }


def cell_permutation_p(scores: np.ndarray, ctrl: np.ndarray, pert: np.ndarray, n_perm: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    idx = np.concatenate([ctrl, pert])
    labels = np.array([0] * len(ctrl) + [1] * len(pert), dtype=np.int8)
    observed = float(scores[pert].mean() - scores[ctrl].mean())
    null = []
    for _ in range(n_perm):
        rng.shuffle(labels)
        pidx = idx[labels == 1]
        cidx = idx[labels == 0]
        null.append(float(scores[pidx].mean() - scores[cidx].mean()))
    null_arr = np.asarray(null)
    return float((np.sum(np.abs(null_arr) >= abs(observed)) + 1) / (n_perm + 1))


def split_by_metadata(obs: pd.DataFrame, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sub = obs.iloc[idx].copy()
    sort_cols = [c for c in ["embryo", "sample", "cell"] if c in sub.columns]
    if sort_cols:
        sub = sub.sort_values(sort_cols)
    ordered = sub.index.to_numpy(dtype=int)
    return ordered[::2], ordered[1::2]


def unit_delta(emb: np.ndarray, ctrl: np.ndarray, pert: np.ndarray) -> np.ndarray:
    delta = emb[pert].mean(axis=0) - emb[ctrl].mean(axis=0)
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-12:
        return np.zeros(emb.shape[1], dtype=np.float64)
    return delta / norm


def projection_diff_for_unit(emb: np.ndarray, unit: np.ndarray, ctrl: np.ndarray, pert: np.ndarray) -> float:
    scores = emb @ unit
    return float(scores[pert].mean() - scores[ctrl].mean())


def crossfit_projection_stat(emb: np.ndarray, obs: pd.DataFrame, ctrl: np.ndarray, pert: np.ndarray) -> dict[str, Any]:
    ctrl_a, ctrl_b = split_by_metadata(obs, ctrl)
    pert_a, pert_b = split_by_metadata(obs, pert)
    unit_a = unit_delta(emb, ctrl_a, pert_a)
    unit_b = unit_delta(emb, ctrl_b, pert_b)
    diff_b = projection_diff_for_unit(emb, unit_a, ctrl_b, pert_b)
    diff_a = projection_diff_for_unit(emb, unit_b, ctrl_a, pert_a)
    return {
        "crossfit_projection_diff": float(np.mean([diff_a, diff_b])),
        "crossfit_fold_a_diff": diff_a,
        "crossfit_fold_b_diff": diff_b,
        "n_ctrl_a": int(len(ctrl_a)),
        "n_ctrl_b": int(len(ctrl_b)),
        "n_pert_a": int(len(pert_a)),
        "n_pert_b": int(len(pert_b)),
        "splits": (ctrl_a, ctrl_b, pert_a, pert_b),
    }


def permuted_pair(ctrl: np.ndarray, pert: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    combined = np.concatenate([ctrl, pert])
    shuffled = rng.permutation(combined)
    return shuffled[: len(ctrl)], shuffled[len(ctrl) :]


def crossfit_permutation_p(
    emb: np.ndarray,
    obs: pd.DataFrame,
    ctrl: np.ndarray,
    pert: np.ndarray,
    observed: float,
    n_perm: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    ctrl_a, ctrl_b = split_by_metadata(obs, ctrl)
    pert_a, pert_b = split_by_metadata(obs, pert)
    null = []
    for _ in range(n_perm):
        cta, pta = permuted_pair(ctrl_a, pert_a, rng)
        ctb, ptb = permuted_pair(ctrl_b, pert_b, rng)
        unit_a = unit_delta(emb, cta, pta)
        unit_b = unit_delta(emb, ctb, ptb)
        diff_b = projection_diff_for_unit(emb, unit_a, ctb, ptb)
        diff_a = projection_diff_for_unit(emb, unit_b, cta, pta)
        null.append(float(np.mean([diff_a, diff_b])))
    null_arr = np.asarray(null, dtype=np.float64)
    return float((np.sum(np.abs(null_arr) >= abs(observed)) + 1) / (n_perm + 1))


def control_split_negative(emb: np.ndarray, ctrl: np.ndarray, n_perm: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_a = len(ctrl) // 2
    null = []
    for _ in range(n_perm):
        shuffled = rng.permutation(ctrl)
        a = shuffled[:n_a]
        b = shuffled[n_a:]
        null.append(float(np.linalg.norm(emb[b].mean(axis=0) - emb[a].mean(axis=0))))
    arr = np.asarray(null, dtype=np.float64)
    return {
        "control_split_l2_q95": float(np.quantile(arr, 0.95)),
        "control_split_l2_mean": float(arr.mean()),
    }


def embryo_pseudobulk_gate(
    emb: np.ndarray,
    obs: pd.DataFrame,
    mask: np.ndarray,
    unit: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    sub = obs[mask].copy()
    sub["_idx"] = np.where(mask)[0]
    rows = []
    for (role, embryo), g in sub.groupby(["selection_role", "embryo"], sort=True):
        idx = g["_idx"].to_numpy(dtype=int)
        mean_emb = emb[idx].mean(axis=0)
        rows.append({"selection_role": role, "embryo": embryo, "score": float(mean_emb @ unit), "n_cells": int(len(idx))})
    df = pd.DataFrame(rows)
    ctrl_scores = df[df["selection_role"].astype(str) == "control"]["score"].to_numpy(dtype=float)
    pert_scores = df[df["selection_role"].astype(str) == "perturb"]["score"].to_numpy(dtype=float)
    out: dict[str, Any] = {
        "n_control_embryos": int(len(ctrl_scores)),
        "n_perturb_embryos": int(len(pert_scores)),
        "embryo_projection_diff": float("nan"),
        "embryo_projection_ci_low": float("nan"),
        "embryo_projection_ci_high": float("nan"),
        "embryo_welch_p": float("nan"),
        "embryo_gate_evaluable": bool(len(ctrl_scores) >= 3 and len(pert_scores) >= 3),
    }
    if not out["embryo_gate_evaluable"]:
        return out
    out["embryo_projection_diff"] = float(pert_scores.mean() - ctrl_scores.mean())
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        c = rng.choice(ctrl_scores, size=len(ctrl_scores), replace=True)
        p = rng.choice(pert_scores, size=len(pert_scores), replace=True)
        boot.append(float(p.mean() - c.mean()))
    boot_arr = np.asarray(boot, dtype=float)
    out["embryo_projection_ci_low"] = float(np.quantile(boot_arr, 0.025))
    out["embryo_projection_ci_high"] = float(np.quantile(boot_arr, 0.975))
    out["embryo_welch_p"] = float(stats.ttest_ind(pert_scores, ctrl_scores, equal_var=False).pvalue)
    return out


def qc_rho(scores: np.ndarray, values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(vals) & np.isfinite(scores)
    if ok.sum() < 8:
        return float("nan")
    return float(stats.spearmanr(scores[ok], vals[ok]).statistic)


def write_waiting_report(out_dir: Path, input_h5ad: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": now_cst(),
        "status": "zscape_uce_danio_latent_continuity_waiting_embedding",
        "input_h5ad": str(input_h5ad),
    }
    (out_dir / "zscape_uce_danio_latent_continuity_gate_20260628.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    report = out_dir / "LATENTFM_ZSCAPE_UCE_DANIO_LATENT_CONTINUITY_GATE_20260628.md"
    report.write_text(
        "\n".join(
            [
                "# LatentFM ZSCAPE UCE Danio Latent Continuity Gate",
                "",
                f"Timestamp: `{payload['timestamp']}`",
                "",
                "Status: `zscape_uce_danio_latent_continuity_waiting_embedding`",
                "",
                f"Waiting for embedded h5ad: `{input_h5ad}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-out-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--n-perm", type=int, default=2000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_h5ad = args.run_out_dir / "zscape_uce_danio_128cell_embedded.h5ad"
    if not input_h5ad.is_file():
        write_waiting_report(args.out_dir, input_h5ad)
        print(json.dumps({"status": "waiting", "input_h5ad": str(input_h5ad)}, indent=2))
        return

    adata = ad.read_h5ad(input_h5ad)
    emb = np.asarray(adata.obsm["uce_danio_emb"], dtype=np.float64)
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
        boot = bootstrap_projection_diff(scores, ctrl, pert, args.n_boot, seed=20260628 + i)
        crossfit = crossfit_projection_stat(emb, obs, ctrl, pert)
        crossfit_p = crossfit_permutation_p(
            emb,
            obs,
            ctrl,
            pert,
            observed=crossfit["crossfit_projection_diff"],
            n_perm=args.n_perm,
            seed=202606280 + i,
        )
        control_neg = control_split_negative(emb, ctrl, args.n_perm, seed=202606290 + i)
        embryo_gate = embryo_pseudobulk_gate(emb, obs, mask, unit, args.n_boot, seed=202606300 + i)
        row = {
            "row_id": row_id,
            "n_control": int(len(ctrl)),
            "n_perturb": int(len(pert)),
            "delta_l2": float(np.linalg.norm(delta)),
            "projection_diff": float(scores[pert].mean() - scores[ctrl].mean()),
            "cell_direction_permutation_p": cell_permutation_p(scores, ctrl, pert, args.n_perm, seed=202606270 + i),
            "crossfit_projection_diff": crossfit["crossfit_projection_diff"],
            "crossfit_fold_a_diff": crossfit["crossfit_fold_a_diff"],
            "crossfit_fold_b_diff": crossfit["crossfit_fold_b_diff"],
            "crossfit_permutation_p": crossfit_p,
            "control_split_l2_q95": control_neg["control_split_l2_q95"],
            "control_split_l2_mean": control_neg["control_split_l2_mean"],
            "delta_beats_control_split_q95": bool(float(np.linalg.norm(delta)) > control_neg["control_split_l2_q95"]),
            "rho_n_umi": qc_rho(scores, obs["n_umi"]),
            "rho_num_genes_expressed": qc_rho(scores, obs["num_genes_expressed"]),
            **embryo_gate,
            **boot,
        }
        row["max_abs_qc_rho"] = float(np.nanmax(np.abs([row["rho_n_umi"], row["rho_num_genes_expressed"]])))
        row["row_pass"] = bool(
            row["crossfit_projection_diff"] > 0
            and row["crossfit_permutation_p"] <= 0.05
            and row["delta_beats_control_split_q95"]
            and row["embryo_gate_evaluable"]
            and row["embryo_projection_ci_low"] > 0
            and row["max_abs_qc_rho"] <= 0.6
        )
        crossfit.pop("splits", None)
        rows.append(row)

    delta_cosine = safe_cosine(deltas[PRIMARY_ROWS[0]], deltas[PRIMARY_ROWS[1]])
    n_rows_pass = sum(bool(r["row_pass"]) for r in rows)
    status = (
        "zscape_uce_danio_latent_continuity_pass_no_training"
        if n_rows_pass == len(PRIMARY_ROWS) and np.isfinite(delta_cosine) and delta_cosine > 0
        else "zscape_uce_danio_latent_continuity_partial_or_fail_no_training"
    )
    payload = {
        "timestamp": now_cst(),
        "status": status,
        "input_h5ad": str(input_h5ad),
        "delta_cosine_noto_smo": delta_cosine,
        "n_rows_pass": int(n_rows_pass),
        "rows": rows,
        "gate": "row pass requires cross-fit projection >0, cross-fit permutation p<=0.05, delta L2 beating control-split q95, embryo pseudobulk CI_low>0, evaluable embryo groups, and max |QC rho|<=0.6; global pass also requires noto/smo delta cosine >0",
    }
    pd.DataFrame(rows).to_csv(args.out_dir / "zscape_uce_danio_latent_continuity_rows.csv", index=False)
    (args.out_dir / "zscape_uce_danio_latent_continuity_gate_20260628.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    report = args.out_dir / "LATENTFM_ZSCAPE_UCE_DANIO_LATENT_CONTINUITY_GATE_20260628.md"
    lines = [
        "# LatentFM ZSCAPE UCE Danio Latent Continuity Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`.",
        "",
        "## Boundary",
        "",
        "- CPU posthoc over the frozen 128-cell UCE Danio smoke output.",
        "- Does not train LatentFM, select checkpoints, use canonical multi selection, or read Track C query.",
        "",
        "## Gate",
        "",
        f"- noto/smo delta cosine: `{delta_cosine}`",
        f"- rows passing: `{n_rows_pass}/{len(PRIMARY_ROWS)}`",
        "- row gate: cross-fit projection, cross-fit label permutation, control-split negative, embryo pseudobulk, and QC rho",
        "",
        "| row | crossfit diff | crossfit p | embryo diff CI | control q95 beaten | max abs QC rho | pass |",
        "|---|---:|---:|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {row_id} | {diff:.6f} | {p:.6f} | [{lo:.6f}, {hi:.6f}] | `{ctrl}` | {qc:.6f} | `{ok}` |".format(
                row_id=row["row_id"],
                diff=row["crossfit_projection_diff"],
                p=row["crossfit_permutation_p"],
                lo=row["embryo_projection_ci_low"],
                hi=row["embryo_projection_ci_high"],
                ctrl=row["delta_beats_control_split_q95"],
                qc=row["max_abs_qc_rho"],
                ok=row["row_pass"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Passing this gate supports using UCE as a ZSCAPE latent diagnostic for biological interpretation.",
            "- It still does not authorize LatentFM training or model promotion without a separate design review.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "delta_cosine_noto_smo": delta_cosine}, indent=2))


if __name__ == "__main__":
    main()
