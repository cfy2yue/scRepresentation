#!/usr/bin/env python3
"""CPU gate for control/background whitening as a LatentFM loss-geometry branch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
MEAN_DIR = ROOT / "reports/latentfm_xverse_nuisance_condition_means_20260624"
ANCHOR_JSON = MEAN_DIR / "condition_family_eval_anchor_internal_means_ode20.json"
CAP120_JSON = MEAN_DIR / "condition_family_eval_cap120_internal_means_ode20.json"
OUT_JSON = ROOT / "reports/latentfm_xverse_control_whitening_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_CONTROL_WHITENING_GATE_20260624.md"
GROUP = "family_gene"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def vec(row: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(row[key], dtype=np.float64)


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    mask = np.isfinite(xx) & np.isfinite(yy)
    if int(mask.sum()) < 3:
        return float("nan")
    rx = rankdata(xx[mask])
    ry = rankdata(yy[mask])
    sx = float(np.std(rx))
    sy = float(np.std(ry))
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def condition_rows(obj: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = obj["groups"][GROUP]["condition_metrics"]
    return {(str(row["dataset"]), str(row["condition"])): row for row in rows}


def dataset_scales(anchor_rows: dict[tuple[str, str], dict[str, Any]]) -> dict[str, np.ndarray]:
    by_ds: dict[str, list[np.ndarray]] = {}
    for (ds, _), row in anchor_rows.items():
        by_ds.setdefault(ds, []).append(vec(row, "gt_mean") - vec(row, "ctrl_mean"))
    scales: dict[str, np.ndarray] = {}
    for ds, arrs in by_ds.items():
        mat = np.stack(arrs, axis=0)
        sd = np.std(mat, axis=0)
        floor = float(np.median(sd[sd > 0])) if np.any(sd > 0) else 1.0
        scales[ds] = np.maximum(sd, max(floor * 0.1, 1e-4))
    return scales


def norm_residual(row: dict[str, Any], scale: np.ndarray | None = None) -> float:
    resid = vec(row, "pred_mean") - vec(row, "gt_mean")
    if scale is not None:
        resid = resid / scale
    return float(np.linalg.norm(resid) / np.sqrt(resid.size))


def top_overlap(values_a: dict[tuple[str, str], float], values_b: dict[tuple[str, str], float], n: int = 10) -> int:
    aa = {k for k, _ in sorted(values_a.items(), key=lambda kv: kv[1], reverse=True)[:n]}
    bb = {k for k, _ in sorted(values_b.items(), key=lambda kv: kv[1], reverse=True)[:n]}
    return len(aa & bb)


def main() -> int:
    anchor = load_json(ANCHOR_JSON)
    cap120 = load_json(CAP120_JSON)
    anchor_rows = condition_rows(anchor)
    cap_rows = condition_rows(cap120)
    keys = sorted(set(anchor_rows) & set(cap_rows))
    scales = dataset_scales(anchor_rows)

    rows = []
    mmd_delta: dict[tuple[str, str], float] = {}
    raw_delta: dict[tuple[str, str], float] = {}
    white_delta: dict[tuple[str, str], float] = {}
    pp_delta: dict[tuple[str, str], float] = {}
    for key in keys:
        ds, cond = key
        a = anchor_rows[key]
        c = cap_rows[key]
        raw_a = norm_residual(a)
        raw_c = norm_residual(c)
        white_a = norm_residual(a, scales[ds])
        white_c = norm_residual(c, scales[ds])
        md = float(c["test_mmd_clamped"]) - float(a["test_mmd_clamped"])
        pp = float(c["pearson_pert"]) - float(a["pearson_pert"])
        mmd_delta[key] = md
        raw_delta[key] = raw_c - raw_a
        white_delta[key] = white_c - white_a
        pp_delta[key] = pp
        rows.append(
            {
                "dataset": ds,
                "condition": cond,
                "mmd_delta": md,
                "pp_delta": pp,
                "raw_residual_delta": raw_delta[key],
                "whitened_residual_delta": white_delta[key],
            }
        )

    raw_s = spearman(list(mmd_delta.values()), list(raw_delta.values()))
    white_s = spearman(list(mmd_delta.values()), list(white_delta.values()))
    pp_s = spearman(list(pp_delta.values()), list(white_delta.values()))
    overlap_raw = top_overlap(mmd_delta, raw_delta, n=10)
    overlap_white = top_overlap(mmd_delta, white_delta, n=10)

    by_ds: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_ds.setdefault(row["dataset"], []).append(row)
    ds_rows = []
    for ds, vals in sorted(by_ds.items()):
        ds_rows.append(
            {
                "dataset": ds,
                "n": len(vals),
                "mean_mmd_delta": float(np.mean([v["mmd_delta"] for v in vals])),
                "mean_pp_delta": float(np.mean([v["pp_delta"] for v in vals])),
                "mean_raw_residual_delta": float(np.mean([v["raw_residual_delta"] for v in vals])),
                "mean_whitened_residual_delta": float(np.mean([v["whitened_residual_delta"] for v in vals])),
            }
        )

    reasons = []
    if not np.isfinite(white_s) or white_s < max(0.25, raw_s + 0.15):
        reasons.append("whitened_residual_not_materially_more_aligned_with_mmd_harm")
    if overlap_white < max(3, overlap_raw + 1):
        reasons.append("whitened_top_mmd_overlap_not_improved")
    if np.mean([r["mean_pp_delta"] for r in ds_rows]) < -0.005:
        reasons.append("cap120_internal_pp_mean_harm_under_whitening_audit")
    decision = {
        "status": "latentfm_xverse_control_whitening_gate_pass_code_gate_next_no_gpu"
        if not reasons
        else "latentfm_xverse_control_whitening_gate_fail_no_gpu",
        "gpu_authorization": "none",
        "action": "design_whitened_loss_smoke_only_if_pass_else_close_normalization",
        "reasons": reasons,
    }
    payload = {
        "status": decision["status"],
        "inputs": {"anchor": str(ANCHOR_JSON), "cap120": str(CAP120_JSON), "group": GROUP},
        "boundary": {
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "heldout_query_read": False,
            "active_log_read": False,
            "gpu_artifact_read": False,
        },
        "metrics": {
            "n_conditions": len(keys),
            "raw_mmd_residual_spearman": raw_s,
            "whitened_mmd_residual_spearman": white_s,
            "whitened_pp_residual_spearman": pp_s,
            "top10_mmd_raw_residual_overlap": overlap_raw,
            "top10_mmd_whitened_residual_overlap": overlap_white,
        },
        "dataset_rows": ds_rows,
        "condition_rows": rows,
        "decision": decision,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM xverse Control/Background Whitening Gate",
        "",
        f"Status: `{decision['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- Uses only train/internal condition means from the nuisance diagnostic.",
        "- Does not read canonical outcomes, canonical multi, held-out query, active logs, or GPU artifacts.",
        "- Tests whether dataset-scale whitening better explains MMD harm than raw residual geometry.",
        "",
        "## Key Metrics",
        "",
        f"- conditions: `{len(keys)}`",
        f"- raw MMD/residual Spearman: `{raw_s:.6f}`",
        f"- whitened MMD/residual Spearman: `{white_s:.6f}`",
        f"- top10 MMD/raw overlap: `{overlap_raw}`",
        f"- top10 MMD/whitened overlap: `{overlap_white}`",
        "",
        "## Dataset Rows",
        "",
        "| dataset | n | mean MMD delta | mean pp delta | raw residual delta | whitened residual delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in ds_rows:
        lines.append(
            f"| {row['dataset']} | {row['n']} | {row['mean_mmd_delta']:+.6f} | "
            f"{row['mean_pp_delta']:+.6f} | {row['mean_raw_residual_delta']:+.6f} | "
            f"{row['mean_whitened_residual_delta']:+.6f} |"
        )
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend([f"- `{r}`" for r in reasons] or ["- none"])
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "gpu_authorization": "none", "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
