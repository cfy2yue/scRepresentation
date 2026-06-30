#!/usr/bin/env python3
"""CPU gate for conservative anchor-to-cap120 update magnitude."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
MEAN_DIR = ROOT / "reports/latentfm_xverse_nuisance_condition_means_20260624"
ANCHOR_JSON = MEAN_DIR / "condition_family_eval_anchor_internal_means_ode20.json"
CAP120_JSON = MEAN_DIR / "condition_family_eval_cap120_internal_means_ode20.json"
OUT_JSON = ROOT / "reports/latentfm_xverse_conservative_update_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_CONSERVATIVE_UPDATE_GATE_20260624.md"
GROUP = "family_gene"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def vec(row: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(row[key], dtype=np.float64)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    aa = a - float(np.mean(a))
    bb = b - float(np.mean(b))
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(aa, bb) / den)


def condition_rows(obj: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["dataset"]), str(row["condition"])): row for row in obj["groups"][GROUP]["condition_metrics"]}


def summarize_alpha(keys: list[tuple[str, str]], anchor_rows: dict[tuple[str, str], dict[str, Any]], cap_rows: dict[tuple[str, str], dict[str, Any]], alpha: float) -> dict[str, Any]:
    by_ds_pp: dict[str, list[float]] = {}
    by_ds_resid: dict[str, list[float]] = {}
    for key in keys:
        ds, _ = key
        a = anchor_rows[key]
        c = cap_rows[key]
        pred = vec(a, "pred_mean") + float(alpha) * (vec(c, "pred_mean") - vec(a, "pred_mean"))
        gt = vec(a, "gt_mean")
        pp = corr(pred, gt)
        resid = float(np.linalg.norm(pred - gt) / np.sqrt(pred.size))
        by_ds_pp.setdefault(ds, []).append(pp)
        by_ds_resid.setdefault(ds, []).append(resid)
    ds_pp = {ds: float(np.nanmean(vals)) for ds, vals in by_ds_pp.items()}
    ds_resid = {ds: float(np.nanmean(vals)) for ds, vals in by_ds_resid.items()}
    return {
        "alpha": alpha,
        "mean_pp": float(np.nanmean(list(ds_pp.values()))),
        "mean_residual": float(np.nanmean(list(ds_resid.values()))),
        "dataset_pp": ds_pp,
        "dataset_residual": ds_resid,
    }


def main() -> int:
    anchor = load_json(ANCHOR_JSON)
    cap120 = load_json(CAP120_JSON)
    anchor_rows = condition_rows(anchor)
    cap_rows = condition_rows(cap120)
    keys = sorted(set(anchor_rows) & set(cap_rows))
    summaries = [summarize_alpha(keys, anchor_rows, cap_rows, alpha) for alpha in ALPHAS]
    anchor_summary = summaries[0]
    cap_summary = summaries[-1]
    rows = []
    for item in summaries:
        pp_delta_anchor = item["mean_pp"] - anchor_summary["mean_pp"]
        pp_delta_cap = item["mean_pp"] - cap_summary["mean_pp"]
        resid_delta_anchor = item["mean_residual"] - anchor_summary["mean_residual"]
        resid_delta_cap = item["mean_residual"] - cap_summary["mean_residual"]
        ds_pp_delta = {
            ds: item["dataset_pp"][ds] - anchor_summary["dataset_pp"][ds]
            for ds in item["dataset_pp"]
        }
        rows.append(
            {
                "alpha": item["alpha"],
                "mean_pp": item["mean_pp"],
                "mean_residual": item["mean_residual"],
                "pp_delta_vs_anchor": pp_delta_anchor,
                "pp_delta_vs_cap120": pp_delta_cap,
                "residual_delta_vs_anchor": resid_delta_anchor,
                "residual_delta_vs_cap120": resid_delta_cap,
                "dataset_pp_delta_min_vs_anchor": float(min(ds_pp_delta.values())),
            }
        )
    candidates = [
        row
        for row in rows
        if 0.0 < row["alpha"] < 1.0
        and row["pp_delta_vs_anchor"] >= 0.005
        and row["dataset_pp_delta_min_vs_anchor"] >= -0.02
        and row["residual_delta_vs_cap120"] <= -0.00005
    ]
    best = max(candidates, key=lambda r: (r["pp_delta_vs_anchor"], -r["mean_residual"]), default=None)
    reasons = []
    if best is None:
        reasons.append("no_interpolation_preserves_internal_pp_gain_while_reducing_residual_proxy")
    decision = {
        "status": "latentfm_xverse_conservative_update_gate_pass_code_gate_next_no_gpu"
        if best is not None
        else "latentfm_xverse_conservative_update_gate_fail_no_gpu",
        "gpu_authorization": "none",
        "action": "design_conservative_update_smoke_only_if_pass_else_close",
        "best_alpha": None if best is None else best["alpha"],
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
        "rows": rows,
        "decision": decision,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# LatentFM xverse Conservative Update Gate",
        "",
        f"Status: `{decision['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- Uses only train/internal condition means for anchor and cap120.",
        "- Does not read canonical outcomes, canonical multi, held-out query, active logs, or GPU artifacts.",
        "- Tests whether prediction-level conservative interpolation suggests a useful update-magnitude mechanism.",
        "",
        "## Rows",
        "",
        "| alpha | pp delta vs anchor | pp delta vs cap120 | residual delta vs anchor | residual delta vs cap120 | dataset pp min vs anchor |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['alpha']:.2f} | {row['pp_delta_vs_anchor']:+.6f} | "
            f"{row['pp_delta_vs_cap120']:+.6f} | {row['residual_delta_vs_anchor']:+.6f} | "
            f"{row['residual_delta_vs_cap120']:+.6f} | {row['dataset_pp_delta_min_vs_anchor']:+.6f} |"
        )
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend([f"- `{r}`" for r in reasons] or ["- none"])
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "gpu_authorization": "none", "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
