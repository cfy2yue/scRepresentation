#!/usr/bin/env python3
"""Deconfounded CPU proxy gate for SciPlex Morgan descriptor semantics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
BASE_SCRIPT = ROOT / "ops/audit_latentfm_sciplex_descriptor_proxy_gate_20260624.py"
OUT_JSON = ROOT / "reports/latentfm_sciplex_descriptor_deconfounded_proxy_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_SCIPLEX_DESCRIPTOR_DECONFOUNDED_PROXY_GATE_20260624.md"


def load_base():
    spec = importlib.util.spec_from_file_location("sciplex_descriptor_proxy_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fmt(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):+.6f}"
    except Exception:
        return str(v)


def build_xy(rows: list[dict[str, Any]], desc: dict[str, np.ndarray] | None, mod, *, use_background: bool) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    xs = []
    ys = []
    kept = []
    for row in rows:
        parts = []
        if desc is not None:
            d = desc.get(row["drug"])
            if d is None:
                continue
            parts.append(np.asarray(d, dtype=np.float64))
        if use_background:
            parts.append(mod.background_features(row["dataset"]))
        if not parts:
            parts.append(np.ones((1,), dtype=np.float64))
        xs.append(np.concatenate(parts))
        ys.append(row["delta"])
        kept.append(row)
    return np.vstack(xs), np.vstack(ys), kept


def run_model(train_rows: list[dict[str, Any]], hold_rows: list[dict[str, Any]], desc: dict[str, np.ndarray] | None, mod, *, use_background: bool) -> dict[str, Any]:
    train_x, train_y, train_kept = build_xy(train_rows, desc, mod, use_background=use_background)
    test_x, test_y, test_kept = build_xy(hold_rows, desc, mod, use_background=use_background)
    pred = mod.kernel_ridge_predict(train_x, train_y, test_x, mod.ALPHA)
    out = mod.eval_pred(pred, test_y, test_kept)
    out["n_train"] = len(train_kept)
    out["n_holdout"] = len(test_kept)
    return out


def delta(model: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    by_ds = {}
    for ds, val in (model.get("dataset_means") or {}).items():
        b = (baseline.get("dataset_means") or {}).get(ds)
        by_ds[ds] = None if val is None or b is None else float(val) - float(b)
    finite = [v for v in by_ds.values() if v is not None]
    return {
        "mean_pp_increment": None
        if model.get("mean_pp_proxy") is None or baseline.get("mean_pp_proxy") is None
        else float(model["mean_pp_proxy"]) - float(baseline["mean_pp_proxy"]),
        "dataset_increment_means": by_ds,
        "dataset_increment_min": min(finite, default=None),
    }


def main() -> int:
    mod = load_base()
    split = mod.load_json(mod.SPLIT)
    train_rows, hold_rows = mod.load_deltas(split)
    desc, scaffolds = mod.load_descriptors()
    bg_only = run_model(train_rows, hold_rows, None, mod, use_background=True)
    actual_bg = run_model(train_rows, hold_rows, desc, mod, use_background=True)
    actual_desc_only = run_model(train_rows, hold_rows, desc, mod, use_background=False)
    shuffled_bg = run_model(train_rows, hold_rows, mod.shuffled_desc(desc, seed=mod.SEED + 11), mod, use_background=True)
    random_bg = run_model(train_rows, hold_rows, mod.random_desc(desc, seed=mod.SEED + 12), mod, use_background=True)

    increments = {
        "actual_morgan_plus_background": delta(actual_bg, bg_only),
        "shuffled_descriptor_plus_background": delta(shuffled_bg, bg_only),
        "random_descriptor_plus_background": delta(random_bg, bg_only),
    }
    hold_scaffolds = {scaffolds.get(row["drug"], "") for row in hold_rows if scaffolds.get(row["drug"], "")}
    train_scaffolds = {scaffolds.get(row["drug"], "") for row in train_rows if scaffolds.get(row["drug"], "")}
    unseen_scaffolds = hold_scaffolds - train_scaffolds

    actual_inc = increments["actual_morgan_plus_background"]["mean_pp_increment"]
    control_inc = max(
        increments["shuffled_descriptor_plus_background"]["mean_pp_increment"] or 0.0,
        increments["random_descriptor_plus_background"]["mean_pp_increment"] or 0.0,
    )
    reasons = []
    if len(hold_rows) < 45:
        reasons.append("too_few_holdout_rows")
    if len(hold_scaffolds) < 30:
        reasons.append("too_few_holdout_scaffolds")
    if actual_inc is None or actual_inc < 0.010:
        reasons.append("actual_increment_over_background_lt_0p010")
    if actual_inc is None or actual_inc - control_inc < 0.010:
        reasons.append("actual_increment_not_0p010_above_control_increment")
    if control_inc >= 0.003:
        reasons.append("control_increment_not_collapsed_below_0p003")
    if increments["actual_morgan_plus_background"]["dataset_increment_min"] is None or increments["actual_morgan_plus_background"]["dataset_increment_min"] < -0.020:
        reasons.append("actual_increment_dataset_tail_below_minus_0p020")
    status = "sciplex_descriptor_deconfounded_proxy_gate_fail_no_gpu" if reasons else "sciplex_descriptor_deconfounded_proxy_gate_pass_external_review_next"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "train_only_sciplex_parent": True,
            "canonical_reference_excluded": True,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "summary": {
            "train_rows": len(train_rows),
            "holdout_rows": len(hold_rows),
            "holdout_scaffolds": len(hold_scaffolds),
            "unseen_holdout_scaffolds": len(unseen_scaffolds),
            "background_only": bg_only,
            "actual_morgan_plus_background": actual_bg,
            "actual_morgan_descriptor_only": actual_desc_only,
            "shuffled_descriptor_plus_background": shuffled_bg,
            "random_descriptor_plus_background": random_bg,
            "increments_vs_background_only": increments,
        },
        "reasons": reasons,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM SciPlex Descriptor Deconfounded Proxy Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only deconfounded descriptor proxy on train-only SciPlex condition means.",
        "- Canonical reference drugs are excluded by the holdout split.",
        "- Does not train, infer, launch GPU, read canonical multi, or read Track C query.",
        "",
        "## Summary",
        "",
        f"- train rows: `{len(train_rows)}`",
        f"- holdout rows: `{len(hold_rows)}`",
        f"- holdout scaffolds: `{len(hold_scaffolds)}`",
        f"- unseen holdout scaffolds: `{len(unseen_scaffolds)}`",
        "",
        "| model | mean pp proxy | dataset min | increment vs background | increment dataset min |",
        "|---|---:|---:|---:|---:|",
        f"| `background_only` | {fmt(bg_only['mean_pp_proxy'])} | {fmt(bg_only['dataset_min'])} | NA | NA |",
        f"| `actual_morgan_plus_background` | {fmt(actual_bg['mean_pp_proxy'])} | {fmt(actual_bg['dataset_min'])} | {fmt(increments['actual_morgan_plus_background']['mean_pp_increment'])} | {fmt(increments['actual_morgan_plus_background']['dataset_increment_min'])} |",
        f"| `shuffled_descriptor_plus_background` | {fmt(shuffled_bg['mean_pp_proxy'])} | {fmt(shuffled_bg['dataset_min'])} | {fmt(increments['shuffled_descriptor_plus_background']['mean_pp_increment'])} | {fmt(increments['shuffled_descriptor_plus_background']['dataset_increment_min'])} |",
        f"| `random_descriptor_plus_background` | {fmt(random_bg['mean_pp_proxy'])} | {fmt(random_bg['dataset_min'])} | {fmt(increments['random_descriptor_plus_background']['mean_pp_increment'])} | {fmt(increments['random_descriptor_plus_background']['dataset_increment_min'])} |",
        f"| `actual_morgan_descriptor_only` | {fmt(actual_desc_only['mean_pp_proxy'])} | {fmt(actual_desc_only['dataset_min'])} | NA | NA |",
        "",
        "## Decision",
        "",
        f"- reasons: `{reasons}`",
        "- GPU authorized: `False`",
        "- A pass would require external review before descriptor-cache training.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "reasons": reasons}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
