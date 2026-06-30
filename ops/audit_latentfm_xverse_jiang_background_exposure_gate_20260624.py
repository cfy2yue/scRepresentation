#!/usr/bin/env python3
"""Audit Jiang condition-level cell-background exposure for xverse splits."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import anndata as ad


ROOT = Path("/data/cyx/1030/scLatent")
BIFLOW = ROOT / "dataset/biFlow_data"
SPLIT_DIR = BIFLOW / "xverse_scaling_splits_v2_20260624"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_jiang_background_exposure_gate_20260624.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_JIANG_BACKGROUND_EXPOSURE_GATE_20260624.md"

JIANG_DATASETS = (
    "Jiang_IFNB",
    "Jiang_IFNG",
    "Jiang_INS",
    "Jiang_TGFB",
    "Jiang_TNFA",
)

ARMS = {
    "cap120_all": SPLIT_DIR / "split_seed42_xverse_trainonly_scaling_cap120_all_v2.json",
    "type_balanced_cap120": SPLIT_DIR / "split_seed42_xverse_trainonly_scaling_type_balanced_cap120_v2.json",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def entropy(counts: dict[str, int]) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    out = 0.0
    for value in counts.values():
        p = float(value) / total
        if p > 0:
            out -= p * math.log(p)
    return out


def normalized_entropy(counts: dict[str, int]) -> float:
    n = sum(1 for v in counts.values() if v > 0)
    if n <= 1:
        return 0.0
    return entropy(counts) / math.log(n)


def obs_counts(dataset: str, biflow_dir: Path) -> dict[str, dict[str, int]]:
    path = biflow_dir / "gt_stack" / f"{dataset}.h5ad"
    a = ad.read_h5ad(path, backed="r")
    try:
        obs = a.obs[["perturbation", "cell_type"]].copy()
        obs["perturbation"] = obs["perturbation"].astype(str)
        obs["cell_type"] = obs["cell_type"].astype(str)
        out: dict[str, dict[str, int]] = {}
        grouped = obs.groupby(["perturbation", "cell_type"], observed=True).size()
        for (cond, bg), count in grouped.items():
            out.setdefault(str(cond), {})[str(bg)] = int(count)
        return out
    finally:
        a.file.close()


def summarize_group(
    conditions: list[str],
    counts_by_cond: dict[str, dict[str, int]],
) -> dict[str, Any]:
    bg_cells: Counter[str] = Counter()
    dominant_bg: Counter[str] = Counter()
    missing = []
    per_condition = []
    for cond in conditions:
        counts = counts_by_cond.get(cond)
        if not counts:
            missing.append(cond)
            continue
        bg_cells.update(counts)
        dom = max(counts.items(), key=lambda kv: kv[1])[0]
        dominant_bg[dom] += 1
        per_condition.append(
            {
                "condition": cond,
                "total_cells": int(sum(counts.values())),
                "n_backgrounds": int(sum(1 for v in counts.values() if v > 0)),
                "dominant_background": dom,
                "dominant_fraction": float(max(counts.values()) / max(1, sum(counts.values()))),
                "background_counts": dict(sorted(counts.items())),
            }
        )
    total_cells = sum(bg_cells.values())
    max_bg_share = 0.0 if total_cells <= 0 else max(bg_cells.values() or [0]) / float(total_cells)
    return {
        "n_conditions_requested": len(conditions),
        "n_conditions_found": len(per_condition),
        "n_missing_conditions": len(missing),
        "missing_conditions": missing[:20],
        "total_gt_cells": int(total_cells),
        "background_cell_counts": dict(sorted(bg_cells.items())),
        "dominant_background_condition_counts": dict(sorted(dominant_bg.items())),
        "n_backgrounds_with_cells": int(sum(1 for v in bg_cells.values() if v > 0)),
        "max_background_cell_share": float(max_bg_share),
        "normalized_entropy": float(normalized_entropy(dict(bg_cells))),
        "per_condition_preview": per_condition[:12],
    }


def summarize_arm(arm: str, split_file: Path, biflow_dir: Path) -> dict[str, Any]:
    split = read_json(split_file)
    rows = []
    aggregate_train_cells: Counter[str] = Counter()
    aggregate_eval_cells: Counter[str] = Counter()
    for ds in JIANG_DATASETS:
        counts = obs_counts(ds, biflow_dir)
        groups = split.get(ds) or {}
        train = [str(c) for c in groups.get("train") or []]
        eval_conds = sorted(
            {
                str(c)
                for key in (
                    "test",
                    "test_single",
                    "internal_val_cross_background_seen_gene_proxy",
                    "internal_val_family_gene_proxy",
                )
                for c in (groups.get(key) or [])
            }
        )
        train_summary = summarize_group(train, counts)
        eval_summary = summarize_group(eval_conds, counts)
        aggregate_train_cells.update(train_summary["background_cell_counts"])
        aggregate_eval_cells.update(eval_summary["background_cell_counts"])
        rows.append(
            {
                "dataset": ds,
                "train": train_summary,
                "eval": eval_summary,
            }
        )
    train_total = sum(aggregate_train_cells.values())
    eval_total = sum(aggregate_eval_cells.values())
    return {
        "arm": arm,
        "split_file": str(split_file),
        "jiang_rows": rows,
        "aggregate_train_background_cell_counts": dict(sorted(aggregate_train_cells.items())),
        "aggregate_eval_background_cell_counts": dict(sorted(aggregate_eval_cells.items())),
        "aggregate_train_max_background_cell_share": (
            0.0 if train_total <= 0 else max(aggregate_train_cells.values() or [0]) / float(train_total)
        ),
        "aggregate_eval_max_background_cell_share": (
            0.0 if eval_total <= 0 else max(aggregate_eval_cells.values() or [0]) / float(eval_total)
        ),
        "aggregate_train_normalized_entropy": normalized_entropy(dict(aggregate_train_cells)),
        "aggregate_eval_normalized_entropy": normalized_entropy(dict(aggregate_eval_cells)),
        "aggregate_train_total_cells": int(train_total),
        "aggregate_eval_total_cells": int(eval_total),
    }


def decide(arms: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {row["arm"]: row for row in arms}
    reasons = []
    for arm, row in by_arm.items():
        if row["aggregate_train_total_cells"] <= 0:
            reasons.append(f"{arm}:no_train_cells")
        if row["aggregate_train_max_background_cell_share"] > 0.55:
            reasons.append(f"{arm}:train_background_max_share_gt_0p55")
        if row["aggregate_train_normalized_entropy"] < 0.80:
            reasons.append(f"{arm}:train_background_entropy_lt_0p80")
    delta = None
    if "cap120_all" in by_arm and "type_balanced_cap120" in by_arm:
        delta = (
            float(by_arm["type_balanced_cap120"]["aggregate_train_max_background_cell_share"])
            - float(by_arm["cap120_all"]["aggregate_train_max_background_cell_share"])
        )
        if delta > 0.02:
            reasons.append("type_balanced_worsens_jiang_background_max_share_gt_0p02")
    return {
        "status": "jiang_background_exposure_gate_pass" if not reasons else "jiang_background_exposure_gate_diagnostic",
        "reasons": reasons,
        "type_balanced_minus_cap120_train_max_background_share": delta,
        "thresholds": {
            "aggregate_train_max_background_cell_share": 0.55,
            "aggregate_train_normalized_entropy": 0.80,
            "type_balanced_worsen_margin": 0.02,
        },
    }


def fmt_float(x: Any) -> str:
    if x is None:
        return "NA"
    return f"{float(x):.4f}"


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Jiang Background Exposure Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only condition-level background exposure audit.",
        "- Reads h5ad `.obs[['perturbation', 'cell_type']]` for Jiang datasets only.",
        "- Does not read expression matrices, canonical outcomes, Track C query, active logs, or model outputs.",
        "",
        "## Arm Summary",
        "",
        "| arm | train cells | train max bg share | train entropy | eval cells | eval max bg share | eval entropy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["arms"]:
        lines.append(
            f"| `{row['arm']}` | {row['aggregate_train_total_cells']} | "
            f"{fmt_float(row['aggregate_train_max_background_cell_share'])} | "
            f"{fmt_float(row['aggregate_train_normalized_entropy'])} | "
            f"{row['aggregate_eval_total_cells']} | "
            f"{fmt_float(row['aggregate_eval_max_background_cell_share'])} | "
            f"{fmt_float(row['aggregate_eval_normalized_entropy'])} |"
        )
    lines += [
        "",
        "## Dataset Rows",
        "",
        "| arm | dataset | train conds | train cells | train max bg share | train entropy | eval conds | eval max bg share |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["arms"]:
        for ds_row in row["jiang_rows"]:
            train = ds_row["train"]
            ev = ds_row["eval"]
            lines.append(
                f"| `{row['arm']}` | `{ds_row['dataset']}` | "
                f"{train['n_conditions_found']} | {train['total_gt_cells']} | "
                f"{fmt_float(train['max_background_cell_share'])} | "
                f"{fmt_float(train['normalized_entropy'])} | "
                f"{ev['n_conditions_found']} | {fmt_float(ev['max_background_cell_share'])} |"
            )
    decision = payload["decision"]
    if decision["reasons"]:
        lines += ["", "Gate reasons:"]
        lines.extend(f"- `{reason}`" for reason in decision["reasons"])
    lines += [
        "",
        "## Interpretation",
        "",
        "- A pass means Jiang condition-level backgrounds are sufficiently broad for the current split family; it does not authorize GPU by itself.",
        "- A diagnostic status means a dedicated background-balanced split should be designed before launching a background-specific GPU branch.",
        "",
        "## JSON",
        "",
        f"`{payload['out_json']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--biflow-dir", type=Path, default=BIFLOW)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = ap.parse_args()

    arms = [summarize_arm(arm, path, args.biflow_dir) for arm, path in sorted(ARMS.items())]
    payload = {
        "biflow_dir": str(args.biflow_dir),
        "jiang_datasets": list(JIANG_DATASETS),
        "arms": arms,
        "decision": decide(arms),
        "out_json": str(args.out_json),
        "out_md": str(args.out_md),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
