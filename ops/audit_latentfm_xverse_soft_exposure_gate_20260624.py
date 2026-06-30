#!/usr/bin/env python3
"""CPU-only gate for soft exposure weighting on the cap120 split.

The gate simulates existing sampler semantics only. It does not train, read
canonical outputs, or change any split.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
COUPLED = ROOT / "CoupledFM"
sys.path.insert(0, str(COUPLED))

from model.latent.dataset import CrossDatasetFMDataset  # noqa: E402


DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
SPLIT_FILE = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
OUT_JSON = ROOT / "reports/latentfm_xverse_soft_exposure_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_SOFT_EXPOSURE_GATE_20260624.md"


SPECS = [
    {"name": "softvisit_p090_no_cap", "ds_alpha": 0.7, "condition_visit_power": 0.90, "condition_visit_cap": 0},
    {"name": "softvisit_p085_no_cap", "ds_alpha": 0.7, "condition_visit_power": 0.85, "condition_visit_cap": 0},
    {"name": "softvisit_p075_no_cap", "ds_alpha": 0.7, "condition_visit_power": 0.75, "condition_visit_cap": 0},
    {"name": "softvisit_p050_no_cap", "ds_alpha": 0.7, "condition_visit_power": 0.50, "condition_visit_cap": 0},
]


def _entropy(vals: list[float]) -> float:
    total = sum(vals)
    if total <= 0:
        return float("nan")
    ps = [v / total for v in vals if v > 0]
    if len(ps) <= 1:
        return 0.0
    return float(-sum(p * math.log(p) for p in ps) / math.log(len(ps)))


def _gini(vals: list[float]) -> float:
    vals = sorted(float(v) for v in vals if v >= 0)
    n = len(vals)
    if n == 0:
        return float("nan")
    total = sum(vals)
    if total <= 0:
        return 0.0
    return float((2.0 * sum((i + 1) * v for i, v in enumerate(vals)) / (n * total)) - (n + 1) / n)


def _simulate(ds: CrossDatasetFMDataset, *, ds_alpha: float, condition_visit_power: float, condition_visit_cap: int) -> dict[str, Any]:
    old_alpha = ds.ds_alpha
    old_power = ds.condition_visit_power
    old_cap = ds.condition_visit_cap
    ds.ds_alpha = float(ds_alpha)
    ds.condition_visit_power = float(condition_visit_power)
    ds.condition_visit_cap = int(condition_visit_cap)
    try:
        by_dataset: dict[str, dict[str, Any]] = {}
        cond_visits: list[int] = []
        total_steps = 0
        for name in ds.ds_names:
            sizes = ds._cond_sizes[name]
            n_eff = ds._n_eff(len(sizes))
            visits = [ds._condition_visits(n_gt) for _, n_gt in sizes.values()]
            selected_visits = sorted(visits, reverse=True)[:n_eff]
            steps = int(sum(selected_visits))
            total_steps += steps
            cond_visits.extend(selected_visits)
            by_dataset[name] = {
                "n_conditions": len(sizes),
                "n_eff": int(n_eff),
                "steps": steps,
                "avg_visit": float(statistics.fmean(selected_visits)) if selected_visits else 0.0,
                "max_visit": max(selected_visits) if selected_visits else 0,
            }
        shares = {k: v["steps"] / max(total_steps, 1) for k, v in by_dataset.items()}
        jiang_share = sum(v for k, v in shares.items() if k.startswith("Jiang_"))
        return {
            "ds_alpha": float(ds_alpha),
            "condition_visit_power": float(condition_visit_power),
            "condition_visit_cap": int(condition_visit_cap),
            "total_steps": total_steps,
            "dataset_entropy": _entropy(list(shares.values())),
            "condition_visit_gini": _gini(cond_visits),
            "max_dataset_share": max(shares.values()) if shares else 0.0,
            "jiang_share": jiang_share,
            "by_dataset": by_dataset,
            "shares": shares,
        }
    finally:
        ds.ds_alpha = old_alpha
        ds.condition_visit_power = old_power
        ds.condition_visit_cap = old_cap


def _delta(spec: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    out = dict(spec)
    out["delta_vs_base"] = {
        "total_steps_frac": spec["total_steps"] / max(base["total_steps"], 1),
        "dataset_entropy": spec["dataset_entropy"] - base["dataset_entropy"],
        "condition_visit_gini": spec["condition_visit_gini"] - base["condition_visit_gini"],
        "max_dataset_share": spec["max_dataset_share"] - base["max_dataset_share"],
        "jiang_share": spec["jiang_share"] - base["jiang_share"],
    }
    return out


def _decision(rows: list[dict[str, Any]], base: dict[str, Any]) -> dict[str, Any]:
    passed: list[str] = []
    rejected: dict[str, list[str]] = {}
    for row in rows:
        d = row["delta_vs_base"]
        reasons: list[str] = []
        if row["condition_visit_cap"] != 0:
            reasons.append("uses_hard_visit_cap")
        if d["total_steps_frac"] < 0.55:
            reasons.append("too_much_compute_or_exposure_removed")
        if d["dataset_entropy"] < -0.005:
            reasons.append("dataset_entropy_decreases")
        if d["condition_visit_gini"] > -0.020:
            reasons.append("condition_visit_gini_not_reduced")
        if d["max_dataset_share"] > 0.005:
            reasons.append("max_dataset_share_increases")
        if row["jiang_share"] < 0.60 * base["jiang_share"]:
            reasons.append("jiang_high_signal_share_not_preserved")
        if reasons:
            rejected[row["name"]] = reasons
        else:
            passed.append(row["name"])
    return {
        "status": "soft_exposure_gate_pass_no_gpu" if passed else "soft_exposure_gate_fail_no_gpu",
        "action": "launch_soft_exposure_smokes" if passed else "do_not_launch_soft_exposure",
        "passed_specs": passed,
        "rejected_specs": rejected,
    }


def main() -> int:
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    ds = CrossDatasetFMDataset(
        data_dir=str(DATA_DIR),
        split=split,
        batch_size=64,
        seed=42,
        mode="train",
        min_cells=32,
        ds_alpha=0.7,
        condition_visit_power=1.0,
        condition_visit_cap=0,
        silent=True,
    )
    base = _simulate(ds, ds_alpha=0.7, condition_visit_power=1.0, condition_visit_cap=0)
    rows = []
    for spec in SPECS:
        sim = _simulate(
            ds,
            ds_alpha=spec["ds_alpha"],
            condition_visit_power=spec["condition_visit_power"],
            condition_visit_cap=spec["condition_visit_cap"],
        )
        sim["name"] = spec["name"]
        rows.append(_delta(sim, base))
    decision = _decision(rows, base)
    payload = {
        "decision": decision,
        "boundary": {
            "source": "cap120 train split condition sizes only",
            "canonical_or_query_used": False,
            "split_file": str(SPLIT_FILE),
        },
        "baseline": base,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM xverse Soft Exposure Gate",
        "",
        "## Boundary",
        "",
        "- CPU-only simulation of existing sampler semantics on cap120 train split.",
        "- No canonical split, Track C query, model outputs, or training.",
        "- Candidate specs use continuous visit power and no hard visit cap.",
        "",
        "## Decision",
        "",
        f"Status: `{decision['status']}`",
        f"Action: `{decision['action']}`",
        "",
        "## Rows",
        "",
        "| spec | steps frac | entropy delta | visit gini delta | max dataset share delta | Jiang share | decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        d = row["delta_vs_base"]
        spec_decision = "pass" if row["name"] in decision["passed_specs"] else ",".join(decision["rejected_specs"].get(row["name"], []))
        lines.append(
            f"| {row['name']} | {d['total_steps_frac']:.3f} | {d['dataset_entropy']:+.4f} | "
            f"{d['condition_visit_gini']:+.4f} | {d['max_dataset_share']:+.4f} | "
            f"{row['jiang_share']:.4f} | {spec_decision} |"
        )
    lines.extend(
        [
            "",
            "## Baseline",
            "",
            f"- total steps: `{base['total_steps']}`",
            f"- dataset entropy: `{base['dataset_entropy']:.4f}`",
            f"- condition visit gini: `{base['condition_visit_gini']:.4f}`",
            f"- max dataset share: `{base['max_dataset_share']:.4f}`",
            f"- Jiang share: `{base['jiang_share']:.4f}`",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": decision["status"], "passed": decision["passed_specs"], "out_md": str(OUT_MD)}, indent=2))
    return 0 if decision["passed_specs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
