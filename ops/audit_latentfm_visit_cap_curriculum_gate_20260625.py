#!/usr/bin/env python3
"""CPU-only gate for LatentFM condition-visit curriculum knobs.

This audit does not train. It simulates the existing dataset sampler's expected
per-epoch exposure under the current baseline and a candidate visit-cap
curriculum, then decides whether the candidate is specific enough to justify a
future bounded GPU smoke.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import h5py


ROOT = Path("/data/cyx/1030/scLatent")
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
MANIFEST = DATA_DIR / "manifest.json"
SPLIT_FILE = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
BUDGET64_DECISION = ROOT / "reports/latentfm_true_cell_count_budget64_tail_stability_6k_decision_20260625.json"
TAIL_SENTINEL = ROOT / "reports/latentfm_scaling_provenance_tail_sentinel_gate_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_visit_cap_curriculum_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_VISIT_CAP_CURRICULUM_GATE_20260625.md"


BASELINE = {
    "name": "baseline_dsalpha0p7_visitpower1_cap0",
    "ds_alpha": 0.7,
    "condition_visit_power": 1.0,
    "condition_visit_cap": 0,
    "batch_size": 64,
}
CANDIDATE = {
    "name": "sublinear_visitpower0p5_cap2",
    "ds_alpha": 0.7,
    "condition_visit_power": 0.5,
    "condition_visit_cap": 2,
    "batch_size": 64,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def decode(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def n_eff(n: int, ds_alpha: float, min_selected: int = 0) -> int:
    if ds_alpha >= 1.0:
        base = n
    else:
        base = max(1, min(int(math.ceil(float(n) ** float(ds_alpha))), n))
    if min_selected > 0:
        base = max(base, min(min_selected, n))
    return base


def condition_visits(n_gt: int, *, batch_size: int, condition_visit_power: float, condition_visit_cap: int) -> int:
    visits = max(1, math.ceil(int(n_gt) / int(batch_size)))
    if float(condition_visit_power) != 1.0:
        visits = max(1, int(math.ceil(float(visits) ** float(condition_visit_power))))
    if int(condition_visit_cap) > 0:
        visits = min(visits, int(condition_visit_cap))
    return visits


def read_condition_sizes(manifest: dict[str, Any], split: dict[str, Any]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for ds, groups in sorted(split.items()):
        h5_path = Path(manifest["datasets"][ds]["out_path"])
        train_conds = {str(c) for c in groups.get("train", [])}
        with h5py.File(h5_path, "r") as handle:
            conds = [decode(x) for x in handle["conditions"][()]]
            offsets = [int(x) for x in handle["gt/offsets"][()]]
        sizes: dict[str, int] = {}
        for i, cond in enumerate(conds):
            if cond in train_conds:
                sizes[cond] = int(offsets[i + 1] - offsets[i])
        out[ds] = sizes
    return out


def summarize_config(condition_sizes: dict[str, dict[str, int]], cfg: dict[str, Any]) -> dict[str, Any]:
    ds_rows = []
    total_selected = 0
    total_train_conditions = 0
    total_expected_steps = 0.0
    high_visit_mass = 0.0
    high_visit_count = 0
    max_visit = 0
    for ds, sizes in sorted(condition_sizes.items()):
        n_train = len(sizes)
        selected = n_eff(n_train, float(cfg["ds_alpha"]))
        visits = [
            condition_visits(
                n,
                batch_size=int(cfg["batch_size"]),
                condition_visit_power=float(cfg["condition_visit_power"]),
                condition_visit_cap=int(cfg["condition_visit_cap"]),
            )
            for n in sizes.values()
        ]
        avg_visits = sum(visits) / max(len(visits), 1)
        expected_steps = selected * avg_visits
        high_visits = [v for v in visits if v >= 4]
        row = {
            "dataset": ds,
            "train_conditions": n_train,
            "selected_conditions_per_epoch": selected,
            "condition_coverage_fraction_per_epoch": selected / max(n_train, 1),
            "avg_visits": avg_visits,
            "max_visits": max(visits) if visits else 0,
            "expected_steps": expected_steps,
            "high_visit_condition_count_ge4": len(high_visits),
            "high_visit_mass_ge4": sum(high_visits),
        }
        ds_rows.append(row)
        total_selected += selected
        total_train_conditions += n_train
        total_expected_steps += expected_steps
        high_visit_mass += sum(high_visits)
        high_visit_count += len(high_visits)
        max_visit = max(max_visit, row["max_visits"])
    return {
        "config": cfg,
        "dataset_rows": ds_rows,
        "total_train_conditions": total_train_conditions,
        "selected_conditions_per_epoch": total_selected,
        "condition_coverage_fraction_per_epoch": total_selected / max(total_train_conditions, 1),
        "expected_epoch_steps": total_expected_steps,
        "high_visit_condition_count_ge4": high_visit_count,
        "high_visit_mass_ge4": high_visit_mass,
        "max_visits": max_visit,
    }


def negative_tail_datasets() -> dict[str, Any]:
    payload: dict[str, Any] = {"budget64_pp_tail_lt_minus_0p020": [], "sentinel_datasets": []}
    if BUDGET64_DECISION.exists():
        data = load_json(BUDGET64_DECISION)
        row = (data.get("matrix_summary") or {}).get("budget_rows", [{}])[0]
        tail = row.get("cross_background_pp_dataset_tail") or {}
        payload["budget64_pp_tail_lt_minus_0p020"] = [
            r["dataset"] for r in tail.get("dataset_rows", []) if float(r.get("mean", 0.0)) < -0.020
        ]
    if TAIL_SENTINEL.exists():
        data = load_json(TAIL_SENTINEL)
        sentinel = set()
        for sim in data.get("simulations", []):
            if not sim.get("pass_gate", False):
                sentinel.update(str(x) for x in sim.get("sentinel_datasets", []))
        payload["sentinel_datasets"] = sorted(sentinel)
        payload["negative_source_backgrounds"] = data.get("negative_source_backgrounds", [])
        payload["negative_source_types"] = data.get("negative_source_types", [])
    return payload


def row_by_dataset(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["dataset"]: r for r in summary["dataset_rows"]}


def decide(base: dict[str, Any], cand: dict[str, Any], tails: dict[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    reasons: list[str] = []
    metrics: dict[str, Any] = {}
    step_ratio = cand["expected_epoch_steps"] / max(base["expected_epoch_steps"], 1e-9)
    coverage_delta = cand["condition_coverage_fraction_per_epoch"] - base["condition_coverage_fraction_per_epoch"]
    high_mass_ratio = cand["high_visit_mass_ge4"] / max(base["high_visit_mass_ge4"], 1e-9)
    metrics.update(
        {
            "expected_epoch_step_ratio": step_ratio,
            "condition_coverage_delta": coverage_delta,
            "high_visit_mass_ratio": high_mass_ratio,
            "max_visit_baseline": base["max_visits"],
            "max_visit_candidate": cand["max_visits"],
        }
    )
    if abs(coverage_delta) > 1e-9:
        reasons.append("condition_coverage_changed")
    if step_ratio < 0.25:
        reasons.append("candidate_too_aggressive_expected_steps_below_25pct")
    if step_ratio > 0.85:
        reasons.append("candidate_too_weak_expected_steps_above_85pct")
    if high_mass_ratio > 0.50:
        reasons.append("high_visit_mass_not_reduced_by_at_least_50pct")
    risk_datasets = sorted(set(tails.get("budget64_pp_tail_lt_minus_0p020", [])) | set(tails.get("sentinel_datasets", [])))
    base_rows = row_by_dataset(base)
    cand_rows = row_by_dataset(cand)
    risk_rows = []
    reductions = []
    for ds in risk_datasets:
        if ds not in base_rows or ds not in cand_rows:
            continue
        b = float(base_rows[ds]["expected_steps"])
        c = float(cand_rows[ds]["expected_steps"])
        ratio = c / max(b, 1e-9)
        reductions.append(1.0 - ratio)
        risk_rows.append({"dataset": ds, "baseline_steps": b, "candidate_steps": c, "ratio": ratio})
    metrics["risk_dataset_rows"] = risk_rows
    metrics["risk_dataset_mean_reduction"] = sum(reductions) / max(len(reductions), 1) if reductions else None
    if reductions and metrics["risk_dataset_mean_reduction"] < 0.20:
        reasons.append("risk_dataset_expected_exposure_reduction_lt_20pct")
    status = "visit_cap_curriculum_gate_pass_one_bounded_smoke_candidate" if not reasons else "visit_cap_curriculum_gate_fail_no_gpu"
    return status, reasons, metrics


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Visit-Cap Curriculum Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only sampler exposure simulation.",
        "- Does not train, infer, read canonical multi, read held-out Track C query, or use GPU.",
        "- A pass here can only authorize preparing one bounded train-only smoke after GPU capacity frees.",
        "",
        "## Configs",
        "",
        f"- baseline: `{payload['baseline']['config']}`",
        f"- candidate: `{payload['candidate']['config']}`",
        "",
        "## Summary Metrics",
        "",
        f"- expected epoch step ratio: `{payload['decision_metrics']['expected_epoch_step_ratio']:.4f}`",
        f"- condition coverage delta: `{payload['decision_metrics']['condition_coverage_delta']:+.6f}`",
        f"- high-visit mass ratio: `{payload['decision_metrics']['high_visit_mass_ratio']:.4f}`",
        f"- max visits baseline/candidate: `{payload['decision_metrics']['max_visit_baseline']}` / `{payload['decision_metrics']['max_visit_candidate']}`",
        f"- risk dataset mean exposure reduction: `{payload['decision_metrics']['risk_dataset_mean_reduction']}`",
        "",
        "## Decision",
        "",
        f"- GPU authorized now: `{payload['gpu_authorized']}`",
        f"- reasons: `{payload['reasons']}`",
        "",
        "## Risk Dataset Rows",
        "",
        "| dataset | baseline steps | candidate steps | ratio |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["decision_metrics"]["risk_dataset_rows"]:
        lines.append(f"| `{row['dataset']}` | {row['baseline_steps']:.2f} | {row['candidate_steps']:.2f} | {row['ratio']:.3f} |")
    lines.extend(["", "## JSON", "", f"`{OUT_JSON}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    manifest = load_json(MANIFEST)
    split = load_json(SPLIT_FILE)
    condition_sizes = read_condition_sizes(manifest, split)
    base = summarize_config(condition_sizes, BASELINE)
    cand = summarize_config(condition_sizes, CANDIDATE)
    tails = negative_tail_datasets()
    status, reasons, decision_metrics = decide(base, cand, tails)
    payload = {
        "status": status,
        "gpu_authorized": False,
        "future_gpu_candidate_if_capacity_frees": status.endswith("one_bounded_smoke_candidate"),
        "reasons": reasons,
        "boundary": {
            "cpu_only": True,
            "gpu": False,
            "training_or_inference": False,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
        },
        "inputs": {
            "manifest": str(MANIFEST),
            "split_file": str(SPLIT_FILE),
            "budget64_decision": str(BUDGET64_DECISION),
            "tail_sentinel": str(TAIL_SENTINEL),
        },
        "baseline": base,
        "candidate": cand,
        "tails": tails,
        "decision_metrics": decision_metrics,
        "next_action": (
            "prepare one bounded train-only smoke launcher after current GPU block frees"
            if status.endswith("one_bounded_smoke_candidate")
            else "do not launch visit-cap GPU; use this as sampler diagnostic only"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
