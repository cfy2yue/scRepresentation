#!/usr/bin/env python3
"""Track C train-only memory readout CPU gate.

This is a stricter follow-up to the support-memory diagnostic. It predicts
support-val multi conditions from Track C ``train_multi`` memories only, so the
support-val conditions remain clean for selection. It reads no held-out query,
canonical posthoc, tmux/log state, or GPU candidate output.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
SUPPORT_MODULE_PATH = ROOT / "ops/audit_latentfm_trackc_support_route_readiness_20260622.py"
MEMORY_MODULE_PATH = ROOT / "ops/audit_latentfm_trackc_support_memory_readout_gate_20260622.py"
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2.json"
DEFAULT_TRAINSELECT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
DEFAULT_PERT_MEANS = (
    ROOT
    / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/"
    "xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz"
)
BASELINES = (
    "support_selected_route",
    "dataset_multi_mean",
    "global_multi_mean",
    "additive_single_mean",
    "additive_single_sum",
    "dataset_single_mean",
)


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def evaluate_trainonly_memory(
    val_rows: list[dict[str, Any]],
    train_memory: list[dict[str, Any]],
    pert_means: dict[str, np.ndarray],
    support: Any,
    memory: Any,
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for target in val_rows:
        scored = {
            "dataset": target["dataset"],
            "condition": target["condition"],
            "genes": target["genes"],
            "nperts": target["nperts"],
            "group": "support_val_multi_trainonly_memory",
        }
        for spec in specs:
            pred = memory.weighted_memory_prediction(
                target,
                train_memory,
                mode=spec["mode"],
                k=spec["k"],
                same_dataset=bool(spec["same_dataset"]),
                min_score=float(spec["min_score"]),
            )
            scored[spec["name"]] = None if pred is None else support.pp_score(target, pred, pert_means)
        out.append(scored)
    return out


def add_trainonly_mmd(
    rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    train_memory: list[dict[str, Any]],
    selected_spec: dict[str, Any],
    single: dict[str, Any],
    multi: dict[str, Any],
    support: Any,
    memory: Any,
) -> None:
    row_by_key = {memory.condition_key(row): row for row in rows}
    for target in val_rows:
        key = memory.condition_key(target)
        scored = row_by_key[key]
        pred = memory.weighted_memory_prediction(
            target,
            train_memory,
            mode=selected_spec["mode"],
            k=selected_spec["k"],
            same_dataset=bool(selected_spec["same_dataset"]),
            min_score=float(selected_spec["min_score"]),
        )
        if pred is not None:
            for metric, value in support.mmd_scores(target, pred).items():
                scored[f"{selected_spec['name']}__{metric}"] = value
        for name, base_pred in support.predict_baselines(target, single, multi).items():
            for metric, value in support.mmd_scores(target, base_pred).items():
                scored[f"{name}__{metric}"] = value


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    selected = payload["selected_model"]
    pp = {
        row["baseline"]: row
        for row in payload.get("paired_deltas") or []
        if row.get("candidate") == selected and row.get("status") == "ok"
    }
    mmd = {
        row["baseline"]: row
        for row in payload.get("paired_mmd_deltas") or []
        if row.get("candidate") == selected and row.get("status") == "ok"
    }
    reasons: list[str] = []
    route = pp.get("support_selected_route", {})
    if float(route.get("delta_mean") or 0.0) < 0.02:
        reasons.append("trainonly_memory_not_materially_better_than_support_route")
    if float(route.get("p_harm") if route.get("p_harm") is not None else 1.0) > 0.20:
        reasons.append("trainonly_memory_harm_risk_vs_support_route")
    by_ds = route.get("by_dataset") or {}
    if by_ds and any(float(value) < -0.02 for value in by_ds.values()):
        reasons.append("trainonly_memory_negative_dataset_delta_vs_support_route")
    for baseline in ("dataset_multi_mean", "additive_single_sum"):
        row = pp.get(baseline, {})
        if row.get("status") != "ok":
            reasons.append(f"{baseline}_comparison_missing")
        elif float(row.get("delta_mean") or 0.0) < 0.0 and float(row.get("p_harm") or 1.0) > 0.20:
            reasons.append(f"trainonly_memory_harm_vs_{baseline}")
    for baseline in ("support_selected_route", "dataset_multi_mean", "additive_single_sum"):
        row = mmd.get(baseline, {})
        if row.get("status") != "ok":
            reasons.append(f"{baseline}_mmd_comparison_missing")
        elif float(row.get("p_harm") if row.get("p_harm") is not None else 1.0) > 0.80:
            reasons.append(f"{baseline}_mmd_hard_harm")
    status = "trainonly_memory_cpu_gate_pass_protocol_candidate" if not reasons else "trainonly_memory_cpu_gate_fail"
    action = (
        "eligible_for_memory_transfer_protocol_review_after_latest_gate"
        if not reasons
        else "do_not_launch_memory_transfer_gpu_branch"
    )
    return {
        "status": status,
        "action": action,
        "reasons": reasons,
        "rules": [
            "selected train-only memory readout pp delta vs support_selected_route >= +0.02",
            "selected train-only memory readout pp p_harm vs support_selected_route <= 0.20",
            "no dataset-level pp delta vs support_selected_route below -0.02",
            "no hard MMD harm vs support_selected_route, dataset_multi_mean, or additive_single_sum",
            "held-out query is not read and cannot tune the selected readout",
        ],
    }


def render(payload: dict[str, Any]) -> str:
    selected = payload["selected_model"]
    lines = [
        "# LatentFM Track C Train-Only Memory Readout CPU Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
        "",
        "## Provenance",
        "",
        f"- split_file: `{payload['split_file']}`",
        f"- trainselect_split_file: `{payload['trainselect_split_file']}`",
        f"- data_dir: `{payload['data_dir']}`",
        f"- leakage_status: `{payload['leakage_status']}`",
        f"- train_multi rows: `{payload['n_train_multi_rows']}`",
        f"- support_val rows: `{payload['n_support_val_rows']}`",
        f"- selected readout: `{selected}`",
        "",
        "## Absolute Support-Val Scores",
        "",
        "| model | equal-dataset pp | equal-dataset MMD clamped |",
        "|---|---:|---:|",
    ]
    for row in payload["absolute_scores"]:
        lines.append(f"| `{row['model']}` | {fmt(row['pp'])} | {fmt(row.get('mmd_clamped'))} |")
    lines.extend(
        [
            "",
            "## Dataset Breakdown",
            "",
            "| dataset | n | selected pp | route pp | dataset_multi pp | additive_sum pp | selected MMD | route MMD |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["dataset_breakdown"]:
        lines.append(
            f"| {row['dataset']} | {row['n_conditions']} | {fmt(row.get(selected))} | "
            f"{fmt(row.get('support_selected_route'))} | {fmt(row.get('dataset_multi_mean'))} | "
            f"{fmt(row.get('additive_single_sum'))} | {fmt(row.get(f'{selected}_mmd_clamped'))} | "
            f"{fmt(row.get('support_selected_route_mmd_clamped'))} |"
        )
    lines.extend(
        [
            "",
            "## Paired PP Deltas",
            "",
            "| candidate | baseline | n cond | n ds | delta | 95% CI | p improve | p harm | dataset deltas |",
            "|---|---|---:|---:|---:|---|---:|---:|---|",
        ]
    )
    for row in payload["paired_deltas"]:
        ci = row.get("ci95") or [None, None]
        by_ds = ", ".join(f"{k}:{fmt(v)}" for k, v in (row.get("by_dataset") or {}).items())
        lines.append(
            f"| `{row['candidate']}` | `{row['baseline']}` | {row.get('n_conditions', 0)} | "
            f"{row.get('n_datasets', 0)} | {fmt(row.get('delta_mean'))} | "
            f"[{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(row.get('p_improve'))} | "
            f"{fmt(row.get('p_harm'))} | {by_ds} |"
        )
    lines.extend(
        [
            "",
            "## Paired MMD Deltas",
            "",
            "Lower MMD is better; delta is candidate minus baseline.",
            "",
            "| candidate | baseline | n cond | n ds | delta | 95% CI | p improve | p harm | dataset deltas |",
            "|---|---|---:|---:|---:|---|---:|---:|---|",
        ]
    )
    for row in payload["paired_mmd_deltas"]:
        ci = row.get("ci95") or [None, None]
        by_ds = ", ".join(f"{k}:{fmt(v)}" for k, v in (row.get("by_dataset") or {}).items())
        lines.append(
            f"| `{row['candidate']}` | `{row['baseline']}` | {row.get('n_conditions', 0)} | "
            f"{row.get('n_datasets', 0)} | {fmt(row.get('delta_mean'))} | "
            f"[{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(row.get('p_improve'))} | "
            f"{fmt(row.get('p_harm'))} | {by_ds} |"
        )
    lines.extend(["", "## Gate Reasons", ""])
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{reason}`" for reason in reasons] if reasons else ["- none"])
    lines.extend(["", "## Rules", ""])
    lines.extend(f"- {rule}" for rule in payload["decision"].get("rules") or [])
    lines.extend(
        [
            "",
            "## Consequence",
            "",
            "- This gate does not authorize GPU while latest-checkpoint posthoc is running.",
            "- A fail closes train-only memory transfer as the next GPU branch.",
            "- A pass only authorizes protocol review; held-out query remains unavailable for tuning.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--trainselect-split-file", type=Path, default=DEFAULT_TRAINSELECT_SPLIT)
    parser.add_argument("--pert-means-file", type=Path, default=DEFAULT_PERT_MEANS)
    parser.add_argument("--max-cells-per-condition", type=int, default=256)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    support = import_module("trackc_support_route_readiness", SUPPORT_MODULE_PATH)
    memory = import_module("trackc_support_memory_readout", MEMORY_MODULE_PATH)
    data_dir = args.data_dir.resolve()
    split = support.load_json(args.split_file)
    trainselect = support.load_json(args.trainselect_split_file)
    manifest = support.load_json(data_dir / "manifest.json")
    metadata = support.load_json(Path(manifest["condition_metadata_file"]))
    pert_means = {k: v.astype(np.float32) for k, v in np.load(args.pert_means_file).items()}

    train_rows = support.collect_role_rows(
        data_dir,
        split,
        metadata,
        "train_multi",
        max_cells=args.max_cells_per_condition,
    )
    val_rows = support.collect_role_rows(
        data_dir,
        split,
        metadata,
        "support_val_multi",
        max_cells=args.max_cells_per_condition,
    )
    trainselect_test = {
        (ds, str(cond))
        for ds, obj in trainselect.items()
        for cond in obj.get("test") or []
    }
    val_keys = {(str(row["dataset"]), str(row["condition"])) for row in val_rows}
    split_guard = {
        "support_val_matches_trainselect_test": val_keys == trainselect_test,
        "n_trainselect_test": len(trainselect_test),
        "n_support_val_rows": len(val_keys),
    }
    single = support.train_single_components(
        data_dir,
        split,
        metadata,
        max_cells=args.max_cells_per_condition,
    )
    multi = support.train_multi_components(train_rows)
    base_rows = memory.add_baseline_scores(val_rows, single, multi, pert_means, support)
    specs = memory.candidate_specs()
    memory_rows = evaluate_trainonly_memory(val_rows, train_rows, pert_means, support, memory, specs)
    selection = memory.select_candidate(memory_rows, specs)
    selected = selection["selected_model"]
    rows = memory.merge_rows(base_rows, memory_rows)
    selected_spec = {spec["name"]: spec for spec in specs}[selected]
    add_trainonly_mmd(rows, val_rows, train_rows, selected_spec, single, multi, support, memory)
    models = [selected, *BASELINES]
    absolute = [
        {
            "model": model,
            "pp": memory.equal_dataset_mean(rows, model),
            "mmd_clamped": memory.equal_dataset_mean(rows, f"{model}__test_mmd_clamped"),
        }
        for model in models
    ]
    by_dataset = memory.dataset_breakdown(rows, models)
    deltas = [
        memory.paired_bootstrap(rows, selected, baseline, metric="pp", n_boot=args.n_boot, seed=args.seed + i)
        for i, baseline in enumerate(BASELINES)
    ]
    mmd_deltas = [
        memory.paired_bootstrap(
            rows,
            selected,
            baseline,
            metric="mmd_clamped",
            n_boot=args.n_boot,
            seed=args.seed + 100 + i,
        )
        for i, baseline in enumerate(BASELINES)
    ]
    payload = {
        "data_dir": str(data_dir),
        "split_file": str(args.split_file),
        "trainselect_split_file": str(args.trainselect_split_file),
        "pert_means_file": str(args.pert_means_file),
        "max_cells_per_condition": args.max_cells_per_condition,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "leakage_status": "trackc_train_multi_memory_to_support_val_only_no_query_no_canonical_no_gpu_candidate",
        "split_guard": split_guard,
        "n_train_multi_rows": len(train_rows),
        "n_support_val_rows": len(val_rows),
        "selected_model": selected,
        "memory_scores": selection["memory_scores"],
        "absolute_scores": absolute,
        "dataset_breakdown": by_dataset,
        "paired_deltas": deltas,
        "paired_mmd_deltas": mmd_deltas,
        "condition_rows": rows,
    }
    payload["decision"] = decide(payload)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "selected_model": selected, "out_md": str(args.out_md)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
