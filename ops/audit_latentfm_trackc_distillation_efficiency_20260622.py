#!/usr/bin/env python3
"""Diagnose whether the Track C routed teacher was actually distilled.

This CPU diagnostic reads only:

* Track C train_multi/support_val_multi from the v2 split;
* existing support-val posthoc JSONs for anchor and candidate;
* train-only perturbation means.

It does not read held-out query_multi.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
SUPPORT_MODULE_PATH = ROOT / "ops/audit_latentfm_trackc_support_route_readiness_20260622.py"
DEFAULT_RUN_ROOT = (
    ROOT
    / "runs/latentfm_xverse_trackc_routed_distill_20260622/"
    "xverse_trackc_route_condprior_w05_replay1_2k_seed42"
)
DEFAULT_ANCHOR_POSTHOC = DEFAULT_RUN_ROOT / "posthoc_eval/support_anchor_split_ode20.json"
DEFAULT_CANDIDATE_POSTHOC = DEFAULT_RUN_ROOT / "posthoc_eval/support_candidate_split_ode20.json"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2.json"
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_PERT_MEANS = (
    ROOT
    / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/"
    "xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz"
)
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_trackc_distillation_efficiency_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_TRACKC_DISTILLATION_EFFICIENCY_20260622.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_support_module() -> Any:
    spec = importlib.util.spec_from_file_location("trackc_support_route_readiness", SUPPORT_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {SUPPORT_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def condition_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["condition"])


def posthoc_condition_metrics(path: Path, group: str) -> dict[tuple[str, str], dict[str, Any]]:
    payload = load_json(path)
    groups = payload.get("groups") or {}
    if group not in groups:
        raise KeyError(f"{path} has no group {group!r}; available={sorted(groups)}")
    out = {}
    for row in groups[group].get("condition_metrics") or []:
        out[condition_key(row)] = row
    return out


def equal_dataset_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = row.get(key)
        if val is not None:
            by_ds[str(row["dataset"])].append(float(val))
    vals = [float(np.mean(v)) for v in by_ds.values() if v]
    return None if not vals else float(np.mean(vals))


def dataset_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for ds in sorted({str(r["dataset"]) for r in rows}):
        ds_rows = [r for r in rows if str(r["dataset"]) == ds]
        item: dict[str, Any] = {"dataset": ds, "n_conditions": len(ds_rows)}
        for key in (
            "anchor_pp",
            "candidate_pp",
            "candidate_delta_pp",
            "route_pp",
            "route_delta_pp",
            "capture_ratio",
            "candidate_mmd_delta",
        ):
            vals = [float(r[key]) for r in ds_rows if r.get(key) is not None and np.isfinite(float(r[key]))]
            item[key] = None if not vals else float(np.mean(vals))
        out.append(item)
    return out


def paired_bootstrap(rows: list[dict[str, Any]], key: str, *, n_boot: int, seed: int) -> dict[str, Any]:
    vals_by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = row.get(key)
        if val is not None and np.isfinite(float(val)):
            vals_by_ds[str(row["dataset"])].append(float(val))
    datasets = sorted(ds for ds, vals in vals_by_ds.items() if vals)
    if not datasets:
        return {"status": "missing", "metric": key}
    point = float(np.mean([np.mean(vals_by_ds[d]) for d in datasets]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sample_ds = rng.choice(datasets, size=len(datasets), replace=True)
        means = []
        for ds in sample_ds:
            vals = np.asarray(vals_by_ds[str(ds)], dtype=np.float64)
            means.append(float(np.mean(rng.choice(vals, size=len(vals), replace=True))))
        boot.append(float(np.mean(means)))
    arr = np.asarray(boot, dtype=np.float64)
    return {
        "status": "ok",
        "metric": key,
        "n_conditions": int(sum(len(vals_by_ds[d]) for d in datasets)),
        "n_datasets": int(len(datasets)),
        "mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_gt_0": float(np.mean(arr > 0.0)),
        "p_lt_0": float(np.mean(arr < 0.0)),
        "p_ge_0p10": float(np.mean(arr >= 0.10)),
        "p_ge_0p25": float(np.mean(arr >= 0.25)),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = {m["metric"]: m for m in payload["bootstrap"] if m.get("status") == "ok"}
    support_delta = metrics.get("candidate_delta_pp", {})
    capture = metrics.get("capture_ratio", {})
    reasons = []
    if float(support_delta.get("mean") or 0.0) < 0.02:
        reasons.append("candidate_support_delta_below_0p02")
    if float(capture.get("mean") or 0.0) < 0.10:
        reasons.append("candidate_captures_less_than_10pct_of_positive_route_gap")
    if float(capture.get("p_ge_0p10") or 0.0) < 0.75:
        reasons.append("capture_ratio_not_bootstrap_supported")
    status = "distillation_inefficient_close_route_distill_mechanism" if reasons else "distillation_efficiency_gate_pass"
    action = (
        "do_not_rerun_same_route_distill; redesign_support_mechanism_or_keep_cpu_route_baseline"
        if reasons
        else "eligible_for_ema_consistent_adapter_smoke_design"
    )
    return {"status": status, "action": action, "reasons": reasons}


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Track C Distillation Efficiency Diagnostic",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
        "",
        "## Provenance",
        "",
        f"- split_file: `{payload['split_file']}`",
        f"- anchor_posthoc: `{payload['anchor_posthoc']}`",
        f"- candidate_posthoc: `{payload['candidate_posthoc']}`",
        f"- leakage_status: `{payload['leakage_status']}`",
        f"- joined support conditions: `{payload['n_joined_conditions']}`",
        "",
        "## Equal-Dataset Means",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in payload["equal_dataset_means"].items():
        lines.append(f"| `{key}` | {fmt(value)} |")
    lines += [
        "",
        "## Dataset Breakdown",
        "",
        "| dataset | n | anchor pp | candidate pp | candidate delta | route pp | route delta | capture ratio | candidate MMD delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["dataset_summary"]:
        lines.append(
            f"| {row['dataset']} | {row['n_conditions']} | {fmt(row.get('anchor_pp'))} | "
            f"{fmt(row.get('candidate_pp'))} | {fmt(row.get('candidate_delta_pp'))} | "
            f"{fmt(row.get('route_pp'))} | {fmt(row.get('route_delta_pp'))} | "
            f"{fmt(row.get('capture_ratio'))} | {fmt(row.get('candidate_mmd_delta'))} |"
        )
    lines += [
        "",
        "## Bootstrap",
        "",
        "| metric | n cond | n ds | mean | 95% CI | p > 0 | p < 0 | p >= 0.10 | p >= 0.25 |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in payload["bootstrap"]:
        ci = row.get("ci95") or [None, None]
        lines.append(
            f"| `{row['metric']}` | {row.get('n_conditions', 0)} | {row.get('n_datasets', 0)} | "
            f"{fmt(row.get('mean'))} | [{fmt(ci[0])}, {fmt(ci[1])}] | "
            f"{fmt(row.get('p_gt_0'))} | {fmt(row.get('p_lt_0'))} | "
            f"{fmt(row.get('p_ge_0p10'))} | {fmt(row.get('p_ge_0p25'))} |"
        )
    lines += ["", "## Gate Reasons", ""]
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{r}`" for r in reasons] if reasons else ["- none"])
    lines += [
        "",
        "## Interpretation",
        "",
        "- `route_delta_pp` is the train/support-only CPU route gap over the anchor on the same support-val conditions.",
        "- `candidate_delta_pp` is the GPU candidate posthoc gap over the anchor.",
        "- `capture_ratio = candidate_delta_pp / route_delta_pp` for conditions where the CPU route gap is positive.",
        "- This diagnostic does not evaluate Track C held-out query.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--pert-means-file", type=Path, default=DEFAULT_PERT_MEANS)
    parser.add_argument("--anchor-posthoc", type=Path, default=DEFAULT_ANCHOR_POSTHOC)
    parser.add_argument("--candidate-posthoc", type=Path, default=DEFAULT_CANDIDATE_POSTHOC)
    parser.add_argument("--group", default="test_multi")
    parser.add_argument("--max-cells-per-condition", type=int, default=256)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    support = load_support_module()
    split = support.load_json(args.split_file)
    manifest = support.load_json(args.data_dir / "manifest.json")
    metadata = support.load_json(Path(manifest["condition_metadata_file"]))
    pert_means = {k: v.astype(np.float32) for k, v in np.load(args.pert_means_file).items()}
    train_rows = support.collect_role_rows(
        args.data_dir,
        split,
        metadata,
        "train_multi",
        max_cells=args.max_cells_per_condition,
    )
    val_rows = support.collect_role_rows(
        args.data_dir,
        split,
        metadata,
        "support_val_multi",
        max_cells=args.max_cells_per_condition,
    )
    single = support.train_single_components(
        args.data_dir,
        split,
        metadata,
        max_cells=args.max_cells_per_condition,
    )
    multi = support.train_multi_components(train_rows)
    route_rows = support.evaluate(val_rows, single, multi, pert_means, compute_mmd=False)
    route_by_key = {condition_key(row): row for row in route_rows}
    anchor = posthoc_condition_metrics(args.anchor_posthoc, args.group)
    candidate = posthoc_condition_metrics(args.candidate_posthoc, args.group)

    joined = []
    for key in sorted(set(route_by_key) & set(anchor) & set(candidate)):
        ds, cond = key
        route_pp = route_by_key[key].get("support_selected_route")
        anchor_pp = anchor[key].get("pearson_pert")
        candidate_pp = candidate[key].get("pearson_pert")
        if route_pp is None or anchor_pp is None or candidate_pp is None:
            continue
        route_delta = float(route_pp) - float(anchor_pp)
        candidate_delta = float(candidate_pp) - float(anchor_pp)
        capture = None
        if route_delta > 1e-8:
            capture = candidate_delta / route_delta
        joined.append(
            {
                "dataset": ds,
                "condition": cond,
                "anchor_pp": float(anchor_pp),
                "candidate_pp": float(candidate_pp),
                "candidate_delta_pp": candidate_delta,
                "route_pp": float(route_pp),
                "route_delta_pp": route_delta,
                "capture_ratio": None if capture is None else float(capture),
                "anchor_mmd_clamped": anchor[key].get("test_mmd_clamped"),
                "candidate_mmd_clamped": candidate[key].get("test_mmd_clamped"),
                "candidate_mmd_delta": (
                    None
                    if anchor[key].get("test_mmd_clamped") is None or candidate[key].get("test_mmd_clamped") is None
                    else float(candidate[key]["test_mmd_clamped"]) - float(anchor[key]["test_mmd_clamped"])
                ),
            }
        )

    bootstrap = [
        paired_bootstrap(joined, "candidate_delta_pp", n_boot=args.n_boot, seed=args.seed),
        paired_bootstrap(joined, "route_delta_pp", n_boot=args.n_boot, seed=args.seed + 1),
        paired_bootstrap(joined, "capture_ratio", n_boot=args.n_boot, seed=args.seed + 2),
        paired_bootstrap(joined, "candidate_mmd_delta", n_boot=args.n_boot, seed=args.seed + 3),
    ]
    payload = {
        "data_dir": str(args.data_dir),
        "split_file": str(args.split_file),
        "pert_means_file": str(args.pert_means_file),
        "anchor_posthoc": str(args.anchor_posthoc),
        "candidate_posthoc": str(args.candidate_posthoc),
        "group": args.group,
        "max_cells_per_condition": args.max_cells_per_condition,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "leakage_status": "trackc_train_multi_and_support_val_only_no_query_multi_no_canonical_test",
        "n_train_multi_rows": len(train_rows),
        "n_support_val_multi_rows": len(val_rows),
        "n_joined_conditions": len(joined),
        "equal_dataset_means": {
            "anchor_pp": equal_dataset_mean(joined, "anchor_pp"),
            "candidate_pp": equal_dataset_mean(joined, "candidate_pp"),
            "candidate_delta_pp": equal_dataset_mean(joined, "candidate_delta_pp"),
            "route_pp": equal_dataset_mean(joined, "route_pp"),
            "route_delta_pp": equal_dataset_mean(joined, "route_delta_pp"),
            "capture_ratio": equal_dataset_mean(joined, "capture_ratio"),
            "candidate_mmd_delta": equal_dataset_mean(joined, "candidate_mmd_delta"),
        },
        "dataset_summary": dataset_summary(joined),
        "condition_rows": joined,
        "bootstrap": bootstrap,
    }
    payload["decision"] = decide(payload)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
