#!/usr/bin/env python3
"""Track C episodic support-memory residual readout CPU gate.

This tests a nonparametric true-multi support mechanism before any new GPU
adapter work: can train/support multi residual memories directly improve
support-val predictions beyond the frozen support-selected route?

Leakage boundary: read only Track C train_multi/support_val_multi plus train
single baselines. Do not read query_multi, canonical test/posthoc, or candidate
model outcomes.
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
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2.json"
DEFAULT_TRAINSELECT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
DEFAULT_PERT_MEANS = (
    ROOT
    / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/"
    "xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz"
)
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_trackc_support_memory_readout_gate_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_MEMORY_READOUT_GATE_20260622.md"

BASELINES = (
    "support_selected_route",
    "dataset_multi_mean",
    "global_multi_mean",
    "additive_single_mean",
    "additive_single_sum",
    "dataset_single_mean",
)


def load_support_module() -> Any:
    spec = importlib.util.spec_from_file_location("trackc_support_route_readiness", SUPPORT_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {SUPPORT_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def gene_set(row: dict[str, Any]) -> set[str]:
    return {str(g).strip().upper() for g in row.get("genes") or [] if str(g).strip()}


def gene_jaccard(a: dict[str, Any], b: dict[str, Any]) -> float:
    ga = gene_set(a)
    gb = gene_set(b)
    if not ga and not gb:
        return 0.0
    return float(len(ga & gb) / max(len(ga | gb), 1))


def gene_overlap_count(a: dict[str, Any], b: dict[str, Any]) -> float:
    return float(len(gene_set(a) & gene_set(b)))


def gene_match_score(a: dict[str, Any], b: dict[str, Any], mode: str) -> float:
    if mode == "jaccard":
        return gene_jaccard(a, b)
    if mode == "overlap":
        return gene_overlap_count(a, b)
    raise ValueError(mode)


def weighted_memory_prediction(
    target: dict[str, Any],
    memory: list[dict[str, Any]],
    *,
    mode: str,
    k: int,
    same_dataset: bool,
    min_score: float,
) -> np.ndarray | None:
    candidates = []
    for row in memory:
        if same_dataset and str(row["dataset"]) != str(target["dataset"]):
            continue
        score = gene_match_score(target, row, mode)
        if score >= min_score:
            candidates.append((score, row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], str(item[1]["dataset"]), str(item[1]["condition"])), reverse=True)
    selected = candidates[: max(int(k), 1)]
    weights = np.asarray([max(s, 1e-6) for s, _ in selected], dtype=np.float64)
    weights = weights / weights.sum()
    arr = np.vstack([np.asarray(row["residual"], dtype=np.float32) for _, row in selected])
    return (weights[:, None] * arr).sum(axis=0).astype(np.float32)


def candidate_specs() -> list[dict[str, Any]]:
    specs = []
    for mode in ("jaccard", "overlap"):
        for same_dataset in (True, False):
            for k in (1, 3, 5, 9):
                for min_score in ((0.0, 0.25, 0.5) if mode == "jaccard" else (0.0, 1.0)):
                    specs.append(
                        {
                            "name": (
                                f"memory_{mode}_k{k}_"
                                f"{'same_ds' if same_dataset else 'all_ds'}_min{min_score:g}"
                            ),
                            "mode": mode,
                            "k": k,
                            "same_dataset": same_dataset,
                            "min_score": float(min_score),
                        }
                    )
    return specs


def condition_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["condition"])


def evaluate_memory_specs(
    val_rows: list[dict[str, Any]],
    train_memory: list[dict[str, Any]],
    pert_means: dict[str, np.ndarray],
    support_module: Any,
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for target in val_rows:
        memory = train_memory + [row for row in val_rows if condition_key(row) != condition_key(target)]
        scored = {
            "dataset": target["dataset"],
            "condition": target["condition"],
            "genes": target["genes"],
            "nperts": target["nperts"],
            "group": "support_val_multi_leave_one",
        }
        for spec in specs:
            pred = weighted_memory_prediction(
                target,
                memory,
                mode=spec["mode"],
                k=spec["k"],
                same_dataset=bool(spec["same_dataset"]),
                min_score=float(spec["min_score"]),
            )
            scored[spec["name"]] = None if pred is None else support_module.pp_score(target, pred, pert_means)
        out.append(scored)
    return out


def equal_dataset_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = row.get(key)
        if val is not None:
            by_ds[str(row["dataset"])].append(float(val))
    vals = [float(np.mean(v)) for v in by_ds.values() if v]
    return None if not vals else float(np.mean(vals))


def add_selected_mmd_scores(
    rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    train_memory: list[dict[str, Any]],
    selected_spec: dict[str, Any],
    single: dict[str, Any],
    multi: dict[str, Any],
    support_module: Any,
) -> None:
    row_by_key = {condition_key(row): row for row in rows}
    for target in val_rows:
        key = condition_key(target)
        scored = row_by_key[key]
        memory = train_memory + [row for row in val_rows if condition_key(row) != key]
        pred = weighted_memory_prediction(
            target,
            memory,
            mode=selected_spec["mode"],
            k=selected_spec["k"],
            same_dataset=bool(selected_spec["same_dataset"]),
            min_score=float(selected_spec["min_score"]),
        )
        if pred is not None:
            for metric, value in support_module.mmd_scores(target, pred).items():
                scored[f"{selected_spec['name']}__{metric}"] = value
        for name, base_pred in support_module.predict_baselines(target, single, multi).items():
            for metric, value in support_module.mmd_scores(target, base_pred).items():
                scored[f"{name}__{metric}"] = value


def paired_bootstrap(
    rows: list[dict[str, Any]],
    candidate: str,
    baseline: str,
    *,
    metric: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    diffs_by_ds: dict[str, list[float]] = defaultdict(list)
    if metric == "pp":
        ck, bk = candidate, baseline
        improve_is_positive = True
    elif metric == "mmd_clamped":
        ck, bk = f"{candidate}__test_mmd_clamped", f"{baseline}__test_mmd_clamped"
        improve_is_positive = False
    else:
        raise ValueError(metric)
    for row in rows:
        a = row.get(ck)
        b = row.get(bk)
        if a is not None and b is not None:
            diffs_by_ds[str(row["dataset"])].append(float(a) - float(b))
    datasets = sorted(ds for ds, vals in diffs_by_ds.items() if vals)
    if not datasets:
        return {"status": "missing", "candidate": candidate, "baseline": baseline, "metric": metric}
    point = float(np.mean([np.mean(diffs_by_ds[ds]) for ds in datasets]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sample_ds = rng.choice(datasets, size=len(datasets), replace=True)
        means = []
        for ds in sample_ds:
            vals = np.asarray(diffs_by_ds[str(ds)], dtype=np.float64)
            means.append(float(np.mean(rng.choice(vals, size=len(vals), replace=True))))
        boot.append(float(np.mean(means)))
    arr = np.asarray(boot, dtype=np.float64)
    by_dataset = {ds: float(np.mean(vals)) for ds, vals in diffs_by_ds.items()}
    if improve_is_positive:
        p_improve = float(np.mean(arr > 0.0))
        p_harm = float(np.mean(arr < 0.0))
    else:
        p_improve = float(np.mean(arr < 0.0))
        p_harm = float(np.mean(arr > 0.0))
    return {
        "status": "ok",
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "n_conditions": int(sum(len(diffs_by_ds[d]) for d in datasets)),
        "n_datasets": int(len(datasets)),
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": p_improve,
        "p_harm": p_harm,
        "by_dataset": by_dataset,
    }


def select_candidate(memory_eval_rows: list[dict[str, Any]], specs: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [{"model": spec["name"], "pp": equal_dataset_mean(memory_eval_rows, spec["name"])} for spec in specs]
    valid = [row for row in scores if row["pp"] is not None]
    if not valid:
        raise ValueError("no valid memory candidates")
    best = max(valid, key=lambda row: float(row["pp"]))
    return {"selected_model": best["model"], "memory_scores": scores}


def add_baseline_scores(
    val_rows: list[dict[str, Any]],
    single: dict[str, Any],
    multi: dict[str, Any],
    pert_means: dict[str, np.ndarray],
    support_module: Any,
) -> list[dict[str, Any]]:
    out = []
    for row in val_rows:
        preds = support_module.predict_baselines(row, single, multi)
        scored = {
            "dataset": row["dataset"],
            "condition": row["condition"],
            "genes": row["genes"],
            "nperts": row["nperts"],
            "group": "support_val_multi",
        }
        for name, pred in preds.items():
            scored[name] = support_module.pp_score(row, pred, pert_means)
        out.append(scored)
    return out


def merge_rows(base_rows: list[dict[str, Any]], memory_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mem = {condition_key(row): row for row in memory_rows}
    out = []
    for row in base_rows:
        merged = dict(row)
        merged.update({k: v for k, v in mem.get(condition_key(row), {}).items() if k not in merged or k.startswith("memory_")})
        out.append(merged)
    return out


def dataset_breakdown(rows: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    out = []
    for ds in sorted({str(r["dataset"]) for r in rows}):
        ds_rows = [r for r in rows if str(r["dataset"]) == ds]
        item: dict[str, Any] = {"dataset": ds, "n_conditions": len(ds_rows)}
        for model in models:
            vals = [float(r[model]) for r in ds_rows if r.get(model) is not None]
            item[model] = None if not vals else float(np.mean(vals))
            mmd_vals = [
                float(r[f"{model}__test_mmd_clamped"])
                for r in ds_rows
                if r.get(f"{model}__test_mmd_clamped") is not None
            ]
            item[f"{model}_mmd_clamped"] = None if not mmd_vals else float(np.mean(mmd_vals))
        out.append(item)
    return out


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    selected = payload["selected_model"]
    deltas = {
        row["baseline"]: row
        for row in payload["paired_deltas"]
        if row.get("candidate") == selected and row.get("status") == "ok"
    }
    reasons = []
    route = deltas.get("support_selected_route", {})
    if not (
        float(route.get("delta_mean") or 0.0) >= 0.02
        or ((route.get("ci95") or [0.0])[0] > 0.0)
    ):
        reasons.append("memory_readout_not_materially_better_than_support_route")
    for baseline in ("dataset_multi_mean", "additive_single_sum"):
        row = deltas.get(baseline, {})
        if row.get("status") != "ok":
            reasons.append(f"{baseline}_comparison_missing")
        elif float(row.get("delta_mean") or 0.0) < 0.0 and float(row.get("p_harm") or 1.0) > 0.20:
            reasons.append(f"memory_readout_harm_vs_{baseline}")
    by_ds = route.get("by_dataset") or {}
    if by_ds and any(float(v) < -0.02 for v in by_ds.values()):
        reasons.append("route_delta_negative_in_at_least_one_dataset")
    mmd_deltas = {
        row["baseline"]: row
        for row in payload.get("paired_mmd_deltas", [])
        if row.get("candidate") == selected and row.get("status") == "ok"
    }
    for baseline in ("support_selected_route", "dataset_multi_mean", "additive_single_sum"):
        row = mmd_deltas.get(baseline, {})
        if row.get("status") != "ok":
            reasons.append(f"{baseline}_mmd_comparison_missing")
        elif float(row.get("p_harm") if row.get("p_harm") is not None else 1.0) > 0.80:
            reasons.append(f"{baseline}_mmd_hard_harm")
    status = "cpu_gate_pass_freeze_support_memory_readout_before_query" if not reasons else "cpu_gate_fail_keep_support_memory_diagnostic"
    action = (
        "freeze_readout_and_consider_one_shot_query_after_protocol_review"
        if not reasons
        else "do_not_query_or_launch_gpu_from_support_memory"
    )
    return {"status": status, "action": action, "reasons": reasons}


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def render(payload: dict[str, Any]) -> str:
    selected = payload["selected_model"]
    lines = [
        "# LatentFM Track C Support-Memory Readout CPU Gate",
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
    lines += [
        "",
        "## Dataset Breakdown",
        "",
        "| dataset | n | selected pp | route pp | dataset_multi pp | additive_sum pp | selected MMD | route MMD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["dataset_breakdown"]:
        lines.append(
            f"| {row['dataset']} | {row['n_conditions']} | {fmt(row.get(selected))} | "
            f"{fmt(row.get('support_selected_route'))} | {fmt(row.get('dataset_multi_mean'))} | "
            f"{fmt(row.get('additive_single_sum'))} | {fmt(row.get(f'{selected}_mmd_clamped'))} | "
            f"{fmt(row.get('support_selected_route_mmd_clamped'))} |"
        )
    lines += [
        "",
        "## Paired Deltas",
        "",
        "| candidate | baseline | n cond | n ds | delta | 95% CI | p improve | p harm | dataset deltas |",
        "|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in payload["paired_deltas"]:
        ci = row.get("ci95") or [None, None]
        by_ds = ", ".join(f"{k}:{fmt(v)}" for k, v in (row.get("by_dataset") or {}).items())
        lines.append(
            f"| `{row['candidate']}` | `{row['baseline']}` | {row.get('n_conditions', 0)} | "
            f"{row.get('n_datasets', 0)} | {fmt(row.get('delta_mean'))} | "
            f"[{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(row.get('p_improve'))} | "
            f"{fmt(row.get('p_harm'))} | {by_ds} |"
        )
    lines += [
        "",
        "## Paired MMD Deltas",
        "",
        "Lower MMD is better; delta is candidate minus baseline.",
        "",
        "| candidate | baseline | n cond | n ds | delta | 95% CI | p improve | p harm | dataset deltas |",
        "|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in payload["paired_mmd_deltas"]:
        ci = row.get("ci95") or [None, None]
        by_ds = ", ".join(f"{k}:{fmt(v)}" for k, v in (row.get("by_dataset") or {}).items())
        lines.append(
            f"| `{row['candidate']}` | `{row['baseline']}` | {row.get('n_conditions', 0)} | "
            f"{row.get('n_datasets', 0)} | {fmt(row.get('delta_mean'))} | "
            f"[{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(row.get('p_improve'))} | "
            f"{fmt(row.get('p_harm'))} | {by_ds} |"
        )
    lines += ["", "## Gate Reasons", ""]
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{r}`" for r in reasons] if reasons else ["- none"])
    lines += [
        "",
        "## Usage Rule",
        "",
        "- This is CPU-only and evaluates support-val with leave-one support memory.",
        "- It does not read Track C query, canonical test/posthoc, or GPU candidate outcomes.",
        "- Passing would freeze a nonparametric readout for protocol review, not launch GPU distillation.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--trainselect-split-file", type=Path, default=DEFAULT_TRAINSELECT_SPLIT)
    parser.add_argument("--pert-means-file", type=Path, default=DEFAULT_PERT_MEANS)
    parser.add_argument("--max-cells-per-condition", type=int, default=256)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    support = load_support_module()
    data_dir = args.data_dir.resolve()
    split = support.load_json(args.split_file)
    trainselect = support.load_json(args.trainselect_split_file)
    manifest = support.load_json(data_dir / "manifest.json")
    metadata = support.load_json(Path(manifest["condition_metadata_file"]))
    pert_means = {k: v.astype(np.float32) for k, v in np.load(args.pert_means_file).items()}

    train_rows = support.collect_role_rows(data_dir, split, metadata, "train_multi", max_cells=args.max_cells_per_condition)
    val_rows = support.collect_role_rows(data_dir, split, metadata, "support_val_multi", max_cells=args.max_cells_per_condition)
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
    single = support.train_single_components(data_dir, split, metadata, max_cells=args.max_cells_per_condition)
    multi = support.train_multi_components(train_rows)
    base_rows = add_baseline_scores(val_rows, single, multi, pert_means, support)
    specs = candidate_specs()
    memory_rows = evaluate_memory_specs(val_rows, train_rows, pert_means, support, specs)
    selection = select_candidate(memory_rows, specs)
    selected = selection["selected_model"]
    rows = merge_rows(base_rows, memory_rows)
    selected_spec = {spec["name"]: spec for spec in specs}[selected]
    add_selected_mmd_scores(rows, val_rows, train_rows, selected_spec, single, multi, support)
    models = [selected, *BASELINES]
    absolute = [
        {
            "model": model,
            "pp": equal_dataset_mean(rows, model),
            "mmd_clamped": equal_dataset_mean(rows, f"{model}__test_mmd_clamped"),
        }
        for model in models
    ]
    by_dataset = dataset_breakdown(rows, models)
    deltas = [
        paired_bootstrap(rows, selected, baseline, metric="pp", n_boot=args.n_boot, seed=args.seed + i)
        for i, baseline in enumerate(BASELINES)
    ]
    mmd_deltas = [
        paired_bootstrap(rows, selected, baseline, metric="mmd_clamped", n_boot=args.n_boot, seed=args.seed + 100 + i)
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
        "leakage_status": "trackc_train_multi_and_leave_one_support_val_only_no_query_no_canonical_no_gpu_candidate",
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
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "selected_model": selected, "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
