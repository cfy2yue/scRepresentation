#!/usr/bin/env python3
"""One-shot Track C held-out query evaluation for frozen support-memory readout.

This script must fail closed before reading query data unless the frozen
support-memory protocol artifacts match the pre-query hashes and invariants.
It does not select a readout rule; it only executes the frozen rule
`memory_overlap_k5_same_ds_min0`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
MEMORY_GATE_SCRIPT = ROOT / "ops/audit_latentfm_trackc_support_memory_readout_gate_20260622.py"
SUPPORT_GATE_JSON = ROOT / "reports/latentfm_trackc_support_memory_readout_gate_20260622.json"
FULL_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2.json"
TRAINSELECT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
PERT_MEANS = (
    ROOT
    / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/"
    "xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz"
)
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_trackc_support_memory_readout_query_eval_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_MEMORY_READOUT_QUERY_EVAL_20260622.md"

EXPECTED_HASHES = {
    str(MEMORY_GATE_SCRIPT): "a3d5e678617b01a480a815b86cc21d02ee2dcadac9a8626340cfcc4200bf3fe0",
    str(SUPPORT_GATE_JSON): "6d17206147474fcf5105c58829e73c04328fc1ed951fc3a19c9e18b055fa1ac7",
    str(TRAINSELECT_SPLIT): "5f29dd5b582a40da3736770ca29950c12c54e46b590c3f7705c1d37da89f4f20",
    str(FULL_SPLIT): "054ed1d7e52df2c03af34351166e92d176437b30277c319a62f940725b303d0a",
    str(PERT_MEANS): "880ae032a9c268ab9d0dd728cf6fa42e15074df34b50075155aac0790f6ff5de",
}

SELECTED_MODEL = "memory_overlap_k5_same_ds_min0"
SELECTED_SPEC = {
    "name": SELECTED_MODEL,
    "mode": "overlap",
    "k": 5,
    "same_dataset": True,
    "min_score": 0.0,
}
PRIMARY_DATASETS = ("NormanWeissman2019_filtered", "Wessels")
EXPECTED_PRIMARY_QUERY_COUNTS = {"NormanWeissman2019_filtered": 92, "Wessels": 79}
EXPECTED_UNSUPPORTED_QUERY_COUNTS = {"GasperiniShendure2019_lowMOI": 3}
BASELINES = (
    "support_selected_route",
    "dataset_multi_mean",
    "global_multi_mean",
    "additive_single_mean",
    "additive_single_sum",
    "dataset_single_mean",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_hashes(paths: list[Path]) -> dict[str, str]:
    observed = {}
    for path in paths:
        key = str(path)
        got = file_sha256(path)
        observed[key] = got
        expected = EXPECTED_HASHES.get(key)
        if expected != got:
            raise AssertionError(f"hash mismatch for {path}: got {got}, expected {expected}")
    return observed


def import_memory_module() -> Any:
    spec = importlib.util.spec_from_file_location("trackc_support_memory_gate", MEMORY_GATE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {MEMORY_GATE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def keys_for(split: dict[str, Any], role: str) -> set[tuple[str, str]]:
    return {
        (str(ds), str(cond))
        for ds, obj in split.items()
        if isinstance(obj, dict)
        for cond in (obj.get(role) or [])
    }


def counts_for(split: dict[str, Any], role: str) -> dict[str, int]:
    return {
        str(ds): len(obj.get(role) or [])
        for ds, obj in split.items()
        if isinstance(obj, dict) and obj.get(role)
    }


def query_strata(split: dict[str, Any]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for ds, obj in split.items():
        if not isinstance(obj, dict):
            continue
        for role in ("query_multi_seen", "query_multi_unseen1", "query_multi_unseen2"):
            for cond in obj.get(role) or []:
                out[(str(ds), str(cond))] = role.replace("query_multi_", "")
    return out


def protocol_assertions() -> dict[str, Any]:
    observed_hashes = assert_hashes([MEMORY_GATE_SCRIPT, SUPPORT_GATE_JSON, TRAINSELECT_SPLIT, FULL_SPLIT, PERT_MEANS])
    gate = load_json(SUPPORT_GATE_JSON)
    full = load_json(FULL_SPLIT)
    trainselect = load_json(TRAINSELECT_SPLIT)

    if gate.get("selected_model") != SELECTED_MODEL:
        raise AssertionError(f"selected_model drifted: {gate.get('selected_model')}")
    if gate.get("split_guard", {}).get("support_val_matches_trainselect_test") is not True:
        raise AssertionError("support-val gate did not record support_val_matches_trainselect_test=True")
    if int(gate.get("max_cells_per_condition")) != 256:
        raise AssertionError("support gate max_cells_per_condition changed")
    if int(gate.get("n_boot")) != 2000 or int(gate.get("seed")) != 42:
        raise AssertionError("support gate bootstrap settings changed")

    full_support = keys_for(full, "support_val_multi")
    trainselect_test = keys_for(trainselect, "test")
    if full_support != trainselect_test:
        raise AssertionError("full support_val_multi keys do not equal trainselect test keys")

    full_query = keys_for(full, "query_multi")
    if full_support & full_query:
        raise AssertionError("support/query overlap is non-empty")
    train_multi = keys_for(full, "train_multi")
    if train_multi & full_query:
        raise AssertionError("train_multi/query overlap is non-empty")

    query_counts = counts_for(full, "query_multi")
    primary_counts = {ds: query_counts.get(ds, 0) for ds in PRIMARY_DATASETS}
    if primary_counts != EXPECTED_PRIMARY_QUERY_COUNTS:
        raise AssertionError(f"primary query counts drifted: {primary_counts}")
    unsupported_counts = {
        ds: count
        for ds, count in query_counts.items()
        if ds not in PRIMARY_DATASETS and count > 0
    }
    if unsupported_counts != EXPECTED_UNSUPPORTED_QUERY_COUNTS:
        raise AssertionError(f"unsupported query coverage drifted: {unsupported_counts}")

    supported_with_memory = {
        ds
        for ds in PRIMARY_DATASETS
        if len((full.get(ds) or {}).get("train_multi") or []) > 0
        and len((full.get(ds) or {}).get("support_val_multi") or []) > 0
    }
    if supported_with_memory != set(PRIMARY_DATASETS):
        raise AssertionError(f"primary support memory missing: {supported_with_memory}")

    return {
        "status": "protocol_assertions_pass",
        "observed_hashes": observed_hashes,
        "selected_model": SELECTED_MODEL,
        "selected_spec": SELECTED_SPEC,
        "primary_query_counts": primary_counts,
        "unsupported_query_counts": unsupported_counts,
        "support_val_count": len(full_support),
        "train_multi_count": len(train_multi),
    }


def equal_dataset_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = row.get(key)
        if val is not None:
            by_ds[str(row["dataset"])].append(float(val))
    vals = [float(np.mean(v)) for v in by_ds.values() if v]
    return None if not vals else float(np.mean(vals))


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


def stratum_breakdown(rows: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    out = []
    for ds in sorted({str(r["dataset"]) for r in rows}):
        for stratum in ("seen", "unseen1", "unseen2"):
            sub = [r for r in rows if str(r["dataset"]) == ds and r.get("query_stratum") == stratum]
            if not sub:
                continue
            item: dict[str, Any] = {"dataset": ds, "stratum": stratum, "n_conditions": len(sub)}
            for model in models:
                vals = [float(r[model]) for r in sub if r.get(model) is not None]
                item[model] = None if not vals else float(np.mean(vals))
                mmd_vals = [
                    float(r[f"{model}__test_mmd_clamped"])
                    for r in sub
                    if r.get(f"{model}__test_mmd_clamped") is not None
                ]
                item[f"{model}_mmd_clamped"] = None if not mmd_vals else float(np.mean(mmd_vals))
            out.append(item)
    return out


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
    point = float(np.mean([np.mean(diffs_by_ds[d]) for d in datasets]))
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


def evaluate_query(args: argparse.Namespace, assertions: dict[str, Any]) -> dict[str, Any]:
    memory = import_memory_module()
    support = memory.load_support_module()
    full = support.load_json(FULL_SPLIT)
    manifest = support.load_json(DATA_DIR / "manifest.json")
    metadata = support.load_json(Path(manifest["condition_metadata_file"]))
    pert_means = {k: v.astype(np.float32) for k, v in np.load(PERT_MEANS).items()}

    train_rows = support.collect_role_rows(DATA_DIR, full, metadata, "train_multi", max_cells=args.max_cells_per_condition)
    support_rows = support.collect_role_rows(DATA_DIR, full, metadata, "support_val_multi", max_cells=args.max_cells_per_condition)
    query_rows = support.collect_role_rows(DATA_DIR, full, metadata, "query_multi", max_cells=args.max_cells_per_condition)
    if len(query_rows) != sum(EXPECTED_PRIMARY_QUERY_COUNTS.values()):
        raise AssertionError(f"primary query HDF5 row count mismatch: {len(query_rows)}")

    stratum_by_key = query_strata(full)
    single = support.train_single_components(DATA_DIR, full, metadata, max_cells=args.max_cells_per_condition)
    multi = support.train_multi_components(train_rows)
    memory_bank = train_rows + support_rows
    rows = []
    for row in query_rows:
        key = (str(row["dataset"]), str(row["condition"]))
        scored = {
            "dataset": row["dataset"],
            "condition": row["condition"],
            "genes": row["genes"],
            "nperts": row["nperts"],
            "query_stratum": stratum_by_key.get(key, "unknown"),
            "group": "heldout_query_multi_primary",
        }
        pred = memory.weighted_memory_prediction(
            row,
            memory_bank,
            mode=SELECTED_SPEC["mode"],
            k=SELECTED_SPEC["k"],
            same_dataset=bool(SELECTED_SPEC["same_dataset"]),
            min_score=float(SELECTED_SPEC["min_score"]),
        )
        if pred is None:
            raise AssertionError(f"selected readout returned no prediction for {key}")
        scored[SELECTED_MODEL] = support.pp_score(row, pred, pert_means)
        for metric, value in support.mmd_scores(row, pred).items():
            scored[f"{SELECTED_MODEL}__{metric}"] = value
        for name, base_pred in support.predict_baselines(row, single, multi).items():
            scored[name] = support.pp_score(row, base_pred, pert_means)
            for metric, value in support.mmd_scores(row, base_pred).items():
                scored[f"{name}__{metric}"] = value
        rows.append(scored)

    models = [SELECTED_MODEL, *BASELINES]
    absolute = [
        {
            "model": model,
            "pp": equal_dataset_mean(rows, model),
            "mmd_clamped": equal_dataset_mean(rows, f"{model}__test_mmd_clamped"),
        }
        for model in models
    ]
    pp_deltas = [
        paired_bootstrap(rows, SELECTED_MODEL, baseline, metric="pp", n_boot=args.n_boot, seed=args.seed + i)
        for i, baseline in enumerate(BASELINES)
    ]
    mmd_deltas = [
        paired_bootstrap(rows, SELECTED_MODEL, baseline, metric="mmd_clamped", n_boot=args.n_boot, seed=args.seed + 100 + i)
        for i, baseline in enumerate(BASELINES)
    ]
    payload = {
        "status": "query_eval_complete_pending_decision",
        "protocol_assertions": assertions,
        "selected_model": SELECTED_MODEL,
        "selected_spec": SELECTED_SPEC,
        "primary_datasets": list(PRIMARY_DATASETS),
        "unsupported_query_counts": EXPECTED_UNSUPPORTED_QUERY_COUNTS,
        "max_cells_per_condition": args.max_cells_per_condition,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "absolute_scores": absolute,
        "dataset_breakdown": dataset_breakdown(rows, models),
        "stratum_breakdown": stratum_breakdown(rows, models),
        "paired_pp_deltas": pp_deltas,
        "paired_mmd_deltas": mmd_deltas,
        "condition_rows": rows,
    }
    payload["decision"] = decide(payload)
    return payload


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    pp_by = {
        row["baseline"]: row
        for row in payload["paired_pp_deltas"]
        if row.get("candidate") == SELECTED_MODEL and row.get("status") == "ok"
    }
    mmd_by = {
        row["baseline"]: row
        for row in payload["paired_mmd_deltas"]
        if row.get("candidate") == SELECTED_MODEL and row.get("status") == "ok"
    }
    route_pp = pp_by.get("support_selected_route", {})
    if float(route_pp.get("delta_mean") or 0.0) < 0.02:
        reasons.append("query_pp_delta_vs_support_route_below_0p02")
    if float(route_pp.get("p_harm") if route_pp.get("p_harm") is not None else 1.0) > 0.20:
        reasons.append("query_pp_harm_risk_vs_support_route")
    for ds, delta in (route_pp.get("by_dataset") or {}).items():
        if float(delta) < -0.02:
            reasons.append(f"{ds}_query_pp_delta_below_minus_0p02_vs_support_route")
    route_mmd = mmd_by.get("support_selected_route", {})
    if float(route_mmd.get("p_harm") if route_mmd.get("p_harm") is not None else 1.0) > 0.80:
        reasons.append("query_mmd_hard_harm_vs_support_route")
    for baseline in ("dataset_multi_mean", "additive_single_sum"):
        row = mmd_by.get(baseline, {})
        if row.get("status") != "ok":
            reasons.append(f"{baseline}_query_mmd_missing")
        elif float(row.get("p_harm") if row.get("p_harm") is not None else 1.0) > 0.80:
            reasons.append(f"query_mmd_hard_harm_vs_{baseline}")
    if payload.get("unsupported_query_counts") != EXPECTED_UNSUPPORTED_QUERY_COUNTS:
        reasons.append("unsupported_query_coverage_not_reported_as_frozen")
    status = (
        "one_shot_query_pass_support_memory_positive"
        if not reasons
        else "one_shot_query_fail_close_support_memory_branch"
    )
    action = (
        "report_as_trackc_inference_time_support_adaptation"
        if not reasons
        else "close_support_memory_as_support_val_only_diagnostic"
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
    lines = [
        "# LatentFM Track C Support-Memory Readout One-Shot Query Evaluation",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
        "",
        "## Protocol Assertions",
        "",
        f"- assertion_status: `{payload['protocol_assertions']['status']}`",
        f"- selected_model: `{payload['selected_model']}`",
        f"- primary_datasets: `{', '.join(payload['primary_datasets'])}`",
        f"- unsupported_query_counts: `{payload['unsupported_query_counts']}`",
        f"- max_cells_per_condition: `{payload['max_cells_per_condition']}`",
        "",
        "## Absolute Primary Query Scores",
        "",
        "| model | equal-dataset pp | equal-dataset MMD clamped |",
        "|---|---:|---:|",
    ]
    for row in payload["absolute_scores"]:
        lines.append(f"| `{row['model']}` | {fmt(row.get('pp'))} | {fmt(row.get('mmd_clamped'))} |")
    lines += [
        "",
        "## Dataset Breakdown",
        "",
        "| dataset | n | selected pp | route pp | dataset_multi pp | additive_sum pp | selected MMD | route MMD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["dataset_breakdown"]:
        lines.append(
            f"| {row['dataset']} | {row['n_conditions']} | {fmt(row.get(SELECTED_MODEL))} | "
            f"{fmt(row.get('support_selected_route'))} | {fmt(row.get('dataset_multi_mean'))} | "
            f"{fmt(row.get('additive_single_sum'))} | {fmt(row.get(f'{SELECTED_MODEL}_mmd_clamped'))} | "
            f"{fmt(row.get('support_selected_route_mmd_clamped'))} |"
        )
    lines += [
        "",
        "## Stratum Breakdown",
        "",
        "| dataset | stratum | n | selected pp | route pp | selected MMD | route MMD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["stratum_breakdown"]:
        lines.append(
            f"| {row['dataset']} | {row['stratum']} | {row['n_conditions']} | "
            f"{fmt(row.get(SELECTED_MODEL))} | {fmt(row.get('support_selected_route'))} | "
            f"{fmt(row.get(f'{SELECTED_MODEL}_mmd_clamped'))} | "
            f"{fmt(row.get('support_selected_route_mmd_clamped'))} |"
        )
    lines += [
        "",
        "## Paired PP Deltas",
        "",
        "| candidate | baseline | n cond | n ds | delta | 95% CI | p improve | p harm | dataset deltas |",
        "|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in payload["paired_pp_deltas"]:
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
        "- This is the frozen one-shot held-out query evaluation.",
        "- Do not rerun or tune the selected readout based on this report.",
        "- Gasperini final-only query rows are unsupported coverage diagnostics, not primary support-adaptation rows.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-check-only", action="store_true")
    parser.add_argument("--max-cells-per-condition", type=int, default=256)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    if args.max_cells_per_condition != 256 or args.n_boot != 2000 or args.seed != 42:
        raise AssertionError("frozen query settings changed")
    assertions = protocol_assertions()
    if args.protocol_check_only:
        print(json.dumps(assertions, indent=2))
        return 0

    payload = evaluate_query(args, assertions)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    if args.out_json.exists() or args.out_md.exists():
        raise FileExistsError("query outputs already exist; refusing to overwrite one-shot artifacts")
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
