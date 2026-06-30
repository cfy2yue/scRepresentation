#!/usr/bin/env python3
"""Screen train-only covariates for routed-response expert legality.

This CPU-only diagnostic asks whether simple deployable covariates can recover
the response expert's clean Wessels-like regime without using held-out
outcomes as route inputs. Route selection in this report is still posthoc
model selection, so passing rows are CPU-gate candidates only.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
METRICS = ("pearson_pert", "test_mmd_clamped")
GROUPS = ("test", "test_multi_unseen2", "family_gene")
HIGHER_IS_BETTER = {"pearson_pert"}
LOWER_IS_BETTER = {"test_mmd_clamped"}


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def fnum(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def coverage_lookup(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, int]]:
    payload = load_json(path)
    rows = {}
    for row in payload.get("rows") or []:
        rows[(str(row["dataset"]), str(row["condition"]))] = row
    counts = {str(k): int(v) for k, v in (payload.get("train_single_counts_by_dataset") or {}).items()}
    return rows, counts


def enrich_rows(rows: list[dict[str, Any]], coverage: dict[tuple[str, str], dict[str, Any]], counts: dict[str, int]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("bundle") != "response_uncapped_full_conditions":
            continue
        cov = coverage.get((str(row["dataset"]), str(row["condition"])), {})
        n_genes = int(cov.get("n_genes") or row.get("nperts_est") or 0)
        same_hits = int(cov.get("same_hits") or 0)
        global_hits = int(cov.get("global_hits") or 0)
        rr = dict(row)
        rr.update(
            {
                "n_genes": n_genes,
                "same_hits": same_hits,
                "global_hits": global_hits,
                "same_hit_frac": same_hits / max(n_genes, 1),
                "global_hit_frac": global_hits / max(n_genes, 1),
                "same_full": bool(cov.get("same_full")),
                "global_full": bool(cov.get("global_full")),
                "dataset_train_single_count": counts.get(str(row["dataset"]), 0),
            }
        )
        out.append(rr)
    return out


def route_defs() -> dict[str, tuple[str, Callable[[dict[str, Any]], bool]]]:
    def target(row: dict[str, Any]) -> bool:
        return truthy(row.get("is_gene")) and truthy(row.get("is_multi"))

    return {
        "same_not_full": ("train_single_coverage", lambda r: target(r) and not truthy(r["same_full"])),
        "same_hit_frac_le_0": ("train_single_coverage", lambda r: target(r) and float(r["same_hit_frac"]) <= 0.0),
        "same_hit_frac_le_0.5": ("train_single_coverage", lambda r: target(r) and float(r["same_hit_frac"]) <= 0.5),
        "same_hit_frac_le_0.75": ("train_single_coverage", lambda r: target(r) and float(r["same_hit_frac"]) <= 0.75),
        "global_full_and_same_not_full": (
            "train_single_coverage",
            lambda r: target(r) and truthy(r["global_full"]) and not truthy(r["same_full"]),
        ),
        "global_full_and_same_hit_frac_le_0.5": (
            "train_single_coverage",
            lambda r: target(r) and truthy(r["global_full"]) and float(r["same_hit_frac"]) <= 0.5,
        ),
        "global_full_and_same_hit_frac_ge_0.5": (
            "train_single_coverage",
            lambda r: target(r) and truthy(r["global_full"]) and float(r["same_hit_frac"]) >= 0.5,
        ),
        "global_full_and_same_hit_frac_ge_1.0": (
            "train_single_coverage",
            lambda r: target(r) and truthy(r["global_full"]) and float(r["same_hit_frac"]) >= 1.0,
        ),
        "global_full_and_train_single_ge_50": (
            "train_single_coverage",
            lambda r: target(r) and truthy(r["global_full"]) and int(r["dataset_train_single_count"]) >= 50,
        ),
        "global_full_and_train_single_ge_80": (
            "train_single_coverage",
            lambda r: target(r) and truthy(r["global_full"]) and int(r["dataset_train_single_count"]) >= 80,
        ),
        "global_full_and_train_single_lt_50": (
            "train_single_coverage",
            lambda r: target(r) and truthy(r["global_full"]) and int(r["dataset_train_single_count"]) < 50,
        ),
        "global_full_and_two_gene": (
            "train_single_coverage",
            lambda r: target(r) and truthy(r["global_full"]) and int(r["n_genes"]) == 2,
        ),
        "global_full_and_complex_gene": (
            "train_single_coverage",
            lambda r: target(r) and truthy(r["global_full"]) and int(r["n_genes"]) > 2,
        ),
        "dataset_train_single_le_10": (
            "dataset_train_size_proxy",
            lambda r: target(r) and int(r["dataset_train_single_count"]) <= 10,
        ),
        "dataset_train_single_le_50": (
            "dataset_train_size_proxy",
            lambda r: target(r) and int(r["dataset_train_single_count"]) <= 50,
        ),
        "dataset_train_single_le_100": (
            "dataset_train_size_proxy",
            lambda r: target(r) and int(r["dataset_train_single_count"]) <= 100,
        ),
        "global_full": ("train_single_coverage", lambda r: target(r) and truthy(r["global_full"])),
    }


def metric_delta(row: dict[str, Any], metric: str, use_response: bool, expert: str) -> float | None:
    base = fnum(row.get(f"anchor__{metric}"))
    cand = fnum(row.get(f"{expert}__{metric}"))
    if base is None or cand is None:
        return None
    return (cand if use_response else base) - base


def bootstrap_route(
    rows: list[dict[str, Any]],
    route_name: str,
    route: Callable[[dict[str, Any]], bool],
    *,
    expert: str,
    n_boot: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    out = []
    for group in GROUPS:
        group_rows = [r for r in rows if r["eval_group"] == group]
        if not group_rows:
            continue
        selected = sum(1 for row in group_rows if route(row))
        for metric in METRICS:
            by_dataset: dict[str, list[float]] = defaultdict(list)
            for row in group_rows:
                delta = metric_delta(row, metric, route(row), expert)
                if delta is not None:
                    by_dataset[str(row["dataset"])].append(delta)
            datasets = sorted(ds for ds, vals in by_dataset.items() if vals)
            if not datasets:
                continue
            observed = float(np.mean([np.mean(by_dataset[ds]) for ds in datasets]))
            samples = []
            for _ in range(n_boot):
                ds_means = []
                for ds in datasets:
                    vals = np.asarray(by_dataset[ds], dtype=float)
                    idx = rng.integers(0, len(vals), size=len(vals))
                    ds_means.append(float(np.mean(vals[idx])))
                samples.append(float(np.mean(ds_means)))
            arr = np.asarray(samples, dtype=float)
            lo, hi = np.quantile(arr, [0.025, 0.975])
            if metric in HIGHER_IS_BETTER:
                p_improve = float(np.mean(arr > 0.0))
                p_harm = float(np.mean(arr < 0.0))
            elif metric in LOWER_IS_BETTER:
                p_improve = float(np.mean(arr < 0.0))
                p_harm = float(np.mean(arr > 0.0))
            else:
                p_improve = float("nan")
                p_harm = float("nan")
            out.append(
                {
                    "route": route_name,
                    "group": group,
                    "metric": metric,
                    "n_conditions": len(group_rows),
                    "selected_conditions": selected,
                    "n_datasets": len(datasets),
                    "delta": observed,
                    "ci95": [float(lo), float(hi)],
                    "p_improve": p_improve,
                    "p_harm": p_harm,
                }
            )
    return out


def bootstrap_route_dataset_resampled(
    rows: list[dict[str, Any]],
    route_name: str,
    route: Callable[[dict[str, Any]], bool],
    *,
    expert: str,
    n_boot: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Bootstrap route deltas by resampling datasets and conditions.

    This is more conservative for route promotion than the within-dataset-only
    bootstrap above because it exposes dataset-composition sensitivity.
    """
    out = []
    for group in GROUPS:
        group_rows = [r for r in rows if r["eval_group"] == group]
        if not group_rows:
            continue
        for metric in METRICS:
            by_dataset: dict[str, list[float]] = defaultdict(list)
            for row in group_rows:
                delta = metric_delta(row, metric, route(row), expert)
                if delta is not None:
                    by_dataset[str(row["dataset"])].append(delta)
            datasets = sorted(ds for ds, vals in by_dataset.items() if vals)
            if not datasets:
                continue
            observed = float(np.mean([np.mean(by_dataset[ds]) for ds in datasets]))
            samples = []
            for _ in range(n_boot):
                picked_datasets = rng.choice(datasets, size=len(datasets), replace=True)
                ds_means = []
                for ds in picked_datasets:
                    vals = np.asarray(by_dataset[str(ds)], dtype=float)
                    idx = rng.integers(0, len(vals), size=len(vals))
                    ds_means.append(float(np.mean(vals[idx])))
                samples.append(float(np.mean(ds_means)))
            arr = np.asarray(samples, dtype=float)
            lo, hi = np.quantile(arr, [0.025, 0.975])
            if metric in HIGHER_IS_BETTER:
                p_improve = float(np.mean(arr > 0.0))
                p_harm = float(np.mean(arr < 0.0))
            elif metric in LOWER_IS_BETTER:
                p_improve = float(np.mean(arr < 0.0))
                p_harm = float(np.mean(arr > 0.0))
            else:
                p_improve = float("nan")
                p_harm = float("nan")
            out.append(
                {
                    "route": route_name,
                    "group": group,
                    "metric": metric,
                    "n_conditions": len(group_rows),
                    "n_datasets": len(datasets),
                    "delta": observed,
                    "ci95": [float(lo), float(hi)],
                    "p_improve": p_improve,
                    "p_harm": p_harm,
                }
            )
    return out


def equal_dataset_delta(
    rows: list[dict[str, Any]],
    route: Callable[[dict[str, Any]], bool],
    *,
    expert: str,
    group: str,
    metric: str,
    drop_dataset: str | None = None,
) -> float | None:
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["eval_group"] != group:
            continue
        dataset = str(row["dataset"])
        if drop_dataset is not None and dataset == drop_dataset:
            continue
        delta = metric_delta(row, metric, route(row), expert)
        if delta is not None:
            by_dataset[dataset].append(delta)
    means = [float(np.mean(vals)) for vals in by_dataset.values() if vals]
    if not means:
        return None
    return float(np.mean(means))


def leave_dataset_sensitivity(
    rows: list[dict[str, Any]],
    route_name: str,
    route: Callable[[dict[str, Any]], bool],
    *,
    expert: str,
) -> list[dict[str, Any]]:
    out = []
    for group in GROUPS:
        datasets = sorted({str(r["dataset"]) for r in rows if r["eval_group"] == group})
        if len(datasets) < 2:
            continue
        for metric in METRICS:
            observed = equal_dataset_delta(rows, route, expert=expert, group=group, metric=metric)
            leaves = {
                dataset: equal_dataset_delta(rows, route, expert=expert, group=group, metric=metric, drop_dataset=dataset)
                for dataset in datasets
            }
            vals = [v for v in leaves.values() if v is not None]
            if observed is None or not vals:
                continue
            if metric in HIGHER_IS_BETTER:
                sign_consistent = all(v >= 0.0 for v in vals)
            elif metric in LOWER_IS_BETTER:
                sign_consistent = all(v <= 0.0 for v in vals)
            else:
                sign_consistent = False
            out.append(
                {
                    "route": route_name,
                    "group": group,
                    "metric": metric,
                    "observed": observed,
                    "leave_one_min": float(min(vals)),
                    "leave_one_max": float(max(vals)),
                    "sign_consistent": sign_consistent,
                    "n_leave_datasets": len(vals),
                    "leave_one_by_dataset": leaves,
                }
            )
    return out


def assess(
    rows: list[dict[str, Any]],
    route_name: str,
    *,
    dataset_resampled_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by = {(r["group"], r["metric"]): r for r in rows if r["route"] == route_name}
    ds_by = {
        (r["group"], r["metric"]): r
        for r in (dataset_resampled_rows or [])
        if r["route"] == route_name
    }
    required = {
        "test_pp": by.get(("test", "pearson_pert")),
        "test_mmd": by.get(("test", "test_mmd_clamped")),
        "family_pp": by.get(("family_gene", "pearson_pert")),
        "family_mmd": by.get(("family_gene", "test_mmd_clamped")),
        "unseen2_pp": by.get(("test_multi_unseen2", "pearson_pert")),
        "unseen2_mmd": by.get(("test_multi_unseen2", "test_mmd_clamped")),
    }
    dataset_resampled_required = {
        "test_pp": ds_by.get(("test", "pearson_pert")),
        "test_mmd": ds_by.get(("test", "test_mmd_clamped")),
        "family_pp": ds_by.get(("family_gene", "pearson_pert")),
        "family_mmd": ds_by.get(("family_gene", "test_mmd_clamped")),
        "unseen2_mmd": ds_by.get(("test_multi_unseen2", "test_mmd_clamped")),
    }
    reasons = []
    for name, row in required.items():
        if row is None:
            reasons.append(f"missing_{name}")
    for name, row in dataset_resampled_required.items():
        if dataset_resampled_rows is not None and row is None:
            reasons.append(f"missing_dataset_resampled_{name}")
    if reasons:
        status = "incomplete"
    else:
        if float(required["test_pp"]["p_improve"]) < 0.90 or float(required["test_pp"]["delta"]) <= 0:
            reasons.append("test_pp_not_supported")
        if float(required["family_pp"]["p_improve"]) < 0.90 or float(required["family_pp"]["delta"]) <= 0:
            reasons.append("family_pp_not_supported")
        if float(required["test_mmd"]["p_harm"]) > 0.80:
            reasons.append("test_mmd_harm")
        if float(required["family_mmd"]["p_harm"]) > 0.80:
            reasons.append("family_mmd_harm")
        unseen_ci = required["unseen2_mmd"].get("ci95") or []
        if float(required["unseen2_mmd"]["p_harm"]) > 0.80 or (len(unseen_ci) >= 1 and float(unseen_ci[0]) > 0):
            reasons.append("unseen2_mmd_hard_harm")
        if dataset_resampled_rows is not None:
            ds_test_pp = dataset_resampled_required["test_pp"]
            ds_family_pp = dataset_resampled_required["family_pp"]
            ds_test_mmd = dataset_resampled_required["test_mmd"]
            ds_family_mmd = dataset_resampled_required["family_mmd"]
            ds_unseen_mmd = dataset_resampled_required["unseen2_mmd"]
            if float(ds_test_pp["p_improve"]) < 0.90 or float(ds_test_pp["delta"]) <= 0:
                reasons.append("dataset_resampled_test_pp_weak")
            if float(ds_family_pp["p_improve"]) < 0.90 or float(ds_family_pp["delta"]) <= 0:
                reasons.append("dataset_resampled_family_pp_weak")
            if float(ds_test_mmd["p_harm"]) > 0.80:
                reasons.append("dataset_resampled_test_mmd_harm")
            if float(ds_family_mmd["p_harm"]) > 0.80:
                reasons.append("dataset_resampled_family_mmd_harm")
            ds_unseen_ci = ds_unseen_mmd.get("ci95") or []
            if float(ds_unseen_mmd["p_harm"]) > 0.80 or (
                len(ds_unseen_ci) >= 1 and float(ds_unseen_ci[0]) > 0
            ):
                reasons.append("dataset_resampled_unseen2_mmd_hard_harm")
        if not reasons:
            status = "strict_trainonly_covariate_gate_pass"
        elif all(str(r).startswith("dataset_resampled_") for r in reasons):
            status = "diagnostic_signal_only"
        else:
            status = "fail"
    return {
        "route": route_name,
        "status": status,
        "reasons": reasons,
        "required": required,
        "dataset_resampled_required": dataset_resampled_required,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Train-Only Router Covariate Audit",
        "",
        "CPU-only posthoc covariate screen. Routes use train-only/deployable covariates as inputs, but route selection in this report is still based on held-out results and is not promotion.",
        "",
        f"Condition CSV: `{payload['condition_csv']}`",
        f"Coverage JSON: `{payload['coverage_json']}`",
        f"Expert: `{payload['expert']}`",
        "",
        "## Gate Summary",
        "",
        "| route | family | status | reasons | test pp | test MMD | unseen2 pp | unseen2 MMD | family pp | family MMD |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    family_by_route = {r["route"]: r["covariate_family"] for r in payload["routes"]}
    for dec in payload["decisions"]:
        req = dec["required"]
        lines.append(
            "| {route} | {family} | {status} | {reasons} | {test_pp} | {test_mmd} | {u2_pp} | {u2_mmd} | {fam_pp} | {fam_mmd} |".format(
                route=dec["route"],
                family=family_by_route.get(dec["route"], ""),
                status=dec["status"],
                reasons=", ".join(dec["reasons"]) or "-",
                test_pp=fmt((req.get("test_pp") or {}).get("delta")),
                test_mmd=fmt((req.get("test_mmd") or {}).get("delta")),
                u2_pp=fmt((req.get("unseen2_pp") or {}).get("delta")),
                u2_mmd=fmt((req.get("unseen2_mmd") or {}).get("delta")),
                fam_pp=fmt((req.get("family_pp") or {}).get("delta")),
                fam_mmd=fmt((req.get("family_mmd") or {}).get("delta")),
            )
        )
    lines.extend(
        [
            "",
            "## Leave-Dataset Sensitivity",
            "",
            "| route | group | metric | observed | leave-one min | leave-one max | sign consistent |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    passing = {
        d["route"]
        for d in payload["decisions"]
        if d["status"] in {"strict_trainonly_covariate_gate_pass", "diagnostic_signal_only"}
    }
    for row in payload.get("leave_dataset_sensitivity", []):
        if row["route"] not in passing:
            continue
        if row["group"] not in {"test", "test_multi_unseen2", "family_gene"}:
            continue
        lines.append(
            "| {route} | {group} | {metric} | {obs} | {lo} | {hi} | {sign} |".format(
                route=row["route"],
                group=row["group"],
                metric=row["metric"],
                obs=fmt(row["observed"]),
                lo=fmt(row["leave_one_min"]),
                hi=fmt(row["leave_one_max"]),
                sign="yes" if row["sign_consistent"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Dataset-Resampled Bootstrap",
            "",
            "| route | group | metric | delta | ci95 | p_improve | p_harm |",
            "|---|---|---|---:|---|---:|---:|",
        ]
    )
    ds_boot_by_route = payload.get("dataset_resampled_bootstrap_rows", [])
    for row in ds_boot_by_route:
        if row["route"] not in passing:
            continue
        if row["group"] not in {"test", "test_multi_unseen2", "family_gene"}:
            continue
        ci = row.get("ci95") or []
        lines.append(
            "| {route} | {group} | {metric} | {delta} | [{lo}, {hi}] | {pi:.3f} | {ph:.3f} |".format(
                route=row["route"],
                group=row["group"],
                metric=row["metric"],
                delta=fmt(row["delta"]),
                lo=fmt(ci[0]) if len(ci) > 0 else "NA",
                hi=fmt(ci[1]) if len(ci) > 1 else "NA",
                pi=float(row.get("p_improve", float("nan"))),
                ph=float(row.get("p_harm", float("nan"))),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `dataset_train_size_proxy` routes are train-only but may still act as dataset proxies; they require extra robustness checks before GPU use.",
            "- A passing route here only permits a predeclared router GPU smoke or a train-only router implementation; it is not a final claim.",
            "- Dataset-resampled bootstrap is the more relevant uncertainty estimate for route promotion; within-dataset condition bootstrap alone is not enough.",
            "- If all simple covariates fail, the Wessels-like response regime needs richer train-only features or should remain mechanism evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition-csv",
        type=Path,
        default=ROOT / "reports/latentfm_existing_expert_route_audit_20260621.conditions.csv",
    )
    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=ROOT / "reports/latentfm_synthetic_combo_prior_coverage_audit_20260621.json",
    )
    parser.add_argument("--expert", default="response_aux0875_uncapped")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", type=Path, default=ROOT / "reports/latentfm_trainonly_router_covariates_20260621")
    args = parser.parse_args()

    coverage, counts = coverage_lookup(args.coverage_json)
    rows = enrich_rows(load_rows(args.condition_csv), coverage, counts)
    rng = np.random.default_rng(args.seed)
    route_meta = []
    boot_rows = []
    dataset_resampled_boot_rows = []
    sensitivity_rows = []
    for name, (family, pred) in route_defs().items():
        route_meta.append({"route": name, "covariate_family": family})
        boot_rows.extend(bootstrap_route(rows, name, pred, expert=args.expert, n_boot=args.n_boot, rng=rng))
        dataset_resampled_boot_rows.extend(
            bootstrap_route_dataset_resampled(rows, name, pred, expert=args.expert, n_boot=args.n_boot, rng=rng)
        )
        sensitivity_rows.extend(leave_dataset_sensitivity(rows, name, pred, expert=args.expert))
    decisions = [
        assess(boot_rows, route["route"], dataset_resampled_rows=dataset_resampled_boot_rows)
        for route in route_meta
    ]
    payload = {
        "condition_csv": str(args.condition_csv),
        "coverage_json": str(args.coverage_json),
        "expert": args.expert,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_input_rows": len(rows),
        "routes": route_meta,
        "bootstrap_rows": boot_rows,
        "dataset_resampled_bootstrap_rows": dataset_resampled_boot_rows,
        "leave_dataset_sensitivity": sensitivity_rows,
        "decisions": decisions,
        "uses_heldout_outcome_for_route_selection": True,
    }
    out_json = args.out_prefix.with_suffix(".json")
    out_md = args.out_prefix.with_suffix(".md")
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
