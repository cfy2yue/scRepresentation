#!/usr/bin/env python3
"""Audit deployable and diagnostic routes across existing LatentFM experts.

This is a CPU-only gate. It reuses condition-level posthoc metrics and never
starts model training or inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
METRICS = ("pearson_pert", "direct_pearson", "test_mmd_clamped")
BOOT_METRICS = ("pearson_pert", "test_mmd_clamped")
GROUPS = ("test", "test_multi_unseen2", "family_gene", "family_drug", "structure_single", "structure_multi")
HIGHER_IS_BETTER = {"pearson_pert", "direct_pearson", "pearson_ctrl"}
LOWER_IS_BETTER = {"test_mmd_clamped"}


@dataclass(frozen=True)
class Expert:
    label: str
    split_json: Path
    family_json: Path


@dataclass(frozen=True)
class Bundle:
    name: str
    baseline: Expert
    experts: dict[str, Expert]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("dataset", "")), str(row.get("condition", ""))


def condition_index(payload: dict[str, Any], group: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = payload.get("groups", {}).get(group, {}).get("condition_metrics") or []
    return {key(r): r for r in rows if isinstance(r, dict)}


def group_membership(payload: dict[str, Any]) -> dict[tuple[str, str], set[str]]:
    out: dict[tuple[str, str], set[str]] = defaultdict(set)
    for group_name, group_obj in (payload.get("groups") or {}).items():
        for row in group_obj.get("condition_metrics") or []:
            if isinstance(row, dict):
                out[key(row)].add(str(group_name))
    return out


def load_manifest(path: Path, label: str | None = None) -> tuple[Expert, Expert]:
    manifest = load_json(path)
    run_name = label or str(manifest.get("run_name") or path.parent.name)
    return (
        Expert(
            label="anchor",
            split_json=Path(manifest["baseline_split_json"]),
            family_json=Path(manifest["baseline_family_json"]),
        ),
        Expert(
            label=run_name,
            split_json=Path(manifest["run_split_json"]),
            family_json=Path(manifest["run_family_json"]),
        ),
    )


def load_uncapped_audit(path: Path, label: str | None = None) -> tuple[Expert, Expert]:
    payload = load_json(path)
    comps = payload.get("comparisons") or []
    if len(comps) != 1:
        raise ValueError(f"expected one comparison in {path}, got {len(comps)}")
    row = comps[0]
    run_label = label or str(row.get("candidate_label") or row.get("name") or path.stem)
    return (
        Expert(
            label="anchor_uncapped",
            split_json=Path(row["baseline_split_json"]),
            family_json=Path(row["baseline_family_json"]),
        ),
        Expert(
            label=run_label,
            split_json=Path(row["candidate_split_json"]),
            family_json=Path(row["candidate_family_json"]),
        ),
    )


def default_bundles() -> list[Bundle]:
    baseline, response0875 = load_uncapped_audit(
        ROOT / "reports/latentfm_response_aux0875_uncapped_route_audit_20260621.json",
        "response_aux0875_uncapped",
    )
    _, response05 = load_uncapped_audit(
        ROOT / "reports/latentfm_response_aux05_uncapped_route_audit_20260621.json",
        "response_aux05_uncapped",
    )
    uncapped_response = Bundle(
        name="response_uncapped_full_conditions",
        baseline=baseline,
        experts={
            response0875.label: response0875,
            response05.label: response05,
        },
    )

    capped_baseline, response1 = load_manifest(
        ROOT / "runs/latentfm_response_normalization_20260621/posthoc_manifest.json",
        "response_aux1_capped",
    )
    _, response0625 = load_manifest(
        ROOT / "runs/latentfm_response_route_sweetspot_20260621/scf_response_dataset_scale_pca32_aux0625_4k/posthoc_manifest.json",
        "response_aux0625_capped",
    )
    _, response075 = load_manifest(
        ROOT / "runs/latentfm_response_route_sweetspot_20260621/scf_response_dataset_scale_pca32_aux075_4k/posthoc_manifest.json",
        "response_aux075_capped",
    )
    _, response0875_capped = load_manifest(
        ROOT / "runs/latentfm_response_route_sweetspot_20260621/scf_response_dataset_scale_pca32_aux0875_4k/posthoc_manifest.json",
        "response_aux0875_capped",
    )
    _, pairwise_full = load_manifest(
        ROOT / "runs/latentfm_pairwise_condition_20260621/posthoc_manifest.json",
        "pairwise_full_capped",
    )
    _, pairwise_adapter = load_manifest(
        ROOT / "runs/latentfm_pairwise_adapter_only_20260621/posthoc_manifest.json",
        "pairwise_adapteronly_capped",
    )
    capped_all = Bundle(
        name="existing_experts_capped_common_conditions",
        baseline=capped_baseline,
        experts={
            response1.label: response1,
            response0625.label: response0625,
            response075.label: response075,
            response0875_capped.label: response0875_capped,
            pairwise_full.label: pairwise_full,
            pairwise_adapter.label: pairwise_adapter,
        },
    )
    return [uncapped_response, capped_all]


def infer_features(condition: str, memberships: set[str]) -> dict[str, bool | str | int]:
    is_single = "structure_single" in memberships or "test_single" in memberships
    is_multi = "structure_multi" in memberships or "test_multi" in memberships or "+" in condition
    is_gene = "family_gene" in memberships
    is_drug = "family_drug" in memberships
    return {
        "is_gene": is_gene,
        "is_drug": is_drug,
        "is_single": is_single,
        "is_multi": is_multi and not is_single,
        "is_test_multi_unseen2": "test_multi_unseen2" in memberships,
        "nperts_est": max(2, condition.count("+") + 1) if is_multi and not is_single else 1,
        "route_features": ",".join(
            name
            for name, flag in (
                ("family_gene", is_gene),
                ("family_drug", is_drug),
                ("structure_single", is_single),
                ("structure_multi", is_multi and not is_single),
            )
            if flag
        ),
    }


def build_rows(bundle: Bundle) -> list[dict[str, Any]]:
    base_split = load_json(bundle.baseline.split_json)
    base_family = load_json(bundle.baseline.family_json)
    base_members = group_membership(base_family)
    split_members = group_membership(base_split)

    expert_payloads: dict[str, tuple[dict[str, Any], dict[str, Any], dict[tuple[str, str], set[str]]]] = {}
    for label, expert in bundle.experts.items():
        split = load_json(expert.split_json)
        family = load_json(expert.family_json)
        expert_payloads[label] = (split, family, group_membership(family))

    rows: list[dict[str, Any]] = []
    for eval_group in GROUPS:
        base_payload = base_split if eval_group in (base_split.get("groups") or {}) else base_family
        if eval_group not in (base_payload.get("groups") or {}):
            continue
        base_group = condition_index(base_payload, eval_group)
        expert_groups: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
        for label, (split, family, _) in expert_payloads.items():
            payload = split if eval_group in (split.get("groups") or {}) else family
            if eval_group in (payload.get("groups") or {}):
                expert_groups[label] = condition_index(payload, eval_group)

        for cond_key in sorted(base_group):
            if not all(cond_key in group for group in expert_groups.values()):
                continue
            dataset, condition = cond_key
            memberships = set(base_members.get(cond_key, set())) | set(split_members.get(cond_key, set()))
            features = infer_features(condition, memberships)
            row: dict[str, Any] = {
                "bundle": bundle.name,
                "eval_group": eval_group,
                "dataset": dataset,
                "condition": condition,
                **features,
            }
            for metric in METRICS:
                row[f"anchor__{metric}"] = fnum(base_group[cond_key].get(metric))
                for label, group in expert_groups.items():
                    row[f"{label}__{metric}"] = fnum(group[cond_key].get(metric))
            rows.append(row)
    return rows


def route_defs(expert_labels: list[str]) -> dict[str, tuple[bool, Callable[[dict[str, Any]], str]]]:
    response = "response_aux0875_uncapped"
    response05 = "response_aux05_uncapped"
    response0875_capped = "response_aux0875_capped"
    response075_capped = "response_aux075_capped"
    response0625_capped = "response_aux0625_capped"
    response1_capped = "response_aux1_capped"
    pairwise = "pairwise_full_capped"
    adapter = "pairwise_adapteronly_capped"
    available = set(expert_labels)

    routes: dict[str, tuple[bool, Callable[[dict[str, Any]], str]]] = {
        "anchor_only": (True, lambda r: "anchor"),
    }
    for route_name, label in (
        ("response0875_gene_multi", response),
        ("response05_gene_multi", response05),
        ("response0875_capped_gene_multi", response0875_capped),
        ("response075_capped_gene_multi", response075_capped),
        ("response0625_capped_gene_multi", response0625_capped),
        ("response1_capped_gene_multi", response1_capped),
        ("pairwise_gene_multi", pairwise),
        ("adapter_gene_multi", adapter),
    ):
        if label in available:
            routes[route_name] = (
                True,
                lambda r, label=label: label if truthy(r["is_gene"]) and truthy(r["is_multi"]) else "anchor",
            )
    if response in available:
        routes["response_wessels_gene_multi_diagnostic"] = (
            False,
            lambda r: response if truthy(r["is_gene"]) and truthy(r["is_multi"]) and r["dataset"] == "Wessels" else "anchor",
        )
        routes["response_focus_not_gasperini_diagnostic"] = (
            False,
            lambda r: response
            if truthy(r["is_gene"])
            and truthy(r["is_multi"])
            and r["dataset"] in {"Wessels", "NormanWeissman2019_filtered"}
            else "anchor",
        )

    def oracle(row: dict[str, Any]) -> str:
        labels = ["anchor", *expert_labels]
        best_label = "anchor"
        best_score = -float("inf")
        for label in labels:
            pp = fnum(row.get(f"{label}__pearson_pert"))
            mmd = fnum(row.get(f"{label}__test_mmd_clamped"))
            if pp is None or mmd is None:
                continue
            score = pp - 0.5 * mmd
            if score > best_score:
                best_score = score
                best_label = label
        return best_label

    routes["oracle_best_pp_minus_half_mmd_diagnostic"] = (False, oracle)
    return routes


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def metric_harm(delta: float, metric: str) -> bool:
    if metric in LOWER_IS_BETTER:
        return delta > 0.0
    return delta < 0.0


def equal_dataset_summary(rows: list[dict[str, Any]], route_name: str, route: Callable[[dict[str, Any]], str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group in GROUPS:
        group_rows = [r for r in rows if r["eval_group"] == group]
        if not group_rows:
            continue
        for metric in METRICS:
            by_dataset: dict[str, list[float]] = defaultdict(list)
            selected_counts: dict[str, int] = defaultdict(int)
            harm_flags: list[bool] = []
            for row in group_rows:
                selected = route(row)
                base = fnum(row.get(f"anchor__{metric}"))
                value = fnum(row.get(f"{selected}__{metric}"))
                if base is None or value is None:
                    continue
                delta = value - base
                by_dataset[str(row["dataset"])].append(delta)
                selected_counts[selected] += 1
                harm_flags.append(metric_harm(delta, metric))
            ds_means = [float(np.mean(values)) for values in by_dataset.values() if values]
            if not ds_means:
                continue
            out.append(
                {
                    "bundle": str(group_rows[0].get("bundle", "")),
                    "route": route_name,
                    "group": group,
                    "metric": metric,
                    "n_conditions": sum(len(v) for v in by_dataset.values()),
                    "n_datasets": len(ds_means),
                    "delta": float(np.mean(ds_means)),
                    "condition_harm_rate": float(np.mean(harm_flags)) if harm_flags else None,
                    "selected_counts": dict(sorted(selected_counts.items())),
                }
            )
    return out


def bootstrap(
    rows: list[dict[str, Any]],
    route_name: str,
    route: Callable[[dict[str, Any]], str],
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for group in ("test", "test_multi_unseen2", "family_gene", "family_drug", "structure_single"):
        group_rows = [r for r in rows if r["eval_group"] == group]
        if not group_rows:
            continue
        for metric in BOOT_METRICS:
            by_dataset: dict[str, list[float]] = defaultdict(list)
            selected_counts: dict[str, int] = defaultdict(int)
            for row in group_rows:
                selected = route(row)
                base = fnum(row.get(f"anchor__{metric}"))
                value = fnum(row.get(f"{selected}__{metric}"))
                if base is None or value is None:
                    continue
                by_dataset[str(row["dataset"])].append(value - base)
                selected_counts[selected] += 1
            datasets = sorted(ds for ds, values in by_dataset.items() if values)
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
                    "bundle": str(group_rows[0].get("bundle", "")),
                    "route": route_name,
                    "group": group,
                    "metric": metric,
                    "n_conditions": sum(len(by_dataset[ds]) for ds in datasets),
                    "n_datasets": len(datasets),
                    "delta": observed,
                    "ci95": [float(lo), float(hi)],
                    "p_improve": p_improve,
                    "p_harm": p_harm,
                    "selected_counts": dict(sorted(selected_counts.items())),
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key_ in row:
            if key_ not in keys:
                keys.append(key_)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Existing-Expert Route Audit",
        "",
        "CPU-only audit across existing posthoc condition metrics. Deployable routes use only condition metadata; diagnostic routes use dataset/outcome-derived choices and cannot be promoted as-is.",
        "",
        f"Condition CSV: `{payload['condition_csv']}`",
        f"Bootstrap: `{payload['n_boot']}` resamples, seed `{payload['seed']}`",
        "",
        "## Bootstrap Gate Table",
        "",
        "| route | deployable | group | metric | n | delta | 95% CI | p improve | p harm | selected experts |",
        "|---|---|---|---|---:|---:|---|---:|---:|---|",
    ]
    deployable = {(r["bundle"], r["route"]): r["deployable"] for r in payload["routes"]}
    for row in payload["bootstrap_rows"]:
        if row["metric"] not in BOOT_METRICS:
            continue
        lines.append(
            "| {route} | {dep} | {group} | {metric} | {n} | {delta} | [{lo}, {hi}] | {pi:.4f} | {ph:.4f} | {counts} |".format(
                route=f"{row['bundle']}::{row['route']}",
                dep="yes" if deployable.get((row["bundle"], row["route"])) else "diagnostic",
                group=row["group"],
                metric=row["metric"],
                n=row["n_conditions"],
                delta=fmt(row["delta"]),
                lo=fmt(row["ci95"][0]),
                hi=fmt(row["ci95"][1]),
                pi=row["p_improve"],
                ph=row["p_harm"],
                counts=json.dumps(row["selected_counts"], sort_keys=True),
            )
        )
    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- `response_uncapped_full_conditions` is the valid full-condition uncapped response-router gate.",
            "- `existing_experts_capped_common_conditions` is a capped/common-condition complementarity diagnostic; it must not be promoted as an uncapped claim.",
            "- A deployable route requires positive test/family pp support and no deterministic MMD harm.",
            "- Diagnostic routes estimate whether expert complementarity exists; they are not legal deployment rules without a train-only router/covariate gate.",
            "- If the oracle is not materially cleaner than simple gene-multi routes, multi-expert routing should be deprioritized.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-prefix", type=Path, default=ROOT / "reports/latentfm_existing_expert_route_audit_20260621")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bundles = default_bundles()
    all_condition_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    route_meta: list[dict[str, Any]] = []
    for bundle in bundles:
        rows = build_rows(bundle)
        all_condition_rows.extend(rows)
        expert_labels = sorted(bundle.experts)
        route_map = route_defs(expert_labels)
        for name, (deployable, route) in route_map.items():
            route_meta.append({"bundle": bundle.name, "route": name, "deployable": deployable})
            summary_rows.extend(
                {
                    **row,
                    "deployable": deployable,
                }
                for row in equal_dataset_summary(rows, name, route)
            )
            boot_rows.extend(
                {
                    **row,
                    "deployable": deployable,
                }
                for row in bootstrap(rows, name, route, args.n_boot, args.seed)
            )

    out_csv = args.out_prefix.with_suffix(".conditions.csv")
    out_json = args.out_prefix.with_suffix(".json")
    out_md = args.out_prefix.with_suffix(".md")
    write_csv(out_csv, all_condition_rows)
    payload = {
        "bundles": [
            {
                "bundle": bundle.name,
                "baseline": bundle.baseline.__dict__
                | {"split_json": str(bundle.baseline.split_json), "family_json": str(bundle.baseline.family_json)},
                "experts": {
                    label: expert.__dict__ | {"split_json": str(expert.split_json), "family_json": str(expert.family_json)}
                    for label, expert in bundle.experts.items()
                },
            }
            for bundle in bundles
        ],
        "routes": route_meta,
        "condition_csv": str(out_csv),
        "summary_rows": summary_rows,
        "bootstrap_rows": boot_rows,
        "n_boot": args.n_boot,
        "seed": args.seed,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), "out_csv": str(out_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
