#!/usr/bin/env python3
"""Condition-level route audit for LatentFM response/pairwise branches."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path("/data/cyx/1030/scLatent")
METRICS = ("pearson_pert", "direct_pearson", "test_mmd_clamped")
GROUPS = ("test", "test_multi_unseen2", "family_gene", "family_drug", "structure_single", "structure_multi")
LOWER_IS_BETTER = {"test_mmd_clamped"}


@dataclass(frozen=True)
class Comparison:
    name: str
    baseline_label: str
    candidate_label: str
    baseline_split_json: Path
    candidate_split_json: Path
    baseline_family_json: Path
    candidate_family_json: Path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    try:
        if value is None:
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
        rows = group_obj.get("condition_metrics") or []
        for row in rows:
            if isinstance(row, dict):
                out[key(row)].add(str(group_name))
    return out


def infer_nperts(condition: str, memberships: set[str]) -> int:
    if "structure_single" in memberships or "test_single" in memberships:
        return 1
    if "structure_multi" in memberships or "test_multi" in memberships:
        return max(2, condition.count("+") + 1)
    if "+" in condition:
        return condition.count("+") + 1
    return 1


def route_allowed_features(memberships: set[str]) -> str:
    bits = []
    for name in ("family_gene", "family_drug", "structure_single", "structure_multi"):
        if name in memberships:
            bits.append(name)
    return ",".join(bits)


def equal_dataset_mean(rows: list[dict[str, Any]], metric: str, routed: bool) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    key_name = f"{metric}_{'routed' if routed else 'base'}"
    for row in rows:
        value = fnum(row.get(key_name))
        if value is not None:
            by_ds[str(row["dataset"])].append(value)
    ds_means = [sum(vals) / len(vals) for vals in by_ds.values() if vals]
    if not ds_means:
        return None
    return sum(ds_means) / len(ds_means)


def metric_harm(delta: float, metric: str) -> bool:
    if metric in LOWER_IS_BETTER:
        return delta > 0.0
    return delta < 0.0


def route_delta_rates(rows: list[dict[str, Any]], metric: str) -> dict[str, float | int | None]:
    deltas: list[float] = []
    candidate_deltas: list[float] = []
    for row in rows:
        base = fnum(row.get(f"{metric}_base"))
        routed = fnum(row.get(f"{metric}_routed"))
        if base is None or routed is None:
            continue
        delta = routed - base
        deltas.append(delta)
        if row.get("route_uses_candidate"):
            candidate_deltas.append(delta)

    def rates(values: list[float]) -> tuple[int, float | None, float | None]:
        if not values:
            return 0, None, None
        harm = sum(1 for value in values if metric_harm(value, metric))
        improve = sum(1 for value in values if value != 0.0 and not metric_harm(value, metric))
        return len(values), harm / len(values), improve / len(values)

    n_all, harm_all, improve_all = rates(deltas)
    n_cand, harm_cand, improve_cand = rates(candidate_deltas)
    return {
        "condition_delta_n": n_all,
        "condition_harm_rate": harm_all,
        "condition_improve_rate": improve_all,
        "candidate_condition_delta_n": n_cand,
        "candidate_condition_harm_rate": harm_cand,
        "candidate_condition_improve_rate": improve_cand,
    }


def build_default_comparisons() -> list[Comparison]:
    response = load_json(ROOT / "runs/latentfm_response_normalization_20260621/posthoc_manifest.json")
    pairwise = load_json(ROOT / "runs/latentfm_pairwise_condition_20260621/posthoc_manifest.json")
    pair_vs_refine = load_json(
        ROOT / "runs/latentfm_pairwise_condition_20260621/pairwise_vs_canonical_refinetune_manifest.json"
    )["launched_runs"][0]
    fewshot = load_json(ROOT / "runs/latentfm_fewshot_multi_calibration_20260621/posthoc_manifest.json")
    canonical = next(r for r in fewshot["launched_runs"] if r.get("arm") == "canonical_refinetune")

    comparisons = [
        Comparison(
            name="canonical_refinetune_vs_anchor",
            baseline_label="anchor",
            candidate_label="canonical_refinetune",
            baseline_split_json=Path(canonical["baseline_split_json"]),
            candidate_split_json=Path(canonical["run_split_json"]),
            baseline_family_json=Path(canonical["baseline_family_json"]),
            candidate_family_json=Path(canonical["run_family_json"]),
        ),
        Comparison(
            name="response_aux1_vs_anchor",
            baseline_label="anchor",
            candidate_label="response_aux1",
            baseline_split_json=Path(response["baseline_split_json"]),
            candidate_split_json=Path(response["run_split_json"]),
            baseline_family_json=Path(response["baseline_family_json"]),
            candidate_family_json=Path(response["run_family_json"]),
        ),
        Comparison(
            name="pairwise_full_vs_anchor",
            baseline_label="anchor",
            candidate_label="pairwise_full",
            baseline_split_json=Path(pairwise["baseline_split_json"]),
            candidate_split_json=Path(pairwise["run_split_json"]),
            baseline_family_json=Path(pairwise["baseline_family_json"]),
            candidate_family_json=Path(pairwise["run_family_json"]),
        ),
        Comparison(
            name="pairwise_full_vs_canonical_refinetune",
            baseline_label="canonical_refinetune",
            candidate_label="pairwise_full",
            baseline_split_json=Path(pair_vs_refine["baseline_split_json"]),
            candidate_split_json=Path(pair_vs_refine["run_split_json"]),
            baseline_family_json=Path(pair_vs_refine["baseline_family_json"]),
            candidate_family_json=Path(pair_vs_refine["run_family_json"]),
        ),
    ]
    optional_manifests = [
        (
            "pairwise_adapteronly_vs_anchor",
            "anchor",
            "pairwise_adapteronly",
            ROOT / "runs/latentfm_pairwise_adapter_only_20260621/posthoc_manifest.json",
        ),
        (
            "response_aux025_vs_anchor",
            "anchor",
            "response_aux025",
            ROOT / "runs/latentfm_response_constrained_20260621/scf_response_dataset_scale_pca32_aux025_4k/posthoc_manifest.json",
        ),
        (
            "response_aux05_vs_anchor",
            "anchor",
            "response_aux05",
            ROOT / "runs/latentfm_response_constrained_20260621/scf_response_dataset_scale_pca32_aux05_4k/posthoc_manifest.json",
        ),
    ]
    for name, baseline_label, candidate_label, path in optional_manifests:
        if not path.is_file():
            continue
        manifest = load_json(path)
        comparisons.append(
            Comparison(
                name=name,
                baseline_label=baseline_label,
                candidate_label=candidate_label,
                baseline_split_json=Path(manifest["baseline_split_json"]),
                candidate_split_json=Path(manifest["run_split_json"]),
                baseline_family_json=Path(manifest["baseline_family_json"]),
                candidate_family_json=Path(manifest["run_family_json"]),
            )
        )
    return comparisons


def comparisons_from_uncapped_index(path: Path) -> list[Comparison]:
    payload = load_json(path)
    rows = payload.get("outputs") or []
    comparisons: list[Comparison] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        run_name = str(row.get("run_name") or f"row{len(comparisons)}")
        comparisons.append(
            Comparison(
                name=f"{run_name}_uncapped_vs_anchor",
                baseline_label="anchor_uncapped",
                candidate_label=f"{run_name}_uncapped",
                baseline_split_json=Path(row["baseline_split_json"]),
                candidate_split_json=Path(row["run_split_json"]),
                baseline_family_json=Path(row["baseline_family_json"]),
                candidate_family_json=Path(row["run_family_json"]),
            )
        )
    return comparisons


def comparison_from_posthoc_manifest(path: Path) -> Comparison:
    manifest = load_json(path)
    run_name = str(manifest.get("run_name") or path.parent.name)
    return Comparison(
        name=f"{run_name}_vs_anchor",
        baseline_label="anchor",
        candidate_label=run_name,
        baseline_split_json=Path(manifest["baseline_split_json"]),
        candidate_split_json=Path(manifest["run_split_json"]),
        baseline_family_json=Path(manifest["baseline_family_json"]),
        candidate_family_json=Path(manifest["run_family_json"]),
    )


def comparison_rows(comp: Comparison) -> list[dict[str, Any]]:
    base_split = load_json(comp.baseline_split_json)
    cand_split = load_json(comp.candidate_split_json)
    base_family = load_json(comp.baseline_family_json)
    cand_family = load_json(comp.candidate_family_json)

    base_members = group_membership(base_family)
    cand_members = group_membership(cand_family)
    split_members = group_membership(base_split)

    rows: list[dict[str, Any]] = []
    for eval_group in GROUPS:
        base_payload = base_split if eval_group in (base_split.get("groups") or {}) else base_family
        cand_payload = cand_split if eval_group in (cand_split.get("groups") or {}) else cand_family
        if eval_group not in (base_payload.get("groups") or {}) or eval_group not in (cand_payload.get("groups") or {}):
            continue
        base_group = condition_index(base_payload, eval_group)
        cand_group = condition_index(cand_payload, eval_group)
        for cond_key in sorted(set(base_group) & set(cand_group)):
            ds, cond = cond_key
            memberships = set(base_members.get(cond_key, set())) | set(split_members.get(cond_key, set()))
            cand_mem = set(cand_members.get(cond_key, set()))
            selected_match = memberships == (cand_mem | set(split_members.get(cond_key, set())))
            row: dict[str, Any] = {
                "comparison": comp.name,
                "eval_group": eval_group,
                "baseline_label": comp.baseline_label,
                "candidate_label": comp.candidate_label,
                "dataset": ds,
                "condition": cond,
                "nperts_est": infer_nperts(cond, memberships),
                "route_features": route_allowed_features(memberships),
                "is_gene": "family_gene" in memberships,
                "is_drug": "family_drug" in memberships,
                "is_single": "structure_single" in memberships,
                "is_multi": "structure_multi" in memberships,
                "is_test_multi_unseen2": "test_multi_unseen2" in memberships,
                "selected_membership_match": selected_match,
            }
            for metric in METRICS:
                b = fnum(base_group[cond_key].get(metric))
                c = fnum(cand_group[cond_key].get(metric))
                row[f"{metric}_base"] = b
                row[f"{metric}_candidate"] = c
                row[f"{metric}_delta"] = None if b is None or c is None else c - b
            rows.append(row)
    return rows


def routes() -> dict[str, Callable[[dict[str, Any]], bool]]:
    return {
        "candidate_all": lambda r: True,
        "candidate_gene_multi": lambda r: bool(r["is_gene"] and r["is_multi"]),
        "candidate_multi_not_drug": lambda r: bool(r["is_multi"] and not r["is_drug"]),
        "candidate_multi_only": lambda r: bool(r["is_multi"]),
        "candidate_focus_multi_diagnostic": lambda r: bool(
            r["is_multi"]
            and r["dataset"] in {"Wessels", "NormanWeissman2019_filtered", "GasperiniShendure2019_lowMOI"}
        ),
        "candidate_unseen2_diagnostic": lambda r: bool(r["is_test_multi_unseen2"]),
    }


def summarize_routes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    route_defs = routes()
    for comp_name in sorted({r["comparison"] for r in rows}):
        comp_rows = [r for r in rows if r["comparison"] == comp_name]
        for route_name, pred in route_defs.items():
            routed_rows = []
            for row in comp_rows:
                rr = dict(row)
                use_cand = pred(row)
                rr["route_uses_candidate"] = use_cand
                for metric in METRICS:
                    rr[f"{metric}_routed"] = row[f"{metric}_{'candidate' if use_cand else 'base'}"]
                routed_rows.append(rr)
            for group in GROUPS:
                subset = [r for r in routed_rows if r["eval_group"] == group]
                if not subset:
                    continue
                for metric in METRICS:
                    base = equal_dataset_mean(subset, metric, routed=False)
                    routed = equal_dataset_mean(subset, metric, routed=True)
                    delta = None if base is None or routed is None else routed - base
                    rates = route_delta_rates(subset, metric)
                    out.append(
                        {
                            "comparison": comp_name,
                            "route": route_name,
                            "group": group,
                            "metric": metric,
                            "n_conditions": len(subset),
                            "n_datasets": len({r["dataset"] for r in subset}),
                            "candidate_conditions": sum(1 for r in subset if r["route_uses_candidate"]),
                            "base_equal_dataset_mean": base,
                            "routed_equal_dataset_mean": routed,
                            "delta": delta,
                            **rates,
                            "route_is_deployable_feature_rule": not route_name.endswith("_diagnostic"),
                        }
                    )
    return out


def by_dataset_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for comp in sorted({r["comparison"] for r in rows}):
        for ds in sorted({r["dataset"] for r in rows if r["comparison"] == comp}):
            subset = [r for r in rows if r["comparison"] == comp and r["dataset"] == ds]
            for flag in GROUPS:
                ss = [r for r in subset if r["eval_group"] == flag]
                if not ss:
                    continue
                item = {"comparison": comp, "dataset": ds, "stratum": flag, "n_conditions": len(ss)}
                for metric in METRICS:
                    vals = [fnum(r.get(f"{metric}_delta")) for r in ss]
                    vals = [v for v in vals if v is not None]
                    item[f"{metric}_delta_mean"] = None if not vals else sum(vals) / len(vals)
                out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def render_md(route_rows: list[dict[str, Any]], dataset_rows: list[dict[str, Any]], out_json: Path, out_csv: Path) -> str:
    lines = [
        "# LatentFM Condition Route Audit",
        "",
        "CPU-only audit. No training or GPU jobs are launched.",
        "",
        f"JSON: `{out_json}`",
        f"Condition CSV: `{out_csv}`",
        "",
        "## Route Summary",
        "",
        "Deployable routes use only condition metadata available before seeing held-out GT.",
        "Diagnostic routes are marked and must not be used as final routing rules.",
        "",
        "| comparison | route | deployable | group | metric | n | cand n | delta | harm rate | cand harm |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    keep_metrics = {"pearson_pert", "test_mmd_clamped"}
    keep_groups = {"test", "test_multi_unseen2", "family_gene", "family_drug", "structure_single"}
    for r in route_rows:
        if r["metric"] not in keep_metrics or r["group"] not in keep_groups:
            continue
        lines.append(
            "| {comparison} | {route} | {dep} | {group} | {metric} | {n} | {cn} | {delta} | {harm} | {charm} |".format(
                comparison=r["comparison"],
                route=r["route"],
                dep="yes" if r["route_is_deployable_feature_rule"] else "diagnostic",
                group=r["group"],
                metric=r["metric"],
                n=r["n_conditions"],
                cn=r["candidate_conditions"],
                delta=fmt(r["delta"]),
                harm=fmt(r.get("condition_harm_rate")),
                charm=fmt(r.get("candidate_condition_harm_rate")),
            )
        )
    lines.extend(["", "## Dataset Highlights", ""])
    focus = {"Wessels", "NormanWeissman2019_filtered", "GasperiniShendure2019_lowMOI"}
    lines.extend(
        [
            "| comparison | dataset | stratum | n | pp delta | MMD delta |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for r in dataset_rows:
        if r["dataset"] not in focus:
            continue
        if r["stratum"] not in {"test", "test_multi_unseen2", "family_gene", "family_drug", "structure_single", "structure_multi"}:
            continue
        lines.append(
            "| {comparison} | {dataset} | {stratum} | {n} | {pp} | {mmd} |".format(
                comparison=r["comparison"],
                dataset=r["dataset"].replace("NormanWeissman2019_filtered", "Norman").replace(
                    "GasperiniShendure2019_lowMOI", "Gasperini"
                ),
                stratum=r["stratum"],
                n=r["n_conditions"],
                pp=fmt(r.get("pearson_pert_delta_mean")),
                mmd=fmt(r.get("test_mmd_clamped_delta_mean")),
            )
        )
    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- `candidate_gene_multi` and `candidate_multi_not_drug` are the first deployable route candidates.",
            "- `candidate_unseen2_diagnostic` estimates the upper-bound of split-aware routing and is not deployable.",
            "- If deployable multi/gene routes still show deterministic aggregate MMD harm, the next GPU branch must use frozen/slow-base adapters rather than routing alone.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=ROOT / "reports/latentfm_condition_route_audit_20260621",
    )
    parser.add_argument(
        "--uncapped-index",
        type=Path,
        action="append",
        default=[],
        help="Optional uncapped_posthoc_index.json files to include as comparisons.",
    )
    parser.add_argument(
        "--posthoc-manifest",
        type=Path,
        action="append",
        default=[],
        help="Optional single-run posthoc_manifest.json files to include as comparisons.",
    )
    parser.add_argument(
        "--include-defaults",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include built-in capped comparison manifests.",
    )
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    comparisons: list[Comparison] = []
    if args.include_defaults:
        comparisons.extend(build_default_comparisons())
    for index_path in args.uncapped_index:
        comparisons.extend(comparisons_from_uncapped_index(index_path))
    for manifest_path in args.posthoc_manifest:
        comparisons.append(comparison_from_posthoc_manifest(manifest_path))
    for comp in comparisons:
        all_rows.extend(comparison_rows(comp))

    route_rows = summarize_routes(all_rows)
    dataset_rows = by_dataset_summary(all_rows)

    out_csv = args.out_prefix.with_suffix(".conditions.csv")
    out_json = args.out_prefix.with_suffix(".json")
    out_md = args.out_prefix.with_suffix(".md")
    write_csv(out_csv, all_rows)
    payload = {
        "comparisons": [c.__dict__ | {k: str(v) for k, v in c.__dict__.items() if isinstance(v, Path)} for c in comparisons],
        "n_condition_group_rows": len(all_rows),
        "route_summary": route_rows,
        "dataset_summary": dataset_rows,
        "condition_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_md(route_rows, dataset_rows, out_json, out_csv), encoding="utf-8")
    print(json.dumps({"out_md": str(out_md), "out_json": str(out_json), "out_csv": str(out_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
