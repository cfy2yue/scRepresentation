#!/usr/bin/env python3
"""CPU-only gate for Track C route-focused sampling options.

The previous routed-distill smoke failed partly because the route teacher was
almost never exposed by the all-dataset sampler. This audit tests whether
existing split/sampler knobs can create a no-query, route-focused training view
that keeps train-single banks available for the Norman additive route.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
REPO = ROOT / "CoupledFM"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from model.latent.dataset import CrossDatasetFMDataset  # noqa: E402
from model.latent.train import _multi_gene_composition_key, _single_gene_composition_key  # noqa: E402


DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
DEFAULT_ROUTE_FILE = ROOT / "reports/latentfm_trackc_support_route_teacher_20260622.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_trackc_route_focused_sampling_gate_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_TRACKC_ROUTE_FOCUSED_SAMPLING_GATE_20260622.md"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def route_dataset_split(split: dict[str, Any], routes: dict[str, str]) -> dict[str, Any]:
    return {ds: deepcopy(split[ds]) for ds in sorted(routes) if ds in split}


def make_dataset(args: argparse.Namespace, split: dict[str, Any], *, ds_alpha: float, min_selected: int) -> CrossDatasetFMDataset:
    return CrossDatasetFMDataset(
        str(args.data_dir),
        {ds: sp for ds, sp in split.items() if sp.get("train")},
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        mode="train",
        min_cells=int(args.min_cells),
        ds_alpha=float(ds_alpha),
        scale_noise=0.01,
        min_selected_conditions_per_dataset=int(min_selected),
        condition_visit_power=1.0,
        condition_visit_cap=0,
        use_pert_condition=True,
        max_pert_genes=16,
        gene_embedding_cache_dir=str(args.gene_cache),
        biflow_dir=str(args.biflow_dir),
        use_h5ad_pert_metadata=False,
        latent_backbone="xverse",
        pert_chem_enabled=True,
        perturbation_family_filter="all",
        silent=True,
    )


def gene_coverage(ds: CrossDatasetFMDataset) -> tuple[dict[str, set[str]], set[str]]:
    by_ds: dict[str, set[str]] = defaultdict(set)
    global_genes: set[str] = set()
    for ds_name in ds.ds_names:
        for cond in ds.ds_conds.get(ds_name, []):
            key = _single_gene_composition_key(ds.metadata_for_condition(ds_name, cond))
            if key is None:
                continue
            gene = str(key[0]).strip().upper()
            if not gene:
                continue
            by_ds[str(ds_name)].add(gene)
            global_genes.add(gene)
    return by_ds, global_genes


def target_status(
    ds: CrossDatasetFMDataset,
    routes: dict[str, str],
    by_ds: dict[str, set[str]],
    global_genes: set[str],
    ds_name: str,
    cond: str,
) -> tuple[bool, str]:
    route = routes.get(str(ds_name), "")
    if not route:
        return False, "dataset_not_routed"
    genes = _multi_gene_composition_key(ds.metadata_for_condition(ds_name, cond))
    if genes is None:
        return False, "not_gene_multi"
    if route == "dataset_multi_mean":
        return True, "dataset_multi_mean"
    if route == "additive_single_sum":
        missing = [
            gene for gene in genes
            if gene not in by_ds.get(str(ds_name), set()) and gene not in global_genes
        ]
        return (False, "additive_gene_missing") if missing else (True, "additive_single_sum")
    return False, f"unsupported_route:{route}"


def count_order(
    ds: CrossDatasetFMDataset,
    routes: dict[str, str],
    by_ds: dict[str, set[str]],
    global_genes: set[str],
    pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    trigger_steps = 0
    reason_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    condition_counts: Counter[tuple[str, str]] = Counter()
    for ds_name, cond in pairs:
        ok, reason = target_status(ds, routes, by_ds, global_genes, ds_name, cond)
        reason_counts[reason] += 1
        if not ok:
            continue
        trigger_steps += 1
        route_counts[routes[str(ds_name)]] += 1
        dataset_counts[str(ds_name)] += 1
        condition_counts[(str(ds_name), str(cond))] += 1
    total = len(pairs)
    return {
        "total_steps": int(total),
        "route_trigger_steps": int(trigger_steps),
        "route_trigger_fraction": float(trigger_steps / total) if total else 0.0,
        "unique_route_conditions": int(len(condition_counts)),
        "by_dataset_steps": dict(sorted(dataset_counts.items())),
        "by_route_steps": dict(sorted(route_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def build_training_prefix(
    ds: CrossDatasetFMDataset,
    *,
    seed: int,
    n_steps: int,
) -> list[tuple[str, str]]:
    """Reconstruct the first training-window condition visits across epochs."""
    out: list[tuple[str, str]] = []
    epoch = 0
    while len(out) < n_steps:
        order = ds._build_epoch_order(np.random.RandomState(seed + epoch))
        if not order:
            break
        out.extend(order)
        epoch += 1
    return out[:n_steps]


def audit_scenario(
    args: argparse.Namespace,
    name: str,
    split: dict[str, Any],
    bank_split: dict[str, Any],
    routes: dict[str, str],
    *,
    ds_alpha: float,
    min_selected: int,
    requires_code: bool = False,
) -> dict[str, Any]:
    ds = make_dataset(args, split, ds_alpha=ds_alpha, min_selected=min_selected)
    bank_ds = ds if bank_split is split else make_dataset(args, bank_split, ds_alpha=0.7, min_selected=0)
    by_ds, global_genes = gene_coverage(bank_ds)
    order = ds._build_epoch_order(np.random.RandomState(int(args.seed)))
    prefix_order = build_training_prefix(ds, seed=int(args.seed), n_steps=int(args.prefix_steps))
    route_multi_conditions: Counter[str] = Counter()
    target_available_conditions: Counter[str] = Counter()
    for ds_name in ds.ds_names:
        for cond in ds.ds_conds.get(ds_name, []):
            genes = _multi_gene_composition_key(ds.metadata_for_condition(ds_name, cond))
            if genes is None:
                continue
            if str(ds_name) in routes:
                route_multi_conditions[str(ds_name)] += 1
                ok, _ = target_status(ds, routes, by_ds, global_genes, ds_name, cond)
                if ok:
                    target_available_conditions[str(ds_name)] += 1
    prefix = count_order(ds, routes, by_ds, global_genes, prefix_order)
    full = count_order(ds, routes, by_ds, global_genes, order)
    return {
        "name": name,
        "split_datasets": sorted(split),
        "train_conditions": int(ds.total_conditions),
        "epoch_steps": int(ds.epoch_steps),
        "ds_alpha": float(ds_alpha),
        "min_selected_conditions_per_dataset": int(min_selected),
        "requires_code": bool(requires_code),
        "bank_source_datasets": sorted(bank_split),
        "bank_source_train_conditions": int(bank_ds.total_conditions),
        "single_gene_coverage_routed_datasets": {key: len(by_ds.get(key, set())) for key in sorted(routes)},
        "route_multi_conditions": dict(sorted(route_multi_conditions.items())),
        "target_available_conditions": dict(sorted(target_available_conditions.items())),
        "prefix_steps": prefix,
        "full_epoch": full,
    }


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    no_code_candidates = []
    design_candidates = []
    for row in payload["scenarios"]:
        prefix = row["prefix_steps"]
        target_total = sum(int(v) for v in row["target_available_conditions"].values())
        no_code = row["name"] in {"route_datasets_dsalpha1", "route_datasets_minall"}
        passed = (
            float(prefix["route_trigger_fraction"]) >= 0.10
            and int(prefix["unique_route_conditions"]) >= 30
            and target_total >= 40
        )
        if passed:
            if no_code and not bool(row.get("requires_code")):
                no_code_candidates.append(row["name"])
            else:
                design_candidates.append(row["name"])
    if not no_code_candidates and not design_candidates:
        reasons.append("no_existing_split_sampler_variant_reaches_route_focus_gate")
    if no_code_candidates:
        status = "route_focused_sampling_gate_pass_existing_config"
        action = "prepare_route_dataset_trainselect_split_and_gpu_smoke_protocol_review"
    elif design_candidates:
        status = "route_focused_sampling_design_gate_pass_requires_bank_split_code"
        action = "implement_default_off_trackc_bank_split_then_protocol_review"
    else:
        status = "route_focused_sampling_gate_fail"
        action = "implement_default_off_route_focused_sampler_before_gpu"
    return {
        "status": status,
        "action": action,
        "passing_existing_config_scenarios": no_code_candidates,
        "passing_design_scenarios": design_candidates,
        "reasons": reasons,
    }


def write_md(payload: dict[str, Any], path: Path) -> None:
    decision = payload["decision"]
    lines = [
        "# LatentFM Track C Route-Focused Sampling Gate",
        "",
        f"Status: `{decision['status']}`",
        f"Recommended action: `{decision['action']}`",
        "",
        "## Provenance",
        "",
        f"- split_file: `{payload['provenance']['split_file']}`",
        f"- route_file: `{payload['provenance']['route_file']}`",
        f"- heldout_query_used: `{payload['provenance']['heldout_query_used']}`",
        f"- prefix_steps: `{payload['prefix_steps_requested']}`",
        "",
        "## Scenario Table",
        "",
        "| scenario | code? | datasets | train conds | bank conds | epoch steps | first-window route steps | fraction | unique route conds | target conds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["scenarios"]:
        prefix = row["prefix_steps"]
        target_total = sum(int(v) for v in row["target_available_conditions"].values())
        lines.append(
            f"| `{row['name']}` | {'yes' if row.get('requires_code') else 'no'} | "
            f"{len(row['split_datasets'])} | {row['train_conditions']} | "
            f"{row['bank_source_train_conditions']} | {row['epoch_steps']} | "
            f"{prefix['route_trigger_steps']} | "
            f"{prefix['route_trigger_fraction']:+.6f} | {prefix['unique_route_conditions']} | "
            f"{target_total} |"
        )
    lines += [
        "",
        "## Gate Reasons",
        "",
    ]
    if decision["reasons"]:
        lines.extend(f"- `{reason}`" for reason in decision["reasons"])
    else:
        lines.append("- none")
    lines += [
        "",
        "## Interpretation",
        "",
        "- A multi-only train split is not sufficient for Norman because the additive route needs train-single gene means.",
        "- The route-dataset-only variants keep Norman/Wessels train singles and train multis together while leaving support-val in `test` for selection.",
        "- The full-bank design keeps the train sampler route-focused while building routed teacher banks from the full trainselect train set; it still does not read held-out query.",
        "- Passing this gate does not authorize query evaluation. It only authorizes protocol review for a new route-focused Track C smoke.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--route-file", type=Path, default=DEFAULT_ROUTE_FILE)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--biflow-dir", type=Path, default=ROOT / "dataset/biFlow_data")
    parser.add_argument("--gene-cache", type=Path, default=ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-cells", type=int, default=16)
    parser.add_argument("--prefix-steps", type=int, default=2000)
    args = parser.parse_args()

    split = load_json(args.split_file)
    route_payload = load_json(args.route_file)
    routes = {str(k): str(v) for k, v in (route_payload.get("route") or route_payload).items()}
    route_split = route_dataset_split(split, routes)
    scenarios = [
        audit_scenario(args, "full_trainselect_current", split, split, routes, ds_alpha=0.7, min_selected=0),
        audit_scenario(args, "full_trainselect_dsalpha1", split, split, routes, ds_alpha=1.0, min_selected=0),
        audit_scenario(args, "route_datasets_current", route_split, route_split, routes, ds_alpha=0.7, min_selected=0),
        audit_scenario(args, "route_datasets_dsalpha1", route_split, route_split, routes, ds_alpha=1.0, min_selected=0),
        audit_scenario(args, "route_datasets_minall", route_split, route_split, routes, ds_alpha=0.7, min_selected=10**9),
        audit_scenario(
            args,
            "route_datasets_current_full_trainselect_bank",
            route_split,
            split,
            routes,
            ds_alpha=0.7,
            min_selected=0,
            requires_code=True,
        ),
        audit_scenario(
            args,
            "route_datasets_dsalpha1_full_trainselect_bank",
            route_split,
            split,
            routes,
            ds_alpha=1.0,
            min_selected=0,
            requires_code=True,
        ),
    ]
    payload = {
        "provenance": {
            "split_file": str(args.split_file),
            "route_file": str(args.route_file),
            "heldout_query_used": False,
            "usage_rule": "trainselect split only; no Track C held-out query and no canonical multi selection",
        },
        "routes": routes,
        "prefix_steps_requested": int(args.prefix_steps),
        "scenarios": scenarios,
    }
    payload["decision"] = decide(payload)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(payload, args.out_md)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
