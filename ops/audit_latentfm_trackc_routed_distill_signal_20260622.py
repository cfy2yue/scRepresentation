#!/usr/bin/env python3
"""Audit why the Track C routed-distill smoke captured little route signal.

This is CPU-only and read-only. It does not evaluate held-out Track C query
conditions. The audit reconstructs the same training dataset/sampler used by
the routed-distill smoke and measures how often the routed teacher loss could
actually fire under the current 2k-step schedule.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
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
DEFAULT_RUN_ROOT = (
    ROOT
    / "runs/latentfm_xverse_trackc_routed_distill_20260622/"
    "xverse_trackc_route_condprior_w05_replay1_2k_seed42"
)
DEFAULT_BANK_SUMMARY = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_trackc_routed_distill_20260622/"
    "xverse_trackc_route_condprior_w05_replay1_2k_seed42/trackc_routed_distill_bank_summary.json"
)
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_trackc_routed_distill_signal_audit_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_TRACKC_ROUTED_DISTILL_SIGNAL_AUDIT_20260622.md"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_dataset(args: argparse.Namespace, split: dict[str, Any]) -> CrossDatasetFMDataset:
    return CrossDatasetFMDataset(
        str(args.data_dir),
        {ds: sp for ds, sp in split.items() if sp.get("train")},
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        mode="train",
        min_cells=int(args.min_cells),
        ds_alpha=float(args.ds_alpha),
        scale_noise=0.01,
        min_selected_conditions_per_dataset=int(args.min_selected_conditions_per_dataset),
        condition_visit_power=float(args.condition_visit_power),
        condition_visit_cap=int(args.condition_visit_cap),
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


def build_gene_coverage(ds: CrossDatasetFMDataset) -> tuple[dict[str, set[str]], set[str]]:
    by_dataset: dict[str, set[str]] = defaultdict(set)
    global_genes: set[str] = set()
    for ds_name in ds.ds_names:
        for cond in ds.ds_conds.get(ds_name, []):
            meta = ds.metadata_for_condition(ds_name, cond)
            single_key = _single_gene_composition_key(meta)
            if single_key is None:
                continue
            gene = str(single_key[0]).strip().upper()
            if not gene:
                continue
            by_dataset[str(ds_name)].add(gene)
            global_genes.add(gene)
    return by_dataset, global_genes


def route_target_available(
    ds: CrossDatasetFMDataset,
    routes: dict[str, str],
    single_by_dataset: dict[str, set[str]],
    global_single: set[str],
    ds_name: str,
    cond: str,
) -> tuple[bool, str, tuple[str, ...]]:
    route = routes.get(str(ds_name), "")
    if not route:
        return False, "dataset_not_routed", ()
    meta = ds.metadata_for_condition(ds_name, cond)
    genes = _multi_gene_composition_key(meta)
    if genes is None:
        return False, "not_gene_multi", ()
    if route == "dataset_multi_mean":
        return True, "dataset_multi_mean", genes
    if route == "additive_single_sum":
        missing = [
            gene for gene in genes
            if gene not in single_by_dataset.get(str(ds_name), set()) and gene not in global_single
        ]
        if missing:
            return False, "additive_gene_missing", genes
        return True, "additive_single_sum", genes
    return False, f"unsupported_route:{route}", genes


def summarize_order(
    ds: CrossDatasetFMDataset,
    routes: dict[str, str],
    single_by_dataset: dict[str, set[str]],
    global_single: set[str],
    order: list[tuple[str, str]],
    *,
    prefix_steps: int,
) -> dict[str, Any]:
    def count_slice(pairs: list[tuple[str, str]]) -> dict[str, Any]:
        reason_counts: Counter[str] = Counter()
        dataset_counts: Counter[str] = Counter()
        route_counts: Counter[str] = Counter()
        condition_counts: Counter[tuple[str, str]] = Counter()
        gene_counts: Counter[str] = Counter()
        trigger_steps = 0
        for ds_name, cond in pairs:
            ok, reason, genes = route_target_available(
                ds,
                routes,
                single_by_dataset,
                global_single,
                ds_name,
                cond,
            )
            reason_counts[reason] += 1
            if not ok:
                continue
            trigger_steps += 1
            dataset_counts[str(ds_name)] += 1
            route_counts[routes[str(ds_name)]] += 1
            condition_counts[(str(ds_name), str(cond))] += 1
            for gene in genes:
                gene_counts[str(gene)] += 1
        total = len(pairs)
        return {
            "total_steps": int(total),
            "route_trigger_steps": int(trigger_steps),
            "route_trigger_fraction": float(trigger_steps / total) if total else 0.0,
            "unique_route_conditions": int(len(condition_counts)),
            "by_dataset_steps": dict(sorted(dataset_counts.items())),
            "by_route_steps": dict(sorted(route_counts.items())),
            "nontrigger_reasons": dict(sorted(reason_counts.items())),
            "top_route_conditions": [
                {"dataset": ds_name, "condition": cond, "visits": int(n)}
                for (ds_name, cond), n in condition_counts.most_common(10)
            ],
            "top_route_genes": [
                {"gene": gene, "visits": int(n)}
                for gene, n in gene_counts.most_common(15)
            ],
        }

    return {
        "prefix_steps": count_slice(order[:prefix_steps]),
        "full_epoch": count_slice(order),
    }


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    prefix = payload["sampler_audit"]["prefix_steps"]
    frac = float(prefix["route_trigger_fraction"])
    n_unique = int(prefix["unique_route_conditions"])
    reasons: list[str] = []
    if frac < 0.05:
        reasons.append("route_loss_trigger_fraction_below_5pct_in_2k_smoke")
    if n_unique < 20:
        reasons.append("route_loss_unique_condition_coverage_below_20_in_2k_smoke")
    if payload["observed_training"].get("avg_trackc_route_final", 0.0) < 0.001:
        reasons.append("observed_avg_trackc_route_loss_near_zero")
    status = (
        "route_distill_signal_diluted_close_same_sampler_rerun"
        if reasons
        else "route_distill_signal_not_diluted_requires_deeper_code_audit"
    )
    action = (
        "do_not_rerun_same_trackc_routed_distill; redesign_route_focused_sampler_or_adapter_then_cpu_audit"
        if reasons
        else "perform_model_path_gradient_audit_before_new_gpu"
    )
    return {"status": status, "action": action, "reasons": reasons}


def parse_final_avg_trackc_route(log_path: Path) -> float:
    if not log_path.is_file():
        return 0.0
    value = 0.0
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        token = "avg_trackc_route="
        if token not in line:
            continue
        after = line.split(token, 1)[1].split()[0]
        try:
            value = float(after)
        except ValueError:
            continue
    return value


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    decision = payload["decision"]
    prefix = payload["sampler_audit"]["prefix_steps"]
    full_epoch = payload["sampler_audit"]["full_epoch"]
    lines = [
        "# LatentFM Track C Routed-Distill Signal Audit",
        "",
        f"Status: `{decision['status']}`",
        f"Recommended action: `{decision['action']}`",
        "",
        "## Provenance",
        "",
        f"- split_file: `{payload['provenance']['split_file']}`",
        f"- route_file: `{payload['provenance']['route_file']}`",
        f"- training_log: `{payload['provenance']['training_log']}`",
        f"- heldout_query_used: `{payload['provenance']['heldout_query_used']}`",
        "",
        "## Routed Teacher Coverage",
        "",
        f"- routes: `{payload['routes']}`",
        f"- train route multi conditions: `{payload['train_route_multi_conditions']}`",
        f"- single-gene coverage by routed dataset: `{payload['single_gene_coverage_routed_datasets']}`",
        f"- observed final avg_trackc_route: `{payload['observed_training']['avg_trackc_route_final']:.6f}`",
        "",
        "## Sampler Exposure",
        "",
        "| window | total steps | route-trigger steps | fraction | unique route conditions |",
        "|---|---:|---:|---:|---:|",
        (
            f"| first {payload['prefix_steps_requested']} steps | {prefix['total_steps']} | "
            f"{prefix['route_trigger_steps']} | {prefix['route_trigger_fraction']:+.6f} | "
            f"{prefix['unique_route_conditions']} |"
        ),
        (
            f"| full epoch | {full_epoch['total_steps']} | {full_epoch['route_trigger_steps']} | "
            f"{full_epoch['route_trigger_fraction']:+.6f} | {full_epoch['unique_route_conditions']} |"
        ),
        "",
        "## First-Window Dataset Breakdown",
        "",
        "| dataset | trigger steps |",
        "|---|---:|",
    ]
    for ds_name, n in prefix["by_dataset_steps"].items():
        lines.append(f"| `{ds_name}` | {n} |")
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
        "- The route file and trainselect split are readable, and route targets are available for the intended Norman/Wessels train multi conditions.",
        "- Under the current all-condition sampler, the routed distillation loss is exposed in only a small fraction of the 2k smoke steps.",
        "- This supports closing exact reruns of the same routed-distill configuration. A future Track C GPU branch needs a new support-only design gate, such as route-focused sampling or a stronger dataset-routed adapter, before launch.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--route-file", type=Path, default=DEFAULT_ROUTE_FILE)
    parser.add_argument("--bank-summary", type=Path, default=DEFAULT_BANK_SUMMARY)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--biflow-dir", type=Path, default=ROOT / "dataset/biFlow_data")
    parser.add_argument("--gene-cache", type=Path, default=ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-cells", type=int, default=16)
    parser.add_argument("--ds-alpha", type=float, default=0.7)
    parser.add_argument("--min-selected-conditions-per-dataset", type=int, default=0)
    parser.add_argument("--condition-visit-power", type=float, default=1.0)
    parser.add_argument("--condition-visit-cap", type=int, default=0)
    parser.add_argument("--prefix-steps", type=int, default=2000)
    args = parser.parse_args()

    split = load_json(args.split_file)
    route_payload = load_json(args.route_file)
    routes = {str(k): str(v) for k, v in (route_payload.get("route") or route_payload).items()}
    ds = make_dataset(args, split)
    single_by_dataset, global_single = build_gene_coverage(ds)
    order = ds._build_epoch_order(np.random.RandomState(int(args.seed)))
    audit = summarize_order(
        ds,
        routes,
        single_by_dataset,
        global_single,
        order,
        prefix_steps=int(args.prefix_steps),
    )

    route_multi_conditions: dict[str, int] = defaultdict(int)
    for ds_name in ds.ds_names:
        for cond in ds.ds_conds.get(ds_name, []):
            ok, _, _ = route_target_available(ds, routes, single_by_dataset, global_single, ds_name, cond)
            if ok:
                route_multi_conditions[str(ds_name)] += 1

    train_log = args.run_root / "logs/xverse_trackc_route_condprior_w05_replay1_2k_seed42.train.log"
    payload = {
        "provenance": {
            "split_file": str(args.split_file),
            "route_file": str(args.route_file),
            "bank_summary": str(args.bank_summary),
            "training_log": str(train_log),
            "heldout_query_used": False,
            "usage_rule": "trainselect split and training log only; no Track C held-out query",
        },
        "routes": routes,
        "prefix_steps_requested": int(args.prefix_steps),
        "train_conditions": int(ds.total_conditions),
        "epoch_steps": int(ds.epoch_steps),
        "train_route_multi_conditions": dict(sorted(route_multi_conditions.items())),
        "single_gene_coverage_routed_datasets": {
            ds_name: len(single_by_dataset.get(ds_name, set()))
            for ds_name in sorted(routes)
        },
        "global_single_gene_coverage": int(len(global_single)),
        "sampler_audit": audit,
        "observed_training": {
            "avg_trackc_route_final": parse_final_avg_trackc_route(train_log),
        },
    }
    payload["decision"] = decide(payload)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, args.out_md)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
