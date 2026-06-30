#!/usr/bin/env python3
"""CPU-only preflight for the Track C residual-operator feasibility gate.

This script does not construct a new operator and does not authorize GPU work.
It summarizes the already-frozen evidence showing why a new support-conditioned
operator is needed: train-only memory/readout targets are strong, while the
closed GPU families barely absorb Wessels route gap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
DEFAULT_MEMORY_JSON = ROOT / "reports/latentfm_trackc_trainonly_memory_readout_gate_20260622.json"
DEFAULT_BOTTLENECK_SUMMARY = ROOT / "reports/latentfm_trackc_memory_transfer_bottleneck_summary_20260622.csv"
DEFAULT_FAILURE_JSON = ROOT / "reports/latentfm_trackc_support_context_failure_analysis_20260622.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_residual_operator_feasibility_preflight_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_RESIDUAL_OPERATOR_FEASIBILITY_PREFLIGHT_20260623.md"


def as_float(value: Any) -> float | None:
    if value is None or value == "" or str(value).lower() == "none":
        return None
    return float(value)


def fmt(value: float | None) -> str:
    value = as_float(value)
    if value is None:
        return "NA"
    return f"{value:+.6f}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_counts(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {
        "sha256": sha256(path),
        "support_val_multi_total": 0,
        "heldout_query_multi_final_only_total": 0,
        "datasets": {},
    }
    for dataset, part in payload.items():
        if not isinstance(part, dict):
            continue
        support_n = len(part.get("support_val_multi") or [])
        query_n = len(part.get("heldout_query_multi_final_only") or [])
        if support_n or query_n:
            out["datasets"][dataset] = {
                "support_val_multi": support_n,
                "test_multi": len(part.get("test_multi") or []),
                "heldout_query_multi_final_only": query_n,
            }
        out["support_val_multi_total"] += support_n
        out["heldout_query_multi_final_only_total"] += query_n
    return out


def load_bottleneck_summary(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def best_closed_family(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairwise = [row for row in rows if row.get("finetune_trainable_scope") == "pairwise_condition_adapter"]
    by_ds: dict[str, list[dict[str, Any]]] = {}
    for row in pairwise:
        by_ds.setdefault(str(row.get("dataset")), []).append(row)

    def best_for(dataset: str) -> dict[str, Any] | None:
        items = by_ds.get(dataset) or []
        if not items:
            return None
        return max(items, key=lambda row: as_float(row.get("mean_delta_pp")) or -999.0)

    norman = best_for("NormanWeissman2019_filtered")
    wessels = best_for("Wessels")
    return {
        "norman_best_pairwise": norman,
        "wessels_best_pairwise": wessels,
        "wessels_delta_gap_to_0p02": None
        if wessels is None
        else 0.02 - (as_float(wessels.get("mean_delta_pp")) or 0.0),
        "wessels_route_closure_gap_to_0p05": None
        if wessels is None
        else 0.05 - (as_float(wessels.get("weighted_route_gap_closed_fraction")) or 0.0),
    }


def memory_readout_signal(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_model")
    datasets = {}
    for row in payload.get("dataset_breakdown") or []:
        ds = str(row.get("dataset"))
        selected_pp = as_float(row.get(selected))
        route_pp = as_float(row.get("support_selected_route"))
        dataset_multi_pp = as_float(row.get("dataset_multi_mean"))
        datasets[ds] = {
            "n_conditions": row.get("n_conditions"),
            "selected_model_pp": selected_pp,
            "support_selected_route_pp": route_pp,
            "dataset_multi_mean_pp": dataset_multi_pp,
            "selected_minus_support_route": None
            if selected_pp is None or route_pp is None
            else selected_pp - route_pp,
            "selected_minus_dataset_multi": None
            if selected_pp is None or dataset_multi_pp is None
            else selected_pp - dataset_multi_pp,
        }
    return {
        "selected_model": selected,
        "decision": payload.get("decision"),
        "split_guard": payload.get("split_guard"),
        "datasets": datasets,
    }


def support_context_failure(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs") or []

    def vals(key: str) -> list[float]:
        out = []
        for row in runs:
            value = as_float(row.get(key))
            if value is not None:
                out.append(value)
        return out

    def mean_or_none(items: list[float]) -> float | None:
        return sum(items) / len(items) if items else None

    support_pp = vals("support_pp_delta")
    wessels_closure = vals("wessels_route_gap_closure")
    return {
        "available": True,
        "decision": payload.get("decision") or {},
        "aggregate": {
            "mean_support_pp_delta": mean_or_none(support_pp),
            "max_support_pp_delta": max(support_pp) if support_pp else None,
            "mean_wessels_route_gap_closure": mean_or_none(wessels_closure),
            "max_wessels_route_gap_closure": max(wessels_closure) if wessels_closure else None,
        },
    }


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    split = payload["split"]
    if split["sha256"] != "5f29dd5b582a40da3736770ca29950c12c54e46b590c3f7705c1d37da89f4f20":
        reasons.append("trainselect_split_hash_mismatch")
    if split["support_val_multi_total"] != 24:
        reasons.append("support_val_count_mismatch")
    memory = payload["memory_readout"]
    wessels_memory = (memory.get("datasets") or {}).get("Wessels") or {}
    if (as_float(wessels_memory.get("selected_minus_support_route")) or -999.0) < 0.02:
        reasons.append("wessels_memory_readout_not_materially_above_support_route")
    closed = payload["closed_family_best"]
    wessels_best = closed.get("wessels_best_pairwise") or {}
    if (as_float(wessels_best.get("mean_delta_pp")) or -999.0) >= 0.02:
        reasons.append("closed_family_wessels_delta_already_hits_gate_unexpected")
    if (as_float(wessels_best.get("weighted_route_gap_closed_fraction")) or -999.0) >= 0.05:
        reasons.append("closed_family_wessels_closure_already_hits_gate_unexpected")
    status = (
        "residual_operator_preflight_supports_new_cpu_gate"
        if not reasons
        else "residual_operator_preflight_fail_closed"
    )
    return {
        "status": status,
        "reasons": reasons,
        "gpu_authorization": "none",
        "next_action": "implement_cpu_residual_operator_gate" if not reasons else "fix_preflight_or_close_candidate",
    }


def render(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    split = payload["split"]
    memory = payload["memory_readout"]
    closed = payload["closed_family_best"]
    sc_fail = payload["support_context_failure"]
    selected = memory.get("selected_model")
    lines = [
        "# Track C Residual-Operator Feasibility Preflight",
        "",
        f"Status: `{decision['status']}`",
        "GPU authorization: `none`",
        "",
        "## Leakage Guard",
        "",
        f"- trainselect split: `{payload['split_file']}`",
        f"- SHA256: `{split['sha256']}`",
        f"- support-val multi total: `{split['support_val_multi_total']}`",
        f"- held-out query multi final-only total in metadata: `{split['heldout_query_multi_final_only_total']}`",
        "- query rows are metadata only and are not read as examples by this preflight.",
        "",
        "## Memory Readout Signal",
        "",
        f"- selected model: `{selected}`",
        "",
        "| dataset | n | selected pp | support route pp | dataset multi pp | selected - route | selected - dataset_multi |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ds, row in sorted((memory.get("datasets") or {}).items()):
        lines.append(
            f"| {ds} | {row.get('n_conditions')} | {fmt(row.get('selected_model_pp'))} | "
            f"{fmt(row.get('support_selected_route_pp'))} | {fmt(row.get('dataset_multi_mean_pp'))} | "
            f"{fmt(row.get('selected_minus_support_route'))} | {fmt(row.get('selected_minus_dataset_multi'))} |"
        )
    lines.extend(
        [
            "",
            "## Closed-Family Absorption",
            "",
            "| dataset | best run | mean pp delta | weighted route-gap closure |",
            "|---|---|---:|---:|",
        ]
    )
    for label, key in (
        ("NormanWeissman2019_filtered", "norman_best_pairwise"),
        ("Wessels", "wessels_best_pairwise"),
    ):
        row = closed.get(key) or {}
        lines.append(
            f"| {label} | `{row.get('run_name', 'NA')}` | "
            f"{fmt(row.get('mean_delta_pp'))} | {fmt(row.get('weighted_route_gap_closed_fraction'))} |"
        )
    lines.extend(
        [
            "",
            "## Gap To Gate",
            "",
            f"- Wessels delta gap to `+0.02`: `{fmt(closed.get('wessels_delta_gap_to_0p02'))}`",
            f"- Wessels closure gap to `+0.05`: `{fmt(closed.get('wessels_route_closure_gap_to_0p05'))}`",
        ]
    )
    if sc_fail.get("available"):
        aggregate = sc_fail.get("aggregate") or {}
        lines.extend(
            [
                "",
                "## Support-Context Failure Reference",
                "",
                f"- mean support pp delta: `{fmt(aggregate.get('mean_support_pp_delta'))}`",
                f"- max support pp delta: `{fmt(aggregate.get('max_support_pp_delta'))}`",
                f"- mean Wessels route-gap closure: `{fmt(aggregate.get('mean_wessels_route_gap_closure'))}`",
                f"- max Wessels route-gap closure: `{fmt(aggregate.get('max_wessels_route_gap_closure'))}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This preflight does not pass a model candidate and does not authorize GPU.",
            "It confirms the next useful work is a CPU residual-operator gate: the",
            "support memory/readout target has material Wessels signal, while the",
            "closed GPU families have not absorbed enough of that route gap.",
        ]
    )
    if decision["reasons"]:
        lines.extend(["", "Reasons:", ""])
        lines.extend(f"- `{reason}`" for reason in decision["reasons"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--memory-json", type=Path, default=DEFAULT_MEMORY_JSON)
    parser.add_argument("--bottleneck-summary", type=Path, default=DEFAULT_BOTTLENECK_SUMMARY)
    parser.add_argument("--support-context-failure-json", type=Path, default=DEFAULT_FAILURE_JSON)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "split_file": str(args.split_file),
        "memory_json": str(args.memory_json),
        "bottleneck_summary": str(args.bottleneck_summary),
        "support_context_failure_json": str(args.support_context_failure_json),
        "leakage_status": "trainselect_support_val_only_no_query_examples_no_canonical_multi_no_tracka_selection",
        "split": split_counts(args.split_file),
        "memory_readout": memory_readout_signal(args.memory_json),
        "closed_family_best": best_closed_family(load_bottleneck_summary(args.bottleneck_summary)),
        "support_context_failure": support_context_failure(args.support_context_failure_json),
    }
    payload["decision"] = decide(payload)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
