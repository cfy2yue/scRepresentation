#!/usr/bin/env python3
"""CPU-only transfer bottleneck diagnostic for corrected Track C memory smokes.

Inputs are completed support-trainselect posthoc JSONs and the train-only
memory readout CPU gate artifact. Held-out query and canonical outputs are not
read.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_READOUT = ROOT / "reports/latentfm_trackc_trainonly_memory_readout_gate_20260622.json"
SELECTED_MEMORY = "memory_jaccard_k3_same_ds_min0.25"


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:+.6f}"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["_manifest"] = str(path)
        rows.append(row)
    return rows


def read_condition_metrics(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = ((payload.get("groups") or {}).get("test") or {}).get("condition_metrics") or []
    return {
        (str(row["dataset"]), str(row["condition"])): row
        for row in rows
    }


def readout_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (str(row["dataset"]), str(row["condition"])): row
        for row in payload.get("condition_rows") or []
    }


def out_root_for(row: dict[str, Any]) -> Path:
    run_status = Path(str(row["run_status"]))
    # .../runs/<root>/<run>/RUN_STATUS.md -> use matching posthoc_eval beside it.
    return run_status.parent


def collect_rows(manifest_paths: list[Path], readout_path: Path) -> list[dict[str, Any]]:
    readout = readout_index(readout_path)
    out: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        for row in load_manifest(manifest_path):
            run = str(row["run_name"])
            run_dir = out_root_for(row)
            anchor_path = run_dir / "posthoc_eval/support_anchor_split_ode20.json"
            candidate_path = run_dir / "posthoc_eval/support_candidate_split_ode20.json"
            anchor = read_condition_metrics(anchor_path)
            candidate = read_condition_metrics(candidate_path)
            for key in sorted(set(anchor) & set(candidate) & set(readout)):
                ds, cond = key
                a = anchor[key]
                c = candidate[key]
                r = readout[key]
                anchor_pp = as_float(a.get("pearson_pert"))
                candidate_pp = as_float(c.get("pearson_pert"))
                route_pp = as_float(r.get("support_selected_route"))
                memory_pp = as_float(r.get(SELECTED_MEMORY))
                delta_pp = None if anchor_pp is None or candidate_pp is None else candidate_pp - anchor_pp
                route_gap = None if anchor_pp is None or route_pp is None else route_pp - anchor_pp
                memory_gap = None if anchor_pp is None or memory_pp is None else memory_pp - anchor_pp
                route_closed = None
                if delta_pp is not None and route_gap is not None and abs(route_gap) > 1e-12:
                    route_closed = delta_pp / route_gap
                memory_closed = None
                if delta_pp is not None and memory_gap is not None and abs(memory_gap) > 1e-12:
                    memory_closed = delta_pp / memory_gap
                out.append(
                    {
                        "run_name": run,
                        "manifest": str(manifest_path),
                        "dataset": ds,
                        "condition": cond,
                        "finetune_trainable_scope": row.get("finetune_trainable_scope"),
                        "pert_pairwise_mode": row.get("pert_pairwise_mode"),
                        "endpoint_weight": row.get("endpoint_weight"),
                        "endpoint_warmup_start": row.get("endpoint_warmup_start"),
                        "endpoint_warmup_end": row.get("endpoint_warmup_end"),
                        "anchor_replay_weight": row.get("anchor_replay_weight"),
                        "anchor_replay_filter": row.get("anchor_replay_filter"),
                        "anchor_pp": anchor_pp,
                        "candidate_pp": candidate_pp,
                        "delta_pp": delta_pp,
                        "support_selected_route_pp": route_pp,
                        f"{SELECTED_MEMORY}_pp": memory_pp,
                        "route_gap_from_anchor_pp": route_gap,
                        "memory_gap_from_anchor_pp": memory_gap,
                        "route_gap_closed_fraction": route_closed,
                        "memory_gap_closed_fraction": memory_closed,
                        "candidate_above_route": bool(candidate_pp is not None and route_pp is not None and candidate_pp > route_pp),
                        "candidate_above_memory": bool(candidate_pp is not None and memory_pp is not None and candidate_pp > memory_pp),
                        "genes": "+".join(str(x) for x in r.get("genes") or []),
                    }
                )
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["run_name"]),
                str(row["dataset"]),
                str(row["finetune_trainable_scope"]),
            )
        ].append(row)
    out: list[dict[str, Any]] = []
    for (run, ds, scope), items in sorted(groups.items()):
        deltas = [as_float(row.get("delta_pp")) for row in items]
        deltas = [x for x in deltas if x is not None]
        route_gaps = [as_float(row.get("route_gap_from_anchor_pp")) for row in items]
        route_gaps = [x for x in route_gaps if x is not None]
        memory_gaps = [as_float(row.get("memory_gap_from_anchor_pp")) for row in items]
        memory_gaps = [x for x in memory_gaps if x is not None]
        route_closed = [as_float(row.get("route_gap_closed_fraction")) for row in items]
        route_closed = [x for x in route_closed if x is not None]
        memory_closed = [as_float(row.get("memory_gap_closed_fraction")) for row in items]
        memory_closed = [x for x in memory_closed if x is not None]
        positive_route = [
            (as_float(row.get("delta_pp")), as_float(row.get("route_gap_from_anchor_pp")))
            for row in items
        ]
        positive_route = [
            (d, g)
            for d, g in positive_route
            if d is not None and g is not None and g > 0
        ]
        positive_memory = [
            (as_float(row.get("delta_pp")), as_float(row.get("memory_gap_from_anchor_pp")))
            for row in items
        ]
        positive_memory = [
            (d, g)
            for d, g in positive_memory
            if d is not None and g is not None and g > 0
        ]
        out.append(
            {
                "run_name": run,
                "dataset": ds,
                "finetune_trainable_scope": scope,
                "n_conditions": len(items),
                "mean_delta_pp": mean(deltas) if deltas else None,
                "median_delta_pp": median(deltas) if deltas else None,
                "mean_route_gap_from_anchor_pp": mean(route_gaps) if route_gaps else None,
                "mean_memory_gap_from_anchor_pp": mean(memory_gaps) if memory_gaps else None,
                "mean_route_gap_closed_fraction": mean(route_closed) if route_closed else None,
                "mean_memory_gap_closed_fraction": mean(memory_closed) if memory_closed else None,
                "weighted_route_gap_closed_fraction": (
                    sum(d for d, _ in positive_route) / sum(g for _, g in positive_route)
                    if positive_route
                    else None
                ),
                "weighted_memory_gap_closed_fraction": (
                    sum(d for d, _ in positive_memory) / sum(g for _, g in positive_memory)
                    if positive_memory
                    else None
                ),
                "candidate_above_route_count": sum(1 for row in items if row["candidate_above_route"]),
                "candidate_above_memory_count": sum(1 for row in items if row["candidate_above_memory"]),
            }
        )
    return out


def decide(summary: list[dict[str, Any]]) -> dict[str, Any]:
    pairwise = [row for row in summary if row["finetune_trainable_scope"] == "pairwise_condition_adapter"]
    wessels = [row for row in pairwise if row["dataset"] == "Wessels"]
    norman = [row for row in pairwise if row["dataset"] == "NormanWeissman2019_filtered"]

    def best(items: list[dict[str, Any]], key: str) -> float | None:
        vals = [as_float(row.get(key)) for row in items]
        vals = [x for x in vals if x is not None]
        return max(vals) if vals else None

    best_wessels_delta = best(wessels, "mean_delta_pp")
    best_wessels_closure = best(wessels, "weighted_route_gap_closed_fraction")
    best_norman_delta = best(norman, "mean_delta_pp")
    reasons = []
    if best_wessels_delta is None or best_wessels_delta < 0.02:
        reasons.append("wessels_pairwise_delta_below_0p02")
    if best_wessels_closure is None or best_wessels_closure < 0.05:
        reasons.append("wessels_route_gap_closure_below_0p05")
    status = "memory_transfer_bottleneck_gate_pass" if not reasons else "memory_transfer_bottleneck_gate_fail"
    return {
        "status": status,
        "reasons": reasons,
        "best_wessels_pairwise_delta": best_wessels_delta,
        "best_wessels_pairwise_route_gap_closure": best_wessels_closure,
        "best_norman_pairwise_delta": best_norman_delta,
    }


def write_md(summary: list[dict[str, Any]], decision: dict[str, Any], path: Path) -> None:
    top = sorted(summary, key=lambda row: as_float(row.get("mean_delta_pp")) or -999, reverse=True)[:16]
    lines = [
        "# Track C Memory-Transfer Bottleneck CPU Gate",
        "",
        "CPU-only diagnostic over corrected mc256 memory-transfer smokes.",
        "Inputs are support-trainselect posthoc JSONs and train-only memory readout rows.",
        "Held-out query and canonical outputs are not read.",
        "",
        f"Status: `{decision['status']}`",
        "",
        "## Decision Evidence",
        "",
        f"- best Wessels pairwise mean support pp delta: `{fmt(decision['best_wessels_pairwise_delta'])}`",
        f"- best Wessels pairwise weighted route-gap closure: `{fmt(decision['best_wessels_pairwise_route_gap_closure'])}`",
        f"- best Norman pairwise mean support pp delta: `{fmt(decision['best_norman_pairwise_delta'])}`",
    ]
    if decision["reasons"]:
        lines.append(f"- reasons: `{';'.join(decision['reasons'])}`")
    lines.extend(
        [
            "",
            "## Top Dataset-Level Rows",
            "",
            "| run | dataset | scope | mean delta | route gap | route closure | memory closure | above route |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['run_name']}`",
                    str(row["dataset"]),
                    str(row["finetune_trainable_scope"]),
                    fmt(as_float(row.get("mean_delta_pp"))),
                    fmt(as_float(row.get("mean_route_gap_from_anchor_pp"))),
                    fmt(as_float(row.get("weighted_route_gap_closed_fraction"))),
                    fmt(as_float(row.get("weighted_memory_gap_closed_fraction"))),
                    str(row.get("candidate_above_route_count")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Gate Consequence",
            "",
            "- This gate is not a query evaluation and cannot promote a model.",
            "- If failed, do not run more train-only memory endpoint/replay sweeps.",
            "- A new GPU branch needs a different mechanism with a prior CPU gate showing Wessels transfer.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--readout-json", default=str(DEFAULT_READOUT))
    parser.add_argument("--out-condition-csv", required=True)
    parser.add_argument("--out-summary-csv", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    rows = collect_rows([Path(x) for x in args.manifest], Path(args.readout_json))
    summary = summarize(rows)
    decision = decide(summary)
    payload = {
        "status": decision["status"],
        "decision": decision,
        "n_condition_rows": len(rows),
        "n_summary_rows": len(summary),
        "inputs": {
            "manifests": args.manifest,
            "readout_json": args.readout_json,
        },
        "heldout_query_used": False,
    }
    write_csv(rows, Path(args.out_condition_csv))
    write_csv(summary, Path(args.out_summary_csv))
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(summary, decision, Path(args.out_md))
    print(json.dumps({"status": decision["status"], "rows": len(rows), "summary_rows": len(summary), "out_md": args.out_md}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
