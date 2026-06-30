#!/usr/bin/env python3
"""CPU-only route-transfer bottleneck diagnostic for Track C support smokes.

This consumes the support-signal ceiling condition table and reports how much
of the fixed support-route/memory-readout gap the GPU adapter actually closes.
It does not read held-out query outputs or run model inference.
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
MEMORY_READOUT = ROOT / "reports/latentfm_trackc_support_memory_readout_gate_20260622.json"


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:+.6f}"


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def memory_condition_index() -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(MEMORY_READOUT.read_text(encoding="utf-8"))
    return {
        (str(row["dataset"]), str(row["condition"])): row
        for row in payload.get("condition_rows") or []
    }


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    memory = memory_condition_index()
    out: list[dict[str, Any]] = []
    for row in rows:
        ds = str(row["dataset"])
        cond = str(row["condition"])
        anchor = to_float(row.get("anchor_pp"))
        candidate = to_float(row.get("candidate_pp"))
        route = to_float(row.get("support_selected_route_pp"))
        memory_k5 = to_float(row.get("memory_overlap_k5_same_ds_min0_pp"))
        delta = None if anchor is None or candidate is None else candidate - anchor
        route_gap = None if route is None or anchor is None else route - anchor
        memory_gap = None if memory_k5 is None or anchor is None else memory_k5 - anchor
        route_closed = None
        if route_gap is not None and delta is not None and abs(route_gap) > 1e-12:
            route_closed = delta / route_gap
        memory_closed = None
        if memory_gap is not None and delta is not None and abs(memory_gap) > 1e-12:
            memory_closed = delta / memory_gap
        mem_row = memory.get((ds, cond), {})
        genes = mem_row.get("genes") or []
        out.append(
            {
                **row,
                "n_genes": len(genes),
                "genes": "+".join(str(x) for x in genes),
                "route_gap_from_anchor_pp": route_gap,
                "memory_gap_from_anchor_pp": memory_gap,
                "route_gap_closed_fraction": route_closed,
                "memory_gap_closed_fraction": memory_closed,
                "candidate_above_anchor": bool(delta is not None and delta > 0),
                "candidate_above_route": bool(candidate is not None and route is not None and candidate > route),
                "candidate_above_memory": bool(candidate is not None and memory_k5 is not None and candidate > memory_k5),
            }
        )
    return out


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["run_name"]), str(row["dataset"]))].append(row)
    out: list[dict[str, Any]] = []
    for (run, ds), items in sorted(groups.items()):
        deltas = [to_float(row.get("delta_pp")) for row in items]
        deltas = [x for x in deltas if x is not None]
        route_gaps = [to_float(row.get("route_gap_from_anchor_pp")) for row in items]
        route_gaps = [x for x in route_gaps if x is not None]
        route_closed = [to_float(row.get("route_gap_closed_fraction")) for row in items]
        route_closed = [x for x in route_closed if x is not None]
        memory_closed = [to_float(row.get("memory_gap_closed_fraction")) for row in items]
        memory_closed = [x for x in memory_closed if x is not None]
        route_positive = [x for x in route_gaps if x > 0]
        useful = [
            (to_float(row.get("delta_pp")), to_float(row.get("route_gap_from_anchor_pp")))
            for row in items
        ]
        useful = [(d, g) for d, g in useful if d is not None and g is not None and g > 0]
        weighted_closed = None
        if useful:
            weighted_closed = sum(d for d, _ in useful) / sum(g for _, g in useful)
        out.append(
            {
                "run_name": run,
                "dataset": ds,
                "n_conditions": len(items),
                "mean_delta_pp": mean(deltas) if deltas else None,
                "median_delta_pp": median(deltas) if deltas else None,
                "mean_route_gap_from_anchor_pp": mean(route_gaps) if route_gaps else None,
                "median_route_gap_from_anchor_pp": median(route_gaps) if route_gaps else None,
                "route_positive_fraction": (len(route_positive) / len(route_gaps)) if route_gaps else None,
                "mean_route_gap_closed_fraction": mean(route_closed) if route_closed else None,
                "median_route_gap_closed_fraction": median(route_closed) if route_closed else None,
                "weighted_route_gap_closed_fraction": weighted_closed,
                "mean_memory_gap_closed_fraction": mean(memory_closed) if memory_closed else None,
                "candidate_above_route_count": sum(1 for row in items if row.get("candidate_above_route") in {True, "True"}),
                "candidate_above_memory_count": sum(1 for row in items if row.get("candidate_above_memory") in {True, "True"}),
            }
        )
    return out


def write_md(rows: list[dict[str, Any]], summary: list[dict[str, Any]], path: Path) -> None:
    norman = [row for row in summary if row["dataset"] == "NormanWeissman2019_filtered"]
    wessels = [row for row in summary if row["dataset"] == "Wessels"]

    def avg(items: list[dict[str, Any]], key: str) -> float | None:
        vals = [to_float(row.get(key)) for row in items]
        vals = [x for x in vals if x is not None]
        return mean(vals) if vals else None

    best = sorted(summary, key=lambda row: to_float(row.get("mean_delta_pp")) or -999, reverse=True)[:8]
    lines = [
        "# Track C Route-Transfer Bottleneck Diagnostic",
        "",
        "CPU-only diagnostic. Inputs are support trainselect smoke posthoc metrics and support-val CPU readouts.",
        "Held-out query outputs are not read.",
        "",
        "## Bottom Line",
        "",
        (
            "Norman has a measurable but partial transfer signal, while Wessels has essentially no transfer. "
            f"Average mean delta by run is `{fmt(avg(norman, 'mean_delta_pp'))}` for Norman and "
            f"`{fmt(avg(wessels, 'mean_delta_pp'))}` for Wessels."
        ),
        (
            "The adapter closes only a small fraction of the fixed support-route gap: "
            f"average weighted route-gap closure is `{fmt(avg(norman, 'weighted_route_gap_closed_fraction'))}` "
            f"for Norman and `{fmt(avg(wessels, 'weighted_route_gap_closed_fraction'))}` for Wessels."
        ),
        "This makes a simple endpoint-weight scale-up poorly justified without a new CPU gate.",
        "",
        "## Dataset-Level Transfer",
        "",
        "| run | dataset | mean delta | route gap | weighted gap closed | candidate > route |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in best:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['run_name']}`",
                    str(row["dataset"]),
                    fmt(to_float(row.get("mean_delta_pp"))),
                    fmt(to_float(row.get("mean_route_gap_from_anchor_pp"))),
                    fmt(to_float(row.get("weighted_route_gap_closed_fraction"))),
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
            "- Do not launch the prepared E GPU scale-up from this evidence alone.",
            "- A new GPU branch needs a CPU gate showing improved route-gap closure, especially on Wessels.",
            "- Candidate mechanisms should target route/memory readout transfer rather than more endpoint weight.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-csv", required=True)
    parser.add_argument("--out-condition-csv", required=True)
    parser.add_argument("--out-summary-csv", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    rows = enrich_rows(read_csv(Path(args.condition_csv)))
    summary = summarize(rows)
    write_csv(rows, Path(args.out_condition_csv))
    write_csv(summary, Path(args.out_summary_csv))
    write_md(rows, summary, Path(args.out_md))
    print(json.dumps({"status": "ok", "rows": len(rows), "summary_rows": len(summary), "out_md": args.out_md}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
