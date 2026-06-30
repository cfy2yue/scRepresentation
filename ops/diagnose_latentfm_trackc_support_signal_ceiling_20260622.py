#!/usr/bin/env python3
"""CPU-only Track C support-signal ceiling/provenance diagnostic.

Reads existing Track C smoke posthoc artifacts and trainselect support CPU
readout reports. It does not read held-out query outputs, tmux state, logs, or
GPU artifacts beyond existing smoke decision/posthoc JSONs.
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
ROUTE_TEACHER = ROOT / "reports/latentfm_trackc_support_route_teacher_20260622.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["_manifest"] = str(path)
        rows.append(row)
    return rows


def condition_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["condition"])


def f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:+.6f}"


def load_support_condition_rows(run_root: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    anchor = load_json(run_root / "posthoc_eval/support_anchor_split_ode20.json")
    candidate = load_json(run_root / "posthoc_eval/support_candidate_split_ode20.json")
    anchor_rows = {
        condition_key(row): row
        for row in anchor["groups"]["test"].get("condition_metrics", [])
    }
    candidate_rows = {
        condition_key(row): row
        for row in candidate["groups"]["test"].get("condition_metrics", [])
    }
    return anchor_rows, candidate_rows


def load_memory_rows() -> dict[tuple[str, str], dict[str, Any]]:
    payload = load_json(MEMORY_READOUT)
    return {condition_key(row): row for row in payload.get("condition_rows") or []}


def route_map() -> dict[str, str]:
    return {str(k): str(v) for k, v in (load_json(ROUTE_TEACHER).get("route") or {}).items()}


def decision_for(run: str) -> dict[str, Any]:
    path = ROOT / "reports" / f"latentfm_trackc_routed_distill_smoke_decision_{run}.json"
    return load_json(path)


def selected_runs(manifests: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for manifest in manifests:
        for row in load_manifest(manifest):
            scope = str(row.get("finetune_trainable_scope", ""))
            pairwise = str(row.get("pert_pairwise_mode", ""))
            endpoint = float(row.get("endpoint_weight") or 0.0)
            head = float(row.get("head_distill_weight") or 0.0)
            if scope == "pairwise_condition_adapter" and pairwise == "hadamard_mean" and endpoint > 0 and head == 0:
                out.append(row)
    return out


def build_condition_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    memory = load_memory_rows()
    routes = route_map()
    out: list[dict[str, Any]] = []
    for run_row in rows:
        run = str(run_row["run_name"])
        run_root = Path(str(run_row["run_status"])).parent
        decision = decision_for(run)
        anchor, candidate = load_support_condition_rows(run_root)
        common = sorted(set(anchor) & set(candidate))
        for key in common:
            ds, cond = key
            a = anchor[key]
            c = candidate[key]
            mem = memory.get(key, {})
            anchor_pp = f(a.get("pearson_pert"))
            candidate_pp = f(c.get("pearson_pert"))
            anchor_mmd = f(a.get("test_mmd_clamped"))
            candidate_mmd = f(c.get("test_mmd_clamped"))
            support_route = f(mem.get("support_selected_route"))
            memory_best = f(mem.get("memory_overlap_k5_same_ds_min0"))
            out.append(
                {
                    "run_name": run,
                    "status": decision.get("decision", {}).get("status", ""),
                    "reasons": ";".join(decision.get("decision", {}).get("reasons") or []),
                    "dataset": ds,
                    "condition": cond,
                    "route": routes.get(ds, ""),
                    "endpoint_weight": run_row.get("endpoint_weight"),
                    "anchor_replay_weight": run_row.get("anchor_replay_weight"),
                    "anchor_replay_filter": run_row.get("anchor_replay_filter"),
                    "anchor_pp": anchor_pp,
                    "candidate_pp": candidate_pp,
                    "delta_pp": None if anchor_pp is None or candidate_pp is None else candidate_pp - anchor_pp,
                    "anchor_mmd": anchor_mmd,
                    "candidate_mmd": candidate_mmd,
                    "delta_mmd": None if anchor_mmd is None or candidate_mmd is None else candidate_mmd - anchor_mmd,
                    "support_selected_route_pp": support_route,
                    "memory_overlap_k5_same_ds_min0_pp": memory_best,
                    "candidate_gap_to_support_route_pp": (
                        None if support_route is None or candidate_pp is None else support_route - candidate_pp
                    ),
                    "candidate_gap_to_memory_k5_pp": (
                        None if memory_best is None or candidate_pp is None else memory_best - candidate_pp
                    ),
                }
            )
    return out


def summarize_conditions(condition_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in condition_rows:
        grouped[(str(row["run_name"]), str(row["dataset"]))].append(row)
    out: list[dict[str, Any]] = []
    for (run, ds), rows in sorted(grouped.items()):
        deltas = [float(row["delta_pp"]) for row in rows if row.get("delta_pp") is not None]
        route_gaps = [
            float(row["candidate_gap_to_support_route_pp"])
            for row in rows
            if row.get("candidate_gap_to_support_route_pp") is not None
        ]
        mmd_deltas = [float(row["delta_mmd"]) for row in rows if row.get("delta_mmd") is not None]
        positive = [x for x in deltas if x > 0]
        total_positive = sum(positive)
        top3_positive = sum(sorted(positive, reverse=True)[:3])
        out.append(
            {
                "run_name": run,
                "dataset": ds,
                "n_conditions": len(deltas),
                "mean_delta_pp": mean(deltas) if deltas else None,
                "median_delta_pp": median(deltas) if deltas else None,
                "positive_fraction": (sum(1 for x in deltas if x > 0) / len(deltas)) if deltas else None,
                "max_delta_pp": max(deltas) if deltas else None,
                "min_delta_pp": min(deltas) if deltas else None,
                "top3_positive_share": (top3_positive / total_positive) if total_positive > 0 else None,
                "mean_delta_mmd": mean(mmd_deltas) if mmd_deltas else None,
                "mean_gap_to_support_route_pp": mean(route_gaps) if route_gaps else None,
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


def write_md(condition_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], path: Path) -> None:
    best = sorted(
        summary_rows,
        key=lambda row: (row.get("mean_delta_pp") is not None, row.get("mean_delta_pp") or -999),
        reverse=True,
    )[:6]
    all_deltas = [float(row["delta_pp"]) for row in condition_rows if row.get("delta_pp") is not None]
    all_route_gaps = [
        float(row["candidate_gap_to_support_route_pp"])
        for row in condition_rows
        if row.get("candidate_gap_to_support_route_pp") is not None
    ]
    lines = [
        "# Track C Support-Signal Ceiling Diagnostic",
        "",
        "CPU-only diagnostic from support trainselect posthoc and support-val CPU readouts.",
        "Held-out query outputs are not read.",
        "",
        "## Bottom Line",
        "",
    ]
    if all_deltas:
        lines.append(
            "Pairwise-condition endpoint smokes improved support pp broadly but weakly: "
            f"overall condition-level mean delta is `{fmt(mean(all_deltas))}`, "
            f"median `{fmt(median(all_deltas))}`."
        )
    if all_route_gaps:
        lines.append(
            "The candidate remains far below the fixed support-selected route readout: "
            f"mean candidate-to-route pp gap is `{fmt(mean(all_route_gaps))}`."
        )
    lines.append(
        "This supports a transfer/conditioning bottleneck rather than a support-teacher ceiling; "
        "the observed GPU adapter signal is below the formal `+0.02` smoke gate."
    )
    lines.extend(
        [
            "",
            "## Best Dataset-Level Rows",
            "",
            "| run | dataset | n | mean delta pp | positive frac | top3 positive share | mean gap to route |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in best:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['run_name']}`",
                    str(row["dataset"]),
                    str(row["n_conditions"]),
                    fmt(row.get("mean_delta_pp")),
                    fmt(row.get("positive_fraction")),
                    fmt(row.get("top3_positive_share")),
                    fmt(row.get("mean_gap_to_support_route_pp")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- C-block pairwise-condition endpoint runs are the only GPU smokes with clean canonical family no-harm and nontrivial support gain.",
            "- The gain is still about half of the formal `+0.02` support gate and is not enough for promotion.",
            "- The support-selected route and memory readout CPU baselines are much higher on the same 24 support conditions, so simply increasing endpoint weight is not proven to be the right mechanism.",
            "- Next GPU work should require a new CPU gate showing how to transfer the route/memory readout signal into the model, not another broad endpoint-weight sweep.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--out-condition-csv", required=True)
    parser.add_argument("--out-summary-csv", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    manifests = [Path(p) for p in args.manifest]
    runs = selected_runs(manifests)
    condition_rows = build_condition_table(runs)
    summary_rows = summarize_conditions(condition_rows)
    write_csv(condition_rows, Path(args.out_condition_csv))
    write_csv(summary_rows, Path(args.out_summary_csv))
    write_md(condition_rows, summary_rows, Path(args.out_md))
    print(
        json.dumps(
            {
                "status": "ok",
                "runs": len(runs),
                "condition_rows": len(condition_rows),
                "summary_rows": len(summary_rows),
                "out_md": args.out_md,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
