#!/usr/bin/env python3
"""Support-only stability gate for Track C memory/readout mechanisms.

This diagnostic reads the frozen support-memory readout CPU gate and summarizes
whether the support-only memory-vs-route signal is broad enough to motivate a
future train/support-only mechanism gate. It does not read held-out query
outputs, tmux state, logs, or GPU candidate outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_MEMORY_JSON = ROOT / "reports/latentfm_trackc_support_memory_readout_gate_20260622.json"


def f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def fmt(value: Any) -> str:
    value = f(value)
    if value is None:
        return "NA"
    return f"{value:+.6f}"


def load_rows(path: Path, memory_key: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for row in payload.get("condition_rows") or []:
        route_pp = f(row.get("support_selected_route"))
        mem_pp = f(row.get(memory_key))
        route_mmd = f(row.get("support_selected_route__test_mmd_clamped"))
        mem_mmd = f(row.get(f"{memory_key}__test_mmd_clamped"))
        if route_pp is None or mem_pp is None:
            continue
        rows.append(
            {
                "dataset": row["dataset"],
                "condition": row["condition"],
                "genes": "+".join(str(x) for x in row.get("genes") or []),
                "route_pp": route_pp,
                "memory_pp": mem_pp,
                "delta_pp_memory_minus_route": mem_pp - route_pp,
                "route_mmd": route_mmd,
                "memory_mmd": mem_mmd,
                "delta_mmd_memory_minus_route": (
                    None if route_mmd is None or mem_mmd is None else mem_mmd - route_mmd
                ),
            }
        )
    return rows


def top_positive_share(values: list[float], k: int = 3) -> float | None:
    positives = sorted([x for x in values if x > 0], reverse=True)
    total = sum(positives)
    if not positives or abs(total) < 1e-12:
        return None
    return sum(positives[:k]) / total


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    datasets = sorted({str(row["dataset"]) for row in rows})
    for ds in datasets:
        items = [row for row in rows if row["dataset"] == ds]
        pp = [float(row["delta_pp_memory_minus_route"]) for row in items]
        mmd = [
            f(row.get("delta_mmd_memory_minus_route"))
            for row in items
            if f(row.get("delta_mmd_memory_minus_route")) is not None
        ]
        loo = [(sum(pp) - x) / (len(pp) - 1) for x in pp] if len(pp) > 1 else []
        out.append(
            {
                "dataset": ds,
                "n_conditions": len(items),
                "mean_delta_pp_memory_minus_route": mean(pp),
                "median_delta_pp_memory_minus_route": median(pp),
                "positive_fraction_pp": sum(1 for x in pp if x > 0) / len(pp),
                "min_delta_pp": min(pp),
                "max_delta_pp": max(pp),
                "leave_one_min_mean_delta_pp": min(loo) if loo else None,
                "top3_positive_share": top_positive_share(pp, 3),
                "mean_delta_mmd_memory_minus_route": mean(mmd) if mmd else None,
                "mmd_harm_fraction": (sum(1 for x in mmd if x > 0) / len(mmd)) if mmd else None,
            }
        )
    return out


def decision(summary: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []
    for row in summary:
        ds = str(row["dataset"])
        if f(row.get("mean_delta_pp_memory_minus_route")) is None or f(row.get("mean_delta_pp_memory_minus_route")) < 0.02:
            reasons.append(f"{ds}_mean_memory_vs_route_delta_below_0p02")
        if f(row.get("positive_fraction_pp")) is None or f(row.get("positive_fraction_pp")) < 0.60:
            reasons.append(f"{ds}_memory_vs_route_positive_fraction_below_0p60")
        if f(row.get("leave_one_min_mean_delta_pp")) is None or f(row.get("leave_one_min_mean_delta_pp")) <= 0.0:
            reasons.append(f"{ds}_leave_one_mean_not_positive")
        if f(row.get("top3_positive_share")) is not None and f(row.get("top3_positive_share")) > 0.80:
            warnings.append(f"{ds}_positive_signal_top3_concentrated")
        if f(row.get("mmd_harm_fraction")) is not None and f(row.get("mmd_harm_fraction")) > 0.50:
            warnings.append(f"{ds}_mmd_harm_fraction_above_0p50")

    if reasons:
        status = "support_memory_stability_gate_fail"
        action = "do_not_launch_memory_transfer_gpu_branch"
    else:
        status = "support_memory_stability_gate_pass_with_warnings" if warnings else "support_memory_stability_gate_pass"
        action = (
            "eligible_for_protocol_review_after_latest_checkpoint_gate; "
            "requires no-query train/support-only mechanism design"
        )
    return {
        "status": status,
        "action": action,
        "reasons": reasons,
        "warnings": warnings,
        "rules": [
            "per-dataset mean memory-minus-route pp delta >= +0.02",
            "per-dataset positive fraction >= 0.60",
            "per-dataset leave-one mean delta remains > 0",
            "top3 positive share > 0.80 is a concentration warning, not a hard fail",
            "held-out query is not read and cannot be used to tune this gate",
        ],
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_md(
    *,
    summary: list[dict[str, Any]],
    details: list[dict[str, Any]],
    decision_payload: dict[str, Any],
    memory_json: Path,
    memory_key: str,
    out_md: Path,
) -> None:
    lines = [
        "# Track C Support-Memory Stability Gate",
        "",
        "CPU-only support/trainselect diagnostic. Held-out Track C query outputs are not read.",
        "",
        "## Bottom Line",
        "",
        f"Status: `{decision_payload['status']}`",
        f"Recommended action: `{decision_payload['action']}`",
        "",
        "This gate asks whether the frozen support-memory readout signal is broad enough to justify",
        "a future no-query train/support-only memory-transfer mechanism, if the currently running",
        "latest-checkpoint gate does not rescue the pairwise endpoint branch.",
        "",
        "## Provenance",
        "",
        f"- memory_json: `{memory_json}`",
        f"- memory_key: `{memory_key}`",
        "- leakage boundary: support train/val memory-readout artifact only; no held-out query.",
        "",
        "## Dataset Stability",
        "",
        "| dataset | n | mean pp delta | median pp delta | positive frac | min pp delta | leave-one min mean | top3 positive share | mean MMD delta | MMD harm frac |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["n_conditions"]),
                    fmt(row["mean_delta_pp_memory_minus_route"]),
                    fmt(row["median_delta_pp_memory_minus_route"]),
                    fmt(row["positive_fraction_pp"]),
                    fmt(row["min_delta_pp"]),
                    fmt(row["leave_one_min_mean_delta_pp"]),
                    fmt(row["top3_positive_share"]),
                    fmt(row["mean_delta_mmd_memory_minus_route"]),
                    fmt(row["mmd_harm_fraction"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Worst Conditions", ""])
    for ds in sorted({str(row["dataset"]) for row in details}):
        worst = sorted(
            [row for row in details if str(row["dataset"]) == ds],
            key=lambda row: float(row["delta_pp_memory_minus_route"]),
        )[:5]
        lines.extend(
            [
                f"### {ds}",
                "",
                "| condition | genes | pp delta | mmd delta |",
                "|---|---|---:|---:|",
            ]
        )
        for row in worst:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['condition']}`",
                        f"`{row['genes']}`",
                        fmt(row["delta_pp_memory_minus_route"]),
                        fmt(row["delta_mmd_memory_minus_route"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(["## Gate Reasons", ""])
    if decision_payload["reasons"]:
        lines.extend(f"- `{x}`" for x in decision_payload["reasons"])
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    if decision_payload["warnings"]:
        lines.extend(f"- `{x}`" for x in decision_payload["warnings"])
    else:
        lines.append("- none")
    lines.extend(["", "## Consequence", ""])
    lines.extend(
        [
            "- This does not authorize a GPU launch while the latest-checkpoint posthoc gate is running.",
            "- If latest-checkpoint fails, a memory-transfer branch may be considered only after protocol review.",
            "- Any such branch must remain train/support-only and cannot use held-out query for tuning or selection.",
            "",
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-json", type=Path, default=DEFAULT_MEMORY_JSON)
    parser.add_argument("--memory-key", default="memory_overlap_k5_same_ds_min0")
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-summary-csv", type=Path, required=True)
    parser.add_argument("--out-condition-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.memory_json, args.memory_key)
    summary = summarize(rows)
    decision_payload = decision(summary)
    payload = {
        "memory_json": str(args.memory_json),
        "memory_key": args.memory_key,
        "summary": summary,
        "decision": decision_payload,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(summary, args.out_summary_csv)
    write_csv(rows, args.out_condition_csv)
    write_md(
        summary=summary,
        details=rows,
        decision_payload=decision_payload,
        memory_json=args.memory_json,
        memory_key=args.memory_key,
        out_md=args.out_md,
    )
    print(json.dumps({"status": decision_payload["status"], "out_md": str(args.out_md)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
