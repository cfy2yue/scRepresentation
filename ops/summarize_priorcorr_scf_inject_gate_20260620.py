#!/usr/bin/env python3
"""Summarize the scFoundation injection prior-correction evaluator.

This script is CPU-only and read-only.  It consumes the group-level summary
emitted by ``evaluate_latentfm_prior_correction_20260619.py`` and makes an
explicit route-level decision.  It intentionally does not sort condition-level
rows to avoid selecting a single easy condition as evidence for the branch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_IN = Path("/data/cyx/1030/scLatent/reports/latentfm_prior_correction_eval_scf_inject_20260620.json")
DEFAULT_OUT_JSON = Path("/data/cyx/1030/scLatent/reports/latentfm_priorcorr_scf_inject_gate_20260620.json")
DEFAULT_OUT_MD = Path("/data/cyx/1030/scLatent/reports/LATENTFM_PRIORCORR_SCF_INJECT_GATE_20260620.md")

PRIORITY_DATASETS = (
    "NormanWeissman2019_filtered",
    "Wessels",
    "GasperiniShendure2019_lowMOI",
)
PRIORITY_GROUPS = (
    "test_multi_seen",
    "test_multi_unseen1",
    "test_multi_unseen2",
)


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def fmt(value: Any) -> str:
    value = as_float(value) if not isinstance(value, str) else value
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "pending_missing_input",
            "input": str(path),
            "reason": "prior-correction evaluator output does not exist yet",
        }
    with path.open(encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict) or "summary" not in obj:
        raise TypeError(f"unexpected evaluator JSON schema: {path}")
    return obj


def key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row.get("dataset")), str(row.get("group")), int(row.get("k", 0)))


def summarize(obj: dict[str, Any]) -> dict[str, Any]:
    if obj.get("status") == "pending_missing_input":
        return obj

    summary = list(obj.get("summary", []))
    rows_by_key: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in summary:
        ds = str(row.get("dataset"))
        group = str(row.get("group"))
        if ds not in PRIORITY_DATASETS or group not in PRIORITY_GROUPS:
            continue
        rows_by_key.setdefault(key(row), []).append(row)

    decisions: list[dict[str, Any]] = []
    for (ds, group, k), rows in sorted(rows_by_key.items()):
        base = next((r for r in rows if as_float(r.get("alpha")) == 0.0), None)
        candidates = [r for r in rows if as_float(r.get("alpha")) and as_float(r.get("alpha")) > 0]
        best = max(candidates, key=lambda r: as_float(r.get("pp")) if as_float(r.get("pp")) is not None else -999.0, default=None)
        base_pp = as_float(base.get("pp")) if base else None
        base_pc = as_float(base.get("pc")) if base else None
        best_pp = as_float(best.get("pp")) if best else None
        best_pc = as_float(best.get("pc")) if best else None
        delta_pp = None if base_pp is None or best_pp is None else best_pp - base_pp
        delta_pc = None if base_pc is None or best_pc is None else best_pc - base_pc
        useful = (
            delta_pp is not None
            and delta_pp >= 0.02
            and best_pp is not None
            and best_pp > 0
            and delta_pc is not None
            and delta_pc >= -0.05
        )
        decisions.append(
            {
                "dataset": ds,
                "group": group,
                "k": k,
                "n_conditions": None if base is None else base.get("n_conditions"),
                "base_alpha": 0.0,
                "base_pp": base_pp,
                "base_pc": base_pc,
                "best_alpha": None if best is None else as_float(best.get("alpha")),
                "best_pp": best_pp,
                "best_pc": best_pc,
                "delta_pp": delta_pp,
                "delta_pc": delta_pc,
                "useful_group_signal": useful,
            }
        )

    norman_unseen2 = [
        d for d in decisions
        if d["dataset"] == "NormanWeissman2019_filtered"
        and d["group"] == "test_multi_unseen2"
        and d["useful_group_signal"]
    ]
    wessels_unseen2 = [
        d for d in decisions
        if d["dataset"] == "Wessels"
        and d["group"] == "test_multi_unseen2"
        and d["useful_group_signal"]
    ]
    gasperini_unseen2 = [
        d for d in decisions
        if d["dataset"] == "GasperiniShendure2019_lowMOI"
        and d["group"] == "test_multi_unseen2"
        and d["useful_group_signal"]
    ]

    # The evaluator is a mean-combination diagnostic and does not report MMD or
    # family_gene.  It can justify mechanism design or a very narrow GPU smoke
    # only if it has strong group-level signal, not manuscript promotion.
    if norman_unseen2 and not wessels_unseen2:
        recommendation = "mechanism_signal_prepare_design_not_gpu_promotion"
    elif norman_unseen2 and wessels_unseen2:
        recommendation = "broad_signal_consider_single_gpu_smoke_after_mmd_plan"
    else:
        recommendation = "no_priorcorr_gpu_smoke"

    return {
        "status": "complete",
        "meta": obj.get("meta", {}),
        "input_summary_rows": len(summary),
        "decisions": decisions,
        "recommendation": recommendation,
        "gate_notes": {
            "mmd_available": False,
            "family_gene_available": False,
            "condition_level_rows_not_used_for_route_decision": True,
            "norman_unseen2_useful": bool(norman_unseen2),
            "wessels_unseen2_useful": bool(wessels_unseen2),
            "gasperini_unseen2_useful": bool(gasperini_unseen2),
        },
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# LatentFM Prior-Correction Gate",
        "",
    ]
    if payload.get("status") == "pending_missing_input":
        lines += [
            f"Status: `{payload['status']}`",
            "",
            f"Input: `{payload['input']}`",
            "",
            payload["reason"],
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines += [
        f"Status: `{payload['status']}`",
        f"Recommendation: `{payload['recommendation']}`",
        "",
        "This is a route-level summary over evaluator group means. Condition-level",
        "rows are not used to select the branch.",
        "",
        "## Gate Notes",
        "",
    ]
    for name, value in payload["gate_notes"].items():
        lines.append(f"- `{name}`: `{value}`")
    lines += [
        "",
        "## Group Decisions",
        "",
        "| dataset | group | k | n | base pp | best alpha | best pp | delta pp | base pc | best pc | delta pc | useful |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["decisions"]:
        lines.append(
            f"| `{row['dataset']}` | `{row['group']}` | {row['k']} | {row['n_conditions']} | "
            f"{fmt(row['base_pp'])} | {fmt(row['best_alpha'])} | {fmt(row['best_pp'])} | "
            f"{fmt(row['delta_pp'])} | {fmt(row['base_pc'])} | {fmt(row['best_pc'])} | "
            f"{fmt(row['delta_pc'])} | `{row['useful_group_signal']}` |"
        )
    lines += [
        "",
        "## Decision Guard",
        "",
        "- This evaluator does not provide MMD or family-gene gates.",
        "- A positive route-level prior-correction signal can justify a narrow model-design smoke, not a promotion claim.",
        "- Wessels, Norman, and Gasperini must stay separated in reports.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    payload = summarize(load_payload(args.input))
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(payload, args.out_md)
    print(json.dumps({"status": payload.get("status"), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
