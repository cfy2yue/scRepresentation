#!/usr/bin/env python3
"""Summarize shifted-MMD prior-correction gate results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_IN = Path("/data/cyx/1030/scLatent/reports/latentfm_prior_correction_shifted_mmd_gate_scf_inject_20260621.json")
DEFAULT_OUT_JSON = Path("/data/cyx/1030/scLatent/reports/latentfm_priorcorr_shifted_mmd_gate_decision_20260621.json")
DEFAULT_OUT_MD = Path("/data/cyx/1030/scLatent/reports/LATENTFM_PRIORCORR_SHIFTED_MMD_GATE_DECISION_20260621.md")

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


def summarize(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "pending_missing_input",
            "input": str(path),
            "recommendation": "wait_for_evaluator",
        }
    obj = json.loads(path.read_text(encoding="utf-8"))
    summary = list(obj.get("summary", []))
    by_key: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in summary:
        ds = str(row.get("dataset"))
        group = str(row.get("group"))
        if ds in PRIORITY_DATASETS and group in PRIORITY_GROUPS:
            by_key.setdefault((ds, group, int(row.get("k", 0))), []).append(row)

    decisions = []
    for (ds, group, k), rows in sorted(by_key.items()):
        base = next((r for r in rows if as_float(r.get("alpha")) == 0.0), None)
        candidates = [r for r in rows if (as_float(r.get("alpha")) or 0.0) > 0.0]
        base_pp = as_float(base.get("pp")) if base else None
        base_pc = as_float(base.get("pc")) if base else None
        base_mmd = as_float(base.get("mmd_clamped")) if base else None

        gated = []
        for cand in candidates:
            pp = as_float(cand.get("pp"))
            pc = as_float(cand.get("pc"))
            mmd = as_float(cand.get("mmd_clamped"))
            delta_pp = None if base_pp is None or pp is None else pp - base_pp
            delta_pc = None if base_pc is None or pc is None else pc - base_pc
            mmd_ratio = None
            if base_mmd is not None and base_mmd > 0 and mmd is not None:
                mmd_ratio = mmd / base_mmd
            pass_gate = (
                delta_pp is not None
                and delta_pp >= 0.02
                and pp is not None
                and pp > 0
                and delta_pc is not None
                and delta_pc >= -0.05
                and mmd_ratio is not None
                and mmd_ratio <= 1.15
            )
            gated.append((pass_gate, delta_pp if delta_pp is not None else -999.0, cand, mmd_ratio, delta_pc))

        viable = [x for x in gated if x[0]]
        if viable:
            _, delta_pp, best, mmd_ratio, delta_pc = max(viable, key=lambda x: x[1])
        elif gated:
            _, delta_pp, best, mmd_ratio, delta_pc = max(gated, key=lambda x: x[1])
        else:
            best = None
            delta_pp = None
            delta_pc = None
            mmd_ratio = None
        best_pp = as_float(best.get("pp")) if best else None
        best_pc = as_float(best.get("pc")) if best else None
        best_mmd = as_float(best.get("mmd_clamped")) if best else None
        pass_group_gate = bool(viable)
        decisions.append(
            {
                "dataset": ds,
                "group": group,
                "k": k,
                "n_conditions": None if base is None else base.get("n_conditions"),
                "base_pp": base_pp,
                "best_alpha": None if best is None else as_float(best.get("alpha")),
                "best_pp": best_pp,
                "delta_pp": delta_pp,
                "base_pc": base_pc,
                "best_pc": best_pc,
                "delta_pc": delta_pc,
                "base_mmd_clamped": base_mmd,
                "best_mmd_clamped": best_mmd,
                "mmd_ratio": mmd_ratio,
                "pass_group_gate": pass_group_gate,
            }
        )

    norman_unseen2 = any(
        d["dataset"] == "NormanWeissman2019_filtered"
        and d["group"] == "test_multi_unseen2"
        and d["pass_group_gate"]
        for d in decisions
    )
    wessels_unseen2 = any(
        d["dataset"] == "Wessels"
        and d["group"] == "test_multi_unseen2"
        and d["pass_group_gate"]
        for d in decisions
    )
    gasperini_unseen2 = any(
        d["dataset"] == "GasperiniShendure2019_lowMOI"
        and d["group"] == "test_multi_unseen2"
        and d["pass_group_gate"]
        for d in decisions
    )
    all_unseen2 = norman_unseen2 and wessels_unseen2 and gasperini_unseen2
    if all_unseen2:
        recommendation = "consider_single_gpu_priorcorr_ppframe_smoke"
    elif norman_unseen2:
        recommendation = "mechanism_signal_only_no_gpu_training"
    else:
        recommendation = "no_priorcorr_gpu_training"

    return {
        "status": "complete",
        "input": str(path),
        "recommendation": recommendation,
        "gate_notes": {
            "uses_group_summary_only": True,
            "shifted_distribution_mmd_available": True,
            "norman_unseen2_pass": norman_unseen2,
            "wessels_unseen2_pass": wessels_unseen2,
            "gasperini_unseen2_pass": gasperini_unseen2,
            "new_gpu_training_requires_all_priority_unseen2": True,
        },
        "decisions": decisions,
    }


def write_md(payload: dict[str, Any], out: Path) -> None:
    lines = [
        "# LatentFM Prior-Correction Shifted-MMD Gate Decision",
        "",
        f"Status: `{payload['status']}`",
        f"Recommendation: `{payload.get('recommendation')}`",
        "",
    ]
    if payload["status"] != "complete":
        lines += [f"Input: `{payload.get('input')}`", ""]
        out.write_text("\n".join(lines), encoding="utf-8")
        return
    lines += [
        "This decision uses group-level summary rows only. The shifted-MMD evaluator",
        "translates each generated cell by `corrected_mean - model_mean` before",
        "computing MMD, so it is a distribution-damage gate for mean-level prior",
        "correction rather than retrained-model evidence.",
        "",
        "## Gate Notes",
        "",
    ]
    for k, v in payload["gate_notes"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines += [
        "",
        "## Group Decisions",
        "",
        "| dataset | group | k | n | base pp | best alpha | best pp | delta pp | base pc | best pc | delta pc | base MMD | best MMD | MMD ratio | pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["decisions"]:
        lines.append(
            f"| `{row['dataset']}` | `{row['group']}` | {row['k']} | {row['n_conditions']} | "
            f"{fmt(row['base_pp'])} | {fmt(row['best_alpha'])} | {fmt(row['best_pp'])} | "
            f"{fmt(row['delta_pp'])} | {fmt(row['base_pc'])} | {fmt(row['best_pc'])} | "
            f"{fmt(row['delta_pc'])} | {fmt(row['base_mmd_clamped'])} | "
            f"{fmt(row['best_mmd_clamped'])} | {fmt(row['mmd_ratio'])} | "
            f"`{row['pass_group_gate']}` |"
        )
    lines += [
        "",
        "## Training Decision",
        "",
        "- `consider_single_gpu_priorcorr_ppframe_smoke` is required before any GPU training launch from this mechanism.",
        "- Otherwise keep this as mechanism evidence and continue CPU design audit.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()
    payload = summarize(args.input)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(payload, args.out_md)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "recommendation": payload.get("recommendation")}, indent=2))


if __name__ == "__main__":
    main()
