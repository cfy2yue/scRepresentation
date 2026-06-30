#!/usr/bin/env python3
"""CPU-only failure analysis for Track C support-context smokes.

Reads capped smoke decisions, route-gap sidecars, configs, and checkpoints.
It does not read held-out Track C query artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_xverse_trackc_support_context_20260622"
OUT_ROOT = ROOT / "CoupledFM/output/latentfm_runs/xverse_trackc_support_context_20260622"
REPORT_DIR = ROOT / "reports"
SUMMARY_JSON = REPORT_DIR / "latentfm_trackc_support_context_smoke_summary_20260622.json"
RUNS = (
    "xverse_trackc_ctx_bridge_fm_2k_seed42",
    "xverse_trackc_ctx_bridge_ep025_2k_seed42",
    "xverse_trackc_ctx_bridge_ep050_2k_seed42",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def checkpoint_adapter_stats(path: Path) -> dict[str, Any]:
    import torch

    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    stats: dict[str, Any] = {
        "checkpoint": str(path),
        "checkpoint_exists": path.is_file(),
        "step": ckpt.get("step") if isinstance(ckpt, dict) else None,
        "best_score": ckpt.get("best_score") if isinstance(ckpt, dict) else None,
    }
    for key in ("support_context_to_c.weight", "support_context_to_c.bias"):
        tensor = state.get(key)
        if tensor is None:
            stats[f"{key}_present"] = False
            continue
        t = tensor.float()
        stats[f"{key}_present"] = True
        stats[f"{key}_shape"] = list(t.shape)
        stats[f"{key}_l2"] = float(t.norm().item())
        stats[f"{key}_absmax"] = float(t.abs().max().item())
        stats[f"{key}_mean"] = float(t.mean().item())
    return stats


def run_row(summary: dict[str, Any], run: str) -> dict[str, Any]:
    for row in summary.get("runs") or []:
        if row.get("run") == run:
            return row
    return {}


def config_subset(path: Path) -> dict[str, Any]:
    cfg = load_json(path)
    keys = [
        "trackc_support_context_use_in_model",
        "trackc_support_context_dim",
        "trackc_support_context_source",
        "finetune_trainable_scope",
        "trackc_routed_distill_loss_weight",
        "trackc_routed_endpoint_loss_weight",
        "anchor_replay_loss_weight",
        "condition_delta_head_use_in_model",
        "total_steps",
    ]
    return {key: cfg.get(key) for key in keys if key in cfg}


def analyze() -> dict[str, Any]:
    summary = load_json(SUMMARY_JSON)
    rows: list[dict[str, Any]] = []
    for run in RUNS:
        base = run_row(summary, run)
        cfg = config_subset(OUT_ROOT / run / "config.json")
        adapter = checkpoint_adapter_stats(OUT_ROOT / run / "best.pt")
        rows.append(
            {
                "run": run,
                "status": base.get("status"),
                "base_status": base.get("base_status"),
                "route_gap_status": base.get("route_gap_status"),
                "support_pp_delta": base.get("support_pp_delta"),
                "support_pp_p_improvement": base.get("support_pp_p_improvement"),
                "wessels_support_pp_delta": base.get("wessels_support_pp_delta"),
                "wessels_route_gap_closure": base.get("wessels_route_gap_closure"),
                "norman_support_pp_delta": base.get("norman_support_pp_delta"),
                "norman_route_gap_closure": base.get("norman_route_gap_closure"),
                "canonical_single_pp_p_harm": base.get("canonical_single_pp_p_harm"),
                "canonical_family_pp_p_harm": base.get("canonical_family_pp_p_harm"),
                "reasons": base.get("reasons") or [],
                "config": cfg,
                "adapter": adapter,
            }
        )
    support_deltas = [fnum(r.get("support_pp_delta")) for r in rows]
    support_deltas = [x for x in support_deltas if x is not None]
    wessels_closures = [fnum(r.get("wessels_route_gap_closure")) for r in rows]
    wessels_closures = [x for x in wessels_closures if x is not None]
    adapter_weight_l2 = [
        fnum((r.get("adapter") or {}).get("support_context_to_c.weight_l2"))
        for r in rows
    ]
    adapter_weight_l2 = [x for x in adapter_weight_l2 if x is not None]
    decision = {
        "status": "support_context_family_failed_close_branch",
        "action": "do_not_run_uncapped_noharm_or_query; require_new_mechanism_before_gpu",
        "reasons": [
            "all_three_smokes_failed_base_support_or_canonical_gate",
            "all_three_route_gap_sidecars_failed_wessels_material_closure",
            "endpoint_dose_0_to_0p5_has_no_meaningful_support_or_wessels_dose_response",
            "adapter_weights_are_nonzero_but_signal_absorption_remains_tiny",
            "canonical_single_harm_risk_above_gate",
        ],
        "rules": [
            "close support-context bridge-only and endpoint-dose variants",
            "do not launch same-family endpoint/replay/dose sweeps",
            "next GPU branch requires a genuinely new support mechanism with train/support-only gate",
        ],
    }
    return {
        "heldout_query_used": False,
        "inputs": {
            "summary_json": str(SUMMARY_JSON),
            "run_root": str(RUN_ROOT),
            "out_root": str(OUT_ROOT),
        },
        "summary": {
            "mean_support_pp_delta": mean(support_deltas) if support_deltas else None,
            "max_support_pp_delta": max(support_deltas) if support_deltas else None,
            "mean_wessels_route_gap_closure": mean(wessels_closures) if wessels_closures else None,
            "max_wessels_route_gap_closure": max(wessels_closures) if wessels_closures else None,
            "adapter_weight_l2_min": min(adapter_weight_l2) if adapter_weight_l2 else None,
            "adapter_weight_l2_max": max(adapter_weight_l2) if adapter_weight_l2 else None,
        },
        "runs": rows,
        "decision": decision,
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Support-Context Failure Analysis",
        "",
        "CPU-only failure analysis. Inputs are capped smoke decisions, route-gap sidecars, configs, and checkpoints.",
        "Held-out Track C query artifacts are not read.",
        "",
        "## Decision",
        "",
        f"- status: `{payload['decision']['status']}`",
        f"- action: `{payload['decision']['action']}`",
        "",
        "## Aggregate",
        "",
    ]
    for key, value in payload["summary"].items():
        lines.append(f"- {key}: `{fmt(value)}`")
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| run | support delta | Wessels delta | Wessels route closure | Norman delta | canonical single pp harm | adapter weight L2 | adapter bias L2 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["runs"]:
        adapter = row.get("adapter") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['run']}`",
                    fmt(row.get("support_pp_delta")),
                    fmt(row.get("wessels_support_pp_delta")),
                    fmt(row.get("wessels_route_gap_closure")),
                    fmt(row.get("norman_support_pp_delta")),
                    fmt(row.get("canonical_single_pp_p_harm")),
                    fmt(adapter.get("support_context_to_c.weight_l2")),
                    fmt(adapter.get("support_context_to_c.bias_l2")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Reasons", ""])
    lines.extend(f"- `{reason}`" for reason in payload["decision"]["reasons"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The support-context adapter did move away from zero, but the effect is tiny and nearly identical across endpoint doses.",
            "The branch does not solve Wessels route-gap absorption and simultaneously trips canonical no-harm gates.",
            "This is negative evidence against more bridge-only or endpoint-dose/replay variants.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    payload = analyze()
    out_json = REPORT_DIR / "latentfm_trackc_support_context_failure_analysis_20260622.json"
    out_md = REPORT_DIR / "LATENTFM_TRACKC_SUPPORT_CONTEXT_FAILURE_ANALYSIS_20260622.md"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
