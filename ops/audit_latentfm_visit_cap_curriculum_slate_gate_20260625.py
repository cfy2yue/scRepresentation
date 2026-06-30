#!/usr/bin/env python3
"""Slate gate for LatentFM condition-visit curriculum candidates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
BASE_SCRIPT = ROOT / "ops/audit_latentfm_visit_cap_curriculum_gate_20260625.py"
OUT_JSON = ROOT / "reports/latentfm_visit_cap_curriculum_slate_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_VISIT_CAP_CURRICULUM_SLATE_GATE_20260625.md"


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("visit_cap_gate_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Visit-Cap Curriculum Slate Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only sampler exposure simulation over predeclared visit-cap candidates.",
        "- Does not train, infer, read canonical multi, read held-out Track C query, or use GPU.",
        "- A pass authorizes only preparing one bounded train-only smoke after the current GPU block frees.",
        "",
        "## Candidate Rows",
        "",
        "| candidate | status | step ratio | high-visit mass ratio | risk exposure reduction | max visit | reasons |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["candidate_rows"]:
        metrics = row["decision_metrics"]
        lines.append(
            f"| `{row['config']['name']}` | `{row['status']}` | "
            f"{metrics['expected_epoch_step_ratio']:.4f} | "
            f"{metrics['high_visit_mass_ratio']:.4f} | "
            f"{metrics['risk_dataset_mean_reduction']:.4f} | "
            f"{metrics['max_visit_candidate']} | {', '.join(row['reasons']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Selected Candidate",
            "",
            f"- selected: `{payload.get('selected_candidate', {}).get('config', {}).get('name')}`",
            f"- GPU authorized now: `{payload['gpu_authorized']}`",
            f"- future GPU candidate after capacity frees: `{payload['future_gpu_candidate_if_capacity_frees']}`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    base = load_base_module()
    manifest = base.load_json(base.MANIFEST)
    split = base.load_json(base.SPLIT_FILE)
    condition_sizes = base.read_condition_sizes(manifest, split)
    baseline = base.summarize_config(condition_sizes, base.BASELINE)
    tails = base.negative_tail_datasets()
    candidates = [
        {"name": "sublinear_visitpower0p5_no_cap", "condition_visit_power": 0.5, "condition_visit_cap": 0},
        {"name": "sublinear_visitpower0p5_cap2", "condition_visit_power": 0.5, "condition_visit_cap": 2},
        {"name": "sublinear_visitpower0p5_cap3", "condition_visit_power": 0.5, "condition_visit_cap": 3},
        {"name": "sublinear_visitpower0p6_cap3", "condition_visit_power": 0.6, "condition_visit_cap": 3},
        {"name": "sublinear_visitpower0p7_cap4", "condition_visit_power": 0.7, "condition_visit_cap": 4},
        {"name": "sublinear_visitpower0p8_cap6", "condition_visit_power": 0.8, "condition_visit_cap": 6},
    ]
    rows = []
    for candidate in candidates:
        cfg = {
            **base.BASELINE,
            **candidate,
            "ds_alpha": 0.7,
            "batch_size": 64,
        }
        summary = base.summarize_config(condition_sizes, cfg)
        status, reasons, metrics = base.decide(baseline, summary, tails)
        rows.append(
            {
                "config": cfg,
                "status": status,
                "reasons": reasons,
                "decision_metrics": metrics,
            }
        )
    passing = [r for r in rows if r["status"].endswith("one_bounded_smoke_candidate")]
    # Prefer the gentlest candidate that still eliminates high-visit mass and
    # keeps expected steps above 25%; this avoids the too-aggressive cap2 case.
    zero_high = [r for r in passing if r["decision_metrics"]["high_visit_mass_ratio"] <= 1e-9]
    selected = None
    if zero_high:
        selected = sorted(
            zero_high,
            key=lambda r: (
                r["decision_metrics"]["expected_epoch_step_ratio"],
                -r["decision_metrics"]["risk_dataset_mean_reduction"],
            ),
        )[0]
    elif passing:
        selected = sorted(passing, key=lambda r: r["decision_metrics"]["expected_epoch_step_ratio"])[0]
    status = "visit_cap_curriculum_slate_gate_pass_one_bounded_smoke_candidate" if selected else "visit_cap_curriculum_slate_gate_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "future_gpu_candidate_if_capacity_frees": bool(selected),
        "baseline": baseline,
        "candidate_rows": rows,
        "selected_candidate": selected,
        "tails": tails,
        "next_action": (
            "prepare a fail-closed one-run launcher for the selected candidate, but launch only after current GPU block frees and fresh resource audit passes"
            if selected
            else "do not launch visit-cap GPU"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
