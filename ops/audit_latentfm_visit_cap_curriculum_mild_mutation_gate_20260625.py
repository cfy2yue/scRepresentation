#!/usr/bin/env python3
"""CPU gate for a milder visit-cap mutation after the cap3 failure."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
BASE_SCRIPT = ROOT / "ops/audit_latentfm_visit_cap_curriculum_gate_20260625.py"
SLATE_JSON = ROOT / "reports/latentfm_visit_cap_curriculum_slate_gate_20260625.json"
FAILED_DECISION_JSON = ROOT / "reports/latentfm_visit_cap_curriculum_smoke_decision_20260625.json"
OUT_JSON = ROOT / "reports/latentfm_visit_cap_curriculum_mild_mutation_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_VISIT_CAP_CURRICULUM_MILD_MUTATION_GATE_20260625.md"


MILD_CANDIDATE = {
    "name": "sublinear_visitpower0p7_cap4",
    "ds_alpha": 0.7,
    "condition_visit_power": 0.7,
    "condition_visit_cap": 4,
    "batch_size": 64,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("visit_cap_gate_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render_md(payload: dict[str, Any]) -> str:
    failed = payload["failed_cap3_metrics"]
    mild = payload["mild_metrics"]
    lines = [
        "# LatentFM Visit-Cap Curriculum Mild Mutation Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only post-failure mutation gate.",
        "- Does not train, infer, read canonical multi, read held-out Track C query, or use GPU.",
        "- This gate only asks whether one milder visit-cap smoke is distinct enough from the failed `power=0.5, cap=3` arm.",
        "",
        "## Failure Being Mutated",
        "",
        "- failed arm: `sublinear_visitpower0p5_cap3`",
        f"- cross pp delta vs anchor: `{failed['cross_pp_delta_vs_anchor']:+.6f}`",
        f"- family_gene pp delta vs anchor: `{failed['family_gene_pp_delta_vs_anchor']:+.6f}`",
        f"- family_gene MMD delta vs anchor: `{failed['family_gene_mmd_delta_vs_anchor']:+.6f}`",
        "",
        "## Mild Candidate",
        "",
        f"- candidate: `{payload['mild_candidate']['name']}`",
        f"- expected step ratio: `{mild['expected_epoch_step_ratio']:.4f}`",
        f"- high-visit mass ratio: `{mild['high_visit_mass_ratio']:.4f}`",
        f"- risk exposure reduction: `{mild['risk_dataset_mean_reduction']:.4f}`",
        f"- max visit: `{mild['max_visit_candidate']}`",
        f"- step-ratio delta vs failed cap3: `{payload['step_ratio_delta_vs_failed_cap3']:+.4f}`",
        "",
        "## Decision",
        "",
        f"- GPU authorized now: `{payload['gpu_authorized']}`",
        f"- future GPU candidate after fresh audit: `{payload['future_gpu_candidate_if_capacity_frees']}`",
        f"- reasons: `{payload['reasons']}`",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    base = load_base_module()
    manifest = base.load_json(base.MANIFEST)
    split = base.load_json(base.SPLIT_FILE)
    condition_sizes = base.read_condition_sizes(manifest, split)
    baseline = base.summarize_config(condition_sizes, base.BASELINE)
    tails = base.negative_tail_datasets()
    mild_summary = base.summarize_config(condition_sizes, MILD_CANDIDATE)
    mild_status, mild_reasons, mild_metrics = base.decide(baseline, mild_summary, tails)

    slate = load_json(SLATE_JSON)
    failed = load_json(FAILED_DECISION_JSON)
    failed_row = failed["rows"][0]
    failed_metrics = failed_row["metrics"]
    failed_cap3 = next(
        row
        for row in slate["candidate_rows"]
        if row["config"]["name"] == "sublinear_visitpower0p5_cap3"
    )
    cap3_step_ratio = failed_cap3["decision_metrics"]["expected_epoch_step_ratio"]
    step_ratio_delta = mild_metrics["expected_epoch_step_ratio"] - cap3_step_ratio

    reasons: list[str] = []
    if failed.get("status") != "internal_fail":
        reasons.append("previous_cap3_smoke_not_internal_fail")
    if float(failed_metrics.get("family_gene_pp_delta_vs_anchor", 0.0)) > -0.020:
        reasons.append("previous_failure_not_strong_family_pp_harm")
    if mild_status != "visit_cap_curriculum_gate_pass_one_bounded_smoke_candidate":
        reasons.append(f"mild_candidate_base_gate_not_pass:{mild_status}:{mild_reasons}")
    if not (0.34 <= float(mild_metrics["expected_epoch_step_ratio"]) <= 0.50):
        reasons.append("mild_candidate_step_ratio_outside_0p34_0p50")
    if float(mild_metrics["high_visit_mass_ratio"]) > 0.35:
        reasons.append("mild_candidate_high_visit_mass_ratio_gt_0p35")
    if step_ratio_delta < 0.05:
        reasons.append("mild_candidate_not_distinct_enough_from_failed_cap3")

    status = (
        "visit_cap_mild_mutation_gate_pass_one_bounded_smoke_candidate"
        if not reasons
        else "visit_cap_mild_mutation_gate_fail_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorized": False,
        "future_gpu_candidate_if_capacity_frees": status.endswith("one_bounded_smoke_candidate"),
        "mild_candidate": MILD_CANDIDATE,
        "mild_metrics": mild_metrics,
        "mild_base_gate_status": mild_status,
        "mild_base_gate_reasons": mild_reasons,
        "failed_cap3_metrics": failed_metrics,
        "failed_cap3_run": failed_row["name"],
        "step_ratio_delta_vs_failed_cap3": step_ratio_delta,
        "reasons": reasons,
        "boundary": {
            "canonical_metrics_read": False,
            "canonical_multi_selection": False,
            "trackc_query_read": False,
            "gpu_used": False,
            "train_selection": "train_only_internal",
        },
        "next_action": (
            "launch exactly one bounded mild visit-cap smoke after fresh resource audit"
            if not reasons
            else "close visit-cap family until a distinct non-exposure mechanism appears"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, indent=2, sort_keys=True))
    return 0 if not reasons else 4


if __name__ == "__main__":
    raise SystemExit(main())
