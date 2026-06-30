#!/usr/bin/env python3
"""Validate endpoint-routed Track C decision metadata without polling jobs.

This reads only static code/checklist files. It must not inspect tmux, logs,
exit-code files, posthoc outputs, or live decision artifacts.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
CHECKLIST = ROOT / "reports/latentfm_trackc_endpoint_routed_decision_checklist_20260622.json"
SUMMARIZER = ROOT / "ops/summarize_latentfm_trackc_routed_distill_smoke_20260622.py"


def load_summarizer():
    spec = importlib.util.spec_from_file_location("trackc_smoke_summary", SUMMARIZER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SUMMARIZER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(*, delta: float = 0.03, p_improvement: float = 0.90, p_harm: float = 0.10):
    return {
        "status": "ok",
        "n_matched_conditions": 8,
        "delta_mean": delta,
        "p_improvement": p_improvement,
        "p_harm": p_harm,
    }


def table_case(*, support_delta=0.03, support_p_improve=0.90, canonical_pp_harm=0.10, canonical_mmd_harm=0.10):
    return {
        "support_split": {
            ("test_multi", "pearson_pert"): row(delta=support_delta, p_improvement=support_p_improve),
            ("test_multi", "test_mmd_clamped"): row(delta=-0.01, p_harm=0.10),
        },
        "canonical_split": {
            ("test_single", "pearson_pert"): row(delta=0.00, p_harm=canonical_pp_harm),
            ("test_single", "test_mmd_clamped"): row(delta=0.00, p_harm=canonical_mmd_harm),
        },
        "canonical_family": {
            ("family_gene", "pearson_pert"): row(delta=0.00, p_harm=canonical_pp_harm),
            ("family_gene", "test_mmd_clamped"): row(delta=0.00, p_harm=canonical_mmd_harm),
        },
    }


def main() -> int:
    checklist = json.loads(CHECKLIST.read_text(encoding="utf-8"))
    smoke = checklist["smoke_decision"]
    summarizer = load_summarizer()

    expected = {
        "pass": "trackc_smoke_support_pass_needs_uncapped_noharm_before_query",
        "support_fail": "trackc_smoke_fail_support_gate_close_branch",
        "canonical_fail": "trackc_smoke_fail_canonical_harm_close_branch",
        "missing": "trackc_smoke_missing_required_metrics_close_branch",
    }
    observed = {
        "pass": summarizer.evaluate_gate(table_case())["status"],
        "support_fail": summarizer.evaluate_gate(table_case(support_delta=0.0))["status"],
        "canonical_fail": summarizer.evaluate_gate(table_case(canonical_mmd_harm=1.0))["status"],
        "missing": summarizer.evaluate_gate({})["status"],
    }
    if observed != expected:
        raise AssertionError({"expected": expected, "observed": observed})

    if smoke["pass_status"] != expected["pass"]:
        raise AssertionError(f"wrong pass status in checklist: {smoke['pass_status']}")
    fail_statuses = set(smoke["fail_statuses"])
    for key in ("support_fail", "canonical_fail", "missing"):
        if expected[key] not in fail_statuses:
            raise AssertionError(f"missing checklist fail status: {expected[key]}")
    if "trackc_smoke_fail_support_close_branch" in fail_statuses:
        raise AssertionError("stale support-fail status remains in checklist")

    next_check = checklist["polling_rule"]["next_manual_check_not_before"]
    if next_check != "2026-06-22 17:40 CST":
        raise AssertionError(f"unexpected next check boundary: {next_check}")

    print(
        json.dumps(
            {
                "status": "endpoint_decision_metadata_validation_pass",
                "observed": observed,
                "next_manual_check_not_before": next_check,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
