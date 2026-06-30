#!/usr/bin/env python3
"""Validate LatentFM full-cap promotion gate logic with synthetic rows.

This is a lightweight guardrail for the decision helper. It does not inspect
large outputs, launch jobs, or query GPUs.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
SUMMARIZER = ROOT / "ops/summarize_latentfm_fullcap_posthoc.py"


def load_summarizer():
    spec = importlib.util.spec_from_file_location("fullcap_summary", SUMMARIZER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load summarizer: {SUMMARIZER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_summarizer()
    primary = {
        "label": "primary_scfoundation",
        "complete_split": True,
        "complete_family": True,
        "family_gene_pp": 0.04,
        "multi_unseen1_pp": -0.01,
        "multi_unseen2_pp": -0.14,
        "test_mmd": 0.027,
    }

    pending = {
        **primary,
        "label": "smoke",
        "complete_split": False,
        "complete_family": False,
    }
    status, recommendations = module.choose_recommendation([primary, pending])
    assert status == "pending", (status, recommendations)

    bad_mmd = {
        **primary,
        "label": "smoke",
        "test_mmd": 0.050,
        "family_gene_pp": 0.05,
        "multi_unseen1_pp": 0.0,
        "multi_unseen2_pp": -0.10,
    }
    status, recommendations = module.choose_recommendation([primary, bad_mmd])
    assert status == "pivot_from_scfoundation_head_smokes", (status, recommendations)
    assert any("test_mmd" in rec for rec in recommendations), recommendations

    good = {**bad_mmd, "test_mmd": 0.028}
    status, recommendations = module.choose_recommendation([primary, good])
    assert status == "promote_candidate", (status, recommendations)
    assert any("test_mmd <=" in rec for rec in recommendations), recommendations

    print("full-cap gate validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
