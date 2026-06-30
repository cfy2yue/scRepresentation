#!/usr/bin/env python3
"""Focused guards for train-only router covariate gate status."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path("/data/cyx/1030/scLatent/ops/audit_latentfm_trainonly_router_covariates_20260621.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("router_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(route: str, group: str, metric: str, *, delta: float, p_improve: float, p_harm: float, ci_low: float, ci_high: float):
    return {
        "route": route,
        "group": group,
        "metric": metric,
        "delta": delta,
        "p_improve": p_improve,
        "p_harm": p_harm,
        "ci95": [ci_low, ci_high],
    }


def _base_rows(route: str = "candidate"):
    return [
        _row(route, "test", "pearson_pert", delta=0.01, p_improve=0.99, p_harm=0.01, ci_low=0.001, ci_high=0.02),
        _row(route, "test", "test_mmd_clamped", delta=-0.001, p_improve=0.99, p_harm=0.01, ci_low=-0.002, ci_high=0.0),
        _row(route, "family_gene", "pearson_pert", delta=0.01, p_improve=0.99, p_harm=0.01, ci_low=0.001, ci_high=0.02),
        _row(route, "family_gene", "test_mmd_clamped", delta=-0.001, p_improve=0.99, p_harm=0.01, ci_low=-0.002, ci_high=0.0),
        _row(route, "test_multi_unseen2", "pearson_pert", delta=0.02, p_improve=0.95, p_harm=0.05, ci_low=0.0, ci_high=0.04),
        _row(route, "test_multi_unseen2", "test_mmd_clamped", delta=-0.001, p_improve=0.95, p_harm=0.05, ci_low=-0.003, ci_high=0.001),
    ]


def test_dataset_resampled_weak_demotes_to_diagnostic_only():
    mod = _load_module()
    rows = _base_rows()
    ds_rows = _base_rows()
    for row in ds_rows:
        if row["metric"] == "pearson_pert":
            row["p_improve"] = 0.85

    dec = mod.assess(rows, "candidate", dataset_resampled_rows=ds_rows)
    assert dec["status"] == "diagnostic_signal_only"
    assert "dataset_resampled_test_pp_weak" in dec["reasons"]
    assert "dataset_resampled_family_pp_weak" in dec["reasons"]


def test_unseen2_hard_harm_remains_fail_even_with_dataset_resampled_reasons():
    mod = _load_module()
    rows = _base_rows()
    for row in rows:
        if row["group"] == "test_multi_unseen2" and row["metric"] == "test_mmd_clamped":
            row["delta"] = 0.01
            row["p_improve"] = 0.0
            row["p_harm"] = 1.0
            row["ci95"] = [0.001, 0.02]
    ds_rows = _base_rows()
    for row in ds_rows:
        if row["metric"] == "pearson_pert":
            row["p_improve"] = 0.85

    dec = mod.assess(rows, "candidate", dataset_resampled_rows=ds_rows)
    assert dec["status"] == "fail"
    assert "unseen2_mmd_hard_harm" in dec["reasons"]
