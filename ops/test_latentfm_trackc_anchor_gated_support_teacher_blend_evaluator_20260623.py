from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path("/data/cyx/1030/scLatent")
SCRIPT = ROOT / "ops/evaluate_latentfm_trackc_anchor_gated_support_teacher_blend_20260623.py"
SUPPORT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
CANONICAL_SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"


def load_module():
    spec = importlib.util.spec_from_file_location("trackc_blend_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scope_guards_accept_only_safe_support_trainselect_group() -> None:
    mod = load_module()

    mod._validate_scope(
        scope="support_trainselect",
        group_kind="split",
        groups=["support_val_multi"],
        split_path=SUPPORT_SPLIT,
    )


def test_scope_guards_reject_support_query_group() -> None:
    mod = load_module()

    with pytest.raises(ValueError, match="held-out query"):
        mod._validate_scope(
            scope="support_trainselect",
            group_kind="split",
            groups=["heldout_query_multi_final_only"],
            split_path=SUPPORT_SPLIT,
        )


def test_scope_guards_reject_support_wrong_split() -> None:
    mod = load_module()

    with pytest.raises(ValueError, match="support_trainselect scope requires"):
        mod._validate_scope(
            scope="support_trainselect",
            group_kind="split",
            groups=["support_val_multi"],
            split_path=CANONICAL_SPLIT,
        )


def test_scope_guards_reject_canonical_multi_selection() -> None:
    mod = load_module()

    with pytest.raises(ValueError, match="canonical_noharm scope forbids canonical multi"):
        mod._validate_scope(
            scope="canonical_noharm",
            group_kind="split",
            groups=["test_multi"],
            split_path=CANONICAL_SPLIT,
        )


def test_scope_guards_accept_canonical_noharm_groups() -> None:
    mod = load_module()

    mod._validate_scope(
        scope="canonical_noharm",
        group_kind="split",
        groups=["test_single"],
        split_path=CANONICAL_SPLIT,
    )
    mod._validate_scope(
        scope="canonical_noharm",
        group_kind="family",
        groups=["family_gene"],
        split_path=CANONICAL_SPLIT,
    )


def test_gate_values_are_scope_determined() -> None:
    mod = load_module()

    assert mod._gate_for_scope("support_trainselect") == 1.0
    assert mod._gate_for_scope("heldout_query_once") == 1.0
    assert mod._gate_for_scope("canonical_noharm") == 0.0
    with pytest.raises(ValueError):
        mod._gate_for_scope("heldout_query")


def test_query_scope_accepts_only_final_query_groups() -> None:
    mod = load_module()

    mod._validate_scope(
        scope="heldout_query_once",
        group_kind="split",
        groups=[
            "heldout_query_multi_final_only",
            "heldout_query_multi_seen_final_only",
            "heldout_query_multi_unseen1_final_only",
            "heldout_query_multi_unseen2_final_only",
        ],
        split_path=SUPPORT_SPLIT,
    )


def test_query_scope_rejects_support_or_canonical_groups() -> None:
    mod = load_module()

    with pytest.raises(ValueError, match="unsupported groups"):
        mod._validate_scope(
            scope="heldout_query_once",
            group_kind="split",
            groups=["support_val_multi"],
            split_path=SUPPORT_SPLIT,
        )
    with pytest.raises(ValueError, match="unsupported groups"):
        mod._validate_scope(
            scope="heldout_query_once",
            group_kind="split",
            groups=["test_single"],
            split_path=SUPPORT_SPLIT,
        )


def test_query_scope_rejects_wrong_split_or_family_kind() -> None:
    mod = load_module()

    with pytest.raises(ValueError, match="heldout_query_once scope requires"):
        mod._validate_scope(
            scope="heldout_query_once",
            group_kind="split",
            groups=["heldout_query_multi_final_only"],
            split_path=CANONICAL_SPLIT,
        )
    with pytest.raises(ValueError, match="requires --group-kind split"):
        mod._validate_scope(
            scope="heldout_query_once",
            group_kind="family",
            groups=["heldout_query_multi_final_only"],
            split_path=SUPPORT_SPLIT,
        )
