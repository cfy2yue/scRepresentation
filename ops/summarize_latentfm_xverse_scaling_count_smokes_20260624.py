#!/usr/bin/env python3
"""Summarize first xverse scaling count smokes on train-only internal gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_xverse_scaling_count_smokes_20260624"
OUT_JSON = ROOT / "reports/latentfm_xverse_scaling_count_smokes_decision_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_SCALING_COUNT_SMOKES_DECISION_20260624.md"

RUNS = [
    {
        "name": "xverse_scaling_cap30_all_3k_seed42",
        "arm": "cap30_all",
    },
    {
        "name": "xverse_scaling_cap120_all_3k_seed42",
        "arm": "cap120_all",
    },
    {
        "name": "xverse_scaling_gene_cap120_allbg_3k_seed42",
        "arm": "gene_cap120_allbg",
    },
    {
        "name": "xverse_scaling_gene_cap120_k562bg_3k_seed42",
        "arm": "gene_cap120_k562bg",
    },
    {
        "name": "xverse_scaling_type_balanced_cap120_3k_seed42",
        "arm": "type_balanced_cap120",
    },
    {
        "name": "xverse_scaling_jiang_exposure_capped_3k_seed42",
        "arm": "jiang_exposure_capped",
    },
    {
        "name": "xverse_scaling_general_exposure_cap_v2_3k_seed42",
        "arm": "general_exposure_cap_v2",
    },
    {
        "name": "xverse_scaling_full_trainonly_3k_seed42",
        "arm": "full_trainonly",
    },
]

GROUPS = [
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
    "test_single",
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_exit(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def group_metrics(payload: dict[str, Any] | None, group: str) -> dict[str, Any]:
    if not payload:
        return {}
    obj = ((payload.get("groups") or {}).get(group) or {})
    return {
        "pearson_pert": obj.get("pearson_pert"),
        "direct_pearson": obj.get("direct_pearson"),
        "test_mmd": obj.get("test_mmd"),
        "test_mse": obj.get("test_mse"),
        "n_conds": obj.get("n_conds"),
        "n_requested": obj.get("n_requested"),
        "status": obj.get("status", "ok" if obj else "missing"),
    }


def better_high(a: float | None, b: float | None, margin: float = 1e-3) -> bool:
    return a is not None and b is not None and float(a) > float(b) + margin


def no_hard_harm(candidate: float | None, anchor: float | None, *, low_is_better: bool) -> bool:
    if candidate is None or anchor is None:
        return False
    c = float(candidate)
    a = float(anchor)
    if low_is_better:
        return c <= max(a * 1.10, a + 1e-4)
    return c >= a - 5e-3


def build_rows() -> list[dict[str, Any]]:
    rows = []
    for run in RUNS:
        run_dir = RUN_ROOT / run["name"]
        eval_dir = run_dir / "posthoc_eval_internal"
        split_anchor = load_json(eval_dir / "split_group_eval_anchor_internal_ode20.json")
        split_cand = load_json(eval_dir / "split_group_eval_candidate_internal_ode20.json")
        train_exit = read_exit(run_dir / f"{run['name']}.EXIT_CODE")
        posthoc_exit = read_exit(run_dir / "POSTHOC_EXIT_CODE")
        row = {
            "name": run["name"],
            "arm": run["arm"],
            "exists": run_dir.is_dir(),
            "train_exit": train_exit,
            "posthoc_exit": posthoc_exit,
            "status": "done" if train_exit == 0 and posthoc_exit == 0 else "pending_or_failed",
            "groups": {},
        }
        for group in GROUPS:
            anchor = group_metrics(split_anchor, group)
            cand = group_metrics(split_cand, group)
            row["groups"][group] = {
                "anchor": anchor,
                "candidate": cand,
                "delta_pearson_pert": None
                if cand.get("pearson_pert") is None or anchor.get("pearson_pert") is None
                else float(cand["pearson_pert"]) - float(anchor["pearson_pert"]),
                "delta_mmd": None
                if cand.get("test_mmd") is None or anchor.get("test_mmd") is None
                else float(cand["test_mmd"]) - float(anchor["test_mmd"]),
            }
        rows.append(row)
    return rows


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count_rows = [row for row in rows if row["arm"] in {"cap30_all", "cap120_all"}]
    if any(row["status"] != "done" for row in count_rows):
        return {"status": "pending", "action": "wait_without_polling"}
    by_arm = {row["arm"]: row for row in rows}
    cap30 = by_arm["cap30_all"]["groups"]["internal_val_cross_background_seen_gene_proxy"]["candidate"]
    cap120 = by_arm["cap120_all"]["groups"]["internal_val_cross_background_seen_gene_proxy"]["candidate"]
    cap120_anchor = by_arm["cap120_all"]["groups"]["internal_val_cross_background_seen_gene_proxy"]["anchor"]
    family120 = by_arm["cap120_all"]["groups"]["internal_val_family_gene_proxy"]["candidate"]
    family120_anchor = by_arm["cap120_all"]["groups"]["internal_val_family_gene_proxy"]["anchor"]
    cap120_vs_cap30_pp = (
        None
        if cap120.get("pearson_pert") is None or cap30.get("pearson_pert") is None
        else float(cap120["pearson_pert"]) - float(cap30["pearson_pert"])
    )
    cap120_vs_anchor_pp = (
        None
        if cap120.get("pearson_pert") is None or cap120_anchor.get("pearson_pert") is None
        else float(cap120["pearson_pert"]) - float(cap120_anchor["pearson_pert"])
    )
    family120_pp_delta = (
        None
        if family120.get("pearson_pert") is None or family120_anchor.get("pearson_pert") is None
        else float(family120["pearson_pert"]) - float(family120_anchor["pearson_pert"])
    )
    family120_mmd_delta = (
        None
        if family120.get("test_mmd") is None or family120_anchor.get("test_mmd") is None
        else float(family120["test_mmd"]) - float(family120_anchor["test_mmd"])
    )
    gate_checks = {
        "cap120_crossbg_pp_minus_cap30": cap120_vs_cap30_pp,
        "cap120_crossbg_pp_minus_anchor": cap120_vs_anchor_pp,
        "cap120_family_pp_minus_anchor": family120_pp_delta,
        "cap120_family_mmd_minus_anchor": family120_mmd_delta,
        "thresholds": {
            "cap120_crossbg_pp_must_exceed_cap30_by": 1e-3,
            "cap120_crossbg_pp_must_exceed_anchor_by": 1e-3,
            "family_pp_hard_harm_floor_delta": -5e-3,
            "family_mmd_hard_harm_rule": "candidate <= max(anchor * 1.10, anchor + 1e-4)",
        },
    }
    reasons = []
    if not better_high(cap120.get("pearson_pert"), cap30.get("pearson_pert")):
        reasons.append("cap120_crossbg_pp_not_better_than_cap30")
    if not better_high(cap120.get("pearson_pert"), cap120_anchor.get("pearson_pert")):
        reasons.append("cap120_crossbg_pp_not_better_than_anchor")
    if not no_hard_harm(family120.get("pearson_pert"), family120_anchor.get("pearson_pert"), low_is_better=False):
        reasons.append("cap120_family_pp_hard_harm")
    if not no_hard_harm(family120.get("test_mmd"), family120_anchor.get("test_mmd"), low_is_better=True):
        reasons.append("cap120_family_mmd_hard_harm")
    if reasons:
        return {
            "status": "all_done_no_count_scaling_pass",
            "action": "stop_count_scaling_or_rethink_training_budget",
            "reasons": reasons,
            "gate_checks": gate_checks,
        }
    return {
        "status": "count_scaling_internal_pass",
        "action": "consider_gene_type_or_background_scaling_then_canonical_noharm_once_route_frozen",
        "reasons": [],
        "gate_checks": gate_checks,
    }


def decide_full_extension(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {row["arm"]: row for row in rows}
    if "full_trainonly" not in by_arm:
        return {"status": "not_configured", "action": "no_full_trainonly_arm"}
    if "cap120_all" not in by_arm or by_arm["cap120_all"]["status"] != "done":
        return {"status": "pending", "action": "wait_for_cap120_reference"}
    full = by_arm["full_trainonly"]
    if full["status"] != "done":
        return {"status": "pending", "action": "wait_without_polling"}
    cap120 = by_arm["cap120_all"]["groups"]["internal_val_cross_background_seen_gene_proxy"]["candidate"]
    full_cross = full["groups"]["internal_val_cross_background_seen_gene_proxy"]["candidate"]
    full_anchor = full["groups"]["internal_val_cross_background_seen_gene_proxy"]["anchor"]
    full_family = full["groups"]["internal_val_family_gene_proxy"]["candidate"]
    full_family_anchor = full["groups"]["internal_val_family_gene_proxy"]["anchor"]
    full_vs_cap120_pp = (
        None
        if full_cross.get("pearson_pert") is None or cap120.get("pearson_pert") is None
        else float(full_cross["pearson_pert"]) - float(cap120["pearson_pert"])
    )
    full_vs_anchor_pp = (
        None
        if full_cross.get("pearson_pert") is None or full_anchor.get("pearson_pert") is None
        else float(full_cross["pearson_pert"]) - float(full_anchor["pearson_pert"])
    )
    full_family_pp_delta = (
        None
        if full_family.get("pearson_pert") is None or full_family_anchor.get("pearson_pert") is None
        else float(full_family["pearson_pert"]) - float(full_family_anchor["pearson_pert"])
    )
    full_family_mmd_delta = (
        None
        if full_family.get("test_mmd") is None or full_family_anchor.get("test_mmd") is None
        else float(full_family["test_mmd"]) - float(full_family_anchor["test_mmd"])
    )
    gate_checks = {
        "full_crossbg_pp_minus_cap120": full_vs_cap120_pp,
        "full_crossbg_pp_minus_anchor": full_vs_anchor_pp,
        "full_family_pp_minus_anchor": full_family_pp_delta,
        "full_family_mmd_minus_anchor": full_family_mmd_delta,
        "thresholds": {
            "full_crossbg_pp_must_exceed_cap120_by": 1e-3,
            "full_crossbg_pp_must_exceed_anchor_by": 1e-3,
            "family_pp_hard_harm_floor_delta": -5e-3,
            "family_mmd_hard_harm_rule": "candidate <= max(anchor * 1.10, anchor + 1e-4)",
        },
    }
    reasons = []
    if not better_high(full_cross.get("pearson_pert"), cap120.get("pearson_pert")):
        reasons.append("full_crossbg_pp_not_better_than_cap120")
    if not better_high(full_cross.get("pearson_pert"), full_anchor.get("pearson_pert")):
        reasons.append("full_crossbg_pp_not_better_than_anchor")
    if not no_hard_harm(full_family.get("pearson_pert"), full_family_anchor.get("pearson_pert"), low_is_better=False):
        reasons.append("full_family_pp_hard_harm")
    if not no_hard_harm(full_family.get("test_mmd"), full_family_anchor.get("test_mmd"), low_is_better=True):
        reasons.append("full_family_mmd_hard_harm")
    if reasons:
        return {
            "status": "full_trainonly_extension_fail",
            "action": "do_not_replace_cap120_with_full_without_new_evidence",
            "reasons": reasons,
            "gate_checks": gate_checks,
        }
    return {
        "status": "full_trainonly_extension_pass",
        "action": "consider_frozen_canonical_noharm_for_full_trainonly_as_separate_candidate",
        "reasons": [],
        "gate_checks": gate_checks,
    }


def decide_type_balance_extension(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {row["arm"]: row for row in rows}
    if "type_balanced_cap120" not in by_arm:
        return {"status": "not_configured", "action": "no_type_balanced_arm"}
    if "cap120_all" not in by_arm or by_arm["cap120_all"]["status"] != "done":
        return {"status": "pending", "action": "wait_for_cap120_reference"}
    type_bal = by_arm["type_balanced_cap120"]
    if type_bal["status"] != "done":
        return {"status": "pending", "action": "wait_without_polling"}
    cap120 = by_arm["cap120_all"]["groups"]["internal_val_cross_background_seen_gene_proxy"]["candidate"]
    tb_cross = type_bal["groups"]["internal_val_cross_background_seen_gene_proxy"]["candidate"]
    tb_anchor = type_bal["groups"]["internal_val_cross_background_seen_gene_proxy"]["anchor"]
    cap120_family = by_arm["cap120_all"]["groups"]["internal_val_family_gene_proxy"]["candidate"]
    tb_family = type_bal["groups"]["internal_val_family_gene_proxy"]["candidate"]
    tb_family_anchor = type_bal["groups"]["internal_val_family_gene_proxy"]["anchor"]
    tb_vs_cap120_pp = (
        None
        if tb_cross.get("pearson_pert") is None or cap120.get("pearson_pert") is None
        else float(tb_cross["pearson_pert"]) - float(cap120["pearson_pert"])
    )
    tb_vs_anchor_pp = (
        None
        if tb_cross.get("pearson_pert") is None or tb_anchor.get("pearson_pert") is None
        else float(tb_cross["pearson_pert"]) - float(tb_anchor["pearson_pert"])
    )
    tb_family_vs_cap120_pp = (
        None
        if tb_family.get("pearson_pert") is None or cap120_family.get("pearson_pert") is None
        else float(tb_family["pearson_pert"]) - float(cap120_family["pearson_pert"])
    )
    tb_family_pp_delta = (
        None
        if tb_family.get("pearson_pert") is None or tb_family_anchor.get("pearson_pert") is None
        else float(tb_family["pearson_pert"]) - float(tb_family_anchor["pearson_pert"])
    )
    tb_family_mmd_delta = (
        None
        if tb_family.get("test_mmd") is None or tb_family_anchor.get("test_mmd") is None
        else float(tb_family["test_mmd"]) - float(tb_family_anchor["test_mmd"])
    )
    gate_checks = {
        "type_balanced_crossbg_pp_minus_cap120": tb_vs_cap120_pp,
        "type_balanced_crossbg_pp_minus_anchor": tb_vs_anchor_pp,
        "type_balanced_family_pp_minus_cap120": tb_family_vs_cap120_pp,
        "type_balanced_family_pp_minus_anchor": tb_family_pp_delta,
        "type_balanced_family_mmd_minus_anchor": tb_family_mmd_delta,
        "thresholds": {
            "crossbg_pp_may_drop_vs_cap120_by_at_most": -5e-3,
            "crossbg_pp_must_exceed_anchor_by": 1e-3,
            "family_pp_must_not_be_worse_than_cap120_by": -2e-3,
            "family_pp_hard_harm_floor_delta": -5e-3,
            "family_mmd_hard_harm_rule": "candidate <= max(anchor * 1.10, anchor + 1e-4)",
        },
    }
    reasons = []
    if tb_vs_cap120_pp is None or tb_vs_cap120_pp < -5e-3:
        reasons.append("type_balanced_crossbg_pp_too_far_below_cap120")
    if not better_high(tb_cross.get("pearson_pert"), tb_anchor.get("pearson_pert")):
        reasons.append("type_balanced_crossbg_pp_not_better_than_anchor")
    if tb_family_vs_cap120_pp is None or tb_family_vs_cap120_pp < -2e-3:
        reasons.append("type_balanced_family_pp_worse_than_cap120")
    if not no_hard_harm(tb_family.get("pearson_pert"), tb_family_anchor.get("pearson_pert"), low_is_better=False):
        reasons.append("type_balanced_family_pp_hard_harm")
    if not no_hard_harm(tb_family.get("test_mmd"), tb_family_anchor.get("test_mmd"), low_is_better=True):
        reasons.append("type_balanced_family_mmd_hard_harm")
    if reasons:
        return {
            "status": "type_balanced_extension_fail",
            "action": "do_not_promote_type_balanced_without_new_mechanism",
            "reasons": reasons,
            "gate_checks": gate_checks,
        }
    return {
        "status": "type_balanced_extension_pass",
        "action": "consider_frozen_canonical_noharm_for_type_balanced_as_separate_candidate",
        "reasons": [],
        "gate_checks": gate_checks,
    }


def decide_jiang_exposure_extension(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {row["arm"]: row for row in rows}
    if "jiang_exposure_capped" not in by_arm:
        return {"status": "not_configured", "action": "no_jiang_exposure_capped_arm"}
    jiang = by_arm["jiang_exposure_capped"]
    if not jiang.get("exists"):
        return {"status": "not_launched", "action": "standby_until_type_balanced_internal_decision"}
    if "type_balanced_cap120" not in by_arm or by_arm["type_balanced_cap120"]["status"] != "done":
        return {"status": "pending", "action": "wait_for_type_balanced_reference"}
    if jiang["status"] != "done":
        return {"status": "pending", "action": "wait_without_polling"}
    type_bal = by_arm["type_balanced_cap120"]
    tb_cross = type_bal["groups"]["internal_val_cross_background_seen_gene_proxy"]["candidate"]
    jiang_cross = jiang["groups"]["internal_val_cross_background_seen_gene_proxy"]["candidate"]
    jiang_anchor = jiang["groups"]["internal_val_cross_background_seen_gene_proxy"]["anchor"]
    tb_family = type_bal["groups"]["internal_val_family_gene_proxy"]["candidate"]
    jiang_family = jiang["groups"]["internal_val_family_gene_proxy"]["candidate"]
    jiang_family_anchor = jiang["groups"]["internal_val_family_gene_proxy"]["anchor"]
    jiang_vs_type_pp = (
        None
        if jiang_cross.get("pearson_pert") is None or tb_cross.get("pearson_pert") is None
        else float(jiang_cross["pearson_pert"]) - float(tb_cross["pearson_pert"])
    )
    jiang_vs_anchor_pp = (
        None
        if jiang_cross.get("pearson_pert") is None or jiang_anchor.get("pearson_pert") is None
        else float(jiang_cross["pearson_pert"]) - float(jiang_anchor["pearson_pert"])
    )
    jiang_family_vs_type_pp = (
        None
        if jiang_family.get("pearson_pert") is None or tb_family.get("pearson_pert") is None
        else float(jiang_family["pearson_pert"]) - float(tb_family["pearson_pert"])
    )
    jiang_family_pp_delta = (
        None
        if jiang_family.get("pearson_pert") is None or jiang_family_anchor.get("pearson_pert") is None
        else float(jiang_family["pearson_pert"]) - float(jiang_family_anchor["pearson_pert"])
    )
    jiang_family_mmd_delta = (
        None
        if jiang_family.get("test_mmd") is None or jiang_family_anchor.get("test_mmd") is None
        else float(jiang_family["test_mmd"]) - float(jiang_family_anchor["test_mmd"])
    )
    gate_checks = {
        "jiang_crossbg_pp_minus_type_balanced": jiang_vs_type_pp,
        "jiang_crossbg_pp_minus_anchor": jiang_vs_anchor_pp,
        "jiang_family_pp_minus_type_balanced": jiang_family_vs_type_pp,
        "jiang_family_pp_minus_anchor": jiang_family_pp_delta,
        "jiang_family_mmd_minus_anchor": jiang_family_mmd_delta,
        "thresholds": {
            "crossbg_pp_may_drop_vs_type_balanced_by_at_most": -5e-3,
            "crossbg_pp_must_exceed_anchor_by": 1e-3,
            "family_pp_may_drop_vs_type_balanced_by_at_most": -2e-3,
            "family_pp_hard_harm_floor_delta": -5e-3,
            "family_mmd_hard_harm_rule": "candidate <= max(anchor * 1.10, anchor + 1e-4)",
        },
    }
    reasons = []
    if jiang_vs_type_pp is None or jiang_vs_type_pp < -5e-3:
        reasons.append("jiang_crossbg_pp_too_far_below_type_balanced")
    if not better_high(jiang_cross.get("pearson_pert"), jiang_anchor.get("pearson_pert")):
        reasons.append("jiang_crossbg_pp_not_better_than_anchor")
    if jiang_family_vs_type_pp is None or jiang_family_vs_type_pp < -2e-3:
        reasons.append("jiang_family_pp_worse_than_type_balanced")
    if not no_hard_harm(jiang_family.get("pearson_pert"), jiang_family_anchor.get("pearson_pert"), low_is_better=False):
        reasons.append("jiang_family_pp_hard_harm")
    if not no_hard_harm(jiang_family.get("test_mmd"), jiang_family_anchor.get("test_mmd"), low_is_better=True):
        reasons.append("jiang_family_mmd_hard_harm")
    if reasons:
        return {
            "status": "jiang_exposure_extension_fail",
            "action": "do_not_promote_jiang_exposure_capped_without_new_mechanism",
            "reasons": reasons,
            "gate_checks": gate_checks,
        }
    return {
        "status": "jiang_exposure_extension_pass",
        "action": "consider_frozen_canonical_noharm_for_jiang_exposure_capped_as_separate_candidate",
        "reasons": [],
        "gate_checks": gate_checks,
    }


def decide_general_exposure_extension(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {row["arm"]: row for row in rows}
    if "general_exposure_cap_v2" not in by_arm:
        return {"status": "not_configured", "action": "no_general_exposure_cap_v2_arm"}
    general = by_arm["general_exposure_cap_v2"]
    if not general.get("exists"):
        return {"status": "not_launched", "action": "standby_until_jiang_internal_decision"}
    if "jiang_exposure_capped" not in by_arm or by_arm["jiang_exposure_capped"]["status"] != "done":
        return {"status": "pending", "action": "wait_for_jiang_reference"}
    if general["status"] != "done":
        return {"status": "pending", "action": "wait_without_polling"}
    jiang = by_arm["jiang_exposure_capped"]
    jiang_cross = jiang["groups"]["internal_val_cross_background_seen_gene_proxy"]["candidate"]
    general_cross = general["groups"]["internal_val_cross_background_seen_gene_proxy"]["candidate"]
    general_anchor = general["groups"]["internal_val_cross_background_seen_gene_proxy"]["anchor"]
    jiang_family = jiang["groups"]["internal_val_family_gene_proxy"]["candidate"]
    general_family = general["groups"]["internal_val_family_gene_proxy"]["candidate"]
    general_family_anchor = general["groups"]["internal_val_family_gene_proxy"]["anchor"]
    general_vs_jiang_pp = (
        None
        if general_cross.get("pearson_pert") is None or jiang_cross.get("pearson_pert") is None
        else float(general_cross["pearson_pert"]) - float(jiang_cross["pearson_pert"])
    )
    general_vs_anchor_pp = (
        None
        if general_cross.get("pearson_pert") is None or general_anchor.get("pearson_pert") is None
        else float(general_cross["pearson_pert"]) - float(general_anchor["pearson_pert"])
    )
    general_family_vs_jiang_pp = (
        None
        if general_family.get("pearson_pert") is None or jiang_family.get("pearson_pert") is None
        else float(general_family["pearson_pert"]) - float(jiang_family["pearson_pert"])
    )
    general_family_pp_delta = (
        None
        if general_family.get("pearson_pert") is None or general_family_anchor.get("pearson_pert") is None
        else float(general_family["pearson_pert"]) - float(general_family_anchor["pearson_pert"])
    )
    general_family_mmd_delta = (
        None
        if general_family.get("test_mmd") is None or general_family_anchor.get("test_mmd") is None
        else float(general_family["test_mmd"]) - float(general_family_anchor["test_mmd"])
    )
    gate_checks = {
        "general_crossbg_pp_minus_jiang": general_vs_jiang_pp,
        "general_crossbg_pp_minus_anchor": general_vs_anchor_pp,
        "general_family_pp_minus_jiang": general_family_vs_jiang_pp,
        "general_family_pp_minus_anchor": general_family_pp_delta,
        "general_family_mmd_minus_anchor": general_family_mmd_delta,
        "thresholds": {
            "crossbg_pp_may_drop_vs_jiang_by_at_most": -5e-3,
            "crossbg_pp_must_exceed_anchor_by": 1e-3,
            "family_pp_may_drop_vs_jiang_by_at_most": -2e-3,
            "family_pp_hard_harm_floor_delta": -5e-3,
            "family_mmd_hard_harm_rule": "candidate <= max(anchor * 1.10, anchor + 1e-4)",
        },
    }
    reasons = []
    if general_vs_jiang_pp is None or general_vs_jiang_pp < -5e-3:
        reasons.append("general_crossbg_pp_too_far_below_jiang")
    if not better_high(general_cross.get("pearson_pert"), general_anchor.get("pearson_pert")):
        reasons.append("general_crossbg_pp_not_better_than_anchor")
    if general_family_vs_jiang_pp is None or general_family_vs_jiang_pp < -2e-3:
        reasons.append("general_family_pp_worse_than_jiang")
    if not no_hard_harm(general_family.get("pearson_pert"), general_family_anchor.get("pearson_pert"), low_is_better=False):
        reasons.append("general_family_pp_hard_harm")
    if not no_hard_harm(general_family.get("test_mmd"), general_family_anchor.get("test_mmd"), low_is_better=True):
        reasons.append("general_family_mmd_hard_harm")
    if reasons:
        return {
            "status": "general_exposure_extension_fail",
            "action": "do_not_promote_general_exposure_cap_v2_without_new_mechanism",
            "reasons": reasons,
            "gate_checks": gate_checks,
        }
    return {
        "status": "general_exposure_extension_pass",
        "action": "consider_frozen_canonical_noharm_for_general_exposure_cap_v2_as_separate_candidate",
        "reasons": [],
        "gate_checks": gate_checks,
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Scaling Count Smokes Decision",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Action: `{payload['decision']['action']}`",
        "",
        "## Boundary",
        "",
        "- Summarizes first nested-v2 count-scaling smokes only.",
        "- Decision gate uses train-only internal validation groups.",
        "- The cross-background proxy is fixed as a named protocol/group; each arm still uses its matching train-only split and pert-mean artifact.",
        "- Canonical held-out metrics are not used for this decision.",
        "",
        "## Gate Rule",
        "",
        "- Primary arm for freeze: `cap120_all`.",
        "- `cap120_all` must exceed `cap30_all` on `internal_val_cross_background_seen_gene_proxy` pearson_pert by `> 0.001`.",
        "- `cap120_all` must exceed its anchor on the same group by `> 0.001`.",
        "- `cap120_all` must not hard-harm `internal_val_family_gene_proxy` pearson_pert or MMD.",
        "- Gene/background/type-balanced/full arms are reported as diagnostics/extensions; they do not replace the primary count-gate winner.",
        "",
        "## Gate Checks",
        "",
    ]
    checks = payload["decision"].get("gate_checks") or {}
    lines += [
        f"- cap120 cross-bg pp minus cap30: `{_fmt(checks.get('cap120_crossbg_pp_minus_cap30'))}`",
        f"- cap120 cross-bg pp minus anchor: `{_fmt(checks.get('cap120_crossbg_pp_minus_anchor'))}`",
        f"- cap120 family pp minus anchor: `{_fmt(checks.get('cap120_family_pp_minus_anchor'))}`",
        f"- cap120 family MMD minus anchor: `{_fmt(checks.get('cap120_family_mmd_minus_anchor'))}`",
        "",
        "## Rows",
        "",
        "| run | status | cross-bg cand pp | cross-bg delta pp vs anchor | cross-bg cand MMD | family cand pp | family delta MMD vs anchor |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        cross = row["groups"]["internal_val_cross_background_seen_gene_proxy"]
        fam = row["groups"]["internal_val_family_gene_proxy"]
        lines.append(
            f"| `{row['name']}` | `{row['status']}` | "
            f"{_fmt(cross['candidate'].get('pearson_pert'))} | {_fmt(cross['delta_pearson_pert'])} | "
            f"{_fmt(cross['candidate'].get('test_mmd'))} | {_fmt(fam['candidate'].get('pearson_pert'))} | "
            f"{_fmt(fam['delta_mmd'])} |"
        )
    if payload["decision"].get("reasons"):
        lines += ["", "Gate reasons:"]
        lines.extend(f"- `{r}`" for r in payload["decision"]["reasons"])
    full_decision = payload.get("full_extension_decision") or {}
    full_checks = full_decision.get("gate_checks") or {}
    lines += [
        "",
        "## Full Train-Only Extension",
        "",
        f"Status: `{full_decision.get('status', 'missing')}`",
        f"Action: `{full_decision.get('action', 'missing')}`",
        "",
        f"- full cross-bg pp minus cap120: `{_fmt(full_checks.get('full_crossbg_pp_minus_cap120'))}`",
        f"- full cross-bg pp minus anchor: `{_fmt(full_checks.get('full_crossbg_pp_minus_anchor'))}`",
        f"- full family pp minus anchor: `{_fmt(full_checks.get('full_family_pp_minus_anchor'))}`",
        f"- full family MMD minus anchor: `{_fmt(full_checks.get('full_family_mmd_minus_anchor'))}`",
    ]
    if full_decision.get("reasons"):
        lines += ["", "Full extension gate reasons:"]
        lines.extend(f"- `{r}`" for r in full_decision["reasons"])
    type_balance_decision = payload.get("type_balance_extension_decision") or {}
    type_balance_checks = type_balance_decision.get("gate_checks") or {}
    lines += [
        "",
        "## Type-Balanced Cap120 Extension",
        "",
        f"Status: `{type_balance_decision.get('status', 'missing')}`",
        f"Action: `{type_balance_decision.get('action', 'missing')}`",
        "",
        f"- type-balanced cross-bg pp minus cap120: `{_fmt(type_balance_checks.get('type_balanced_crossbg_pp_minus_cap120'))}`",
        f"- type-balanced cross-bg pp minus anchor: `{_fmt(type_balance_checks.get('type_balanced_crossbg_pp_minus_anchor'))}`",
        f"- type-balanced family pp minus cap120: `{_fmt(type_balance_checks.get('type_balanced_family_pp_minus_cap120'))}`",
        f"- type-balanced family pp minus anchor: `{_fmt(type_balance_checks.get('type_balanced_family_pp_minus_anchor'))}`",
        f"- type-balanced family MMD minus anchor: `{_fmt(type_balance_checks.get('type_balanced_family_mmd_minus_anchor'))}`",
    ]
    if type_balance_decision.get("reasons"):
        lines += ["", "Type-balanced extension gate reasons:"]
        lines.extend(f"- `{r}`" for r in type_balance_decision["reasons"])
    jiang_decision = payload.get("jiang_exposure_extension_decision") or {}
    jiang_checks = jiang_decision.get("gate_checks") or {}
    lines += [
        "",
        "## Jiang Exposure-Capped Extension",
        "",
        f"Status: `{jiang_decision.get('status', 'missing')}`",
        f"Action: `{jiang_decision.get('action', 'missing')}`",
        "",
        f"- Jiang exposure cross-bg pp minus type-balanced: `{_fmt(jiang_checks.get('jiang_crossbg_pp_minus_type_balanced'))}`",
        f"- Jiang exposure cross-bg pp minus anchor: `{_fmt(jiang_checks.get('jiang_crossbg_pp_minus_anchor'))}`",
        f"- Jiang exposure family pp minus type-balanced: `{_fmt(jiang_checks.get('jiang_family_pp_minus_type_balanced'))}`",
        f"- Jiang exposure family pp minus anchor: `{_fmt(jiang_checks.get('jiang_family_pp_minus_anchor'))}`",
        f"- Jiang exposure family MMD minus anchor: `{_fmt(jiang_checks.get('jiang_family_mmd_minus_anchor'))}`",
    ]
    if jiang_decision.get("reasons"):
        lines += ["", "Jiang exposure extension gate reasons:"]
        lines.extend(f"- `{r}`" for r in jiang_decision["reasons"])
    general_decision = payload.get("general_exposure_extension_decision") or {}
    general_checks = general_decision.get("gate_checks") or {}
    lines += [
        "",
        "## General Exposure-Cap v2 Extension",
        "",
        f"Status: `{general_decision.get('status', 'missing')}`",
        f"Action: `{general_decision.get('action', 'missing')}`",
        "",
        f"- general exposure cross-bg pp minus Jiang: `{_fmt(general_checks.get('general_crossbg_pp_minus_jiang'))}`",
        f"- general exposure cross-bg pp minus anchor: `{_fmt(general_checks.get('general_crossbg_pp_minus_anchor'))}`",
        f"- general exposure family pp minus Jiang: `{_fmt(general_checks.get('general_family_pp_minus_jiang'))}`",
        f"- general exposure family pp minus anchor: `{_fmt(general_checks.get('general_family_pp_minus_anchor'))}`",
        f"- general exposure family MMD minus anchor: `{_fmt(general_checks.get('general_family_mmd_minus_anchor'))}`",
    ]
    if general_decision.get("reasons"):
        lines += ["", "General exposure extension gate reasons:"]
        lines.extend(f"- `{r}`" for r in general_decision["reasons"])
    lines += [
        "",
        "## Decision JSON",
        "",
        f"`{OUT_JSON}`",
        "",
    ]
    return "\n".join(lines)


def _fmt(x: Any) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


def main() -> int:
    rows = build_rows()
    payload = {
        "run_root": str(RUN_ROOT),
        "decision": decide(rows),
        "full_extension_decision": decide_full_extension(rows),
        "type_balance_extension_decision": decide_type_balance_extension(rows),
        "jiang_exposure_extension_decision": decide_jiang_exposure_extension(rows),
        "general_exposure_extension_decision": decide_general_exposure_extension(rows),
        "rows": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
