from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
SCRIPT = ROOT / "ops/summarize_latentfm_trackc_anchor_gated_support_teacher_blend_posthoc_20260623.py"


def load_module():
    spec = importlib.util.spec_from_file_location("trackc_blend_posthoc_summary", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(dataset: str, condition: str, *, pp: float, mmd: float, mmd_biased: float, noop_delta: float = 0.0) -> dict:
    return {
        "dataset": dataset,
        "condition": condition,
        "blend_delta_vs_anchor_pearson_pert": pp,
        "blend_delta_vs_anchor_test_mmd": mmd,
        "blend_delta_vs_anchor_test_mmd_biased": mmd_biased,
        "blend_delta_vs_anchor_test_mmd_clamped": noop_delta,
        "blend_delta_vs_anchor_direct_pearson": noop_delta,
        "blend_delta_vs_anchor_pearson_ctrl": noop_delta,
    }


def payload(group: str, rows: list[dict], *, clean_safety: bool = True) -> dict:
    return {
        "safety": {
            "heldout_query_read": False if clean_safety else True,
            "canonical_multi_selection": False,
        },
        "groups": {group: {"condition_metrics": rows}},
    }


def write_json(path: Path, obj: dict) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def args_for(tmp_path: Path, support: dict, single: dict, family: dict, *, n_boot: int = 128) -> argparse.Namespace:
    return argparse.Namespace(
        support_json=write_json(tmp_path / "support.json", support),
        canonical_test_single_json=write_json(tmp_path / "single.json", single),
        canonical_family_gene_json=write_json(tmp_path / "family.json", family),
        support_group="support_val_multi",
        out_json=tmp_path / "out.json",
        out_md=tmp_path / "out.md",
        n_boot=n_boot,
        noop_tol=1e-8,
    )


def passing_payloads() -> tuple[dict, dict, dict]:
    support_rows = [
        row("Wessels", "W1+W2", pp=0.04, mmd=-0.01, mmd_biased=-0.01),
        row("Wessels", "W3+W4", pp=0.03, mmd=-0.02, mmd_biased=-0.02),
        row("NormanWeissman2019_filtered", "N1+N2", pp=0.03, mmd=-0.01, mmd_biased=-0.01),
        row("NormanWeissman2019_filtered", "N3+N4", pp=0.03, mmd=-0.01, mmd_biased=-0.01),
    ]
    single_rows = [
        row("DatasetA", "G1", pp=0.0, mmd=0.0, mmd_biased=0.0),
        row("DatasetB", "G2", pp=0.0, mmd=0.0, mmd_biased=0.0),
    ]
    family_rows = [
        row("DatasetA", "G1", pp=0.0, mmd=0.0, mmd_biased=0.0),
        row("DatasetB", "G2", pp=0.0, mmd=0.0, mmd_biased=0.0),
    ]
    return (
        payload("support_val_multi", support_rows),
        payload("test_single", single_rows),
        payload("family_gene", family_rows),
    )


def test_blend_posthoc_summary_passes_clean_support_and_exact_noop(tmp_path: Path) -> None:
    mod = load_module()
    support, single, family = passing_payloads()

    decision = mod.summarize(args_for(tmp_path, support, single, family))

    assert decision["status"] == mod.PASS_STATUS
    assert decision["reasons"] == []
    assert decision["support"]["pearson_pert_delta"]["observed"] >= 0.02
    assert decision["support"]["test_mmd_delta"]["observed"] <= 0.005


def test_blend_posthoc_summary_fails_support_mmd_harm(tmp_path: Path) -> None:
    mod = load_module()
    support, single, family = passing_payloads()
    for support_row in support["groups"]["support_val_multi"]["condition_metrics"]:
        support_row["blend_delta_vs_anchor_test_mmd"] = 0.02
        support_row["blend_delta_vs_anchor_test_mmd_biased"] = 0.02

    decision = mod.summarize(args_for(tmp_path, support, single, family))

    assert decision["status"] != mod.PASS_STATUS
    assert "support_unbiased_mmd_delta_above_0p005" in decision["reasons"]
    assert "support_biased_mmd_delta_above_0p005" in decision["reasons"]


def test_blend_posthoc_summary_fails_canonical_non_noop(tmp_path: Path) -> None:
    mod = load_module()
    support, single, family = passing_payloads()
    single["groups"]["test_single"]["condition_metrics"][0]["blend_delta_vs_anchor_pearson_pert"] = 1e-4

    decision = mod.summarize(args_for(tmp_path, support, single, family))

    assert decision["status"] != mod.PASS_STATUS
    assert "canonical_test_single_blend_delta_vs_anchor_pearson_pert_not_exact_noop" in decision["reasons"]


def test_blend_posthoc_summary_fails_unclean_safety_flag(tmp_path: Path) -> None:
    mod = load_module()
    support, single, family = passing_payloads()
    support["safety"]["heldout_query_read"] = True

    decision = mod.summarize(args_for(tmp_path, support, single, family))

    assert decision["status"] != mod.PASS_STATUS
    assert "payload_safety_flags_not_clean" in decision["reasons"]
