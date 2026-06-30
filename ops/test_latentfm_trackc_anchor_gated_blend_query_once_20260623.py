from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
SCRIPT = ROOT / "ops/summarize_latentfm_trackc_anchor_gated_blend_query_once_20260623.py"


def load_module():
    spec = importlib.util.spec_from_file_location("trackc_blend_query_summary", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(dataset: str, condition: str, *, pp_delta: float, mmd_delta: float) -> dict:
    return {
        "dataset": dataset,
        "condition": condition,
        "anchor_pearson_pert": 0.10,
        "blend_pearson_pert": 0.10 + pp_delta,
        "blend_delta_vs_anchor_pearson_pert": pp_delta,
        "anchor_test_mmd_clamped": 0.05,
        "blend_test_mmd_clamped": 0.05 + mmd_delta,
        "blend_delta_vs_anchor_test_mmd_clamped": mmd_delta,
    }


def group_rows(pp_delta: float = 0.03, mmd_delta: float = -0.01) -> list[dict]:
    return [
        row("Wessels", "W1+W2", pp_delta=pp_delta, mmd_delta=mmd_delta),
        row("Wessels", "W3+W4", pp_delta=pp_delta, mmd_delta=mmd_delta),
        row("NormanWeissman2019_filtered", "N1+N2", pp_delta=pp_delta, mmd_delta=mmd_delta),
        row("NormanWeissman2019_filtered", "N3+N4", pp_delta=pp_delta, mmd_delta=mmd_delta),
    ]


def query_payload(*, scope: str = "heldout_query_once", safety_clean: bool = True, pp_delta: float = 0.03, mmd_delta: float = -0.01) -> dict:
    groups = {
        group: {"condition_metrics": group_rows(pp_delta=pp_delta, mmd_delta=mmd_delta)}
        for group in (
            "heldout_query_multi_final_only",
            "heldout_query_multi_seen_final_only",
            "heldout_query_multi_unseen1_final_only",
            "heldout_query_multi_unseen2_final_only",
        )
    }
    return {
        "scope": scope,
        "safety": {
            "heldout_query_read": True if safety_clean else False,
            "canonical_multi_selection": False,
            "query_result_may_select_or_tune": False,
        },
        "groups": groups,
    }


def write_json(path: Path, obj: dict) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def args_for(tmp_path: Path, query: dict, posthoc_status: str, *, n_boot: int = 128) -> argparse.Namespace:
    return argparse.Namespace(
        query_json=write_json(tmp_path / "query.json", query),
        posthoc_gate_json=write_json(tmp_path / "posthoc.json", {"status": posthoc_status}),
        out_json=tmp_path / "out.json",
        out_md=tmp_path / "out.md",
        n_boot=n_boot,
    )


def test_query_summary_candidate_supported_for_clean_positive_payload(tmp_path: Path) -> None:
    mod = load_module()

    payload = mod.summarize(args_for(tmp_path, query_payload(), mod.PASS_POSTHOC_STATUS))

    assert payload["status"] == mod.SUPPORTED_STATUS
    assert payload["decision"]["reasons"] == []
    assert payload["groups"]["heldout_query_multi_final_only"]["delta_pearson_pert"]["observed"] > 0.0


def test_query_summary_fails_closed_when_posthoc_gate_not_pass(tmp_path: Path) -> None:
    mod = load_module()

    payload = mod.summarize(args_for(tmp_path, query_payload(), "failed"))

    assert payload["status"] == mod.FAIL_CLOSED_STATUS
    assert any(reason.startswith("posthoc_gate_not_pass") for reason in payload["decision"]["reasons"])


def test_query_summary_fails_closed_for_wrong_scope_or_safety(tmp_path: Path) -> None:
    mod = load_module()

    wrong_scope = mod.summarize(args_for(tmp_path, query_payload(scope="support_trainselect"), mod.PASS_POSTHOC_STATUS))
    assert wrong_scope["status"] == mod.FAIL_CLOSED_STATUS
    assert any(reason.startswith("query_payload_wrong_scope") for reason in wrong_scope["decision"]["reasons"])

    bad_safety = mod.summarize(args_for(tmp_path, query_payload(safety_clean=False), mod.PASS_POSTHOC_STATUS))
    assert bad_safety["status"] == mod.FAIL_CLOSED_STATUS
    assert "query_payload_does_not_mark_heldout_query_read" in bad_safety["decision"]["reasons"]


def test_query_summary_negative_query_closes_branch_not_precondition(tmp_path: Path) -> None:
    mod = load_module()

    payload = mod.summarize(
        args_for(
            tmp_path,
            query_payload(pp_delta=-0.03, mmd_delta=0.02),
            mod.PASS_POSTHOC_STATUS,
        )
    )

    assert payload["status"] == mod.NOT_SUPPORTED_STATUS
    assert "query_multi_pp_delta_not_positive" in payload["decision"]["reasons"]
    assert "query_multi_mmd_hard_harm_probability_gt_0p80" in payload["decision"]["reasons"]


def test_query_summary_records_worst_query_rows(tmp_path: Path) -> None:
    mod = load_module()
    query = query_payload()
    rows = query["groups"]["heldout_query_multi_final_only"]["condition_metrics"]
    rows[0]["condition"] = "worst_pp"
    rows[0]["blend_delta_vs_anchor_pearson_pert"] = -0.20
    rows[1]["condition"] = "worst_mmd"
    rows[1]["blend_delta_vs_anchor_test_mmd_clamped"] = 0.30

    payload = mod.summarize(args_for(tmp_path, query, mod.PASS_POSTHOC_STATUS))
    primary = payload["groups"]["heldout_query_multi_final_only"]

    assert primary["worst_pp_delta_rows"][0]["condition"] == "worst_pp"
    assert primary["worst_mmd_delta_rows"][0]["condition"] == "worst_mmd"
    rendered = mod.render(payload)
    assert "## Worst Query Rows" in rendered
    assert "`Wessels` / `worst_pp`" in rendered
    assert "`Wessels` / `worst_mmd`" in rendered
