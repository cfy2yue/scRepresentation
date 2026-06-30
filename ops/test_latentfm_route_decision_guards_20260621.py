#!/usr/bin/env python3
"""Self-check strict routed-expert decision guards."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT = Path("/data/cyx/1030/scLatent/ops/summarize_latentfm_route_decision_20260621.py")
PYTHON = Path("/data/cyx/software/miniconda3/envs/scdfm/bin/python")


def row(group: str, metric: str, delta: float, p_improve: float, p_harm: float, ci_low: float, ci_high: float) -> dict:
    return {
        "comparison": "synthetic_vs_anchor",
        "route": "candidate_gene_multi",
        "group": group,
        "metric": metric,
        "delta": delta,
        "ci95": [ci_low, ci_high],
        "p_improve": p_improve,
        "p_harm": p_harm,
    }


def test_unseen2_mmd_hard_harm_fails(tmp_path: Path) -> None:
    payload = {
        "rows": [
            row("test", "pearson_pert", 0.01, 1.0, 0.0, 0.001, 0.02),
            row("test", "test_mmd_clamped", -0.001, 1.0, 0.0, -0.002, -0.0001),
            row("family_gene", "pearson_pert", 0.01, 1.0, 0.0, 0.001, 0.02),
            row("family_gene", "test_mmd_clamped", -0.001, 1.0, 0.0, -0.002, -0.0001),
            row("test_multi_unseen2", "pearson_pert", 0.02, 1.0, 0.0, 0.001, 0.03),
            row("test_multi_unseen2", "test_mmd_clamped", 0.003, 0.0, 1.0, 0.001, 0.005),
            row("family_drug", "pearson_pert", 0.0, 0.0, 0.0, 0.0, 0.0),
            row("structure_single", "pearson_pert", 0.0, 0.0, 0.0, 0.0, 0.0),
        ]
    }
    inp = tmp_path / "route_bootstrap.json"
    out_json = tmp_path / "decision.json"
    out_md = tmp_path / "decision.md"
    inp.write_text(json.dumps(payload), encoding="utf-8")
    subprocess.run(
        [
            str(PYTHON),
            str(SCRIPT),
            "--route-bootstrap-json",
            str(inp),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        check=True,
    )
    result = json.loads(out_json.read_text(encoding="utf-8"))
    assert result["overall_status"] == "no_route_candidate"
    decision = result["decisions"][0]
    assert decision["status"] == "fail"
    assert "unseen2_mmd_hard_harm" in decision["reasons"]
