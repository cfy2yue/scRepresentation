#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
SUMMARY = ROOT / "runs/latentfm_stack_composite_selection_20260618/summarize_stack_composite_selection.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stack_summary", SUMMARY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load summary script: {SUMMARY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def split_payload(pp: float, mmd: float, pc: float, u1: float, u2: float) -> dict:
    return {
        "checkpoint_step": 123,
        "groups": {
            "test": {"test_mmd": mmd, "pearson_ctrl": pc, "pearson_pert": pp, "n_conds": 787},
            "test_multi_seen": {"pearson_pert": 0.1},
            "test_multi_unseen1": {"pearson_pert": u1},
            "test_multi_unseen2": {"pearson_pert": u2},
        },
    }


def family_payload(gene: float, drug: float) -> dict:
    return {
        "groups": {
            "family_gene": {"pearson_pert": gene},
            "family_drug": {"pearson_pert": drug},
            "structure_multi": {"pearson_pert": 0.01},
        }
    }


def iid_payload(pp: float, mmd: float, pc: float) -> dict:
    return {"test_mmd": mmd, "pearson_ctrl": pc, "pearson_pert": pp, "direct_pearson": 0.9}


def main() -> int:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        module.BASE = tmp_path / "out"
        module.REPORT = tmp_path / "report.md"
        module.CSV_OUT = tmp_path / "summary.csv"
        module.JSON_OUT = tmp_path / "status.json"
        module.RUNS = [
            ("a", "run_a", "candidate a"),
            ("b", "run_b", "candidate b"),
        ]

        # Missing artifacts should produce pending status and a nonzero summary rc.
        rc = module.main()
        if rc != 2:
            raise AssertionError(f"missing-artifact summary rc should be 2, got {rc}")
        pending = json.loads(module.JSON_OUT.read_text(encoding="utf-8"))
        if pending["status"] != "pending" or not pending["missing"]:
            raise AssertionError("pending summary did not record missing artifacts")

        # Complete fake outputs should produce complete status and rank b higher.
        for tag, pp, mmd, pc, u1, u2, gene, drug in [
            ("run_a", 0.02, 0.04, 0.2, -0.03, -0.18, 0.03, -0.01),
            ("run_b", 0.04, 0.03, 0.3, 0.01, -0.08, 0.05, 0.00),
        ]:
            run_dir = module.BASE / tag
            write_json(run_dir / "iid_eval_results.json", iid_payload(pp, mmd, pc))
            write_json(run_dir / "posthoc_eval/split_group_eval_best_ode20_mse2048_mmd2048.json", split_payload(pp, mmd, pc, u1, u2))
            write_json(run_dir / "posthoc_eval/condition_family_eval_best_ode20_mse2048_mmd2048.json", family_payload(gene, drug))
        rc = module.main()
        if rc != 0:
            raise AssertionError(f"complete summary rc should be 0, got {rc}")
        complete = json.loads(module.JSON_OUT.read_text(encoding="utf-8"))
        if complete["status"] != "complete":
            raise AssertionError("complete summary did not report complete status")
        rows = {row["short"]: row for row in complete["rows"]}
        if rows["b"]["selection_score"] <= rows["a"]["selection_score"]:
            raise AssertionError("expected candidate b to rank above candidate a")
        if "Best completed Stack branch" not in module.REPORT.read_text(encoding="utf-8"):
            raise AssertionError("complete report missing interpretation section")

    print("stack summary validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
