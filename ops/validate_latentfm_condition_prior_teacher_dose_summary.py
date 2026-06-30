#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
SUMMARY = ROOT / "ops/summarize_latentfm_condition_prior_teacher_dose_20260619.py"


def load_summary():
    spec = importlib.util.spec_from_file_location("dose_summary", SUMMARY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load summary: {SUMMARY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_run(base: Path, run: str, *, pp: float, u1: float, u2: float, gene: float, mmd: float) -> None:
    run_dir = base / run
    write_json(
        run_dir / "iid_eval_results.json",
        {"test_mmd": mmd, "pearson_ctrl": 0.2, "pearson_pert": pp},
    )
    split_groups = {
        "test": {"test_mmd": mmd, "pearson_ctrl": 0.25, "pearson_pert": pp},
        "test_multi_seen": {"pearson_pert": 0.22},
        "test_multi_unseen1": {"pearson_pert": u1},
        "test_multi_unseen2": {"pearson_pert": u2},
    }
    write_json(
        run_dir / "posthoc_eval/split_group_eval_best_ode20_mse1024_mmd1024.json",
        {"checkpoint_step": 4000, "groups": split_groups},
    )
    family_groups = {
        "family_gene": {"pearson_pert": gene},
        "family_drug": {"pearson_pert": -0.005},
    }
    write_json(
        run_dir / "posthoc_eval/condition_family_eval_best_ode20_mse1024_mmd1024.json",
        {"groups": family_groups},
    )
    write_json(
        run_dir / "posthoc_eval/condition_residual_full128_best.json",
        {
            "rows": [
                {
                    "pred_target_cosine": 0.05,
                    "groups": "test_multi_seen,family_gene",
                },
                {
                    "pred_target_cosine": -0.05,
                    "groups": "test_multi_unseen1,family_gene",
                },
                {
                    "pred_target_cosine": -0.10,
                    "groups": "test_multi_unseen2,family_gene",
                },
            ]
        },
    )


def test_pending(module) -> None:
    with tempfile.TemporaryDirectory(prefix="dose_summary_pending_") as tmp_s:
        tmp = Path(tmp_s)
        module.BASE = tmp / "runs"
        module.REPORT = tmp / "dose.md"
        module.CSV_OUT = tmp / "dose.csv"
        module.JSON_OUT = tmp / "dose.json"
        code = module.main()
        assert code == 2, code
        payload = json.loads(module.JSON_OUT.read_text(encoding="utf-8"))
        assert payload["status"] == "pending", payload
        assert all(not row["complete"] for row in payload["rows"]), payload


def test_complete(module) -> None:
    with tempfile.TemporaryDirectory(prefix="dose_summary_complete_") as tmp_s:
        tmp = Path(tmp_s)
        module.BASE = tmp / "runs"
        module.REPORT = tmp / "dose.md"
        module.CSV_OUT = tmp / "dose.csv"
        module.JSON_OUT = tmp / "dose.json"
        write_run(module.BASE, "scf_prior002_e2_4k", pp=0.04, u1=0.02, u2=-0.10, gene=0.050, mmd=0.027)
        write_run(module.BASE, "scf_prior005_e2_4k", pp=0.08, u1=0.05, u2=-0.05, gene=0.055, mmd=0.028)
        write_run(module.BASE, "scf_prior010_e2_4k", pp=0.06, u1=0.03, u2=-0.08, gene=0.045, mmd=0.031)
        code = module.main()
        assert code == 0, code
        payload = json.loads(module.JSON_OUT.read_text(encoding="utf-8"))
        assert payload["status"] == "complete", payload
        assert payload["best"] == "scf_prior005_e2_4k", payload
        rows = {row["run"]: row for row in payload["rows"]}
        assert rows["scf_prior005_e2_4k"]["decision"] == "repeat_candidate", rows
        text = module.REPORT.read_text(encoding="utf-8")
        assert "Best completed branch" in text, text
        assert module.CSV_OUT.is_file() and module.CSV_OUT.stat().st_size > 0


def main() -> int:
    module = load_summary()
    test_pending(module)
    module = load_summary()
    test_complete(module)
    print("dose summary validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
