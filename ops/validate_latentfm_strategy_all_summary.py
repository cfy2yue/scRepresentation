#!/usr/bin/env python3
"""Pure-Python checks for the combined LatentFM strategy decision helper.

The validation uses temporary CSV files and monkeypatched output paths. It does
not read training logs, inspect GPUs, or overwrite real reports.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
SUMMARY = ROOT / "ops/summarize_latentfm_strategy_all_20260619.py"


def load_summary():
    spec = importlib.util.spec_from_file_location("latentfm_strategy_all_summary", SUMMARY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load summary helper: {SUMMARY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def base_row(run: str, backbone: str) -> dict[str, object]:
    ref = load_summary().REFERENCE[backbone]
    return {
        "run": run,
        "backbone": backbone,
        "complete": True,
        "checkpoint_step": 4000,
        "test_mmd": ref["test_mmd"],
        "test_pc": 0.2,
        "test_pp": ref["test_pp"],
        "multi_seen_pp": ref["multi_seen_pp"],
        "multi_unseen1_pp": ref["multi_unseen1_pp"],
        "multi_unseen2_pp": ref["multi_unseen2_pp"],
        "family_gene_pp": ref["family_gene_pp"],
        "family_drug_pp": ref["family_drug_pp"],
        "resid_cosine": ref["resid_cosine"],
        "resid_unseen2_cosine": ref["resid_unseen2_cosine"],
        "desc": "",
        "run_dir": f"/tmp/{run}",
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "run",
        "backbone",
        "complete",
        "checkpoint_step",
        "test_mmd",
        "test_pc",
        "test_pp",
        "multi_seen_pp",
        "multi_unseen1_pp",
        "multi_unseen2_pp",
        "family_gene_pp",
        "family_drug_pp",
        "resid_cosine",
        "resid_unseen2_cosine",
        "desc",
        "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def test_classification(module) -> None:
    repeat = base_row("strict_repeat", "scfoundation")
    repeat.update(
        {
            "test_mmd": 0.026,
            "test_pp": 0.040,
            "multi_seen_pp": 0.230,
            "multi_unseen1_pp": 0.010,
            "multi_unseen2_pp": -0.110,
            "family_gene_pp": 0.043,
        }
    )
    diagnostic = base_row("seen_only_diag", "stack")
    diagnostic.update(
        {
            "test_mmd": 0.041,
            "test_pp": 0.000,
            "multi_seen_pp": 0.180,
            "multi_unseen1_pp": 0.010,
            "multi_unseen2_pp": -0.080,
            "family_gene_pp": 0.000,
        }
    )
    reject = base_row("flat_reject", "stack")
    reject.update(
        {
            "test_mmd": 0.050,
            "test_pp": -0.010,
            "multi_seen_pp": 0.100,
            "multi_unseen1_pp": -0.010,
            "multi_unseen2_pp": -0.100,
            "family_gene_pp": -0.010,
        }
    )
    unknown = base_row("unknown_backbone", "stack")
    unknown["backbone"] = "missing_model"

    normalized = {row["run"]: row for row in module.normalize([repeat, diagnostic, reject, unknown])}
    assert normalized["strict_repeat"]["decision"] == "repeat_candidate", normalized["strict_repeat"]
    assert normalized["seen_only_diag"]["decision"] == "diagnostic_candidate", normalized["seen_only_diag"]
    assert normalized["flat_reject"]["decision"] == "reject_as_is", normalized["flat_reject"]
    assert normalized["unknown_backbone"]["decision"] == "needs_manual_review", normalized["unknown_backbone"]


def test_end_to_end_outputs(module) -> None:
    with tempfile.TemporaryDirectory(prefix="latentfm_strategy_summary_") as tmp_s:
        tmp = Path(tmp_s)
        four = tmp / "four.csv"
        expanded = tmp / "expanded.csv"
        report = tmp / "decision.md"
        csv_out = tmp / "decision.csv"
        json_out = tmp / "decision.json"

        repeat = base_row("strict_repeat", "scfoundation")
        repeat.update(
            {
                "test_mmd": 0.026,
                "test_pp": 0.040,
                "multi_seen_pp": 0.230,
                "multi_unseen1_pp": 0.010,
                "multi_unseen2_pp": -0.110,
                "family_gene_pp": 0.043,
            }
        )
        reject = base_row("flat_reject", "stack")
        reject.update({"test_mmd": 0.050, "test_pp": -0.010, "multi_seen_pp": 0.100})
        write_rows(four, [repeat])
        write_rows(expanded, [reject])

        module.INPUTS = [("four_run", four), ("expanded", expanded)]
        module.REPORT = report
        module.CSV_OUT = csv_out
        module.JSON_OUT = json_out

        code = module.main()
        assert code == 0
        text = report.read_text(encoding="utf-8")
        assert "Status: `complete`" in text, text
        assert "Repeat/deepen `strict_repeat` first" in text, text
        assert "Run at least one repeat seed" in text, text
        rows = list(csv.DictReader(csv_out.open(newline="", encoding="utf-8")))
        assert {row["decision"] for row in rows} == {"repeat_candidate", "reject_as_is"}, rows
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        assert payload["missing_inputs"] == [], payload


def test_pending_output(module) -> None:
    with tempfile.TemporaryDirectory(prefix="latentfm_strategy_summary_pending_") as tmp_s:
        tmp = Path(tmp_s)
        module.INPUTS = [("four_run", tmp / "missing_four.csv"), ("expanded", tmp / "missing_expanded.csv")]
        module.REPORT = tmp / "decision.md"
        module.CSV_OUT = tmp / "decision.csv"
        module.JSON_OUT = tmp / "decision.json"
        code = module.main()
        assert code == 0
        text = module.REPORT.read_text(encoding="utf-8")
        assert "Status: `pending`" in text, text
        assert "Do not launch more LatentFM strategy jobs" in text, text
        payload = json.loads(module.JSON_OUT.read_text(encoding="utf-8"))
        assert len(payload["missing_inputs"]) == 2, payload


def test_empty_csv_treated_as_missing(module) -> None:
    with tempfile.TemporaryDirectory(prefix="latentfm_strategy_summary_empty_") as tmp_s:
        tmp = Path(tmp_s)
        empty = tmp / "empty.csv"
        empty.write_text("", encoding="utf-8")
        module.INPUTS = [("four_run", empty), ("expanded", tmp / "missing_expanded.csv")]
        module.REPORT = tmp / "decision.md"
        module.CSV_OUT = tmp / "decision.csv"
        module.JSON_OUT = tmp / "decision.json"
        code = module.main()
        assert code == 0
        text = module.REPORT.read_text(encoding="utf-8")
        assert f"`four_run`: `{empty}` (missing)" in text, text
        payload = json.loads(module.JSON_OUT.read_text(encoding="utf-8"))
        assert str(empty) in payload["missing_inputs"], payload
        assert len(payload["rows"]) == 0, payload


def test_partial_pending_output(module) -> None:
    with tempfile.TemporaryDirectory(prefix="latentfm_strategy_summary_partial_") as tmp_s:
        tmp = Path(tmp_s)
        four = tmp / "four.csv"
        repeat = base_row("diagnostic_visible", "stack")
        repeat.update(
            {
                "test_mmd": 0.041,
                "test_pp": 0.000,
                "multi_seen_pp": 0.180,
                "multi_unseen1_pp": 0.030,
                "multi_unseen2_pp": -0.080,
                "family_gene_pp": 0.020,
            }
        )
        write_rows(four, [repeat])
        module.INPUTS = [("four_run", four), ("expanded", tmp / "missing_expanded.csv")]
        module.REPORT = tmp / "decision.md"
        module.CSV_OUT = tmp / "decision.csv"
        module.JSON_OUT = tmp / "decision.json"
        code = module.main()
        assert code == 0
        text = module.REPORT.read_text(encoding="utf-8")
        assert "Status: `pending`" in text, text
        assert "Partial evidence:" in text, text
        assert "Present strategy CSV inputs: 1 / 2" in text, text
        assert "Do not launch more LatentFM strategy jobs from a partial table" in text, text
        assert "until at least one upstream strategy CSV exists" not in text, text
        payload = json.loads(module.JSON_OUT.read_text(encoding="utf-8"))
        assert len(payload["missing_inputs"]) == 1, payload


def main() -> int:
    module = load_summary()
    test_classification(module)
    module = load_summary()
    test_end_to_end_outputs(module)
    module = load_summary()
    test_pending_output(module)
    module = load_summary()
    test_empty_csv_treated_as_missing(module)
    module = load_summary()
    test_partial_pending_output(module)
    print("latentfm strategy all-summary validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
