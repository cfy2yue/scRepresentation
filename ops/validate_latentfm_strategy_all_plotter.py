#!/usr/bin/env python3
"""Pure-Python checks for the LatentFM strategy decision plotter.

The validation uses temporary CSV files and monkeypatched paths. It does not
read training logs, inspect GPUs, or overwrite real report figures.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
PLOTTER = ROOT / "ops/plot_latentfm_strategy_all_decision_20260619.py"


FIELDS = [
    "run",
    "backbone",
    "complete",
    "decision",
    "test_mmd",
    "mmd_ratio_to_ref",
    "test_pp",
    "delta_test_pp",
    "delta_multi_seen_pp",
    "delta_multi_unseen1_pp",
    "delta_multi_unseen2_pp",
    "delta_family_gene_pp",
    "delta_family_drug_pp",
    "score",
]


def load_plotter():
    spec = importlib.util.spec_from_file_location("latentfm_strategy_all_plotter", PLOTTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load plotter: {PLOTTER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in FIELDS})


def complete_row(run: str, *, score: float, decision: str, backbone: str) -> dict[str, object]:
    return {
        "run": run,
        "backbone": backbone,
        "complete": "True",
        "decision": decision,
        "test_mmd": 0.025 + abs(score) * 0.002,
        "mmd_ratio_to_ref": 0.95,
        "test_pp": 0.04 + score * 0.01,
        "delta_test_pp": 0.005 + score * 0.001,
        "delta_multi_seen_pp": 0.02 + score * 0.001,
        "delta_multi_unseen1_pp": 0.01 + score * 0.001,
        "delta_multi_unseen2_pp": 0.02 + score * 0.001,
        "delta_family_gene_pp": -0.001,
        "delta_family_drug_pp": 0.002,
        "score": score,
    }


def test_complete_plot() -> None:
    module = load_plotter()
    with tempfile.TemporaryDirectory(prefix="latentfm_strategy_plotter_") as tmp_s:
        tmp = Path(tmp_s)
        module.CSV_IN = tmp / "decision.csv"
        module.OUT_BASE = tmp / "decision_plot"
        rows = [
            complete_row("strict_repeat", score=0.20, decision="repeat_candidate", backbone="scfoundation"),
            complete_row("diagnostic", score=0.10, decision="diagnostic_candidate", backbone="stack"),
            complete_row("reject", score=-0.05, decision="reject_as_is", backbone="stack"),
        ]
        write_csv(module.CSV_IN, rows)
        module.OUT_BASE.with_suffix(".txt").write_text("stale placeholder\n", encoding="utf-8")
        loaded = module.load_rows()
        assert [row["run"] for row in loaded] == ["strict_repeat", "diagnostic", "reject"], loaded
        assert module.main() == 0
        for suffix in (".pdf", ".png", ".svg"):
            out = module.OUT_BASE.with_suffix(suffix)
            assert out.is_file() and out.stat().st_size > 0, out
        assert not module.OUT_BASE.with_suffix(".txt").exists()


def test_placeholder_when_no_complete_rows() -> None:
    module = load_plotter()
    with tempfile.TemporaryDirectory(prefix="latentfm_strategy_plotter_empty_") as tmp_s:
        tmp = Path(tmp_s)
        module.CSV_IN = tmp / "decision.csv"
        module.OUT_BASE = tmp / "decision_plot"
        write_csv(
            module.CSV_IN,
            [
                {"run": "incomplete", "complete": "False", "score": 1.0},
                {"run": "missing_mmd", "complete": "True", "score": 1.0, "test_pp": 0.1},
            ],
        )
        assert module.load_rows() == []
        assert module.main() == 0
        placeholder = module.OUT_BASE.with_suffix(".txt")
        assert placeholder.is_file(), placeholder
        assert "No complete LatentFM strategy rows" in placeholder.read_text(encoding="utf-8")


def main() -> int:
    test_complete_plot()
    test_placeholder_when_no_complete_rows()
    print("latentfm strategy all-plotter validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
