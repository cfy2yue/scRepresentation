#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
PLOTTER = ROOT / "ops/plot_latentfm_condition_prior_teacher_dose_20260619.py"


def load_plotter():
    spec = importlib.util.spec_from_file_location("dose_plotter", PLOTTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load plotter: {PLOTTER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, *, complete_rows: int) -> None:
    fields = [
        "run",
        "weight",
        "complete",
        "decision",
        "checkpoint_step",
        "test_mmd",
        "test_pc",
        "test_pp",
        "multi_seen_pp",
        "multi_unseen1_pp",
        "multi_unseen2_pp",
        "family_gene_pp",
        "family_drug_pp",
    ]
    rows = [
        {
            "run": "scf_prior002_e2_4k",
            "weight": "0.02",
            "complete": "True",
            "decision": "diagnostic_candidate",
            "checkpoint_step": "4000",
            "test_mmd": "0.027",
            "test_pc": "0.22",
            "test_pp": "0.04",
            "multi_seen_pp": "0.20",
            "multi_unseen1_pp": "0.02",
            "multi_unseen2_pp": "-0.10",
            "family_gene_pp": "0.05",
            "family_drug_pp": "-0.01",
        },
        {
            "run": "scf_prior005_e2_4k",
            "weight": "0.05",
            "complete": "True",
            "decision": "repeat_candidate",
            "checkpoint_step": "4000",
            "test_mmd": "0.028",
            "test_pc": "0.25",
            "test_pp": "0.08",
            "multi_seen_pp": "0.22",
            "multi_unseen1_pp": "0.05",
            "multi_unseen2_pp": "-0.05",
            "family_gene_pp": "0.055",
            "family_drug_pp": "0.00",
        },
        {
            "run": "scf_prior010_e2_4k",
            "weight": "0.10",
            "complete": "True",
            "decision": "diagnostic_candidate",
            "checkpoint_step": "4000",
            "test_mmd": "0.031",
            "test_pc": "0.24",
            "test_pp": "0.06",
            "multi_seen_pp": "0.19",
            "multi_unseen1_pp": "0.03",
            "multi_unseen2_pp": "-0.08",
            "family_gene_pp": "0.045",
            "family_drug_pp": "-0.02",
        },
    ][:complete_rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_pending(module) -> None:
    with tempfile.TemporaryDirectory(prefix="dose_plot_pending_") as tmp_s:
        tmp = Path(tmp_s)
        module.CSV_IN = tmp / "dose.csv"
        module.OUT_BASE = tmp / "dose_plot"
        module.META_OUT = tmp / "dose_plot.figure_meta.json"
        write_csv(module.CSV_IN, complete_rows=2)
        code = module.main()
        assert code == 2, code
        meta = json.loads(module.META_OUT.read_text(encoding="utf-8"))
        assert meta["status"] == "pending", meta
        assert not module.OUT_BASE.with_suffix(".pdf").exists()


def test_complete(module) -> None:
    with tempfile.TemporaryDirectory(prefix="dose_plot_complete_") as tmp_s:
        tmp = Path(tmp_s)
        module.CSV_IN = tmp / "dose.csv"
        module.OUT_BASE = tmp / "dose_plot"
        module.META_OUT = tmp / "dose_plot.figure_meta.json"
        write_csv(module.CSV_IN, complete_rows=3)
        code = module.main()
        assert code == 0, code
        meta = json.loads(module.META_OUT.read_text(encoding="utf-8"))
        assert meta["status"] == "complete", meta
        assert len(meta["outputs"]) == 3, meta
        for suffix in ("pdf", "svg", "png"):
            path = module.OUT_BASE.with_suffix(f".{suffix}")
            assert path.is_file() and path.stat().st_size > 0, path


def main() -> int:
    module = load_plotter()
    test_pending(module)
    module = load_plotter()
    test_complete(module)
    print("dose plotter validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
