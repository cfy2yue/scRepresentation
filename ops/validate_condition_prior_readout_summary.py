#!/usr/bin/env python3
"""Validate condition-prior one-shot readout summarizer with temp files."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
SCRIPT = ROOT / "ops/summarize_condition_prior_one_shot_readout.py"


def load_module():
    spec = importlib.util.spec_from_file_location("condition_prior_readout", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_module()
    with tempfile.TemporaryDirectory(prefix="condition_prior_readout_") as tmp_s:
        tmp = Path(tmp_s)
        module.ONE_SHOT = tmp / "one_shot.md"
        module.DOSE_JSON = tmp / "dose.json"
        module.DOSE_REPORT = tmp / "dose.md"
        module.NEXT_ACTIONS = tmp / "next.md"
        module.OUT = tmp / "summary.md"
        code = module.main()
        assert code == 2, code
        text = module.OUT.read_text(encoding="utf-8")
        assert "Dose JSON 尚未生成" in text, text

        payload = {
            "status": "complete",
            "best": "scf_prior005_e2_4k",
            "rows": [
                {
                    "run": "scf_prior002_e2_4k",
                    "complete": True,
                    "decision": "diagnostic_candidate",
                    "test_mmd": 0.028,
                    "test_pp": 0.04,
                    "multi_unseen1_pp": 0.02,
                    "multi_unseen2_pp": -0.10,
                    "family_gene_pp": 0.05,
                    "score": -0.01,
                },
                {
                    "run": "scf_prior005_e2_4k",
                    "complete": True,
                    "decision": "repeat_candidate",
                    "test_mmd": 0.027,
                    "test_pp": 0.08,
                    "multi_unseen1_pp": 0.05,
                    "multi_unseen2_pp": -0.05,
                    "family_gene_pp": 0.055,
                    "score": 0.02,
                },
            ],
        }
        module.DOSE_JSON.write_text(json.dumps(payload), encoding="utf-8")
        module.ONE_SHOT.write_text("# one shot\n", encoding="utf-8")
        module.DOSE_REPORT.write_text("# dose\n", encoding="utf-8")
        code = module.main()
        assert code == 0, code
        text = module.OUT.read_text(encoding="utf-8")
        assert "Repeat candidates: 1" in text, text
        assert "至少一个 dose 分支达到" in text, text

    print("condition-prior readout summary validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
