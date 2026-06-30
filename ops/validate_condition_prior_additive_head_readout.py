#!/usr/bin/env python3
"""Validate additive-head readout summarizer with temporary files."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
SCRIPT = ROOT / "ops/summarize_condition_prior_additive_head_readout.py"


def load_module():
    spec = importlib.util.spec_from_file_location("condition_prior_additive_readout", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_module()
    with tempfile.TemporaryDirectory(prefix="condition_prior_additive_readout_") as tmp_s:
        tmp = Path(tmp_s)
        module.ONE_SHOT = tmp / "one_shot.md"
        module.ADD_JSON = tmp / "additive.json"
        module.ADD_REPORT = tmp / "additive.md"
        module.LAUNCH_REPORT = tmp / "launch.md"
        module.OUT = tmp / "summary.md"
        code = module.main()
        assert code == 2, code
        text = module.OUT.read_text(encoding="utf-8")
        assert "Additive-head JSON 尚未生成" in text, text

        pending_payload = {
            "status": "pending",
            "best": "scf_prior010_inject_e2_4k",
            "rows": [
                {"run": "scf_prior010_inject_e2_4k", "complete": True, "decision": "diagnostic_candidate"},
                {
                    "run": "scf_prioradd005_prior010_inject_e2_4k",
                    "complete": False,
                    "decision": "pending",
                    "missing": "iid,split,family,residual",
                },
            ],
        }
        module.ADD_JSON.write_text(json.dumps(pending_payload), encoding="utf-8")
        code = module.main()
        assert code == 2, code
        text = module.OUT.read_text(encoding="utf-8")
        assert "Additive-head 分支仍未完成" in text, text
        assert "iid,split,family,residual" in text, text

        repeat_payload = {
            "status": "complete",
            "best": "scf_prioradd005_prior010_inject_e2_4k",
            "rows": [
                {
                    "run": "scf_prioradd005_prior010_inject_e2_4k",
                    "complete": True,
                    "decision": "repeat_candidate",
                    "test_mmd": 0.026,
                    "test_pp": 0.08,
                    "multi_unseen1_pp": 0.04,
                    "multi_unseen2_pp": -0.04,
                    "family_gene_pp": 0.05,
                    "score": 0.04,
                    "decomp_wessels_unseen2_combo_additive_cosine": 0.7,
                    "decomp_wessels_unseen2_additive_norm_ratio": 1.1,
                    "decomp_wessels_unseen2_interaction_norm_ratio": 0.2,
                }
            ],
        }
        module.ADD_JSON.write_text(json.dumps(repeat_payload), encoding="utf-8")
        code = module.main()
        assert code == 0, code
        text = module.OUT.read_text(encoding="utf-8")
        assert "Repeat candidates: 1" in text, text
        assert "至少一个分支达到" in text, text

    print("condition-prior additive-head readout validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
