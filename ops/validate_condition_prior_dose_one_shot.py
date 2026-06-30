#!/usr/bin/env python3
"""Validate the condition-prior dose one-shot checker without touching jobs."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
CHECKER = ROOT / "ops/check_condition_prior_dose_once.sh"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="dose_one_shot_") as tmp_s:
        tmp = Path(tmp_s)
        ops = tmp / "ops"
        ops.mkdir()
        checker = ops / CHECKER.name
        checker.write_text(CHECKER.read_text(encoding="utf-8"), encoding="utf-8")
        checker.chmod(0o755)

        run_names = [
            "latentfm_condition_prior_teacher_prior002_20260619",
            "latentfm_condition_prior_teacher_probe_20260619",
            "latentfm_condition_prior_teacher_prior010_20260619",
            "latentfm_condition_prior_teacher_posthoc_20260619",
            "latentfm_condition_prior_teacher_sister_posthoc_20260619",
            "latentfm_condition_prior_teacher_dose_summary_20260619",
        ]
        for name in run_names:
            run_root = tmp / "runs" / name
            write(run_root / "RUN_STATUS.md", f"# {name}\n\nStatus: synthetic\n")

        dose_json = tmp / "reports/latentfm_condition_prior_teacher_dose_20260619.json"
        dose_payload = {
            "status": "complete_no_repeat_candidate",
            "best": {"run": "scf_prior002_e2_4k"},
            "rows": [
                {"complete": True, "decision": "diagnostic_candidate"},
                {"complete": True, "decision": "reject_as_is"},
                {"complete": False, "decision": "pending"},
            ],
        }
        write(dose_json, json.dumps(dose_payload))
        write(
            tmp / "reports/LATENTFM_CONDITION_PRIOR_TEACHER_DOSE_20260619.md",
            "# Synthetic Dose Report\n\nStatus: `complete_no_repeat_candidate`\n",
        )
        write(tmp / "reports/latentfm_condition_prior_teacher_dose_20260619.csv", "run,complete\n")

        env = os.environ.copy()
        env["ROOT"] = str(tmp)
        result = subprocess.run(
            [str(checker)],
            cwd=tmp,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        report = tmp / "reports/LATENTFM_CONDITION_PRIOR_DOSE_ONE_SHOT_STATUS_20260619.md"
        text = report.read_text(encoding="utf-8")
        assert "complete=2/3" in text, text
        assert "repeat_candidates=0" in text, text
        assert "best=scf_prior002_e2_4k" in text, text
        assert "does not tail training logs" in text, text
        assert str(report) in result.stdout, result.stdout

    print("condition-prior dose one-shot validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
