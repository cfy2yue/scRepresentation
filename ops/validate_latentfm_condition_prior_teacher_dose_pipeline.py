#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
PYTHON = Path(sys.executable)
CHECKS = [
    ROOT / "ops/validate_latentfm_condition_prior_teacher_dose_summary.py",
    ROOT / "ops/validate_latentfm_condition_prior_teacher_dose_plotter.py",
    ROOT / "ops/validate_condition_prior_dose_one_shot.py",
    ROOT / "ops/validate_condition_prior_dose_watchers.py",
    ROOT / "ops/validate_condition_prior_readout_summary.py",
]


def main() -> int:
    for script in CHECKS:
        print(f"[validate] {script}")
        subprocess.check_call([str(PYTHON), str(script)], cwd=str(ROOT))
    print("condition-prior dose pipeline validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
