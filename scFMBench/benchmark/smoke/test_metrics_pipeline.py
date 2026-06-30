#!/usr/bin/env python3
"""Smoke: post_process → geometry → perturb summaries → aggregate_report (synthetic)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))

from metrics.geometry import run_geometry_metrics
from metrics.post_process import center_embeddings, centerscale_on_controls, tvn_on_controls
from metrics.perturb_geom import centroid_shift_metrics
from metrics.perturb_xcellline import summarize_xcellline_by_line

from cli import aggregate_report


def main() -> int:
    rng = np.random.default_rng(0)
    n = 400
    d = 16
    Z = rng.standard_normal((n, d)).astype(np.float64)

    meta = pd.DataFrame({
        "pert": np.array(["ctrl"] * 150 + ["p1"] * 125 + ["p2"] * 125),
        "is_control": np.array([True] * 150 + [False] * 250),
        "batch": np.tile(np.array(["A", "B"]), n // 2),
        "cell_line": np.repeat(["K562", "Jurkat"], n // 2),
        "cell_type": np.random.choice(["T", "B"], n),
    })

    Zc = center_embeddings(Z, meta, "pert", "ctrl", batch_col="batch")
    Zcs = centerscale_on_controls(Z, meta, "pert", "ctrl", batch_col="batch")
    Ztvn = tvn_on_controls(Z, meta, "pert", "ctrl", batch_col="batch")
    assert Zc.shape == Z.shape == Zcs.shape == Ztvn.shape

    geom = run_geometry_metrics(Ztvn, meta, label_col="cell_type", batch_col="batch", seed=1)
    assert "G1_participation_ratio" in geom and "LDM_proxy_score" in geom

    cmet = centroid_shift_metrics(Ztvn, meta, "pert", is_control_col="is_control")
    assert cmet["n_perts"] == 2.0

    xc = summarize_xcellline_by_line(Ztvn, meta, "cell_line", "pert", is_control_col="is_control", ot_max_n=64)
    assert xc["n_lines"] == 2.0

    td = Path(tempfile.mkdtemp())
    p1 = td / "run_a.json"
    p2 = td / "subdir" / "run_b.json"
    p2.parent.mkdir(parents=True, exist_ok=True)
    with open(p1, "w") as f:
        json.dump({"model": "m1", "acc": 0.9, "geom": geom["LDM_proxy_score"]}, f)
    with open(p2, "w") as f:
        json.dump({"model": "m2", "acc": 0.85}, f)
    rows = aggregate_report.aggregate([p1, p2])
    assert len(rows) == 2 and rows[0]["model"] == "m1"
    print("metrics pipeline smoke OK")
    print("LDM_proxy", geom["LDM_proxy_score"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
