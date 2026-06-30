#!/usr/bin/env python3
"""Smoke: run_metrics_one.py CLI on a tiny synthetic embedding export."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _write_export(raw_dir: Path) -> None:
    rng = np.random.default_rng(0)
    n = 64
    d = 12
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.save(raw_dir / "latent.npy", rng.normal(size=(n, d)).astype(np.float32))
    obs = pd.DataFrame(
        {
            "pert": ["ctrl"] * 24 + ["p1"] * 20 + ["p2"] * 20,
            "is_control": [True] * 24 + [False] * 40,
            "batch": ["A", "B"] * (n // 2),
            "cell_line": ["K562"] * 32 + ["Jurkat"] * 32,
            "cell_type": ["T", "B", "Mono", "NK"] * (n // 4),
        }
    )
    obs.to_csv(raw_dir / "obs.csv.gz", index=False)
    with open(raw_dir / "meta.json", "w") as f:
        json.dump(
            {
                "model": "synthetic",
                "dataset_id": "tiny",
                "obs_artifact": "obs.csv.gz",
            },
            f,
            indent=2,
        )


def _run_cli(bench_root: Path, raw_dir: Path, out_dir: Path, latent_space: str) -> None:
    env = os.environ.copy()
    fm_root = bench_root.parent / "fm"
    env["PYTHONPATH"] = f"{bench_root}:{fm_root}" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    cmd = [
        sys.executable,
        str(bench_root / "cli" / "run_metrics_one.py"),
        "--emb-dir",
        str(raw_dir),
        "--out-dir",
        str(out_dir),
        "--latent-space",
        latent_space,
        "--skip",
        "atlas",
    ]
    subprocess.run(cmd, cwd=bench_root.parent, env=env, check=True)
    summary = out_dir / "summary.json"
    if not summary.is_file():
        raise AssertionError(f"missing {summary}")
    with open(summary) as f:
        payload = json.load(f)
    assert payload["latent_space"] == latent_space
    assert payload["n_cells"] == 64
    assert payload["geometry"]["n_cells"] == 64
    assert (out_dir / "geometry.json").is_file()
    assert (out_dir / "perturb.json").is_file()


def main() -> int:
    bench_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="scfm_metrics_cli_") as td:
        root = Path(td)
        raw_dir = root / "output" / "embeddings" / "synthetic" / "tiny" / "raw"
        _write_export(raw_dir)
        _run_cli(bench_root, raw_dir, root / "metrics_raw", "raw")
        _run_cli(bench_root, raw_dir, root / "metrics_pca128", "pca128")
    print("metrics CLI smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
