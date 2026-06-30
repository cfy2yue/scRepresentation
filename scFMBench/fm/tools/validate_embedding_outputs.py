#!/usr/bin/env python3
"""Validate output/embeddings/<model>/<dataset>/raw/{latent.npy, obs.*, meta.json}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model_registry import DEFAULT_EMBEDDING_EXPORT_ROOT

import numpy as np
import pandas as pd


def check_triplet(raw_dir: Path) -> Dict[str, Any]:
    latent_f = raw_dir / "latent.npy"
    meta_f = raw_dir / "meta.json"
    rec: Dict[str, Any] = {"raw_dir": str(raw_dir), "ok": True, "issues": []}
    if not latent_f.is_file():
        rec["ok"] = False
        rec["issues"].append("missing latent.npy")
        return rec
    if not meta_f.is_file():
        rec["ok"] = False
        rec["issues"].append("missing meta.json")
        return rec
    obs_parquet = raw_dir / "obs.parquet"
    obs_gz = raw_dir / "obs.csv.gz"
    obs_f: Path | None = None
    if obs_parquet.is_file():
        obs_f = obs_parquet
    elif obs_gz.is_file():
        obs_f = obs_gz
    else:
        try:
            with open(meta_f) as f:
                meta_hint = json.load(f)
            art = meta_hint.get("obs_artifact")
            if art == "obs.csv.gz" and obs_gz.is_file():
                obs_f = obs_gz
            elif art == "obs.parquet" and obs_parquet.is_file():
                obs_f = obs_parquet
        except Exception:
            pass
    if obs_f is None:
        rec["ok"] = False
        rec["issues"].append("missing obs.parquet or obs.csv.gz")
        return rec
    z = np.load(latent_f)
    if obs_f.suffix == ".gz":
        obs = pd.read_csv(obs_f, index_col=0)
    else:
        obs = pd.read_parquet(obs_f)
    with open(meta_f) as f:
        meta = json.load(f)
    n_z = int(z.shape[0])
    n_o = int(obs.shape[0])
    if n_z != n_o:
        rec["ok"] = False
        rec["issues"].append(f"row mismatch latent {n_z} vs obs {n_o}")
    if np.isnan(z).any() or np.isinf(z).any():
        rec["ok"] = False
        rec["issues"].append("NaN/Inf in latent")
    mo = meta.get("n_obs")
    if mo is not None and int(mo) != n_z:
        rec["ok"] = False
        rec["issues"].append(f"meta n_obs {mo} vs latent {n_z}")
    rec["n_cells"] = n_z
    rec["latent_dim"] = int(z.shape[1]) if z.ndim == 2 else -1
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-root", type=Path, default=DEFAULT_EMBEDDING_EXPORT_ROOT)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    results: List[Dict[str, Any]] = []
    root = args.export_root
    if not root.is_dir():
        print(json.dumps({"error": f"no such dir {root}"}))
        return 1
    for meta_path in root.glob("*/*/raw/meta.json"):
        raw_dir = meta_path.parent
        results.append(check_triplet(raw_dir))
    ok_n = sum(1 for r in results if r.get("ok"))
    summary = {"checked": len(results), "ok": ok_n, "failed": len(results) - ok_n, "details": results}
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"checked": len(results), "ok": ok_n, "failed": len(results) - ok_n}))
    if summary["checked"] == 0:
        return 0
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
