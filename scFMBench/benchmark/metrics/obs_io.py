"""Load obs sidecar tables written alongside embedding exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_obs_table(path: Path) -> pd.DataFrame:
    """Support ``obs.parquet`` and ``obs.csv.gz`` (and plain ``.csv``)."""
    p = Path(path)
    suf = p.suffix.lower()
    name = p.name.lower()
    if suf == ".parquet":
        return pd.read_parquet(p)
    if name.endswith(".csv.gz") or suf == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported obs table: {path} (expected .parquet or .csv.gz)")
