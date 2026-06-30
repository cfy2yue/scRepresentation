"""Shared metric helpers."""

import numpy as np


def pearson_np(x: np.ndarray, y: np.ndarray) -> float:
    x = x.reshape(-1).astype(np.float64)
    y = y.reshape(-1).astype(np.float64)
    if x.size == 0:
        return float("nan")
    xm, ym = x.mean(), y.mean()
    num = ((x - xm) * (y - ym)).sum()
    den = np.sqrt(((x - xm) ** 2).sum() * ((y - ym) ** 2).sum())
    return float(num / den) if den > 1e-12 else float("nan")
