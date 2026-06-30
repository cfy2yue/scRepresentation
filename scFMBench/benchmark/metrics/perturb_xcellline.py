"""xCellLine / multi-cell-line chemical perturbation summaries in latent space."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .perturb_geom import centroid_shift_metrics, summarize_ot_deltas


def summarize_xcellline_by_line(
    latent: np.ndarray,
    obs: pd.DataFrame,
    cell_line_col: str,
    pert_col: str,
    *,
    is_control_col: str = "is_control",
    ot_max_perts: int = 20,
    ot_max_n: int = 512,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Per cell line: ``centroid_shift_metrics`` and ``summarize_ot_deltas`` on the subset.

    Expects ``obs[is_control_col]`` and ``obs[pert_col]`` consistent within each line.
    """
    lines: List[str] = sorted(obs[cell_line_col].astype(str).unique())
    per_line: Dict[str, Any] = {}
    for line in lines:
        m = obs[cell_line_col].astype(str) == line
        Zl = latent[m.to_numpy()]
        obl = obs.loc[m].reset_index(drop=True)
        if len(obl) < 8:
            per_line[line] = {"skipped": True, "n_cells": len(obl)}
            continue
        cent = centroid_shift_metrics(Zl, obl, pert_col, is_control_col=is_control_col)
        if is_control_col not in obl.columns or not obl[is_control_col].astype(bool).any():
            ot = {"emd_mean": None, "emd_median": None, "n_computed": 0, "pot_available": False, "note": "no controls"}
        else:
            ot = summarize_ot_deltas(
                Zl, obl, pert_col, is_control_col=is_control_col, max_perts=ot_max_perts, max_n=ot_max_n, seed=seed
            )
        per_line[line] = {"centroid": cent, "ot": ot, "n_cells": len(obl)}

    agg_l2 = []
    agg_emd = []
    for v in per_line.values():
        if isinstance(v, dict) and "centroid" in v and "mean_l2_to_control" in v["centroid"]:
            x = v["centroid"]["mean_l2_to_control"]
            if x == x:
                agg_l2.append(x)
        if isinstance(v, dict) and "ot" in v and v["ot"].get("emd_mean") is not None:
            agg_emd.append(v["ot"]["emd_mean"])
    return {
        "per_cell_line": per_line,
        "xcellline_mean_l2_across_lines": float(np.mean(agg_l2)) if agg_l2 else None,
        "xcellline_mean_emd_across_lines": float(np.mean(agg_emd)) if agg_emd else None,
        "n_lines": float(len(lines)),
    }
