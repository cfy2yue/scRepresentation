"""
Shared helpers for **protected-gene coverage** (latent_bench protocol).

Indices refer to ``adata`` column positions; they constrain sampling or
truncation so listed genes are not dropped from the encoded gene/token set.
This is not a separate perturbation **condition** input to model forward passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np


@dataclass
class PerCellPertSpec:
    """Per-cell protected gene column indices in **dataset var** space (``obsm['pert_var_idx']`` rows)."""

    indices_per_cell: List[Set[int]] = field(default_factory=list)

    @classmethod
    def from_pert_var_idx(cls, pert_var_idx: Optional[np.ndarray], n_cells: int) -> "PerCellPertSpec":
        out: List[Set[int]] = []
        if pert_var_idx is None:
            return cls(indices_per_cell=[set() for _ in range(n_cells)])
        pert_var_idx = np.asarray(pert_var_idx, dtype=np.int64)
        for i in range(n_cells):
            row = pert_var_idx[i] if i < pert_var_idx.shape[0] else []
            out.append({int(x) for x in np.asarray(row).ravel() if x >= 0})
        return cls(indices_per_cell=out)


def project_pert_to_vocab_ids(
    pert_var_indices: Sequence[int],
    gene_ids: np.ndarray,
) -> List[int]:
    """
    Map dataset column indices to vocab token ids where ``gene_ids[j]`` is the model vocab id.

    Skips indices outside range or negative ``gene_ids``.
    """
    g = np.asarray(gene_ids)
    out: List[int] = []
    for j in pert_var_indices:
        if j < 0 or j >= len(g):
            continue
        tid = int(g[j])
        if tid >= 0:
            out.append(tid)
    return out


def force_pert_head_tokens(
    pert_token_ids: Sequence[int],
    rest_tokens: Sequence[int],
    rest_values: Sequence[float],
    *,
    cls_token_id: int,
    max_len: int,
    pad_token_id: int,
    pad_value: float,
) -> Tuple[List[int], List[float]]:
    """
    Build ``[cls] + head_tokens + rest ...`` up to ``max_len``.

    **Not used** by current latent_bench encoders (expression-only + tail
    coverage). Kept for experiments; placing tokens immediately after ``<cls>``
    violates the benchmark “no perturbation prefix” contract if interpreted as
    a condition stream.
    """
    seen: Set[int] = {cls_token_id}
    tokens: List[int] = [cls_token_id]
    values: List[float] = [pad_value]  # match scGPT: cls uses pad_value for expr

    for tid in pert_token_ids:
        if tid in seen:
            continue
        if len(tokens) >= max_len:
            break
        seen.add(tid)
        tokens.append(tid)
        values.append(0.0)  # placeholder; caller should overwrite from expression

    for t, v in zip(rest_tokens, rest_values):
        if t in seen:
            continue
        if len(tokens) >= max_len:
            break
        seen.add(t)
        tokens.append(int(t))
        values.append(float(v))

    while len(tokens) < max_len:
        tokens.append(pad_token_id)
        values.append(pad_value)
    return tokens[:max_len], values[:max_len]


def histogram_pert_kept(per_cell_counts: Sequence[int]) -> Dict[str, int]:
    """Bucket counts 0 / 1 / 2 / 3+ for manifest metadata."""
    h = {"0": 0, "1": 0, "2": 0, "3+": 0}
    for c in per_cell_counts:
        if c <= 0:
            h["0"] += 1
        elif c == 1:
            h["1"] += 1
        elif c == 2:
            h["2"] += 1
        else:
            h["3+"] += 1
    return h
