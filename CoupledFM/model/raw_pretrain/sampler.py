"""Pair enumeration + fair round-robin across tissues."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, List, Sequence, Tuple


def linear_pair_to_ij(linear_l: int, n_cells: int) -> Tuple[int, int]:
    """Map linear index L in [0, N*(N-1)) to (i,j), i!=j.

    Order: i-major; within each i, j excludes i via skip-index mapping.
    """
    if n_cells < 2:
        raise ValueError("n_cells must be >= 2")
    denom = n_cells - 1
    i = linear_l // denom
    pos = linear_l % denom
    j = pos if pos < i else pos + 1
    return int(i), int(j)


def total_pairs(n_cells: int) -> int:
    return max(0, int(n_cells) * max(0, int(n_cells) - 1))


@dataclass(frozen=True)
class TissueShardMeta:
    tissue_idx: int
    n_cells: int
    pair_begin: int  # global linear pair offset


def build_shard_metas(pair_counts: Sequence[int]) -> List[TissueShardMeta]:
    """Global pair indexing: concatenate tissues in order."""
    out: List[TissueShardMeta] = []
    off = 0
    for ti, nc in enumerate(pair_counts):
        npairs = total_pairs(nc)
        out.append(TissueShardMeta(ti, nc, off))
        off += npairs
    return out


def global_pair_to_tissue_local(
    global_pair_idx: int, metas: Sequence[TissueShardMeta],
) -> Tuple[int, int]:
    """Return (tissue_idx, local_linear_L)."""
    if global_pair_idx < 0:
        raise ValueError("global_pair_idx must be non-negative")
    for k in range(len(metas)):
        m = metas[k]
        npairs = total_pairs(m.n_cells)
        end = m.pair_begin + npairs
        if global_pair_idx < end:
            return m.tissue_idx, global_pair_idx - m.pair_begin
        if k == len(metas) - 1:
            break
    raise IndexError(f"global_pair_idx {global_pair_idx} out of range")


class RoundRobinPairGenerator:
    """Yield (tissue_idx, local_L) cycling tissues until all exhausted."""

    def __init__(self, metas: Sequence[TissueShardMeta]):
        self._metas = list(metas)
        self._cursor = [0 for _ in self._metas]
        self._alive = [total_pairs(m.n_cells) > 0 for m in self._metas]

    def __iter__(self) -> Generator[Tuple[int, int], None, None]:
        ptr = 0
        while any(self._alive):
            n_alive = sum(self._alive)
            if n_alive == 0:
                break
            advanced = False
            steps = 0
            while steps < len(self._alive):
                ti = ptr % len(self._alive)
                ptr += 1
                steps += 1
                if not self._alive[ti]:
                    continue
                m = self._metas[ti]
                max_l = total_pairs(m.n_cells)
                if self._cursor[ti] >= max_l:
                    self._alive[ti] = False
                    continue
                L = self._cursor[ti]
                self._cursor[ti] += 1
                yield m.tissue_idx, L
                advanced = True
                break
            if not advanced:
                break


def ddp_yield_indices(
    seq: Iterable[Tuple[int, int]],
    *,
    rank: int,
    world_size: int,
) -> Generator[Tuple[int, int], None, None]:
    """Stride-select elements for distributed training."""
    ws = max(1, int(world_size))
    r = int(rank) % ws
    for m, item in enumerate(seq):
        if m % ws == r:
            yield item


__all__ = [
    "linear_pair_to_ij",
    "total_pairs",
    "TissueShardMeta",
    "build_shard_metas",
    "global_pair_to_tissue_local",
    "RoundRobinPairGenerator",
    "ddp_yield_indices",
]
