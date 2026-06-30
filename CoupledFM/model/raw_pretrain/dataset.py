"""Pairwise cell–cell dataset for raw FM pretraining."""
from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from model.raw_pretrain.data_source import TissueShardSource

logger = logging.getLogger(__name__)


def pair_from_linear_index(n: int, k: int) -> Tuple[int, int]:
    """Map linear index k in [0, n*(n-1)) to ordered pair (i, j), i != j."""
    if k < 0 or k >= n * (n - 1):
        raise IndexError(k)
    denom = n - 1
    i = k // denom
    pos = k % denom
    j = pos if pos < i else pos + 1
    return int(i), int(j)


def _gumbel_top_k(
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Weighted sampling without replacement using Gumbel–top-k."""
    if k >= len(weights):
        return np.arange(len(weights), dtype=np.int64)
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    eps = 1e-12
    g = -np.log(-np.log(rng.uniform(0.0, 1.0, size=weights.shape) + eps) + eps)
    score = np.log(weights + eps) + g
    idx = np.argpartition(-score, k - 1)[:k]
    return idx.astype(np.int64)


class PairwisePretrainDataset(IterableDataset):
    """Round-robin across tissues; DDP/world-size sharding of global pair indices."""

    def __init__(
        self,
        sources: List[TissueShardSource],
        *,
        rank: int,
        world_size: int,
        max_pert_genes: int = 24,
        cond_tau: float = 1.0,
        cond_alpha: float = 1.0,
        pseudo_delta_min: float = 0.0,
        batch_size: int = 64,
        seed: int = 42,
    ):
        super().__init__()
        if not sources:
            raise ValueError("PairwisePretrainDataset: empty sources")
        self.sources = sources
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.max_pert_genes = int(max_pert_genes)
        self.cond_tau = float(cond_tau)
        self.cond_alpha = float(cond_alpha)
        self.pseudo_delta_min = float(pseudo_delta_min)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        self._pair_counts = [int(s.n_units * (s.n_units - 1)) for s in sources]

    def _global_stride(self) -> int:
        wi = get_worker_info()
        nw = int(wi.num_workers) if wi else 1
        return max(1, self.world_size * nw)

    def _effective_id(self) -> int:
        wi = get_worker_info()
        wid = int(wi.id) if wi else 0
        nw = int(wi.num_workers) if wi else 1
        return self.rank + wid * self.world_size

    def _tissue_pair_generator(
        self,
        tissue_idx: int,
        *,
        start_offset: int = 0,
    ) -> Iterator[int]:
        src = self.sources[tissue_idx]
        n = src.n_units
        p = n * (n - 1)
        stride = self._global_stride()
        eff = self._effective_id()
        if eff >= p:
            return
        local_total = ((p - 1 - eff) // stride) + 1
        # Smallest local linear index k>=0 with k % stride == eff. Tissue
        # batches are homogeneous, so global offsets are intentionally not used.
        r = eff % stride
        start = max(0, min(int(start_offset), p - 1))
        k = r if start <= r else start + ((eff - (start % stride)) % stride)
        if k >= p:
            k = r
        yielded = 0
        while k < p:
            yield k
            yielded += 1
            k += stride
        if start_offset:
            k = eff % stride
            while k < start and yielded < local_total:
                yield k
                yielded += 1
                k += stride

    def _sample_perturbation(
        self,
        delta: np.ndarray,
        gene_ids_row_vocab: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
        """Return sel_idx (K,), signs (+1/-1), mags [0,1], mask (Kmax,) float."""
        nz = np.nonzero(np.abs(delta) > self.pseudo_delta_min)[0].astype(np.int64)
        n_nz = int(nz.size)
        kmax = self.max_pert_genes
        if n_nz == 0:
            raise RuntimeError("delta has no pseudo target genes (caller should skip)")

        k_upper = min(kmax, n_nz)
        k = int(rng.integers(1, k_upper + 1))

        absd = np.abs(delta[nz]).astype(np.float64)
        w = np.log1p(absd / self.cond_tau) ** self.cond_alpha
        sub = _gumbel_top_k(w, k, rng)
        sel = nz[sub]

        signs = np.sign(delta[sel]).astype(np.float32)
        mags = np.abs(delta[sel]).astype(np.float32)
        mags = np.clip(mags, 0.0, 1.0)

        pad = kmax - sel.size
        if pad > 0:
            sel = np.pad(sel, (0, pad), constant_values=-1)
            signs = np.pad(signs, (0, pad), constant_values=0.0)
            mags = np.pad(mags, (0, pad), constant_values=0.0)
        mask = np.zeros((kmax,), dtype=np.float32)
        mask[:k] = 1.0

        pert_ids = sel.astype(np.int64)
        emb_ids = np.full_like(pert_ids, fill_value=-1, dtype=np.int64)
        valid = pert_ids >= 0
        emb_ids[valid] = gene_ids_row_vocab[pert_ids[valid]]

        return (
            emb_ids.astype(np.int64),
            signs,
            mags,
            torch.from_numpy(mask),
        )

    def _one_sample(
        self,
        tissue_idx: int,
        k_lin: int,
        rng: np.random.Generator,
    ) -> Dict[str, torch.Tensor]:
        src = self.sources[tissue_idx]
        gene_ids_np = src.gene_ids_cellnavi()
        n = src.n_units
        i, j = pair_from_linear_index(n, k_lin)
        x_ctrl = src.get_expr(int(i))
        x_gt = src.get_expr(int(j))
        delta = x_gt.astype(np.float32) - x_ctrl.astype(np.float32)
        if not np.any(delta != 0):
            raise RuntimeError("delta all-zero pair")

        pert_ids, signs, mags, pmask = self._sample_perturbation(delta, gene_ids_np, rng)

        gene_ids_t = torch.from_numpy(np.asarray(gene_ids_np, dtype=np.int64))

        return {
            "x_ctrl": torch.from_numpy(x_ctrl.astype(np.float32, copy=False)),
            "x_gt": torch.from_numpy(x_gt.astype(np.float32, copy=False)),
            "pert_gene_ids": torch.from_numpy(pert_ids),
            "pert_signs": torch.from_numpy(signs),
            "pert_mags": torch.from_numpy(mags),
            "pert_mask": pmask,
            "gene_ids": gene_ids_t,
        }

    @staticmethod
    def _collate_one_tissue(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return collate_pretrain_batch(batch)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        wi = get_worker_info()
        wid = int(wi.id) if wi else 0
        seed = self.seed + 100003 * self.rank + 7919 * wid
        rng = np.random.default_rng(seed)

        order = np.arange(len(self.sources), dtype=np.int64)
        rng.shuffle(order)
        gens: Dict[int, Iterator[int]] = {}
        active = set(int(x) for x in order)
        for ti in order:
            p = self._pair_counts[int(ti)]
            start = int(rng.integers(0, max(p, 1))) if p > 0 else 0
            gens[int(ti)] = iter(self._tissue_pair_generator(int(ti), start_offset=start))

        bsz = max(1, int(self.batch_size))
        while active:
            progressed = False
            for ti in list(order):
                ti_i = int(ti)
                if ti_i not in active:
                    continue
                rows: List[Dict[str, torch.Tensor]] = []
                while len(rows) < bsz:
                    try:
                        k_lin = next(gens[ti_i])
                    except StopIteration:
                        active.remove(ti_i)
                        break
                    try:
                        rows.append(self._one_sample(ti_i, k_lin, rng))
                    except RuntimeError:
                        continue
                if rows:
                    progressed = True
                    yield self._collate_one_tissue(rows)
            if not progressed:
                break


def collate_pretrain_batch(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if k == "gene_ids":
            ref = vals[0]
            for vb in vals[1:]:
                if vb.shape != ref.shape or not torch.equal(vb, ref):
                    raise ValueError(
                        "gene_ids mismatch inside a batch (mixed tissues with different "
                        "gene axes). Use strict_same_genes=True with a unified top-gene "
                        "list across shards, or train a single tissue."
                    )
            out[k] = ref
            continue
        if isinstance(vals[0], torch.Tensor) and vals[0].dim() > 0:
            out[k] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals[0]
    return out


__all__ = [
    "PairwisePretrainDataset",
    "collate_pretrain_batch",
    "pair_from_linear_index",
]
