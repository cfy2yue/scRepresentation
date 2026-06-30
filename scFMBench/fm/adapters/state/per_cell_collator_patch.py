"""
Runtime patch for latent_bench: ``VCIDatasetSentenceCollator`` applies
**per-cell** protected-gene coverage (from ``pert_var_matrix`` or
``sampling_pert_per_cell``) instead of computing a **batch union** of condition
genes (see upstream loader.py ~499–519). This keeps coverage constraints from
leaking cross-cell while still not introducing a separate metadata condition
stream—only ``adata.X`` drives expression ordering.

Also exposes :func:`install_input_mode` to deterministically force the
``log1p`` vs. ``raw`` branch of
``VCIDatasetSentenceCollator.sample_cell_sentences`` — bypassing the
magnitude-based heuristic ``is_raw_integer_counts``
(loader.py ~lines 579-601, with RAW_COUNT_HEURISTIC_THRESHOLD=35 and
EXPONENTIATED_UMIS_LIMIT=5_000_000).

Trade-off: enforcing the branch is safer when we *know* the ingest
convention (latent_bench uses ``log1p(normalize_total)``) because the
heuristic can in principle misfire on unusual inputs (e.g. heavily
downsampled raw counts with max<=35 and row-sum<=5e6 would be silently
treated as log1p). The cost is that callers must pass ``input_is_log1p``
correctly; if they lie, the pipeline will silently apply the wrong
transform.

Does not modify files under third_party; applied once when importing state adapter.
"""

from __future__ import annotations

from typing import Optional

import torch


def install() -> None:
    from state.emb.data.loader import VCIDatasetSentenceCollator
    from state.emb import utils

    if getattr(VCIDatasetSentenceCollator, "_latent_bench_per_cell_installed", False):
        return

    _orig_call = VCIDatasetSentenceCollator.__call__

    def __call__(self, batch):  # noqa: N802 — match upstream name
        num_aug = getattr(self.cfg.model, "num_downsample", 1)
        if num_aug > 1 and self.training:
            batch = [item for item in batch for _ in range(num_aug)]

        batch_size = len(batch)

        batch_sentences = torch.zeros((batch_size, self.pad_length), dtype=torch.int32)
        batch_sentences_counts = torch.zeros((batch_size, self.pad_length))
        masks = torch.zeros((batch_size, self.pad_length), dtype=torch.bool)

        idxs = torch.zeros(batch_size, dtype=torch.int32)
        if self.cfg.loss.name == "tabular":
            task_num = self.P + self.N + self.S
        else:
            task_num = self.P + self.N
        Xs = torch.zeros((batch_size, (task_num)), dtype=torch.int32)
        Ys = torch.zeros((batch_size, (task_num)))

        largest_cnt = max([x[0].shape[1] for x in batch])
        batch_weights = torch.zeros((batch_size, largest_cnt))

        total_counts_all = None
        if self.cfg.model.rda:
            total_counts_all = torch.zeros(batch_size)

        if self.cfg.loss.name == "tabular":
            if "global_size" not in self.__dict__:
                self.global_size = utils.get_embedding_cfg(self.cfg).num
            shared_genes = torch.randint(
                low=0, high=self.global_size, size=(self.S,), device=masks.device, dtype=torch.long
            )
        else:
            shared_genes = None

        dataset_nums = torch.zeros(batch_size, dtype=torch.int32)

        # Per-cell condition indices (replaces batch union)
        max_len = 0
        tasks = []
        for i, (counts, idx, dataset, dataset_num) in enumerate(batch):
            valid_mask = self.valid_gene_mask.get(dataset) if self.valid_gene_mask else None
            downsample_fraction = 1.0 if (num_aug > 1 and i % num_aug == 0 and self.training) else None
            # Column indices in dataset var space — used only inside sentence sampling
            # as ``condition_indices`` (upstream name) i.e. protected coverage, not a batch label.
            cell_cond: Optional[set] = None
            if self.pert_var_matrix is not None and int(idx) < self.pert_var_matrix.shape[0]:
                row = self.pert_var_matrix[int(idx)]
                cell_cond = {int(x) for x in row if int(x) >= 0}
            elif self.sampling_pert_per_cell is not None and int(idx) < len(self.sampling_pert_per_cell):
                cell_cond = set(self.sampling_pert_per_cell[int(idx)])
            tasks.append(
                (i, counts, idx, dataset, dataset_num, valid_mask, downsample_fraction, shared_genes, cell_cond)
            )

        if self.n_collate_workers <= 1:
            for tup in tasks:
                i, bs, xx, yy, batch_weight, mask, cell_total_counts, cell_sentence_counts = self._collate_one_cell(
                    *tup
                )
                batch_sentences[i, :] = bs
                masks[i, :] = mask
                batch_weights[i, : len(batch_weight)] = batch_weight
                max_len = max(max_len, self.cfg.dataset.pad_length)
                idxs[i] = tup[2]
                Xs[i] = xx
                Ys[i] = yy.squeeze()
                dataset_nums[i] = tup[4]
                if total_counts_all is not None and cell_total_counts is not None:
                    total_counts_all[i] = cell_total_counts[0]
                if cell_sentence_counts is not None:
                    batch_sentences_counts[i, :] = cell_sentence_counts
        else:
            results = list(self._executor.map(lambda t: self._collate_one_cell(*t), tasks))
            for (i, bs, xx, yy, batch_weight, mask, cell_total_counts, cell_sentence_counts) in results:
                batch_sentences[i, :] = bs
                masks[i, :] = mask
                batch_weights[i, : len(batch_weight)] = batch_weight
                max_len = max(max_len, self.cfg.dataset.pad_length)
                idxs[i] = batch[i][1]
                Xs[i] = xx
                Ys[i] = yy.squeeze()
                dataset_nums[i] = batch[i][3]
                if total_counts_all is not None and cell_total_counts is not None:
                    total_counts_all[i] = cell_total_counts[0]
                if cell_sentence_counts is not None:
                    batch_sentences_counts[i, :] = cell_sentence_counts

        if self.precision is not None:
            Ys = Ys.to(dtype=self.precision)
            batch_weights = batch_weights.to(dtype=self.precision)
            if total_counts_all is not None:
                total_counts_all = total_counts_all.to(dtype=self.precision)
            if batch_sentences_counts is not None:
                batch_sentences_counts = batch_sentences_counts.to(dtype=self.precision)

        return (
            batch_sentences[:, :max_len],
            Xs,
            Ys,
            idxs,
            batch_weights,
            masks,
            total_counts_all if self.cfg.model.rda else None,
            batch_sentences_counts if self.cfg.model.counts else None,
            dataset_nums if self.use_dataset_info else None,
        )

    VCIDatasetSentenceCollator.__call__ = __call__
    VCIDatasetSentenceCollator._latent_bench_per_cell_installed = True
    VCIDatasetSentenceCollator._latent_bench_orig_call = _orig_call


def install_input_mode(is_log1p: Optional[bool]) -> None:
    """Deterministically force the ``sample_cell_sentences`` branch.

    Parameters
    ----------
    is_log1p
        - ``True``:  force the log1p branch (skip ``torch.log1p`` call;
          use ``expm1`` to recover per-gene expression weights).
        - ``False``: force the raw-integer-counts branch (apply
          ``torch.log1p`` internally).
        - ``None``:  restore the original heuristic-based behavior
          (``is_raw_integer_counts``).

    Safe to call repeatedly; the last value wins. Uses a class-level
    monkey-patch on ``VCIDatasetSentenceCollator.is_raw_integer_counts``
    (the only caller of the heuristic is
    ``VCIDatasetSentenceCollator.sample_cell_sentences``, loader.py ~line 616).
    """
    from state.emb.data.loader import VCIDatasetSentenceCollator

    if not hasattr(VCIDatasetSentenceCollator, "_latent_bench_orig_is_raw"):
        VCIDatasetSentenceCollator._latent_bench_orig_is_raw = (
            VCIDatasetSentenceCollator.is_raw_integer_counts
        )

    if is_log1p is None:
        VCIDatasetSentenceCollator.is_raw_integer_counts = (
            VCIDatasetSentenceCollator._latent_bench_orig_is_raw
        )
        VCIDatasetSentenceCollator._latent_bench_input_is_log1p = None
        return

    forced_is_raw = not bool(is_log1p)

    def is_raw_integer_counts(self, counts: torch.Tensor) -> bool:  # noqa: D401
        return forced_is_raw

    VCIDatasetSentenceCollator.is_raw_integer_counts = is_raw_integer_counts
    VCIDatasetSentenceCollator._latent_bench_input_is_log1p = bool(is_log1p)
