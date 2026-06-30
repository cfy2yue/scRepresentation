"""Focused tests for LatentFM condition-prior teacher helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from model.latent.train import (
    _aggregate_condition_prior_records,
    _make_gene_combo_perturbation_batch,
    _summarize_condition_prior_bank,
    sample_condition_prior_teacher,
)


class _TinyGeneCache:
    """Minimal lookup surface used by PerturbationBatch.from_metadata_list."""

    pad_index = 0
    unk_index = 1

    def __init__(self) -> None:
        self._map = {
            "A": 2,
            "B": 3,
            "C": 4,
        }

    def lookup(self, symbol: str) -> int:
        return self._map.get(str(symbol).strip().upper(), self.unk_index)


def test_make_gene_combo_records_actual_nperts() -> None:
    cache = _TinyGeneCache()

    pb2 = _make_gene_combo_perturbation_batch(
        genes=("B", "A", "A"),
        perturbation_type_raw="CRISPRi",
        batch_size=3,
        cache=cache,
        max_genes=8,
        max_chem_keys=4,
    )
    gid2, mask2, _tid2, npt2, _cid2, _chem2, _cmask2 = pb2
    assert gid2.shape == (3, 8)
    assert mask2.shape == (3, 8)
    assert npt2.tolist() == [2, 2, 2]
    assert mask2.sum(dim=1).tolist() == [2, 2, 2]
    assert gid2[0, :2].tolist() == [2, 3]

    pb3 = _make_gene_combo_perturbation_batch(
        genes=("C", "B", "A"),
        perturbation_type_raw=None,
        batch_size=2,
        cache=cache,
        max_genes=8,
        max_chem_keys=4,
    )
    _gid3, mask3, _tid3, npt3, _cid3, _chem3, _cmask3 = pb3
    assert npt3.tolist() == [3, 3]
    assert mask3.sum(dim=1).tolist() == [3, 3]


def test_global_gene_mean_prior_aggregation_collapses_across_datasets() -> None:
    records = {
        "__global__": [
            ("A", "CRISPRi", torch.tensor([1.0, 3.0])),
            ("A", "CRISPRi", torch.tensor([3.0, 5.0])),
            ("B", "CRISPRi", torch.tensor([10.0, 20.0])),
        ]
    }
    bank = _aggregate_condition_prior_records(records, aggregation="gene_mean")

    assert list(bank) == ["__global__"]
    rows = {gene: (ptype, delta) for gene, ptype, delta in bank["__global__"]}
    assert set(rows) == {"A", "B"}
    assert rows["A"][0] == "CRISPRi"
    assert torch.allclose(rows["A"][1], torch.tensor([2.0, 4.0]))
    assert torch.allclose(rows["B"][1], torch.tensor([10.0, 20.0]))


def test_sample_condition_prior_teacher_uses_global_fallback_bank() -> None:
    cache = _TinyGeneCache()
    bank = {
        "__global__": [
            ("A", "CRISPRi", torch.tensor([1.0, 2.0])),
            ("B", "CRISPRi", torch.tensor([3.0, 5.0])),
            ("C", "CRISPRi", torch.tensor([7.0, 11.0])),
        ]
    }

    target, pb = sample_condition_prior_teacher(
        bank=bank,
        ds_name="Wessels",
        step=0,
        cond="synthetic",
        batch_size=2,
        cache=cache,
        max_genes=8,
        max_chem_keys=4,
        num_genes=2,
    )

    assert target is not None
    assert pb is not None
    gid, mask, _tid, npt, _cid, _chem, _cmask = pb
    assert gid.shape == (2, 8)
    assert mask.sum(dim=1).tolist() == [2, 2]
    assert npt.tolist() == [2, 2]
    possible = {
        (4.0, 7.0),
        (8.0, 13.0),
        (10.0, 16.0),
    }
    assert tuple(float(x) for x in target.tolist()) in possible


def test_prior_bank_summary_preserves_source_dataset_provenance() -> None:
    raw = {
        "DatasetA": [("A", "CRISPRi", torch.tensor([1.0, 3.0]))],
        "DatasetB": [
            ("A", "CRISPRi", torch.tensor([3.0, 5.0])),
            ("B", "CRISPRi", torch.tensor([10.0, 20.0])),
        ],
    }
    final = _aggregate_condition_prior_records(
        {"__global__": [record for records in raw.values() for record in records]},
        aggregation="gene_mean",
    )

    summary = _summarize_condition_prior_bank(
        raw_bank=raw,
        final_bank=final,
        scope="global",
        aggregation="gene_mean",
        split_file="/tmp/split.json",
        max_cells=512,
        min_norm=1e-6,
        skipped=7,
        raw_records=3,
    )

    assert summary["scope"] == "global"
    assert summary["raw_records_by_dataset"] == {"DatasetA": 1, "DatasetB": 2}
    assert summary["final_records_by_bank"] == {"__global__": 2}
    assert summary["genes"]["A"]["raw_condition_count"] == 2
    assert summary["genes"]["A"]["source_datasets"] == {"DatasetA": 1, "DatasetB": 1}
    assert summary["genes"]["A"]["bank"] == "__global__"


def test_prior_bank_summary_records_jiang_lowcount_guard() -> None:
    raw = {
        "Jiang_IFNG": [("A", "CRISPRi", torch.tensor([1.0, 3.0]))],
        "Other": [("A", "CRISPRi", torch.tensor([3.0, 5.0]))],
    }
    final = _aggregate_condition_prior_records(
        raw,
        aggregation="gene_shrink_k2_jiang_lowcount_mask",
    )
    summary = _summarize_condition_prior_bank(
        raw_bank=raw,
        final_bank=final,
        scope="same_dataset",
        aggregation="gene_shrink_k2_jiang_lowcount_mask",
        split_file="",
        max_cells=512,
        min_norm=1e-6,
        skipped=0,
        raw_records=2,
    )
    assert summary["guarded_fallback"] == {
        "mode": "jiang_lowcount_mask",
        "fallback_datasets": ["Jiang_IFNG", "Jiang_TNFA"],
        "gene_train_count_threshold": 1,
        "fallback_target": "dataset_mean",
    }


def test_prior_bank_summary_records_dataset_negative_guard() -> None:
    raw = {
        "NormanWeissman2019_filtered": [("A", "CRISPRi", torch.tensor([1.0, 3.0]))],
        "Other": [("A", "CRISPRi", torch.tensor([3.0, 5.0]))],
    }
    final = _aggregate_condition_prior_records(
        raw,
        aggregation="gene_shrink_k2_dataset_negative_mask",
    )
    summary = _summarize_condition_prior_bank(
        raw_bank=raw,
        final_bank=final,
        scope="same_dataset",
        aggregation="gene_shrink_k2_dataset_negative_mask",
        split_file="",
        max_cells=512,
        min_norm=1e-6,
        skipped=0,
        raw_records=2,
    )
    assert summary["guarded_fallback"]["mode"] == "dataset_negative_mask"
    assert "NormanWeissman2019_filtered" in summary["guarded_fallback"]["fallback_datasets"]
    assert summary["guarded_fallback"]["fallback_target"] == "dataset_mean"


if __name__ == "__main__":
    test_make_gene_combo_records_actual_nperts()
    test_global_gene_mean_prior_aggregation_collapses_across_datasets()
    test_sample_condition_prior_teacher_uses_global_fallback_bank()
    test_prior_bank_summary_preserves_source_dataset_provenance()
    test_prior_bank_summary_records_jiang_lowcount_guard()
    test_prior_bank_summary_records_dataset_negative_guard()
    print("condition-prior teacher helper test passed")
