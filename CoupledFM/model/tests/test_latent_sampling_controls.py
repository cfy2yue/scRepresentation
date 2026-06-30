"""Tests for default-off LatentFM condition sampling controls."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from model.latent.dataset import CrossDatasetFMDataset


def _stub_dataset(
    *,
    batch_size: int = 64,
    ds_alpha: float = 0.7,
    min_selected_conditions_per_dataset: int = 0,
    condition_visit_power: float = 1.0,
    condition_visit_cap: int = 0,
) -> CrossDatasetFMDataset:
    ds = CrossDatasetFMDataset.__new__(CrossDatasetFMDataset)
    ds.batch_size = batch_size
    ds.ds_alpha = ds_alpha
    ds.min_selected_conditions_per_dataset = min_selected_conditions_per_dataset
    ds.condition_visit_power = condition_visit_power
    ds.condition_visit_cap = condition_visit_cap
    ds.handles = {}
    return ds


def test_sampling_controls_preserve_legacy_defaults() -> None:
    ds = _stub_dataset()

    assert ds._n_eff(100) == 26
    assert ds._n_eff(4) == 3
    assert ds._condition_visits(63) == 1
    assert ds._condition_visits(64) == 1
    assert ds._condition_visits(65) == 2
    assert ds._condition_visits(512) == 8


def test_sampling_controls_can_floor_and_cap_visits() -> None:
    ds = _stub_dataset(
        min_selected_conditions_per_dataset=32,
        condition_visit_power=0.5,
        condition_visit_cap=4,
    )

    assert ds._n_eff(100) == 32
    assert ds._n_eff(8) == 8
    assert ds._condition_visits(64) == 1
    assert ds._condition_visits(512) == 3
    assert ds._condition_visits(4096) == 4


if __name__ == "__main__":
    test_sampling_controls_preserve_legacy_defaults()
    test_sampling_controls_can_floor_and_cap_visits()
    print("latent sampling controls tests passed")
