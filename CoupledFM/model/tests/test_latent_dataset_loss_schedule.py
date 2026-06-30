"""Tests for LatentFM per-dataset loss weighting schedule."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from model.latent.config import Config
from model.latent.train import dataset_loss_weights_active


def test_dataset_loss_weights_are_independent_of_mmd_warmup() -> None:
    cfg = Config(
        ds_loss_alpha=0.5,
        ds_loss_warmup_start=0,
        gamma_warmup_start=50_000,
    )

    assert dataset_loss_weights_active(0, cfg)
    assert dataset_loss_weights_active(4_000, cfg)


def test_dataset_loss_weights_have_their_own_warmup() -> None:
    cfg = Config(ds_loss_alpha=0.5, ds_loss_warmup_start=100)

    assert not dataset_loss_weights_active(99, cfg)
    assert dataset_loss_weights_active(100, cfg)


def test_dataset_loss_weights_remain_off_when_alpha_is_zero() -> None:
    cfg = Config(ds_loss_alpha=0.0, ds_loss_warmup_start=0)

    assert not dataset_loss_weights_active(1_000, cfg)
