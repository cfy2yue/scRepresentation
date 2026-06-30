"""Tests for stable LatentFM eval condition selection."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from model.latent.config import Config
from model.latent.dataset import CrossDatasetFMDataset
from model.latent.train import (
    _select_eval_condition_pairs,
    checkpoint_ema_is_active,
    evaluate,
    load_model_weights_only,
)
from model.latent.eval_split_groups import _resolve_means_file


def _stub_dataset(ds_conds: dict[str, list[str]]) -> CrossDatasetFMDataset:
    ds = CrossDatasetFMDataset.__new__(CrossDatasetFMDataset)
    ds.ds_names = sorted(ds_conds)
    ds.ds_conds = {k: list(v) for k, v in ds_conds.items()}
    ds.handles = {}
    return ds


def test_eval_condition_selection_is_independent_of_input_order() -> None:
    cfg = Config(
        seed=42,
        eval_max_conditions_per_dataset=3,
        eval_max_conditions=5,
    )
    a = _stub_dataset(
        {
            "ds_b": ["cond_9", "cond_1", "cond_4", "cond_7"],
            "ds_a": ["cond_3", "cond_2", "cond_8", "cond_5"],
        }
    )
    b = _stub_dataset(
        {
            "ds_a": ["cond_5", "cond_8", "cond_2", "cond_3"],
            "ds_b": ["cond_7", "cond_4", "cond_1", "cond_9"],
        }
    )

    pairs_a, n_a = _select_eval_condition_pairs(a, cfg)
    pairs_b, n_b = _select_eval_condition_pairs(b, cfg)

    assert n_a == n_b == 8
    assert pairs_a == pairs_b
    assert len(pairs_a) == 5
    assert pairs_a == sorted(pairs_a)


def test_eval_condition_selection_keeps_all_when_caps_off() -> None:
    cfg = Config(seed=7, eval_max_conditions_per_dataset=0, eval_max_conditions=0)
    ds = _stub_dataset({"ds": ["b", "a", "a", "c"]})

    pairs, n_available = _select_eval_condition_pairs(ds, cfg)

    assert n_available == 3
    assert pairs == [("ds", "a"), ("ds", "b"), ("ds", "c")]


def test_evaluate_returns_condition_metric_schema_when_no_conditions() -> None:
    cfg = Config(seed=7, emb_dim=4)
    ds = _stub_dataset({"ds": []})
    model = torch.nn.Linear(4, 4)

    metrics = evaluate(
        model,
        ds,
        path=None,
        cfg=cfg,
        device=torch.device("cpu"),
    )

    assert metrics["n_conds"] == 0
    assert metrics["n_available_conditions"] == 0
    assert "test_mmd_biased" in metrics
    assert "test_mmd_clamped" in metrics
    assert "per_ds_mmd_biased" in metrics
    assert "per_ds_mmd_clamped" in metrics
    assert metrics["selected_conditions"] == []
    assert metrics["condition_metrics"] == []
    assert metrics["eval_caps"]["condition_selection"] == "stable_hash_dataset_condition"
    assert metrics["eval_caps"]["cell_selection"] == "stable_hash_dataset_condition_metric"
    assert metrics["eval_caps"]["aggregation"] == "condition_mean_then_dataset_equal_mean"


def test_checkpoint_ema_is_inactive_before_update_after() -> None:
    cfg = Config(use_ema=True, ema_update_after=1000)

    inactive = {
        "step": 870,
        "ema": {"__meta__": torch.tensor([0.999, 1000, 1, 0], dtype=torch.float64)},
    }
    active_by_updates = {
        "step": 870,
        "ema": {"__meta__": torch.tensor([0.999, 1000, 1, 3], dtype=torch.float64)},
    }
    active_by_step = {
        "step": 1200,
        "ema": {},
    }

    assert checkpoint_ema_is_active(inactive, cfg) is False
    assert checkpoint_ema_is_active(active_by_updates, cfg) is True
    assert checkpoint_ema_is_active(active_by_step, cfg) is True


def test_load_model_weights_only_can_prefer_active_ema(tmp_path: Path) -> None:
    raw = torch.nn.Linear(3, 2)
    with torch.no_grad():
        raw.weight.fill_(1.0)
        raw.bias.fill_(0.5)
    ckpt = {
        "step": 1200,
        "model": raw.state_dict(),
        "ema": {
            "shadow.weight": torch.full_like(raw.weight, 2.0),
            "shadow.bias": torch.full_like(raw.bias, -1.0),
            "__meta__": torch.tensor([0.999, 1000, 1, 5], dtype=torch.float64),
        },
    }
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)

    model_raw = torch.nn.Linear(3, 2)
    load_model_weights_only(path, model_raw, torch.device("cpu"), prefer_ema=False)
    assert torch.allclose(model_raw.weight, torch.full_like(model_raw.weight, 1.0))
    assert torch.allclose(model_raw.bias, torch.full_like(model_raw.bias, 0.5))

    model_ema = torch.nn.Linear(3, 2)
    load_model_weights_only(path, model_ema, torch.device("cpu"), prefer_ema=True)
    assert torch.allclose(model_ema.weight, torch.full_like(model_ema.weight, 2.0))
    assert torch.allclose(model_ema.bias, torch.full_like(model_ema.bias, -1.0))


def test_eval_split_groups_pert_means_override_path_resolution(tmp_path: Path) -> None:
    data_dir = tmp_path / "bundle"
    data_dir.mkdir()

    default_path = _resolve_means_file("", data_dir=data_dir, default_name="pert_means.npz")
    assert default_path == data_dir / "pert_means.npz"

    override = tmp_path / "trainonly_pert_means.npz"
    resolved = _resolve_means_file(str(override), data_dir=data_dir, default_name="pert_means.npz")
    assert resolved == override
