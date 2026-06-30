"""Tests for LatentFM condition-delta target frames."""
from __future__ import annotations

import unittest

import torch

from model.latent.config import Config
from model.latent.fm_ot import CondOTPath
from model.latent.train import train_step


class _DummyConditionDeltaModel(torch.nn.Module):
    use_pert_condition = True
    condition_delta_head = True

    def forward(self, x_t, t, x_0, **kwargs):
        del t, x_0, kwargs
        return torch.zeros_like(x_t)

    def predict_condition_delta(self, **kwargs):
        gid = kwargs["pert_gene_ids"]
        return torch.zeros(gid.shape[0], 2, dtype=torch.float32, device=gid.device)

    def predict_additive_condition_delta(self, **kwargs):
        gid = kwargs["pert_gene_ids"]
        return torch.zeros(gid.shape[0], 2, dtype=torch.float32, device=gid.device)


def _pert_batch(batch_size: int) -> tuple:
    return (
        torch.zeros(batch_size, 1, dtype=torch.long),
        torch.ones(batch_size, 1, dtype=torch.float32),
        torch.zeros(batch_size, dtype=torch.long),
        torch.ones(batch_size, dtype=torch.long),
        torch.zeros(batch_size, dtype=torch.long),
        None,
        None,
    )


class TestLatentConditionDeltaTarget(unittest.TestCase):
    def _run(self, target: str):
        cfg = Config(
            emb_dim=2,
            use_mmd=False,
            use_amp=False,
            condition_delta_head_loss_weight=1.0,
            condition_delta_head_target=target,
            time_sampling="uniform",
        )
        src = torch.zeros(2, 2)
        gt = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
        return train_step(
            src,
            gt,
            _DummyConditionDeltaModel(),
            CondOTPath(),
            cfg,
            torch.device("cpu"),
            gamma_t=0.0,
            condition_delta_head_weight_t=1.0,
            perturbation_batch=_pert_batch(2),
            pert_mean_ref=torch.tensor([1.0, 0.0]),
        )

    def test_endpoint_delta_target_is_legacy_default_frame(self):
        out = self._run("endpoint_delta")
        self.assertAlmostEqual(float(out["condition_delta_head"]), 2.0, places=6)

    def test_pert_residual_target_matches_pearson_pert_frame(self):
        out = self._run("pert_residual")
        self.assertAlmostEqual(float(out["condition_delta_head"]), 0.5, places=6)

    def test_unknown_condition_delta_target_rejected(self):
        with self.assertRaises(ValueError):
            self._run("unknown")

    def test_condition_prior_additive_delta_loss_trains_head_atoms(self):
        cfg = Config(
            emb_dim=2,
            use_mmd=False,
            use_amp=False,
            condition_prior_additive_delta_loss_weight=1.0,
            time_sampling="uniform",
        )
        src = torch.zeros(2, 2)
        gt = torch.zeros(2, 2)
        out = train_step(
            src,
            gt,
            _DummyConditionDeltaModel(),
            CondOTPath(),
            cfg,
            torch.device("cpu"),
            gamma_t=0.0,
            condition_prior_additive_delta_weight_t=1.0,
            condition_prior_delta_target=torch.tensor([2.0, 0.0]),
            condition_prior_perturbation_batch=_pert_batch(2),
            perturbation_batch=_pert_batch(2),
        )
        self.assertAlmostEqual(float(out["condition_prior_additive_delta"]), 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
