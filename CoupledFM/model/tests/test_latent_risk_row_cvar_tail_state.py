"""Tests for default-off LatentFM risk-row CVaR tail-state controls."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from model.latent.config import Config
from model.latent.fm_ot import CondOTPath
from model.latent.train import (
    RiskRowCvarTailState,
    risk_row_cvar_batch_control,
    risk_row_cvar_loss_schedule,
    train_step,
)


class _ZeroVelocity(torch.nn.Module):
    def forward(self, x_t, t, x_0=None, *, support_context=None):  # noqa: ANN001, ARG002
        return torch.zeros_like(x_t)


def test_risk_row_cvar_defaults_are_off() -> None:
    cfg = Config()

    assert cfg.risk_row_cvar_loss_weight == 0.0
    assert cfg.train_eval_enabled is True
    assert risk_row_cvar_loss_schedule(1_000, cfg) == 0.0


def test_risk_row_cvar_schedule_uses_own_warmup() -> None:
    cfg = Config(
        risk_row_cvar_loss_weight=0.5,
        risk_row_cvar_loss_warmup_start=10,
        risk_row_cvar_loss_warmup_end=20,
    )

    assert risk_row_cvar_loss_schedule(9, cfg) == 0.0
    assert risk_row_cvar_loss_schedule(20, cfg) == 0.5


def test_tail_state_applies_only_after_high_tail_history() -> None:
    state = RiskRowCvarTailState(history_size=8, min_history=4, top_frac=0.25, threshold=0.005)

    for cond, value in [("a", 0.001), ("b", 0.002), ("c", 0.003)]:
        state.update("RiskDS", cond, value)
    assert not state.should_apply("RiskDS", "a")

    state.update("RiskDS", "tail", 0.020)
    assert state.should_apply("RiskDS", "tail")
    assert not state.should_apply("RiskDS", "a")


def test_batch_control_respects_dataset_filter_exclusion() -> None:
    cfg = Config(
        risk_row_cvar_loss_weight=0.5,
        risk_row_cvar_dataset_filter="RiskDS",
    )
    state = RiskRowCvarTailState(history_size=8, min_history=2, top_frac=0.5, threshold=0.005)
    state.update("OtherDS", "tail", 0.030)
    state.update("OtherDS", "low", 0.001)

    observe, weight = risk_row_cvar_batch_control(1, cfg, state, "OtherDS", "tail")

    assert observe is False
    assert weight == 0.0


def test_batch_control_observes_then_applies_nonzero_tail_weight() -> None:
    cfg = Config(
        risk_row_cvar_loss_weight=0.5,
        risk_row_cvar_dataset_filter="RiskDS",
    )
    state = RiskRowCvarTailState(history_size=8, min_history=4, top_frac=0.25, threshold=0.005)

    observe, weight = risk_row_cvar_batch_control(1, cfg, state, "RiskDS", "tail")
    assert observe is True
    assert weight == 0.0

    for cond, value in [("low_a", 0.001), ("low_b", 0.002), ("low_c", 0.003), ("tail", 0.030)]:
        state.update("RiskDS", cond, value)

    observe, weight = risk_row_cvar_batch_control(2, cfg, state, "RiskDS", "tail")

    assert observe is True
    assert weight == 0.5


def test_train_step_can_observe_tail_mmd_without_extra_loss_weight() -> None:
    cfg = Config(emb_dim=2, model_type="mlp", use_mmd=True, mmd_estimator="biased")
    model = _ZeroVelocity()
    path = CondOTPath()
    src = torch.zeros(8, 2)
    gt = torch.ones(8, 2)
    device = torch.device("cpu")

    no_observe = train_step(
        src,
        gt,
        model,
        path,
        cfg,
        device,
        ds_name="RiskDS",
        gamma_t=0.0,
        risk_row_cvar_weight_t=0.0,
        risk_row_cvar_observe=False,
    )
    observe = train_step(
        src,
        gt,
        model,
        path,
        cfg,
        device,
        ds_name="RiskDS",
        gamma_t=0.0,
        risk_row_cvar_weight_t=0.0,
        risk_row_cvar_observe=True,
    )

    assert float(no_observe["mmd"]) == 0.0
    assert float(observe["mmd"]) > 0.0
    assert float(observe["risk_row_cvar_weight"]) == 0.0


if __name__ == "__main__":
    test_risk_row_cvar_defaults_are_off()
    test_risk_row_cvar_schedule_uses_own_warmup()
    test_tail_state_applies_only_after_high_tail_history()
    test_batch_control_respects_dataset_filter_exclusion()
    test_batch_control_observes_then_applies_nonzero_tail_weight()
    test_train_step_can_observe_tail_mmd_without_extra_loss_weight()
    print("latent risk-row CVaR tail-state tests passed")
