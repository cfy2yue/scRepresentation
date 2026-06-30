"""Learning-rate schedulers compatible with torch optimizers.

Adds two schedulers not in torch core:

- :func:`get_cosine_with_min_lr_schedule_with_warmup` — HuggingFace-style
  cosine decay that floors at ``min_lr = base_lr * min_lr_ratio`` rather than
  decaying all the way to 0. Keeps gradient information alive even near the
  end of long training runs.

- :func:`get_linear_warmup_then_const` — linear warmup then constant LR. Useful
  as a baseline or for Flow-Matching-style never-ending training.

All schedulers are returned as :class:`torch.optim.lr_scheduler.LambdaLR`
instances (so checkpointing works out of the box).
"""

from __future__ import annotations

import math
from typing import List, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _as_list(x: Union[float, List[float]], n: int) -> List[float]:
    if isinstance(x, (list, tuple)):
        if len(x) != n:
            raise ValueError(f"expected {n} values, got {len(x)}")
        return [float(v) for v in x]
    return [float(x)] * n


def get_cosine_with_min_lr_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: Union[float, List[float]] = 0.1,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """Linear warmup, then half-cosine decay to ``min_lr_ratio * base_lr``.

    ``min_lr_ratio`` can be a scalar (applied to all param groups) or a list
    of the same length as ``optimizer.param_groups`` for per-group floors.
    """
    ratios = _as_list(min_lr_ratio, len(optimizer.param_groups))

    def lr_lambda(pg_idx: int):
        r = ratios[pg_idx]

        def fn(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            progress = min(1.0, progress)
            cos = 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress))
            # cos in [0, 1]; floor at r
            return r + (1.0 - r) * cos

        return fn

    lr_lambdas = [lr_lambda(i) for i in range(len(optimizer.param_groups))]
    return LambdaLR(optimizer, lr_lambdas, last_epoch=last_epoch)


def get_linear_warmup_then_const(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    def fn(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    n = len(optimizer.param_groups)
    return LambdaLR(optimizer, [fn] * n, last_epoch=last_epoch)


def lr_warmup_cosine_to_eta_min(
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    eta_min: float,
) -> float:
    """Absolute LR for latent FM-style schedule: warmup then cosine decay to ``eta_min``."""
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
    return eta_min + 0.5 * (peak_lr - eta_min) * (1.0 + math.cos(math.pi * progress))


def lr_warmup_cosine_ratio_floor_absolute(
    step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr_ratio: float,
) -> float:
    """Absolute LR matching coupled FM param-group cosine with floor ``base_lr * min_lr_ratio``.

    Warmup ramps absolute LR from 0 toward ``base_lr``; cosine phase respects per-group ``base_lr``.
    """
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(1.0, max(0.0, progress))
    floor = base_lr * float(min_lr_ratio)
    return floor + 0.5 * (base_lr - floor) * (1.0 + math.cos(math.pi * progress))


def get_ode_prob_curriculum(
    current_step: int,
    warmup_steps: int,
    anneal_steps: int,
    max_prob: float = 1.0,
) -> float:
    """ODE-vs-interp curriculum for teacher forcing.

    - step < warmup: 0 (pure teacher forcing via interp)
    - warmup ≤ step < warmup+anneal: linearly ramp 0 -> max_prob
    - step >= warmup+anneal: max_prob

    Use in the data/train loop to decide whether a given sample uses
    ``latent_z_mode=interp`` (p = 1 - ode_prob) or ``latent_z_mode=ode``.
    """
    if current_step < warmup_steps:
        return 0.0
    if anneal_steps <= 0:
        return float(max_prob)
    progress = (current_step - warmup_steps) / float(anneal_steps)
    return float(max_prob) * min(1.0, max(0.0, progress))
