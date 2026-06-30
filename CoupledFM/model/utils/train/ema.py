"""Exponential Moving Average (EMA) helper for Flow Matching / diffusion models.

Usage
-----
Train-side:

    from model.utils.train.ema import ModelEMA

    ema = ModelEMA(model, decay=0.999, device=device, update_after=1000)
    ...
    for step, batch in enumerate(loader):
        loss = train_step(model, batch)
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        ema.update(model, step=step)

Eval / checkpoint-side:

    # inference with EMA weights
    with ema.apply_to(model):
        metrics = evaluate(model, val_loader)
    # context manager restores original weights on exit

    # or save a separate ema checkpoint
    torch.save(ema.state_dict(), "ckpt_ema.pt")

Design notes
------------
- Keeps a **CPU or GPU shadow** of parameter tensors; no autograd.
- `update_after` delays EMA update until after LR warmup, so early noisy weights
  don't pollute the shadow.
- Handles DDP by always taking the underlying `model.module` if present.
- `apply_to` / `restore` is a context manager; **do not** keep EMA weights loaded
  during training.
- DOES NOT save buffers (running stats of BN etc.) by default; override
  `include_buffers=True` if needed. For pure FM MLP / transformer models with
  LayerNorm there are no running stats, so default is correct.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn


def _unwrap(model: nn.Module) -> nn.Module:
    """Return underlying module (strip DDP wrappers)."""
    inner = getattr(model, "module", None)
    return inner if isinstance(inner, nn.Module) else model


class ModelEMA:
    """Maintain an exponential moving average of model parameters.

    Parameters
    ----------
    model : nn.Module
        The *live* model whose parameters will be averaged.
    decay : float
        EMA decay. Typical: 0.999 for FM/diffusion. Closer to 1 = smoother but
        slower to catch up.
    update_after : int
        Skip updates for the first ``update_after`` calls (use warmup-end step
        to avoid averaging initial noisy weights).
    update_every : int
        Only update every N calls (rest are no-ops). Default 1.
    device : Optional[torch.device]
        Where to keep the shadow tensors. `None` uses model's device. Use `cpu`
        if GPU memory is tight (shadow doubles parameter memory).
    include_buffers : bool
        If True also average persistent buffers (BatchNorm running stats).
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        update_after: int = 0,
        update_every: int = 1,
        device: Optional[torch.device] = None,
        include_buffers: bool = False,
        dynamic: bool = False,
    ) -> None:
        self.decay = float(decay)
        self.update_after = int(update_after)
        self.update_every = max(1, int(update_every))
        self.include_buffers = bool(include_buffers)
        self.dynamic = bool(dynamic)
        self.num_updates = 0

        inner = _unwrap(model)
        shadow_device = device if device is not None else next(inner.parameters()).device
        self.device = shadow_device

        # Deepcopy parameter dict to CPU/GPU shadow. Do NOT attach to optimizer.
        with torch.no_grad():
            self._shadow: Dict[str, torch.Tensor] = {
                name: p.detach().clone().to(shadow_device)
                for name, p in inner.named_parameters()
                if p.requires_grad  # frozen params don't need EMA
            }
            if self.include_buffers:
                self._buffers: Dict[str, torch.Tensor] = {
                    name: b.detach().clone().to(shadow_device)
                    for name, b in inner.named_buffers()
                }
            else:
                self._buffers = {}

        self._backup: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def add_missing_parameters(
        self,
        model: nn.Module,
        requires_grad_only: bool = True,
    ) -> int:
        """Register parameters that became trainable after EMA init.

        This keeps EMA aligned with staged fine-tuning where some params are
        frozen at startup and unfrozen later.
        """
        inner = _unwrap(model)
        added = 0
        for name, p in inner.named_parameters():
            if name in self._shadow:
                continue
            if requires_grad_only and not p.requires_grad:
                continue
            self._shadow[name] = p.detach().clone().to(self.device)
            added += 1
        return added

    # ------------------------------------------------------------------ update
    @torch.no_grad()
    def update(self, model: nn.Module, step: Optional[int] = None) -> None:
        """Update shadow weights. Safe to call every step."""
        if step is not None and step < self.update_after:
            return
        self.num_updates += 1
        if (self.num_updates - 1) % self.update_every != 0:
            return

        d = self.decay
        if self.dynamic and step is not None:
            d = min(d, (1.0 + float(step)) / (10.0 + float(step)))
        inner = _unwrap(model)
        for name, p in inner.named_parameters():
            if name not in self._shadow:  # newly unfrozen param? skip silently
                continue
            s = self._shadow[name]
            if s.device != p.device:
                p_ = p.detach().to(s.device, non_blocking=True)
            else:
                p_ = p.detach()
            s.mul_(d).add_(p_, alpha=1.0 - d)
        if self.include_buffers:
            for name, b in inner.named_buffers():
                if name in self._buffers:
                    sb = self._buffers[name]
                    if sb.dtype.is_floating_point:
                        sb.mul_(d).add_(b.detach().to(sb.device), alpha=1.0 - d)
                    else:  # int buffers: just copy latest
                        sb.copy_(b.detach().to(sb.device))

    # ----------------------------------------------------------------- apply
    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite ``model``'s parameters with EMA shadow (in-place)."""
        inner = _unwrap(model)
        for name, p in inner.named_parameters():
            if name in self._shadow:
                p.data.copy_(self._shadow[name].to(p.device, non_blocking=True))
        if self.include_buffers:
            for name, b in inner.named_buffers():
                if name in self._buffers:
                    b.data.copy_(self._buffers[name].to(b.device, non_blocking=True))

    @torch.no_grad()
    def _backup_params(self, model: nn.Module) -> None:
        inner = _unwrap(model)
        self._backup = {
            name: p.detach().clone() for name, p in inner.named_parameters() if name in self._shadow
        }

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore live params from backup (undo apply_to)."""
        if self._backup is None:
            return
        inner = _unwrap(model)
        for name, p in inner.named_parameters():
            if name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup = None

    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[None]:
        """Context manager: temporarily swap live params with EMA shadow."""
        self._backup_params(model)
        try:
            self.copy_to(model)
            yield
        finally:
            self.restore(model)

    # --------------------------------------------------------- serialization
    def state_dict(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {f"shadow.{k}": v for k, v in self._shadow.items()}
        for k, v in self._buffers.items():
            out[f"buffer.{k}"] = v
        out["__meta__"] = torch.tensor([self.decay, self.update_after, self.update_every, self.num_updates], dtype=torch.float64)
        return out

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> None:
        shadow = {k[len("shadow."):]: v for k, v in state_dict.items() if k.startswith("shadow.")}
        buffers = {k[len("buffer."):]: v for k, v in state_dict.items() if k.startswith("buffer.")}
        if strict:
            miss = set(self._shadow) - set(shadow)
            extra = set(shadow) - set(self._shadow)
            if miss or extra:
                raise RuntimeError(f"EMA state mismatch. missing={miss} unexpected={extra}")
        for k, v in shadow.items():
            if k in self._shadow:
                self._shadow[k].copy_(v.to(self._shadow[k].device))
        if self.include_buffers:
            for k, v in buffers.items():
                if k in self._buffers:
                    self._buffers[k].copy_(v.to(self._buffers[k].device))
        meta = state_dict.get("__meta__", None)
        if meta is not None:
            self.num_updates = int(meta[-1].item())
