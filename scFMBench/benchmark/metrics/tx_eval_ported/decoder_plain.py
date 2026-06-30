"""MLP decoder for reconstruction without Lightning."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .lr_scheduler import LinearWarmupCosineAnnealingLR
from .reconstruction_metrics import evaluate_reconstruction_metrics


def _build_mlp(embedding_dim: int, output_dim: int, hidden_dims: List[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    d_in = embedding_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d_in, h))
        layers.append(nn.ReLU())
        d_in = h
    layers.append(nn.Linear(d_in, output_dim))
    return nn.Sequential(*layers)


class DecoderMLP(nn.Module):
    def __init__(self, embedding_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.net = _build_mlp(embedding_dim, output_dim, hidden_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_decoder_mlp(
    train_emb: np.ndarray,
    train_expr: np.ndarray,
    *,
    hidden_dims: Optional[List[int]] = None,
    model_depth: int = 3,
    max_epochs: int = 50,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler_name: str = "none",
    warmup_epochs: int = 5,
    min_lr: float = 1e-5,
    warmup_start_lr: float = 1e-4,
    seed: int = 42,
    device: Optional[torch.device] = None,
    loss_type: str = "mse",
    num_workers: int = 0,
) -> Tuple[DecoderMLP, Dict[str, float]]:
    """
    Train decoder embedding -> expression. Returns model and training-set reconstruction metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = np.asarray(train_emb, dtype=np.float32)
    expr = np.asarray(train_expr, dtype=np.float32)
    n, d_e = emb.shape
    _, d_g = expr.shape
    if hidden_dims is None:
        hidden_dims = [
            int(d_e + (d_g - d_e) * i / (model_depth + 1)) for i in range(1, model_depth + 1)
        ]

    ds = TensorDataset(torch.from_numpy(emb), torch.from_numpy(expr))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=dev.type == "cuda",
    )

    model = DecoderMLP(d_e, d_g, hidden_dims).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(loader))
    total_steps = max_epochs * steps_per_epoch

    if scheduler_name == "warmup_cosine":
        sched = LinearWarmupCosineAnnealingLR(
            opt,
            warmup_epochs=warmup_epochs * steps_per_epoch,
            max_epochs=total_steps,
            warmup_start_lr=warmup_start_lr if warmup_epochs > 0 else lr,
            eta_min=min_lr,
        )
        step_sched_each_batch = True
    else:
        sched = None
        step_sched_each_batch = False

    if loss_type == "mse":
        loss_fn = nn.functional.mse_loss
    elif loss_type == "mae":
        loss_fn = nn.functional.l1_loss
    else:
        raise ValueError("loss_type must be mse or mae")

    global_step = 0
    model.train()
    for _ in range(max_epochs):
        for xb, yb in loader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            if step_sched_each_batch and sched is not None:
                sched.step()
            global_step += 1
        if sched is not None and not step_sched_each_batch:
            sched.step()

    model.eval()
    with torch.no_grad():
        all_p, all_t = [], []
        for xb, yb in loader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            all_p.append(model(xb).cpu())
            all_t.append(yb.cpu())
        preds = torch.cat(all_p, dim=0)
        targ = torch.cat(all_t, dim=0)
    metrics = evaluate_reconstruction_metrics(preds, targ)
    return model, metrics
