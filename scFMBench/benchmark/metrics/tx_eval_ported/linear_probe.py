"""Linear probing without Lightning (Tx-Evaluation-style)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics_utils import accuracy_at_k, weighted_mean

_OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}


def fit_linear_probe(
    train_X: np.ndarray,
    train_y: Union[np.ndarray, List[str]],
    val_X: np.ndarray,
    val_y: Union[np.ndarray, List[str]],
    *,
    unique_labels: Optional[Sequence[str]] = None,
    max_epochs: int = 100,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    optimizer_name: str = "adamw",
    seed: int = 42,
    device: Optional[torch.device] = None,
    num_workers: int = 0,
) -> Dict[str, float]:
    """
    Train a single linear layer on embeddings; return validation top-1 / top-5 accuracy (%).

    Labels may be strings; ``unique_labels`` fixes class order (e.g. train∪test union).
    """
    if optimizer_name not in _OPTIMIZERS:
        raise ValueError(f"optimizer_name must be one of {list(_OPTIMIZERS)}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_y = np.asarray(train_y, dtype=object)
    val_y = np.asarray(val_y, dtype=object)

    if unique_labels is None:
        labels = sorted(set(train_y.tolist()) | set(val_y.tolist()))
    else:
        labels = list(unique_labels)
    label_to_idx = {str(lab): i for i, lab in enumerate(labels)}
    num_classes = len(labels)
    in_dim = int(train_X.shape[1])

    def _encode(y: np.ndarray) -> torch.Tensor:
        return torch.tensor([label_to_idx[str(v)] for v in y], dtype=torch.long)

    Xt = torch.from_numpy(np.asarray(train_X, dtype=np.float32))
    Yt = _encode(train_y)
    Xv = torch.from_numpy(np.asarray(val_X, dtype=np.float32))
    Yv = _encode(val_y)

    train_loader = DataLoader(
        TensorDataset(Xt, Yt),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=dev.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(Xv, Yv),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=dev.type == "cuda",
    )

    model = nn.Linear(in_dim, num_classes).to(dev)
    opt = _OPTIMIZERS[optimizer_name](model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    val_rows: List[Dict] = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            logits = model(xb)
            acc1, acc5 = accuracy_at_k(logits, yb, top_k=(1, 5))
            val_rows.append(
                {
                    "batch_size": xb.size(0),
                    "val_acc1": acc1,
                    "val_acc5": acc5,
                }
            )

    acc1 = float(weighted_mean(val_rows, "val_acc1", "batch_size"))
    acc5 = float(weighted_mean(val_rows, "val_acc5", "batch_size"))
    return {
        "linear_acc1": acc1,
        "linear_acc5": acc5,
        "num_classes": float(num_classes),
        "input_dim": float(in_dim),
    }
