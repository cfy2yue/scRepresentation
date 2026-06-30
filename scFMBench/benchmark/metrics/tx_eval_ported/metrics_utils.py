"""Subset of Tx-Evaluation biomodalities.utils.metrics (no Lightning)."""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F


def accuracy_at_k(
    outputs: torch.Tensor, targets: torch.Tensor, top_k: Sequence[int] = (1, 5)
) -> List[torch.Tensor]:
    with torch.no_grad():
        num_classes = outputs.size(1)
        maxk = min(max(top_k), num_classes)
        batch_size = targets.size(0)
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        res = []
        for k in top_k:
            kk = min(k, num_classes)
            correct_k = correct[:kk].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def weighted_mean(outputs: List[Dict], key: str, batch_size_key: str) -> torch.Tensor:
    value = 0.0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    return (value / n).squeeze(0)


def multiclass_logits_loss_batch(
    logits: torch.Tensor, targets: torch.Tensor, loss_fn: torch.nn.Module
) -> Dict[str, torch.Tensor]:
    loss = loss_fn(logits, targets)
    acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, 5))
    return {"loss": loss, "acc1": acc1, "acc5": acc5, "batch_size": logits.size(0)}
