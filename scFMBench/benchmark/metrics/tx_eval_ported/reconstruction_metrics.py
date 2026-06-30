"""Reconstruction metrics from Tx-Evaluation (torchmetrics only, no Lightning)."""

from __future__ import annotations

from typing import Dict

import logging
import numpy as np
import torch
import torchmetrics
import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def evaluate_reconstruction_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """MSE, MAE, mean Pearson/Spearman, R² over feature dimension."""
    targets = targets.to(predictions.device)
    mse_metric = torchmetrics.MeanSquaredError().to(predictions.device)
    mae_metric = torchmetrics.MeanAbsoluteError().to(predictions.device)
    pearson_metric = torchmetrics.PearsonCorrCoef(num_outputs=predictions.shape[1]).to(predictions.device)
    spearman_metric = torchmetrics.SpearmanCorrCoef(num_outputs=predictions.shape[1]).to(predictions.device)

    mse = mse_metric(predictions, targets).item()
    mae = mae_metric(predictions, targets).item()
    average_pearson = pearson_metric(predictions, targets).mean().item()
    average_spearman = spearman_metric(predictions, targets).mean().item()

    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r_squared = (1 - ss_res / ss_tot).item()

    mse_metric.reset()
    mae_metric.reset()
    pearson_metric.reset()
    spearman_metric.reset()

    return {
        "MSE": mse,
        "MAE": mae,
        "Average Pearson Correlation": average_pearson,
        "Average Spearman Correlation": average_spearman,
        "R-squared": r_squared,
    }


class StructuralTranscriptomeDistance(torchmetrics.Metric):
    """Batch- and control-centered Frobenius-style integrity (Tx-Evaluation)."""

    def __init__(self, distance_function: str = "frobenius", compute_on_cpu: bool = False):
        super().__init__(compute_on_cpu=compute_on_cpu)
        self.distance_function = distance_function
        self.add_state("total_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_max_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_elements", default=torch.tensor(0), dist_reduce_fx="sum")

    def adjust_data(self, transcriptomes, batch_ids, is_control, control_label):
        if not isinstance(is_control, np.ndarray):
            is_control = np.array(is_control)
        control_mask = is_control == control_label
        if isinstance(batch_ids, torch.Tensor):
            batch_ids = batch_ids.detach().cpu().numpy()
        unique_batches = np.unique(batch_ids)
        adjusted_transcriptomes = transcriptomes.clone()
        device = transcriptomes.device
        for batch in unique_batches:
            batch_mask = batch_ids == batch
            control_batch_mask = batch_mask & control_mask
            if np.any(control_batch_mask):
                control_idx = torch.from_numpy(control_batch_mask).to(device=device)
                batch_idx = torch.from_numpy(batch_mask).to(device=device)
                mean_control = transcriptomes[control_idx].mean(dim=0)
                adjusted_transcriptomes[batch_idx] -= mean_control
        return adjusted_transcriptomes

    def update(self, preds, target, batch_ids, is_control, control_label):
        pred_adjusted = self.adjust_data(preds, batch_ids, is_control, control_label)
        target_adjusted = self.adjust_data(target, batch_ids, is_control, control_label)
        if isinstance(batch_ids, torch.Tensor):
            batch_ids = batch_ids.detach().cpu().numpy()
        unique_batches = np.unique(batch_ids)
        device = preds.device
        for batch in tqdm.tqdm(unique_batches):
            batch_mask = batch_ids == batch
            batch_idx = torch.from_numpy(batch_mask).to(device=device)
            pred_batch = pred_adjusted[batch_idx]
            target_batch = target_adjusted[batch_idx]
            if len(pred_batch) > 0 and self.distance_function == "frobenius":
                distance = torch.norm(pred_batch - target_batch, p="fro") / target_batch.shape[0]
                max_distance = torch.norm(target_batch - 0, p="fro") / target_batch.shape[0]
                self.total_distance += distance
                self.total_max_distance += max_distance
                self.num_elements += 1

    def compute(self):
        if self.num_elements > 0:
            distance = self.total_distance / self.num_elements
            max_distance = 2 * (self.total_max_distance / self.num_elements)
        else:
            distance = self.total_distance
            max_distance = 2 * self.total_max_distance
        integrity = 1 - (distance / max_distance)
        return integrity.item()
