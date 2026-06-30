"""
Tx-Evaluation code paths vendored without Lightning / WandB.

Optional: ``geomloss`` for ``pert_signal_magnitude_*`` (install via pip).
"""

from __future__ import annotations

from .bmdb import (
    aggregate,
    known_relationship_benchmark,
    pert_signal_consistency_benchmark,
    pert_signal_magnitude_benchmark,
)
from .decoder_plain import DecoderMLP, fit_decoder_mlp
from .ilisi import TorchILISIMetric
from .knn import WeightedKNNClassifier
from .linear_probe import fit_linear_probe
from .reconstruction_metrics import StructuralTranscriptomeDistance, evaluate_reconstruction_metrics
from .visual import OfflineVIZ

__all__ = [
    "aggregate",
    "known_relationship_benchmark",
    "pert_signal_consistency_benchmark",
    "pert_signal_magnitude_benchmark",
    "DecoderMLP",
    "fit_decoder_mlp",
    "TorchILISIMetric",
    "WeightedKNNClassifier",
    "fit_linear_probe",
    "StructuralTranscriptomeDistance",
    "evaluate_reconstruction_metrics",
    "OfflineVIZ",
]
