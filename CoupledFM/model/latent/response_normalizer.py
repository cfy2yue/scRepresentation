"""Train-only response-space transforms for LatentFM auxiliary losses.

The normalizer is intentionally artifact-driven: fitting happens in a separate
script using train conditions only. Training/evaluation code only loads the
artifact and applies deterministic tensor transforms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    h = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ResponseNormalizer:
    """Invertible-ish linear response transform selected by dataset.

    Supported modes:
    - ``dataset_scale``: normalize residual magnitude by a train-only
      per-dataset median-norm factor.
    - ``pca_subspace``: whiten top response PCs while passing the orthogonal
      residual through.
    - ``dataset_scale_pca``: dataset scale, then PCA transform.
    """

    mode: str
    emb_dim: int
    dataset_scales: dict[str, float]
    global_scale: float
    pca_mean: torch.Tensor | None = None
    pca_components: torch.Tensor | None = None
    pca_scales: torch.Tensor | None = None
    artifact_path: str = ""
    metadata: dict[str, Any] | None = None
    eps: float = 1e-6

    @staticmethod
    def is_enabled_mode(mode: str) -> bool:
        return str(mode or "off").strip().lower() not in {"", "off", "none", "false", "0"}

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        mode: str,
        device: torch.device | str = "cpu",
        strict_split_file: str | Path | None = None,
        strict_emb_dim: int | None = None,
    ) -> "ResponseNormalizer":
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"response normalization artifact not found: {path}")
        mode = str(mode or "off").strip().lower()
        if not cls.is_enabled_mode(mode):
            raise ValueError("ResponseNormalizer.from_npz called with disabled mode")
        if mode not in {"dataset_scale", "pca_subspace", "dataset_scale_pca"}:
            raise ValueError(f"unsupported response_normalization_mode={mode!r}")

        obj = np.load(str(path), allow_pickle=False)
        metadata = json.loads(str(obj["metadata_json"].item()))
        emb_dim = int(metadata["emb_dim"])
        if strict_emb_dim is not None and int(strict_emb_dim) != emb_dim:
            raise ValueError(f"response normalizer emb_dim {emb_dim} != config emb_dim {strict_emb_dim}")
        if strict_split_file is not None:
            split_path = Path(strict_split_file).expanduser().resolve()
            expected = str(metadata.get("split_sha256", ""))
            actual = sha256_file(split_path)
            if expected and expected != actual:
                raise ValueError(
                    "response normalizer split hash mismatch: "
                    f"artifact={expected}, split={actual}, split_file={split_path}"
                )

        ds_scales = json.loads(str(obj["dataset_scale_factors_json"].item()))
        pca_mean = None
        pca_components = None
        pca_scales = None
        if mode in {"pca_subspace", "dataset_scale_pca"}:
            pca_mean = torch.as_tensor(obj["pca_mean"], dtype=torch.float32, device=device)
            pca_components = torch.as_tensor(obj["pca_components"], dtype=torch.float32, device=device)
            pca_scales = torch.as_tensor(obj["pca_scales"], dtype=torch.float32, device=device)
            if pca_components.ndim != 2 or pca_components.shape[1] != emb_dim:
                raise ValueError(
                    f"invalid pca_components shape {tuple(pca_components.shape)} for emb_dim={emb_dim}"
                )
        return cls(
            mode=mode,
            emb_dim=emb_dim,
            dataset_scales={str(k): float(v) for k, v in ds_scales.items()},
            global_scale=float(metadata.get("global_median_norm", 1.0) or 1.0),
            pca_mean=pca_mean,
            pca_components=pca_components,
            pca_scales=pca_scales,
            artifact_path=str(path),
            metadata=metadata,
        )

    def to(self, device: torch.device | str) -> "ResponseNormalizer":
        if self.pca_components is None:
            return self
        return ResponseNormalizer(
            mode=self.mode,
            emb_dim=self.emb_dim,
            dataset_scales=self.dataset_scales,
            global_scale=self.global_scale,
            pca_mean=None if self.pca_mean is None else self.pca_mean.to(device),
            pca_components=self.pca_components.to(device),
            pca_scales=None if self.pca_scales is None else self.pca_scales.to(device),
            artifact_path=self.artifact_path,
            metadata=self.metadata,
            eps=self.eps,
        )

    def _scale_for_dataset(self, ds_name: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        scale = float(self.dataset_scales.get(str(ds_name), 1.0))
        scale = max(scale, self.eps)
        return torch.as_tensor(scale, dtype=dtype, device=device)

    def transform_delta(self, ds_name: str, delta: torch.Tensor) -> torch.Tensor:
        """Return delta in the configured train-response coordinates."""
        if int(delta.shape[-1]) != self.emb_dim:
            raise ValueError(f"delta dim {int(delta.shape[-1])} != response normalizer emb_dim {self.emb_dim}")
        x = delta
        if self.mode in {"dataset_scale", "dataset_scale_pca"}:
            x = x / self._scale_for_dataset(str(ds_name), device=x.device, dtype=x.dtype)
        if self.mode in {"pca_subspace", "dataset_scale_pca"}:
            if self.pca_components is None or self.pca_scales is None or self.pca_mean is None:
                raise RuntimeError("PCA response normalizer tensors are missing")
            comps = self.pca_components.to(device=x.device, dtype=x.dtype)
            scales = self.pca_scales.to(device=x.device, dtype=x.dtype).clamp_min(self.eps)
            mean = self.pca_mean.to(device=x.device, dtype=x.dtype)
            centered = x - mean
            coords = centered @ comps.t()
            recon_top = coords @ comps
            coords_scaled = coords / scales
            x = mean + (centered - recon_top) + coords_scaled @ comps
        return x

    def inverse_delta(self, ds_name: str, delta: torch.Tensor) -> torch.Tensor:
        """Invert ``transform_delta`` for model-output target-normalization experiments."""
        if int(delta.shape[-1]) != self.emb_dim:
            raise ValueError(f"delta dim {int(delta.shape[-1])} != response normalizer emb_dim {self.emb_dim}")
        x = delta
        if self.mode in {"pca_subspace", "dataset_scale_pca"}:
            if self.pca_components is None or self.pca_scales is None or self.pca_mean is None:
                raise RuntimeError("PCA response normalizer tensors are missing")
            comps = self.pca_components.to(device=x.device, dtype=x.dtype)
            scales = self.pca_scales.to(device=x.device, dtype=x.dtype).clamp_min(self.eps)
            mean = self.pca_mean.to(device=x.device, dtype=x.dtype)
            centered = x - mean
            coords = centered @ comps.t()
            recon_top = coords @ comps
            coords_unscaled = coords * scales
            x = mean + (centered - recon_top) + coords_unscaled @ comps
        if self.mode in {"dataset_scale", "dataset_scale_pca"}:
            x = x * self._scale_for_dataset(str(ds_name), device=x.device, dtype=x.dtype)
        return x
