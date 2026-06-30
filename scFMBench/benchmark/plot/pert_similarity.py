"""Perturbation effect-similarity analysis.

For a chosen dataset and the top-K perturbations (ranked by raw expression
centroid L2), we compute:

* In raw expression space (ground truth): per-perturbation diff vector
  ``mu_pert - mu_control`` over the staging log1p X (G genes), giving a
  (K, G) matrix; from it a (K, K) similarity matrix.
* In each model's latent space: same construction over the cached
  ``latent.npy`` (d-dim), giving a (K, K) similarity matrix per model.

The two matrices encode the **structure of perturbation relationships**
(co-functionality, pathway co-membership, etc.). We summarize how well a
model preserves that structure with a Mantel-style Spearman correlation
between the upper-triangle vectors of the GT and latent similarity matrices
(also report Pearson for comparison). Cosine and Pearson similarity are both
supported; cosine is used for the displayed matrices.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
FM_ROOT = Path(__file__).resolve().parents[2] / "fm"
if str(FM_ROOT) not in sys.path:
    sys.path.insert(0, str(FM_ROOT))
import paths


# ---------------- diff matrices ----------------------------------------

def _load_obs(emb_dir: Path) -> Optional[pd.DataFrame]:
    pq = emb_dir / "obs.parquet"
    cz = emb_dir / "obs.csv.gz"
    if pq.is_file():
        try:
            return pd.read_parquet(pq)
        except Exception as exc:
            LOG.warning("Could not read %s: %s", pq, exc)
    if cz.is_file():
        try:
            return pd.read_csv(cz)
        except Exception as exc:
            LOG.warning("Could not read %s: %s", cz, exc)
    return None


def compute_latent_pert_diff_matrix(
    scfm_root: Path,
    model: str,
    dataset_id: str,
    perts: Sequence[str],
    *,
    pert_col: str = "perturbation",
    control_label: str = "control",
) -> Optional[np.ndarray]:
    """Return ``(K, d)`` latent diff matrix for the requested ``perts``."""
    emb_dir = paths.output_root() / "embeddings" / model / dataset_id / "raw"
    z_path = emb_dir / "latent.npy"
    if not z_path.is_file():
        return None
    obs = _load_obs(emb_dir)
    if obs is None or pert_col not in obs.columns:
        return None
    Z = np.load(z_path, mmap_mode="r")
    if Z.shape[0] != len(obs):
        LOG.warning("Latent/obs row mismatch for %s/%s: %d vs %d",
                    model, dataset_id, Z.shape[0], len(obs))
        return None

    pert = obs[pert_col].astype(str).to_numpy()
    if "is_control" in obs.columns:
        is_ctrl = obs["is_control"].to_numpy().astype(bool)
    elif "control" in obs.columns and obs["control"].dtype.kind in "biu":
        is_ctrl = obs["control"].to_numpy().astype(bool)
    else:
        is_ctrl = pert == control_label

    ctrl_idx = np.where(is_ctrl)[0]
    if ctrl_idx.size < 2:
        return None
    mu_ctrl = np.asarray(Z[ctrl_idx]).mean(axis=0)

    K, d = len(perts), int(Z.shape[1])
    out = np.full((K, d), np.nan, dtype=np.float32)
    for i, p in enumerate(perts):
        idx = np.where((pert == p) & (~is_ctrl))[0]
        if idx.size < 2:
            continue
        mu_p = np.asarray(Z[idx]).mean(axis=0)
        out[i] = (mu_p - mu_ctrl).astype(np.float32)
    return out


# ---------------- similarity / consistency -----------------------------

def pert_similarity_matrix(diff: np.ndarray, *, kind: str = "cosine") -> np.ndarray:
    """``(K, K)`` similarity matrix among diff vectors. NaN-safe (rows that
    are all-NaN remain NaN in their row/column).
    """
    K = diff.shape[0]
    out = np.full((K, K), np.nan, dtype=np.float64)
    valid = ~np.any(np.isnan(diff), axis=1)
    if valid.sum() < 2:
        return out
    M = diff[valid].astype(np.float64)

    if kind == "cosine":
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms = np.where(norms == 0, np.nan, norms)
        Mn = M / norms
        S = Mn @ Mn.T
    elif kind == "pearson":
        Mc = M - M.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(Mc, axis=1, keepdims=True)
        norms = np.where(norms == 0, np.nan, norms)
        Mn = Mc / norms
        S = Mn @ Mn.T
    else:
        raise ValueError(f"unknown similarity kind {kind!r}")

    idx = np.where(valid)[0]
    for i, ii in enumerate(idx):
        for j, jj in enumerate(idx):
            out[ii, jj] = float(S[i, j])
    return out


def _upper_tri(A: np.ndarray) -> np.ndarray:
    K = A.shape[0]
    iu = np.triu_indices(K, k=1)
    return A[iu]


def mantel_correlation(
    A: np.ndarray, B: np.ndarray, *, kind: str = "spearman"
) -> float:
    """Correlation between upper-triangle vectors of two square matrices.

    Returns NaN when fewer than 4 valid pairs are available.
    """
    from scipy.stats import pearsonr, spearmanr

    a = _upper_tri(A)
    b = _upper_tri(B)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 4:
        return float("nan")
    if kind == "spearman":
        rho, _ = spearmanr(a[ok], b[ok])
    elif kind == "pearson":
        rho, _ = pearsonr(a[ok], b[ok])
    else:
        raise ValueError(kind)
    return float(rho)


# ---------------- top-K resolution -------------------------------------

def select_top_perts(
    scfm_root: Path, out_dir: Path, dataset_id: str, k: int = 10,
) -> List[str]:
    """Return the GT top-K perturbation names (highest raw centroid L2)."""
    from .raw_pert_shifts import compute_raw_pert_shifts

    gt = compute_raw_pert_shifts(scfm_root, out_dir, dataset_id)
    if not gt:
        return []
    return [k for k, _ in sorted(gt.items(), key=lambda kv: -kv[1])][:k]


# ---------------- per-(dataset, models) computation ---------------------

def compute_similarity_set(
    scfm_root: Path,
    out_dir: Path,
    dataset_id: str,
    models: Sequence[str],
    *,
    k: int = 10,
    kinds: Sequence[str] = ("cosine", "pearson"),
) -> Dict:
    """Compute GT and per-model (K, K) similarity matrices and consistency
    summaries. Returns a dict::

        {
          "perts": [..K..],
          "sim": {kind: {"GT": (K,K), model_a: (K,K), ...}},
          "consistency": {model: {"mantel_spearman_<kind>": rho,
                                   "mantel_pearson_<kind>": rho}},
        }
    """
    from .raw_pert_shifts import compute_raw_pert_diff_matrix

    perts = select_top_perts(scfm_root, out_dir, dataset_id, k=k)
    if not perts:
        return {"perts": [], "sim": {}, "consistency": {}}

    raw_diff = compute_raw_pert_diff_matrix(scfm_root, dataset_id, perts)
    if raw_diff is None:
        return {"perts": perts, "sim": {}, "consistency": {}}

    sim: Dict[str, Dict[str, np.ndarray]] = {k: {} for k in kinds}
    for kind in kinds:
        sim[kind]["GT"] = pert_similarity_matrix(raw_diff, kind=kind)

    consistency: Dict[str, Dict[str, float]] = {}
    for model in models:
        latent_diff = compute_latent_pert_diff_matrix(
            scfm_root, model, dataset_id, perts,
        )
        cons: Dict[str, float] = {}
        if latent_diff is None:
            for kind in kinds:
                sim[kind][model] = np.full((len(perts), len(perts)), np.nan)
                cons[f"mantel_spearman_{kind}"] = float("nan")
                cons[f"mantel_pearson_{kind}"] = float("nan")
            consistency[model] = cons
            continue
        for kind in kinds:
            S_lat = pert_similarity_matrix(latent_diff, kind=kind)
            sim[kind][model] = S_lat
            cons[f"mantel_spearman_{kind}"] = mantel_correlation(
                sim[kind]["GT"], S_lat, kind="spearman",
            )
            cons[f"mantel_pearson_{kind}"] = mantel_correlation(
                sim[kind]["GT"], S_lat, kind="pearson",
            )
        consistency[model] = cons

    return {"perts": perts, "sim": sim, "consistency": consistency}


# ---------------- caching ---------------------------------------------

def _cache_path(out_dir: Path, dataset_id: str, k: int) -> Path:
    return out_dir / "_cache_pert_similarity" / f"{dataset_id}__top{k}.json"


def _np_to_list(a: np.ndarray) -> List:
    return [[None if np.isnan(v) else float(v) for v in row] for row in a]


def _list_to_np(L: List) -> np.ndarray:
    return np.array([[np.nan if v is None else float(v) for v in row] for row in L],
                    dtype=np.float64)


def compute_or_load(
    scfm_root: Path,
    out_dir: Path,
    dataset_id: str,
    models: Sequence[str],
    *,
    k: int = 10,
    use_cache: bool = True,
) -> Dict:
    cache = _cache_path(out_dir, dataset_id, k)
    if use_cache and cache.is_file():
        try:
            payload = json.loads(cache.read_text())
            if set(payload.get("models", [])) >= set(models):
                sim = {
                    kind: {name: _list_to_np(L) for name, L in d.items()}
                    for kind, d in payload["sim"].items()
                }
                return {
                    "perts": payload["perts"],
                    "sim": sim,
                    "consistency": payload["consistency"],
                }
        except Exception as exc:
            LOG.warning("Ignoring unreadable similarity cache %s: %s", cache, exc)

    res = compute_similarity_set(scfm_root, out_dir, dataset_id, models, k=k)
    payload = {
        "dataset_id": dataset_id,
        "k": k,
        "models": list(models),
        "perts": res["perts"],
        "sim": {
            kind: {name: _np_to_list(M) for name, M in d.items()}
            for kind, d in res["sim"].items()
        },
        "consistency": res["consistency"],
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload, indent=2))
    return res
