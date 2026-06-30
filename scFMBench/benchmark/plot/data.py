"""Load and reshape ``summary_all.csv`` for plotting."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

FM_ROOT = Path(__file__).resolve().parents[2] / "fm"
if str(FM_ROOT) not in sys.path:
    sys.path.insert(0, str(FM_ROOT))
import paths

from .metrics import ALL_METRICS, Metric, by_column
from .style import ALL_MODELS, BASELINE_MODELS, FM_MODELS

_DATASET_ORDER: Tuple[str, ...] = (
    # atlas (staging)
    "Skin", "Heart", "BoneMarrow", "Blood", "Lung", "LymphNode", "TS_Immune_xtissue",
    # atlas TS (raw counts)
    "TS_Skin_filtered", "TS_Heart_filtered", "TS_Bone_Marrow_filtered",
    "TS_Blood_filtered", "TS_Lung_filtered", "TS_Lymph_Node_filtered",
    # chempert
    "sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7", "sciplex3_xCellLine",
    # genepert
    "Adamson", "NormanWeissman2019_filtered__single",
    "ReplogleWeissman2022_K562_gwps", "Replogle_RPE1essential",
    "TianActivation", "TianInhibition",
)

GENEPERT_DATASETS: Tuple[str, ...] = (
    "Adamson",
    "NormanWeissman2019_filtered__single",
    "ReplogleWeissman2022_K562_gwps",
    "Replogle_RPE1essential",
    "TianActivation",
    "TianInhibition",
)

DATASET_DISPLAY: Dict[str, str] = {
    "Skin": "Skin",
    "Heart": "Heart",
    "BoneMarrow": "BoneMarrow",
    "Blood": "Blood",
    "Lung": "Lung",
    "LymphNode": "LymphNode",
    "TS_Immune_xtissue": "Immune-x",
    "TS_Skin_filtered": "Skin·TS",
    "TS_Heart_filtered": "Heart·TS",
    "TS_Bone_Marrow_filtered": "BoneMarrow·TS",
    "TS_Blood_filtered": "Blood·TS",
    "TS_Lung_filtered": "Lung·TS",
    "TS_Lymph_Node_filtered": "LymphNode·TS",
    "sciplex3_A549": "sciplex3·A549",
    "sciplex3_K562": "sciplex3·K562",
    "sciplex3_MCF7": "sciplex3·MCF7",
    "sciplex3_xCellLine": "sciplex3·xLine",
    "Adamson": "Adamson",
    "NormanWeissman2019_filtered__single": "Norman2019",
    "ReplogleWeissman2022_K562_gwps": "Replogle_K562",
    "Replogle_RPE1essential": "Replogle_RPE1",
    "TianActivation": "TianAct",
    "TianInhibition": "TianInh",
}


def _category_for(dataset_id: str) -> str:
    if dataset_id.startswith("sciplex3_"):
        return "chempert"
    if dataset_id in GENEPERT_DATASETS:
        return "genepert"
    if dataset_id.startswith("TS_") and dataset_id.endswith("_filtered"):
        return "atlas_TS"
    return "atlas"


def load_wide(scfm_root: Path) -> pd.DataFrame:
    """Load ``output/metrics/summary_all.csv`` and add helper columns.

    Adds: ``category``, ``is_baseline``, ``model_display``, ``dataset_display``.
    Restricts to canonical models and ordered atlas / atlas_TS / chempert / genepert datasets.
    """
    csv = paths.output_root() / "metrics" / "summary_all.csv"
    df = pd.read_csv(csv)
    df = df[df["model"].isin(ALL_MODELS) & df["dataset_id"].isin(_DATASET_ORDER)].copy()
    df["category"] = df["dataset_id"].map(_category_for)
    df["is_baseline"] = df["model"].isin(BASELINE_MODELS)
    df["model_display"] = df["model"]  # caller can apply display map
    df["dataset_display"] = df["dataset_id"].map(DATASET_DISPLAY).fillna(df["dataset_id"])
    return df


def melt_metrics(
    df: pd.DataFrame,
    metrics: Sequence[Metric] | None = None,
) -> pd.DataFrame:
    """Melt selected metric columns into long format with metadata."""
    metrics = list(metrics) if metrics is not None else list(ALL_METRICS)
    cols = [m.column for m in metrics if m.column in df.columns]
    id_cols = [
        "model", "dataset_id", "latent_space", "category",
        "is_baseline", "n_cells", "latent_dim_input",
    ]
    id_cols = [c for c in id_cols if c in df.columns]
    long = df[id_cols + cols].melt(
        id_vars=id_cols, value_vars=cols, var_name="metric", value_name="value"
    )
    reg = by_column()
    long["family"] = long["metric"].map(lambda c: reg[c].family if c in reg else "")
    long["direction"] = long["metric"].map(lambda c: reg[c].direction if c in reg else "up")
    long["short"] = long["metric"].map(lambda c: reg[c].short if c in reg else c.split(".")[-1])
    return long


def normalize_per_dataset(
    long: pd.DataFrame,
    *,
    method: str = "rank",
) -> pd.DataFrame:
    """Per (dataset_id, metric, latent_space): convert to a 0-1 score where 1 = best.

    ``rank``  : average rank divided by N (then flip if direction=down so 1 is best)
    ``zscore``: per-dataset z-score, sign-flipped for ``down`` metrics so + is good
    ``minmax``: (x - min) / (max - min); for ``down`` metrics use 1 - that
    """
    df = long.dropna(subset=["value"]).copy()
    keys = ["dataset_id", "metric", "latent_space"]

    if method == "rank":
        ranks = df.groupby(keys)["value"].rank(method="average")
        sizes = df.groupby(keys)["value"].transform("size")
        normed = ranks / sizes
        flip = df["direction"].eq("down")
        normed = np.where(flip, 1.0 - normed + 1.0 / sizes, normed)
        df["score"] = normed
    elif method == "zscore":
        mu = df.groupby(keys)["value"].transform("mean")
        sd = df.groupby(keys)["value"].transform("std").replace(0, np.nan)
        z = (df["value"] - mu) / sd
        z = np.where(df["direction"].eq("down"), -z, z)
        df["score"] = z
    elif method == "minmax":
        mn = df.groupby(keys)["value"].transform("min")
        mx = df.groupby(keys)["value"].transform("max")
        rng = (mx - mn).replace(0, np.nan)
        s = (df["value"] - mn) / rng
        s = np.where(df["direction"].eq("down"), 1.0 - s, s)
        df["score"] = s
    else:
        raise ValueError(f"unknown method {method!r}")

    return df


def aggregate_model_score(
    normed: pd.DataFrame,
    *,
    by: Sequence[str] = ("model", "latent_space"),
) -> pd.DataFrame:
    """Mean of normalized score across datasets+metrics within ``by`` groups."""
    return (
        normed.groupby(list(by))["score"].mean().reset_index().rename(columns={"score": "mean_score"})
    )


def order_models_by(score: pd.DataFrame, *, latent_space: str = "raw") -> List[str]:
    """Return models sorted by mean_score descending for the given latent_space."""
    s = score[score["latent_space"] == latent_space].sort_values("mean_score", ascending=False)
    return s["model"].tolist()


def augment_with_topk_spearman(
    df: pd.DataFrame,
    scfm_root: Path,
    out_dir: Path,
    per_pert_df: pd.DataFrame,
    *,
    k_chempert: int = 50,
    k_genepert: int = 30,
    column_name: str = "perturb.topk_spearman_vs_gt",
) -> pd.DataFrame:
    """Inject the Top-K Spearman headline metric into ``df`` (in place + return).

    For each (model, dataset_id) on ``latent_space == "raw"``, computes the
    Spearman correlation between the model's scale-normalized centroid shifts
    and the raw expression-space centroid L2 (ground truth) over the GT top-K
    perturbations. K is per-dataset (chempert/genepert defaults), capped at the
    number of available perturbations. pca128 rows receive NaN by design.
    """
    import numpy as np

    from .raw_pert_shifts import compute_raw_pert_shifts
    from .figures import (  # local import to avoid circular dep at module load
        _scale_normalize_per_pert,
        _topk_spearman_per_model,
    )

    if column_name not in df.columns:
        df[column_name] = np.nan

    raw = df[df["latent_space"] == "raw"]
    chempert_ds = sorted(set(raw[raw["category"] == "chempert"]["dataset_id"]))
    genepert_ds = sorted(set(raw[raw["category"] == "genepert"]["dataset_id"]))

    for ds_list, K in ((chempert_ds, k_chempert), (genepert_ds, k_genepert)):
        for ds in ds_list:
            gt = compute_raw_pert_shifts(
                scfm_root=scfm_root, out_dir=out_dir, dataset_id=ds,
            )
            if not gt:
                continue
            pp = _scale_normalize_per_pert(per_pert_df, out_dir=out_dir, dataset_id=ds)
            if pp.empty:
                continue
            models = sorted(pp["model"].unique())
            tbl = _topk_spearman_per_model(
                pp, gt, models=models, ks=(min(K, len(gt)),),
            )
            for _, row in tbl.iterrows():
                mask = (
                    (df["model"] == row["model"]) &
                    (df["dataset_id"] == ds) &
                    (df["latent_space"] == "raw")
                )
                df.loc[mask, column_name] = row["spearman"]
    return df


def augment_with_mantel_spearman(
    df: pd.DataFrame,
    scfm_root: Path,
    out_dir: Path,
    *,
    k: int = 10,
    column_name: str = "perturb.mantel_spearman_cos_vs_gt",
) -> pd.DataFrame:
    """Inject Mantel-Spearman of the cosine perturbation-similarity matrices
    (top-K perturbations, raw expression GT vs model latent) into ``df``.

    Computed per (model, dataset_id) for raw latent space; pca128 rows get NaN.
    Iterates over every chempert / genepert dataset present in the wide table.
    """
    import numpy as np
    from . import pert_similarity as PS

    if column_name not in df.columns:
        df[column_name] = np.nan

    raw = df[df["latent_space"] == "raw"]
    chempert_ds = sorted(set(raw[raw["category"] == "chempert"]["dataset_id"]))
    genepert_ds = sorted(set(raw[raw["category"] == "genepert"]["dataset_id"]))

    for ds in chempert_ds + genepert_ds:
        models = sorted(set(raw[raw["dataset_id"] == ds]["model"]))
        if not models:
            continue
        res = PS.compute_or_load(scfm_root, out_dir, ds, models, k=k)
        cons = res.get("consistency", {})
        for m, cs in cons.items():
            val = cs.get("mantel_spearman_cosine", np.nan)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            mask = (
                (df["model"] == m) &
                (df["dataset_id"] == ds) &
                (df["latent_space"] == "raw")
            )
            df.loc[mask, column_name] = float(val)
    return df


def per_perturb_table(scfm_root: Path) -> pd.DataFrame:
    """Build a (model, dataset, perturbation) -> L2 table from per-task summary.json.

    Used by Fig 4d (per-perturbation rank heatmap).
    """
    import json

    rows = []
    base = paths.output_root() / "metrics"
    for model_dir in base.iterdir():
        if not model_dir.is_dir() or model_dir.name not in ALL_MODELS:
            continue
        for ds_dir in model_dir.iterdir():
            is_perturb = ds_dir.name.startswith("sciplex3_") or ds_dir.name in GENEPERT_DATASETS
            if not ds_dir.is_dir() or not is_perturb:
                continue
            sp_dir = ds_dir / "raw"
            f = sp_dir / "summary.json"
            if not f.is_file():
                continue
            try:
                m = json.loads(f.read_text())
            except Exception:
                continue
            per = (
                m.get("perturb", {})
                .get("centroid_shift", {})
                .get("per_pert_l2", {})
            )
            for pert, l2 in per.items():
                rows.append(
                    dict(
                        model=model_dir.name,
                        dataset_id=ds_dir.name,
                        latent_space="raw",
                        pert=pert,
                        l2=float(l2),
                    )
                )
    return pd.DataFrame(rows)
