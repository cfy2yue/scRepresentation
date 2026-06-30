"""Per-dataset Atlas UMAP panels (GT vs model embeddings, Nature Immunity–style)."""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

from . import style as ST
FM_ROOT = Path(__file__).resolve().parents[2] / "fm"
if str(FM_ROOT) not in sys.path:
    sys.path.insert(0, str(FM_ROOT))
import paths

# benchmark ID -> Tabula Sapiens filtered h5ad in CoupledFM/data/raw/atlas_TS
ATLAS_DATASETS: Dict[str, str] = {
    "Blood": "TS_Blood_filtered.h5ad",
    "BoneMarrow": "TS_Bone_Marrow_filtered.h5ad",
    "Heart": "TS_Heart_filtered.h5ad",
    "Lung": "TS_Lung_filtered.h5ad",
    "LymphNode": "TS_Lymph_Node_filtered.h5ad",
    "Skin": "TS_Skin_filtered.h5ad",
}

DEFAULT_LABEL_COL = "cell_type"
LEGEND_MAX_ITEMS = 40
OTHER_COLOR = "#bbbbbb"


def _scfm_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _atlas_ts_dir() -> Path:
    return paths.data_root() / "raw" / "atlas_TS"


def _embeddings_root() -> Path:
    return paths.output_root() / "embeddings"


def cell_type_display(raw: str) -> str:
    """Human-readable label for legend (light touch)."""
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none"):
        return "Unknown"
    acronyms = {
        "nk cell": "NK cell",
        "t cell": "T cell",
        "b cell": "B cell",
        "tregs": "Tregs",
        "treg": "Treg",
        "cd4": "CD4",
        "cd8": "CD8",
        "dc": "DC",
    }
    lower = s.lower()
    if lower in acronyms:
        return acronyms[lower]
    t = s.replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip()
    parts = []
    for w in t.split():
        if re.fullmatch(r"cd\d+[\+-]?", w, re.I):
            parts.append(w.upper().replace("CD", "CD"))
        elif w.upper() in ("NK", "B", "T", "DC"):
            parts.append(w.upper())
        else:
            parts.append(w.capitalize())
    return " ".join(parts) if parts else s


def load_gt(
    dataset_id: str,
    *,
    label_col: str = DEFAULT_LABEL_COL,
    atlas_ts: Optional[Path] = None,
) -> Tuple[np.ndarray, pd.Series, pd.Index]:
    """GT UMAP coords (n,2), cell-type labels, obs index from Tabula Sapiens h5ad."""
    import anndata as ad

    atlas_ts = atlas_ts or _atlas_ts_dir()
    fname = ATLAS_DATASETS.get(dataset_id)
    if not fname:
        raise KeyError(f"unknown dataset_id {dataset_id!r}")
    path = atlas_ts / fname
    if not path.is_file():
        raise FileNotFoundError(path)
    adata = ad.read_h5ad(path)
    if "X_umap" not in adata.obsm:
        raise KeyError(f"No obsm['X_umap'] in {path}")
    xy = np.asarray(adata.obsm["X_umap"], dtype=np.float32)
    if label_col not in adata.obs:
        raise KeyError(f"No {label_col!r} in obs: {path}")
    labels = adata.obs[label_col].astype(str).copy()
    return xy, labels, adata.obs_names.copy()


_BARCODE_COL_CANDIDATES = ("obs_names", "cell_id", "cell_barcode", "barcode", "Unnamed: 0", "index")


def _restore_obs_index(obs: pd.DataFrame, gt_index: Optional[pd.Index]) -> pd.DataFrame:
    """If obs has a default RangeIndex but a column matches gt_index entries, promote it."""
    if gt_index is None:
        return obs
    if obs.index.equals(gt_index):
        return obs
    is_default = isinstance(obs.index, pd.RangeIndex) or obs.index.dtype.kind in "iu"
    if not is_default:
        return obs
    gt_set = set(map(str, gt_index))
    for col in _BARCODE_COL_CANDIDATES:
        if col in obs.columns:
            cand = obs[col].astype(str)
            if cand.is_unique and cand.isin(gt_set).mean() > 0.99:
                obs = obs.set_index(cand.rename(None))
                if col in obs.columns:
                    obs = obs.drop(columns=[col])
                return obs
    return obs


def load_emb(
    model: str,
    dataset_id: str,
    *,
    label_col: str = DEFAULT_LABEL_COL,
    gt_index: Optional[pd.Index] = None,
) -> Tuple[np.ndarray, pd.Series]:
    """Load latent + labels; optionally reorder to match GT obs order."""
    root = _embeddings_root() / model / dataset_id / "raw"
    lat_p = root / "latent.npy"
    obs_p = root / "obs.parquet"
    if not obs_p.is_file():
        obs_p = root / "obs.csv.gz"
    if not lat_p.is_file() or not obs_p.is_file():
        raise FileNotFoundError(f"missing {lat_p} or {obs_p}")
    z = np.asarray(np.load(lat_p), dtype=np.float32)
    if str(obs_p).endswith(".parquet"):
        obs = pd.read_parquet(obs_p)
    else:
        obs = pd.read_csv(obs_p)
    obs = _restore_obs_index(obs, gt_index)
    if label_col not in obs.columns:
        raise KeyError(f"{label_col!r} missing in obs for {model}/{dataset_id}")
    if gt_index is not None and not obs.index.equals(gt_index):
        missing = gt_index.difference(obs.index)
        if len(missing):
            raise ValueError(
                f"{model}/{dataset_id}: {len(missing)} obs missing vs GT index "
                f"(obs cols={list(obs.columns)[:8]}…)"
            )
        order = obs.index.get_indexer(gt_index)
        obs = obs.iloc[order]
        z = z[order]
    labels = obs[label_col].astype(str)
    if len(labels) != z.shape[0]:
        raise ValueError(
            f"{model}/{dataset_id}: n_obs {len(labels)} != latent {z.shape[0]}"
        )
    return z, labels


def apply_pca128(z: np.ndarray) -> np.ndarray:
    """Match run_metrics_one: StandardScaler + PCA on all cells (atlas, no ctrl col)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    z = z.astype(np.float64, copy=False)
    k = min(128, z.shape[1])
    scaler = StandardScaler().fit(z)
    pca = PCA(n_components=k, random_state=0).fit(scaler.transform(z))
    return pca.transform(scaler.transform(z)).astype(np.float32)


def compute_umap(
    z: np.ndarray,
    cache_path: Path,
    *,
    force: bool = False,
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    random_state: int = 0,
) -> np.ndarray:
    cache_path = cache_path.resolve()
    if cache_path.is_file() and not force:
        return np.load(cache_path)

    import umap

    reducer = umap.UMAP(
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=random_state,
        n_components=2,
        verbose=False,
    )
    emb = reducer.fit_transform(z.astype(np.float32, copy=False))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb.astype(np.float32))
    return emb


def _tab20_series_colors() -> List[str]:
    colors: List[str] = []
    for cmi in (plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")):
        n = cmi.N
        for j in range(n):
            colors.append(mpl.colors.to_hex(cmi(j / max(n - 1, 1))))
    return colors


def build_palette(
    labels: pd.Series,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], int]:
    """Full palette, display names, plot palette (greys rare types), n_rare."""
    counts = labels.value_counts()
    ordered = counts.index.tolist()
    colors_list = _tab20_series_colors()
    display_map: Dict[str, str] = {}
    palette: Dict[str, str] = {}
    for i, raw in enumerate(ordered):
        display_map[raw] = cell_type_display(raw)
        palette[raw] = colors_list[i % len(colors_list)]

    n_types = len(ordered)
    plot_palette = dict(palette)
    n_rare = 0
    if n_types > LEGEND_MAX_ITEMS:
        top = set(ordered[:LEGEND_MAX_ITEMS])
        for raw in ordered[LEGEND_MAX_ITEMS:]:
            plot_palette[raw] = OTHER_COLOR
            n_rare += 1

    return palette, display_map, plot_palette, n_rare


def _scatter_umap_ax(
    ax: plt.Axes,
    xy: np.ndarray,
    labels: pd.Series,
    palette: Mapping[str, str],
    *,
    n_cells: int,
    title: str = "",
    show_umap_arrows: bool = False,
) -> None:
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    s_pts = float(np.clip(12_000.0 / max(n_cells, 1), 0.6, 5.0))
    rng = np.random.default_rng(0)
    order = rng.permutation(len(xy))
    lab_arr = labels.to_numpy()[order]
    xo = xy[order, 0]
    yo = xy[order, 1]
    for raw in pd.unique(labels):
        m = lab_arr == raw
        if not np.any(m):
            continue
        c = palette.get(raw, OTHER_COLOR)
        ax.scatter(
            xo[m],
            yo[m],
            s=s_pts,
            c=c,
            alpha=0.85,
            linewidths=0,
            rasterized=True,
        )
    ax.set_aspect("equal", adjustable="datalim")
    if title:
        ax.set_title(title, fontsize=7.0, fontweight="regular", pad=2)

    if show_umap_arrows:
        _draw_umap_corner_arrows(ax)


def _draw_umap_corner_arrows(ax: plt.Axes) -> None:
    trans = ax.transAxes
    ax.add_patch(
        FancyArrowPatch(
            (0.05, 0.05),
            (0.13, 0.05),
            transform=trans,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            facecolor="#444444",
            edgecolor="#444444",
            clip_on=False,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.05, 0.05),
            (0.05, 0.13),
            transform=trans,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            facecolor="#444444",
            edgecolor="#444444",
            clip_on=False,
        )
    )
    ax.text(0.135, 0.038, "UMAP1", transform=trans, fontsize=6, color="#444444", va="top")
    ax.text(
        0.038,
        0.135,
        "UMAP2",
        transform=trans,
        fontsize=6,
        color="#444444",
        ha="right",
        rotation=90,
        va="bottom",
    )


def _legend_ax(
    ax: plt.Axes,
    labels: pd.Series,
    palette: Mapping[str, str],
    display_map: Mapping[str, str],
    *,
    n_rare: int,
) -> None:
    ax.set_axis_off()
    ax.set_facecolor("white")
    counts = labels.value_counts()
    items: List[Tuple[str, str]] = [
        (raw, display_map.get(raw, cell_type_display(raw))) for raw in counts.index
    ]
    if len(items) > LEGEND_MAX_ITEMS:
        items = items[:LEGEND_MAX_ITEMS]
        items.append(("__OTHER__", f"Other ({n_rare} types)"))
    handles = []
    for raw, disp in items:
        color = OTHER_COLOR if raw == "__OTHER__" else palette.get(raw, OTHER_COLOR)
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=disp,
                markerfacecolor=color,
                markersize=5,
                markeredgecolor="none",
                linestyle="None",
            )
        )
    ax.legend(
        handles=handles,
        loc="upper left",
        ncol=2,
        frameon=False,
        fontsize=6.5,
        labelspacing=0.45,
        columnspacing=1.6,
        handletextpad=0.5,
    )


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    svg = out_dir / f"{name}.svg"
    meta = out_dir / f"{name}.meta.json"
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    meta.write_text(
        json.dumps(
            {
                "figure": name,
                "artifacts": {"pdf": pdf.name, "png": png.name, "svg": svg.name},
                "dpi_png": 600,
                "figsize_inches": [float(x) for x in fig.get_size_inches()],
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n"
    )
    plt.close(fig)
    return pdf, png


def build_gt_figure(
    dataset_id: str,
    out_dir: Path,
    *,
    label_col: str = DEFAULT_LABEL_COL,
    atlas_ts: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Standalone GT UMAP figure for one atlas dataset (UMAP + legend)."""
    ST.apply_rcparams()
    xy_gt, labels_gt, _ = load_gt(dataset_id, label_col=label_col, atlas_ts=atlas_ts)
    palette, display_map, plot_palette, n_rare = build_palette(labels_gt)

    fig = plt.figure(figsize=(ST.mm(180), ST.mm(110)), facecolor="white")
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.05, 0.95],
                  wspace=0.06, left=0.04, right=0.99, top=0.90, bottom=0.05)
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    _scatter_umap_ax(ax_gt, xy_gt, labels_gt, plot_palette,
                     n_cells=len(labels_gt),
                     title="Tabula Sapiens (reference UMAP)",
                     show_umap_arrows=True)
    _legend_ax(ax_leg, labels_gt, palette, display_map, n_rare=n_rare)
    fig.suptitle(f"Atlas · {dataset_id}  ·  ground-truth UMAP",
                 fontsize=10, fontweight="bold", y=0.97)
    return _save(fig, out_dir, f"atlas_umap_{dataset_id}_gt")


def _grid_dims(n_panels: int) -> Tuple[int, int]:
    """4 cols, ceil(n/4) rows for a clean horizontal layout."""
    cols = 4
    rows = int(np.ceil(n_panels / cols))
    return rows, cols


def build_embedding_figure(
    dataset_id: str,
    space: str,
    out_dir: Path,
    *,
    models: Sequence[str] = ST.ALL_MODELS,
    label_col: str = DEFAULT_LABEL_COL,
    cache_dir: Optional[Path] = None,
    force_umap: bool = False,
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    max_workers: int = 1,
    atlas_ts: Optional[Path] = None,
) -> Tuple[Path, Path, List[str]]:
    """Per-model UMAP grid for one (dataset, space). Returns (pdf, png, missing_models)."""
    if space not in ("raw", "pca128"):
        raise ValueError(f"space must be 'raw' or 'pca128', got {space!r}")
    ST.apply_rcparams()
    out_dir = Path(out_dir).resolve()
    cache_dir = (cache_dir or (out_dir / "cache")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    xy_gt, labels_gt, gt_index = load_gt(dataset_id, label_col=label_col, atlas_ts=atlas_ts)
    n_cells = len(labels_gt)
    palette, display_map, plot_palette, n_rare = build_palette(labels_gt)

    payloads: Dict[str, Tuple[np.ndarray, pd.Series]] = {}
    skipped: List[Tuple[str, str]] = []
    for m in models:
        try:
            z, lab = load_emb(m, dataset_id, label_col=label_col, gt_index=gt_index)
        except Exception as exc:  # pragma: no cover - propagate via skipped list
            skipped.append((m, f"{type(exc).__name__}: {exc}"))
            continue
        if not lab.equals(labels_gt):
            skipped.append((m, "cell_type order mismatch vs GT"))
            continue
        payloads[m] = (z, lab)

    tasks: List[Tuple[str, Path]] = []
    for m in payloads:
        tasks.append((m, cache_dir / f"{dataset_id}__{m}__{space}.npy"))

    def _one(job: Tuple[str, Path]) -> Tuple[str, np.ndarray]:
        model, cpath = job
        z0, _ = payloads[model]
        zz = z0 if space == "raw" else apply_pca128(z0)
        xy = compute_umap(
            zz, cpath, force=force_umap,
            n_neighbors=n_neighbors, min_dist=min_dist,
        )
        return model, xy

    umap_xy: Dict[str, np.ndarray] = {}
    if max_workers <= 1:
        for job in tasks:
            k, xy = _one(job)
            umap_xy[k] = xy
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_one, job) for job in tasks]
            for fut in as_completed(futs):
                k, xy = fut.result()
                umap_xy[k] = xy

    models_list = list(models)
    rows, cols = _grid_dims(len(models_list))
    fig_w = ST.mm(45 * cols + 18)
    fig_h = ST.mm(45 * rows + 22)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = GridSpec(rows, cols, figure=fig,
                  hspace=0.42, wspace=0.18,
                  left=0.04, right=0.99, top=0.90, bottom=0.04)

    for j, m in enumerate(models_list):
        r, c = divmod(j, cols)
        ax = fig.add_subplot(gs[r, c])
        disp = ST.MODEL_DISPLAY.get(m, m)
        if m not in payloads or m not in umap_xy:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                    transform=ax.transAxes, color="#888888", fontsize=7)
            ax.set_title(disp, fontsize=8, pad=4)
            continue
        _, lab = payloads[m]
        _scatter_umap_ax(ax, umap_xy[m], lab, plot_palette,
                         n_cells=n_cells, title=disp)

    space_disp = "raw embedding" if space == "raw" else "PCA-128 projection"
    fig.suptitle(f"Atlas · {dataset_id}  ·  {space_disp}",
                 fontsize=10, fontweight="bold", y=0.965)

    name = f"atlas_umap_{dataset_id}_{space}"
    pdf, png = _save(fig, out_dir, name)
    missing = [f"{m} ({why})" for m, why in skipped]
    return pdf, png, missing


def build_dataset_figures(
    dataset_id: str,
    out_dir: Path,
    *,
    models: Sequence[str] = ST.ALL_MODELS,
    label_col: str = DEFAULT_LABEL_COL,
    cache_dir: Optional[Path] = None,
    force_umap: bool = False,
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    max_workers: int = 1,
    atlas_ts: Optional[Path] = None,
) -> Dict[str, object]:
    """Build all 3 figures (gt, raw, pca128) for one dataset."""
    pdf_gt, png_gt = build_gt_figure(
        dataset_id, out_dir, label_col=label_col, atlas_ts=atlas_ts,
    )
    pdf_r, png_r, miss_r = build_embedding_figure(
        dataset_id, "raw", out_dir,
        models=models, label_col=label_col, cache_dir=cache_dir,
        force_umap=force_umap, n_neighbors=n_neighbors, min_dist=min_dist,
        max_workers=max_workers, atlas_ts=atlas_ts,
    )
    pdf_p, png_p, miss_p = build_embedding_figure(
        dataset_id, "pca128", out_dir,
        models=models, label_col=label_col, cache_dir=cache_dir,
        force_umap=force_umap, n_neighbors=n_neighbors, min_dist=min_dist,
        max_workers=max_workers, atlas_ts=atlas_ts,
    )
    return {
        "dataset": dataset_id,
        "gt": {"pdf": str(pdf_gt), "png": str(png_gt)},
        "raw": {"pdf": str(pdf_r), "png": str(png_r), "skipped": miss_r},
        "pca128": {"pdf": str(pdf_p), "png": str(png_p), "skipped": miss_p},
    }


# Legacy combined-figure builder removed; use ``build_dataset_figures`` instead.
