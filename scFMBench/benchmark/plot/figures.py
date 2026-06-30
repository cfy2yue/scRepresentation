"""Figure builders for the scFM benchmark.

Each ``fig_*`` function returns a saved PDF/PNG path pair.

Story:
- Fig 1  Overview          (headline rank + scoreboard + family leaderboard)
- Fig 2  Atlas integration (NMI/iLISI tradeoff + per-metric strip + best-of)
- Fig 3  Latent geometry   (PR + isotropy + raw-vs-pca128 paired)
- Fig 4  Chempert          (centroid shift + EMD + cross-cell-line)
- Fig 5  Robustness        (rank stability + dim sensitivity + heatmap)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec

from . import metrics as M
from . import style as ST
from .data import (
    DATASET_DISPLAY,
    GENEPERT_DATASETS,
    aggregate_model_score,
    melt_metrics,
    normalize_per_dataset,
    order_models_by,
    per_perturb_table,
)
from .raw_pert_shifts import compute_raw_pert_shifts
from . import pert_similarity as PS


LOG = logging.getLogger(__name__)
_MISSING_LATENT_WARNED: set[str] = set()


def _three_pillar_balanced(
    rows: pd.DataFrame,
    *,
    restrict_atlas_geo_to: Tuple[str, ...] | None = None,
    headline_only: bool = True,
) -> float:
    """Mean of atlas pillar, geometry pillar, and perturb (chem/gene averaged).

    Perturb pillar = mean(chempert mean, genepert mean) when both exist.

    ``restrict_atlas_geo_to``: if given, atlas/geometry pillars only average rows
    with those dataset categories (e.g. atlas + atlas_TS for full-metric mode).

    ``headline_only``: when True (default), restrict every pillar to
    ``HEADLINE_*`` metrics so old per-task surrogates (mean L2, EMD, xCellLine
    L2/EMD) do not dilute the new perturbation-faithfulness headline metrics
    (Top-K Spearman, Mantel ρ_S). Set False to revert to the historical
    "all metrics" aggregation.
    """
    if headline_only:
        head_atlas = set(M.HEADLINE_ATLAS)
        head_geo = set(M.HEADLINE_GEOMETRY)
        head_pert = set(M.HEADLINE_PERTURB)
        rows = rows[rows["metric"].isin(head_atlas | head_geo | head_pert)]

    if restrict_atlas_geo_to is None:
        atlas_mask = rows["family"].eq("atlas")
        geo_mask = rows["family"].eq("geometry")
    else:
        bc = rows["category"].isin(restrict_atlas_geo_to)
        atlas_mask = bc & rows["family"].eq("atlas")
        geo_mask = bc & rows["family"].eq("geometry")
    pert = rows[rows["family"].eq("perturb")]
    pchem = pert[pert["category"].eq("chempert")]["score"].mean()
    pgene = pert[pert["category"].eq("genepert")]["score"].mean()
    if pd.notna(pchem) and pd.notna(pgene):
        perts = 0.5 * (float(pchem) + float(pgene))
    elif pd.notna(pchem):
        perts = float(pchem)
    elif pd.notna(pgene):
        perts = float(pgene)
    else:
        perts = np.nan
    vals: list[float] = []
    if atlas_mask.any():
        v = rows.loc[atlas_mask, "score"].mean()
        if pd.notna(v):
            vals.append(float(v))
    if geo_mask.any():
        v = rows.loc[geo_mask, "score"].mean()
        if pd.notna(v):
            vals.append(float(v))
    if pd.notna(perts):
        vals.append(perts)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _balanced_overall_from_normed(normed: pd.DataFrame, *, latent_space: str) -> pd.DataFrame:
    """Per-model balanced overall (atlas+atlas_TS geometry, chem+gene perturb equally)."""
    sub = normed[normed["latent_space"] == latent_space]
    rows: list[dict] = []
    for m in sorted(sub["model"].unique(), key=lambda x: ST.ALL_MODELS.index(x) if x in ST.ALL_MODELS else 99):
        r = sub[sub["model"] == m]
        s = _three_pillar_balanced(r, restrict_atlas_geo_to=("atlas", "atlas_TS"))
        if np.isnan(s):
            continue
        rows.append(dict(model=m, mean_score=s))
    return pd.DataFrame(rows)


def _headline_balanced_agg(normed: pd.DataFrame) -> pd.DataFrame:
    """Mean rank score per (model, latent_space) for headline Fig.1 (balanced perturb)."""
    rows_out: list[dict] = []
    for ls in sorted(normed["latent_space"].unique()):
        sub = normed[normed["latent_space"] == ls]
        for m in ST.ALL_MODELS:
            if m not in set(sub["model"]):
                continue
            r = sub[sub["model"] == m]
            ms = _three_pillar_balanced(r, restrict_atlas_geo_to=None)
            if np.isnan(ms):
                continue
            rows_out.append(dict(model=m, latent_space=ls, mean_score=ms))
    return pd.DataFrame(rows_out)


# ----------------------- low-level helpers --------------------------------

def _save(fig: plt.Figure, out_dir: Path, name: str) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    svg = out_dir / f"{name}.svg"
    meta = out_dir / f"{name}.meta.json"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    fig.savefig(svg)
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


def _scfm_root_from_out_dir(out_dir: Path) -> Path:
    """Resolve ``.../<output-root>/figures`` back to the scFM delivery root."""
    return out_dir.resolve().parents[1]


def _embeddings_root_from_out_dir(out_dir: Path) -> Path:
    """Resolve the embedding root that shares the same output root as figures."""
    return out_dir.resolve().parent / "embeddings"


def _latent_norm_cache_path(out_dir: Path) -> Path:
    return out_dir / "_cache_latent_norms.json"


def _load_latent_norm_cache(out_dir: Path) -> dict[str, float]:
    path = _latent_norm_cache_path(out_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        LOG.warning("Ignoring unreadable latent norm cache %s: %s", path, exc)
        return {}
    return {str(k): float(v) for k, v in data.items() if pd.notna(v)}


def _write_latent_norm_cache(out_dir: Path, cache: dict[str, float]) -> None:
    path = _latent_norm_cache_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _median_cell_l2_norm(
    *,
    scfm_root: Path,
    out_dir: Path,
    model: str,
    dataset_id: str,
    cache: dict[str, float],
    persist_cache: bool = True,
) -> float:
    key = f"{model}/{dataset_id}"
    if key in cache:
        return float(cache[key])
    latent_path = _embeddings_root_from_out_dir(out_dir) / model / dataset_id / "raw" / "latent.npy"
    if not latent_path.is_file():
        warn_key = str(latent_path)
        if warn_key not in _MISSING_LATENT_WARNED:
            LOG.warning("Missing latent for norm cache: %s", latent_path)
            _MISSING_LATENT_WARNED.add(warn_key)
        return float("nan")
    Z = np.load(latent_path, mmap_mode="r")
    norms = np.linalg.norm(Z, axis=1)
    val = float(np.median(norms))
    cache[key] = val
    if persist_cache:
        _write_latent_norm_cache(out_dir, cache)
    return val


def _add_scale_normalized_perturb_columns(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Divide raw perturb distances by per-run median per-cell L2 norm."""
    out = df.copy()
    scfm_root = _scfm_root_from_out_dir(out_dir)
    cache = _load_latent_norm_cache(out_dir)
    norm_map: dict[tuple[str, str], float] = {}
    for model, dataset_id in out[["model", "dataset_id"]].drop_duplicates().itertuples(index=False):
        norm_map[(model, dataset_id)] = _median_cell_l2_norm(
            scfm_root=scfm_root,
            out_dir=out_dir,
            model=str(model),
            dataset_id=str(dataset_id),
            cache=cache,
            persist_cache=False,
        )
    _write_latent_norm_cache(out_dir, cache)
    run_norm = out.apply(lambda r: norm_map.get((r["model"], r["dataset_id"]), np.nan), axis=1)
    for col in (
        "perturb.centroid_shift.mean_l2_to_control",
        "perturb.centroid_shift.median_l2_to_control",
        "perturb.ot_summary.emd_mean",
        "perturb.ot_summary.emd_median",
    ):
        if col in out.columns:
            out[f"{col}.scale_norm"] = out[col] / run_norm.replace(0, np.nan)
    return out


def _scale_normalize_per_pert(
    per_pert_df: pd.DataFrame,
    *,
    out_dir: Path,
    dataset_id: str,
) -> pd.DataFrame:
    """Add ``l2_scale_norm = l2 / median ||z||`` to a slice of ``per_perturb_table``."""
    sub = per_pert_df[per_pert_df["dataset_id"] == dataset_id].copy()
    if sub.empty:
        sub["l2_scale_norm"] = []
        return sub
    scfm_root = _scfm_root_from_out_dir(out_dir)
    cache = _load_latent_norm_cache(out_dir)
    sub["median_cell_l2_norm"] = [
        _median_cell_l2_norm(
            scfm_root=scfm_root,
            out_dir=out_dir,
            model=str(m),
            dataset_id=str(d),
            cache=cache,
            persist_cache=False,
        )
        for m, d in sub[["model", "dataset_id"]].itertuples(index=False)
    ]
    _write_latent_norm_cache(out_dir, cache)
    sub["l2_scale_norm"] = sub["l2"] / sub["median_cell_l2_norm"].replace(0, np.nan)
    return sub


def _topk_spearman_per_model(
    per_pert_scaled: pd.DataFrame,
    gt_shifts: Dict[str, float],
    *,
    models: Sequence[str],
    ks: Sequence[int] = (10, 20, 50),
    score_col: str = "l2_scale_norm",
) -> pd.DataFrame:
    """For each (model, K), Spearman between the model's scale-norm L2 ranking
    of the **GT top-K** perturbations and the GT raw-expression shift ranking.

    Returns long DF with columns ``model, K, spearman, n_used``.
    """
    from scipy.stats import spearmanr

    if not gt_shifts:
        return pd.DataFrame(columns=["model", "K", "spearman", "n_used"])

    gt = pd.Series(gt_shifts, name="gt").astype(float).sort_values(ascending=False)

    rows: list[dict] = []
    for K in ks:
        topk = gt.head(min(K, len(gt))).index.tolist()
        gt_vals = gt.loc[topk].values
        for m in models:
            sub = per_pert_scaled[per_pert_scaled["model"] == m]
            if sub.empty:
                continue
            d = dict(zip(sub["pert"].astype(str), sub[score_col]))
            model_vals = np.array([d.get(p, np.nan) for p in topk], dtype=float)
            ok = ~np.isnan(model_vals)
            if ok.sum() < 4:
                rows.append(dict(model=m, K=int(K), spearman=np.nan,
                                 n_used=int(ok.sum())))
                continue
            rho, _ = spearmanr(gt_vals[ok], model_vals[ok])
            rows.append(dict(model=m, K=int(K), spearman=float(rho),
                             n_used=int(ok.sum())))
    return pd.DataFrame(rows)


def _topk_spearman_per_model_multi(
    per_pert_scaled: pd.DataFrame,
    gt_per_dataset: Dict[str, Dict[str, float]],
    *,
    models: Sequence[str],
    ks: Sequence[int],
    score_col: str = "l2_scale_norm",
) -> pd.DataFrame:
    """Per-dataset Spearman, returned long with column ``dataset_id``."""
    parts = []
    for ds, gt in gt_per_dataset.items():
        sub = per_pert_scaled[per_pert_scaled["dataset_id"] == ds]
        if not gt or sub.empty:
            continue
        t = _topk_spearman_per_model(sub, gt, models=models, ks=ks, score_col=score_col)
        t["dataset_id"] = ds
        parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["model", "K", "spearman", "n_used", "dataset_id"]
    )


def _grouped_bars_spearman(
    ax: plt.Axes,
    table: pd.DataFrame,
    *,
    models: Sequence[str],
    ks: Sequence[int],
    ylabel: str = "Spearman ρ (top-K vs raw GT)",
    points_df: pd.DataFrame | None = None,
) -> None:
    """Grouped vertical bars: x = model, hue = K. Bars are colored by model
    palette with hatch / alpha differentiating K. Empty cells render as gaps.
    """
    n_k = len(ks)
    width = 0.78 / max(n_k, 1)
    alphas = np.linspace(0.45, 0.95, n_k)
    hatches = ["", "//", "xx", ".."][:n_k]
    x = np.arange(len(models))

    for j, K in enumerate(ks):
        offsets = (j - (n_k - 1) / 2) * width
        vals = []
        for m in models:
            r = table[(table["model"] == m) & (table["K"] == K)]["spearman"]
            vals.append(float(r.iloc[0]) if len(r) and pd.notna(r.iloc[0]) else np.nan)
        bars = ax.bar(
            x + offsets, vals, width=width * 0.95,
            color=[mcolors.to_rgba(ST.MODEL_PALETTE[m], float(alphas[j])) for m in models],
            edgecolor="black", linewidth=0.35, hatch=hatches[j],
            label=f"K={K}",
        )
        if points_df is not None and len(points_df):
            rng = np.random.default_rng(42 + j)
            for i, m in enumerate(models):
                pts = points_df[(points_df["model"] == m) & (points_df["K"] == K)]
                yv = pts["spearman"].dropna().to_numpy()
                if not len(yv):
                    continue
                jitter = rng.uniform(-width * 0.32, width * 0.32, size=len(yv))
                ax.scatter(
                    np.full(len(yv), x[i] + offsets) + jitter, yv,
                    s=4.5, color="black", alpha=0.65, linewidths=0, zorder=3,
                )

    ax.axhline(0, color="grey", lw=0.4, ls="--", alpha=0.6)
    _model_axis(ax, models, axis="x")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.4, 1.05)
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    ax.legend(fontsize=5.5, loc="lower right", frameon=False, ncol=n_k,
              handlelength=1.2, handletextpad=0.4, columnspacing=0.8)


def _label_scatter_no_overlap(ax: plt.Axes, *, xs, ys, labels,
                              log_x: bool = False, log_y: bool = False,
                              fontsize: float = 6.0) -> None:
    """Place labels around scatter points minimising overlap.

    Greedy approach: try a set of offset directions for each point and pick
    the first that does not collide (approximately) with already placed labels
    in screen pixel space.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed_bboxes = []
    # candidate offsets in points (dx, dy, ha, va)
    candidates = [
        (8, 0, "left", "center"),
        (-8, 0, "right", "center"),
        (8, 8, "left", "bottom"),
        (-8, 8, "right", "bottom"),
        (8, -8, "left", "top"),
        (-8, -8, "right", "top"),
        (0, 12, "center", "bottom"),
        (0, -12, "center", "top"),
        (14, 4, "left", "center"),
        (-14, 4, "right", "center"),
        (14, -4, "left", "center"),
        (-14, -4, "right", "center"),
    ]
    for x, y, lab in zip(xs, ys, labels):
        chosen = None
        for dx, dy, ha, va in candidates:
            txt = ax.annotate(lab, (x, y),
                              xytext=(dx, dy), textcoords="offset points",
                              fontsize=fontsize, ha=ha, va=va,
                              arrowprops=dict(arrowstyle="-", lw=0.3, color="#888"))
            fig.canvas.draw()
            bb = txt.get_window_extent(renderer=renderer)
            collide = any(bb.overlaps(b) for b in placed_bboxes)
            if collide:
                txt.remove()
                continue
            placed_bboxes.append(bb)
            chosen = txt
            break
        if chosen is None:
            ax.annotate(lab, (x, y), xytext=(8, 0), textcoords="offset points",
                        fontsize=fontsize, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.3, color="#888"))


def _model_axis(ax: plt.Axes, models: Sequence[str], *, axis: str = "x") -> None:
    labels = [ST.MODEL_DISPLAY.get(m, m) for m in models]
    if axis == "x":
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
    else:
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(labels)


def _stripbox(
    ax: plt.Axes,
    long: pd.DataFrame,
    *,
    models: Sequence[str],
    value_col: str = "score",
    show_baseline_band: bool = True,
) -> None:
    data = [long.loc[long["model"] == m, value_col].dropna().values for m in models]
    bp = ax.boxplot(
        data,
        positions=range(len(models)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=0.8),
        whiskerprops=dict(linewidth=0.5),
        capprops=dict(linewidth=0.5),
    )
    for patch, m in zip(bp["boxes"], models):
        c = ST.MODEL_PALETTE[m]
        patch.set_facecolor(mcolors.to_rgba(c, 0.35))
        patch.set_edgecolor(c)
        patch.set_linewidth(0.8)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        if len(vals) == 0:
            continue
        x = i + (rng.uniform(-0.18, 0.18, size=len(vals)))
        ax.scatter(x, vals, s=4, color=ST.MODEL_PALETTE[models[i]], alpha=0.85, linewidths=0)

    if show_baseline_band:
        # baseline reference band: PCA + scVI median across all points
        ref = long[long["model"].isin(ST.BASELINE_MODELS)][value_col].dropna()
        if len(ref):
            ax.axhspan(ref.quantile(0.25), ref.quantile(0.75),
                       color="grey", alpha=0.10, lw=0, zorder=0)
            ax.axhline(ref.median(), color="grey", lw=0.5, ls="--", alpha=0.6, zorder=0)

    _model_axis(ax, models, axis="x")
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)


def _heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
    cbar_fraction: float = 0.025,
    annotate: bool = False,
) -> None:
    if vmin is None or vmax is None:
        absmax = float(np.nanmax(np.abs(matrix.values)))
        vmin = vmin if vmin is not None else -absmax
        vmax = vmax if vmax is not None else +absmax
    im = ax.imshow(matrix.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([ST.MODEL_DISPLAY.get(m, m) for m in matrix.index])
    if annotate:
        midpoint = 0.5 * (vmin + vmax)
        half = max(1e-9, 0.5 * (vmax - vmin))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix.values[i, j]
                if np.isnan(v):
                    continue
                # near-mid (light cell on diverging cmap) -> black, extremes -> white
                rel = abs(v - midpoint) / half
                color = "white" if rel > 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5, color=color)
    cb = plt.colorbar(im, ax=ax, fraction=cbar_fraction, pad=0.02)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6, length=2, width=0.4)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=6.5)


def _legend_models(fig: plt.Figure, models: Sequence[str], *,
                   loc: str = "lower center", bbox: Tuple[float, float] = (0.5, -0.02),
                   ncol: int | None = None) -> None:
    handles = [
        mpatches.Patch(facecolor=ST.MODEL_PALETTE[m], edgecolor=ST.MODEL_PALETTE[m],
                       label=ST.MODEL_DISPLAY.get(m, m))
        for m in models
    ]
    if ncol is None:
        ncol = min(len(models), 11)
    fig.legend(handles=handles, loc=loc, bbox_to_anchor=bbox, ncol=ncol, frameon=False,
               fontsize=6.5, handlelength=1.2, handletextpad=0.5, columnspacing=1.0)


# ============================ FIG 1: Overview =============================

def fig1_overview(df: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path]:
    """Hero figure: headline rank, metric scoreboard, and raw leaderboard."""
    headline = M.HEADLINE_ATLAS + M.HEADLINE_GEOMETRY + M.HEADLINE_PERTURB
    reg = M.by_column()
    selected = [reg[c] for c in headline if c in reg]
    long = melt_metrics(df, selected)
    normed = normalize_per_dataset(long, method="rank")

    agg = _headline_balanced_agg(normed)

    order = order_models_by(agg, latent_space="raw")

    fig = plt.figure(figsize=(ST.mm(180), ST.mm(135)))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[0.92, 1.12],
                  hspace=1.05, wspace=0.78,
                  left=0.10, right=0.985, top=0.93, bottom=0.22)

    # (a) raw vs pca128 mean-score paired bar ---------------------------
    ax_a = fig.add_subplot(gs[0, :])
    _paired_bar_score(ax_a, agg, models=order)
    ax_a.set_title("a  Mean rank-score across all metrics (per latent space)", loc="left", pad=14)

    # (b) headline metric heatmap (model x metric, raw) ------------------
    ax_b = fig.add_subplot(gs[1, :2])
    mat = (
        normed[normed["latent_space"] == "raw"]
        .pivot_table(index="model", columns="short", values="score", aggfunc="mean")
        .reindex(order)
    )
    short_order = [reg[c].short for c in headline if c in reg]
    mat = mat.reindex(columns=[s for s in short_order if s in mat.columns])
    _heatmap(ax_b, mat, cmap="RdBu_r", vmin=0.0, vmax=1.0,
             cbar_label="rank score (1 = best)", annotate=True, cbar_fraction=0.012)
    # family separators + ribbon below x-axis
    fam_of = {reg[c].short: reg[c].family for c in headline if c in reg}
    fam_colors = {"atlas": "#1F77B4", "geometry": "#2CA02C", "perturb": "#E15759"}
    last_fam = None
    for j, short in enumerate(mat.columns):
        fam = fam_of.get(short, "")
        if last_fam is not None and fam != last_fam:
            ax_b.axvline(j - 0.5, color="white", lw=1.4)
            ax_b.axvline(j - 0.5, color="black", lw=0.5)
        last_fam = fam
    # color x-tick labels by family (atlas / geometry / perturb)
    for tick_label, short in zip(ax_b.get_xticklabels(), mat.columns):
        fam = fam_of.get(short, "")
        tick_label.set_color(fam_colors.get(fam, "#444"))
        tick_label.set_fontweight("bold")
    # inline family legend beneath title
    fam_legend_handles = [
        mpatches.Patch(facecolor=fam_colors[k],
                       label={"atlas": "Atlas integration",
                              "geometry": "Latent geometry",
                              "perturb": "Perturbation"}[k])
        for k in ("atlas", "geometry", "perturb")
    ]
    ax_b.legend(
        handles=fam_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=3,
        frameon=False,
        fontsize=6.5,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.4,
    )
    ax_b.set_title(
        "b  Headline metric scoreboard (raw embedding)",
        loc="left",
        pad=16,
    )
    ax_b.set_xlabel("")
    ax_b.set_ylabel("")
    ax_b.tick_params(axis="x", labelsize=5.5, rotation=45)

    # (c) family leaderboard, raw embedding -----------------------------
    ax_c = fig.add_subplot(gs[1, 2])
    _family_leaderboard(ax_c, normed, models=order)
    ax_c.set_title("c  Leaderboard (raw embedding)", loc="left", pad=14)
    ax_c.tick_params(axis="x", pad=8)

    return _save(fig, out_dir, "fig1_overview")


def _paired_bar_score(ax: plt.Axes, agg: pd.DataFrame, models: Sequence[str]) -> None:
    width = 0.35
    x = np.arange(len(models))
    raw_vals = [
        agg.loc[(agg["model"] == m) & (agg["latent_space"] == "raw"), "mean_score"].mean()
        for m in models
    ]
    pca_vals = [
        agg.loc[(agg["model"] == m) & (agg["latent_space"] == "pca128"), "mean_score"].mean()
        for m in models
    ]
    ax.bar(x - width / 2, raw_vals, width, color=ST.LATENT_PALETTE["raw"],
           alpha=0.85, edgecolor="white", linewidth=0.4, label=ST.LATENT_DISPLAY["raw"])
    ax.bar(x + width / 2, pca_vals, width, color=ST.LATENT_PALETTE["pca128"],
           alpha=0.85, edgecolor="white", linewidth=0.4, label=ST.LATENT_DISPLAY["pca128"])
    labels = [ST.MODEL_DISPLAY.get(m, m) for m in models]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.0)
    # push xlim left so rotated labels at x=0 aren't clipped
    ax.set_xlim(-0.65, len(models) - 0.5)
    ax.set_ylabel("mean rank score (higher = better)")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="grey", lw=0.5, ls=":", alpha=0.6)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.99, 0.97),
        ncol=1,
        fontsize=6.8,
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        handletextpad=0.4,
        borderpad=0.25,
        labelspacing=0.25,
    )
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)


def _family_leaderboard(ax: plt.Axes, normed: pd.DataFrame, models: Sequence[str]) -> None:
    raw = normed[normed["latent_space"] == "raw"]
    fams = ["atlas", "geometry", "perturb"]
    fam_colors = {"atlas": "#1F77B4", "geometry": "#2CA02C", "perturb": "#E15759"}
    atlas = raw[raw["family"] == "atlas"].groupby("model")["score"].mean()
    geometry = raw[raw["family"] == "geometry"].groupby("model")["score"].mean()
    pert = raw[raw["family"] == "perturb"]
    pchem = pert[pert["category"] == "chempert"].groupby("model")["score"].mean()
    pgene = pert[pert["category"] == "genepert"].groupby("model")["score"].mean()
    perturb_bal = pd.Series(index=list(models), dtype=float)
    for m in models:
        a_, b_ = pchem.get(m, np.nan), pgene.get(m, np.nan)
        if pd.notna(a_) and pd.notna(b_):
            perturb_bal[m] = 0.5 * (float(a_) + float(b_))
        elif pd.notna(a_):
            perturb_bal[m] = float(a_)
        elif pd.notna(b_):
            perturb_bal[m] = float(b_)
        else:
            perturb_bal[m] = np.nan
    mat = pd.DataFrame(
        dict(
            atlas=atlas.reindex(models),
            geometry=geometry.reindex(models),
            perturb=perturb_bal.reindex(models),
        )
    ).fillna(0.0)
    x = np.arange(len(models))
    bottom = np.zeros(len(models), dtype=float)
    for fam in fams:
        vals = mat[fam].to_numpy()
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=fam_colors[fam],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.3,
            label={"atlas": "Atlas", "geometry": "Geometry", "perturb": "Perturbation"}[fam],
        )
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([ST.MODEL_DISPLAY.get(m, m) for m in models], rotation=45, ha="right", fontsize=6.0)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylabel("summed family rank score")
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        ncol=1,
        fontsize=5.8,
        frameon=True,
        framealpha=0.9,
        edgecolor="#dddddd",
        handlelength=1.0,
        handletextpad=0.35,
        labelspacing=0.2,
    )


# ============================ FIG 2: Atlas ================================

def fig2_atlas(df: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path]:
    reg = M.by_column()
    atlas_metrics = [m for m in M.ATLAS_METRICS if m.column in df.columns]
    long = melt_metrics(df, atlas_metrics)
    normed = normalize_per_dataset(long, method="rank")
    agg = aggregate_model_score(normed[normed["family"] == "atlas"], by=("model", "latent_space"))
    order = order_models_by(agg, latent_space="raw")

    fig = plt.figure(figsize=(ST.mm(180), ST.mm(140)))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.05],
                  width_ratios=[1, 1], hspace=0.62, wspace=0.42,
                  left=0.07, right=0.97, top=0.93, bottom=0.13)

    # (a) NMI vs iLISI bio-batch tradeoff (raw) ---------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    _scatter_tradeoff(
        ax_a,
        df[df["latent_space"] == "raw"],
        x_col="atlas.A1_nmi",
        y_col="atlas.A3_ilisi",
        x_label="NMI (cell-type cluster agreement)",
        y_label="iLISI (batch mixing)",
    )
    ax_a.set_title("a  Bio-batch trade-off (raw)", loc="left", pad=10)

    # (b) per-metric strip+box (raw) -------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    sub = normed[(normed["latent_space"] == "raw") & (normed["family"] == "atlas")]
    _stripbox(ax_b, sub, models=order)
    ax_b.set_ylabel("rank score (across atlas metrics)")
    ax_b.set_title("b  Atlas score distribution per model", loc="left")
    ax_b.set_ylim(0, 1.05)

    # (c) heatmap model x dataset (mean atlas rank score, raw) -----------
    ax_c = fig.add_subplot(gs[1, 0])
    mat = (
        sub.groupby(["model", "dataset_id"])["score"]
        .mean()
        .unstack("dataset_id")
        .reindex(order)
    )
    mat.columns = [DATASET_DISPLAY.get(c, c) for c in mat.columns]
    _heatmap(ax_c, mat, cmap="RdBu_r", vmin=0, vmax=1,
             cbar_label="mean atlas rank score", annotate=True)
    ax_c.set_title("c  Per-dataset atlas score (raw)", loc="left", pad=12)

    # (d) raw vs pca128 paired arrow per model ---------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    _paired_dot(ax_d, agg, models=order, ylabel="atlas rank score")
    ax_d.set_title("d  Raw vs PCA-128 (atlas)", loc="left")

    return _save(fig, out_dir, "fig2_atlas")


def _scatter_tradeoff(ax: plt.Axes, df: pd.DataFrame, *,
                      x_col: str, y_col: str, x_label: str, y_label: str) -> None:
    sub = df[[x_col, y_col, "model", "dataset_id"]].dropna()
    for m, sd in sub.groupby("model"):
        ax.scatter(sd[x_col], sd[y_col], color=ST.MODEL_PALETTE[m],
                   s=10, alpha=0.40, edgecolor="white", linewidth=0.2)
    centroids = (
        sub.groupby("model")[[x_col, y_col]].mean()
        .sort_values(x_col)
    )
    # numbered circles for centroids to avoid label overlap
    for i, (m, row) in enumerate(centroids.iterrows(), start=1):
        ax.scatter(
            row[x_col],
            row[y_col],
            color=mcolors.to_rgba(ST.MODEL_PALETTE[m], 0.78),
            s=110,
            edgecolor="black",
            linewidth=0.65,
            zorder=5,
        )
        ax.text(
            row[x_col],
            row[y_col],
            str(i),
            ha="center",
            va="center",
            fontsize=6.0,
            fontweight="bold",
            color="white",
            alpha=1.0,
            zorder=6,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, lw=0.3, alpha=0.4)
    # legend
    handles = [
        plt.Line2D([], [], marker="o", lw=0,
                   markerfacecolor=ST.MODEL_PALETTE[m],
                   markeredgecolor="black", markeredgewidth=0.5,
                   markersize=6,
                   label=f"{i}  {ST.MODEL_DISPLAY.get(m, m)}")
        for i, m in enumerate(centroids.index, start=1)
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.96),
        fontsize=5.2,
        frameon=False,
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.6,
        borderaxespad=0.35,
    )


def _paired_dot(ax: plt.Axes, agg: pd.DataFrame, models: Sequence[str], ylabel: str) -> None:
    right_vals = []
    for m in models:
        a = agg.loc[(agg["model"] == m) & (agg["latent_space"] == "raw"), "mean_score"].mean()
        b = agg.loc[(agg["model"] == m) & (agg["latent_space"] == "pca128"), "mean_score"].mean()
        c = ST.MODEL_PALETTE[m]
        ax.plot([0, 1], [a, b], color=c, alpha=0.7, lw=1.0)
        ax.scatter([0, 1], [a, b], color=c, s=22, edgecolor="black", linewidth=0.4, zorder=4)
        right_vals.append((b, m, c))
    # stagger labels on the right to avoid overlap
    right_vals.sort(key=lambda x: x[0])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["raw", "PCA-128"])
    ax.set_xlim(-0.08, 1.45)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    if right_vals:
        ymin, ymax = ax.get_ylim()
        slot_min = ymin + 0.05 * (ymax - ymin)
        slot_max = ymax - 0.05 * (ymax - ymin)
        slots = np.linspace(slot_min, slot_max, len(right_vals))
        for slot_y, (_, m, c) in zip(slots, right_vals):
            ax.text(1.06, slot_y, ST.MODEL_DISPLAY.get(m, m),
                    ha="left", va="center", fontsize=5.4,
                    color=c, fontweight="semibold")


# ============================ FIG 3: Geometry =============================

def fig3_geometry(df: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path]:
    fig = plt.figure(figsize=(ST.mm(130), ST.mm(125)))
    gs = GridSpec(2, 2, figure=fig, hspace=0.6, wspace=0.5,
                  left=0.10, right=0.98, top=0.92, bottom=0.16)

    raw = df[df["latent_space"] == "raw"]
    order = sorted(df["model"].unique(), key=lambda m: ST.ALL_MODELS.index(m))

    # (a) participation ratio (raw, all datasets pooled)
    ax_a = fig.add_subplot(gs[0, 0])
    _model_value_box(ax_a, raw, "geometry.G1_participation_ratio",
                     models=order, ylabel="Participation ratio (raw)")
    ax_a.set_title("a  Effective dimensionality", loc="left")

    # (b) anisotropy lambda_max / trace (raw, lower better)
    ax_b = fig.add_subplot(gs[0, 1])
    _model_value_box(ax_b, raw, "geometry.G3_lambda_max_over_trace",
                     models=order, ylabel="λ$_{max}$ / trace  (lower = isotropic)",
                     log_y=False)
    ax_b.set_title("b  Anisotropy", loc="left")

    # (c) kNN-LC boxplot (raw)
    ax_c = fig.add_subplot(gs[1, 0])
    _model_value_box(ax_c, raw, "geometry.G2_knn_label_consistency",
                     models=order, ylabel="kNN label consistency (raw)")
    ax_c.set_title("c  kNN label consistency", loc="left")

    # (d) LDM proxy composite per model (raw)
    ax_d = fig.add_subplot(gs[1, 1])
    _model_value_box(ax_d, raw, "geometry.LDM_proxy_score",
                     models=order, ylabel="LDM proxy (composite)")
    ax_d.set_title("d  LDM proxy score", loc="left")

    return _save(fig, out_dir, "fig3_geometry")


def _model_value_box(
    ax: plt.Axes,
    df: pd.DataFrame,
    column: str,
    *,
    models: Sequence[str],
    ylabel: str,
    log_y: bool = False,
) -> None:
    data = [df.loc[df["model"] == m, column].dropna().values for m in models]
    bp = ax.boxplot(
        data, positions=range(len(models)), widths=0.55, patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=0.8),
        whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5),
    )
    for patch, m in zip(bp["boxes"], models):
        c = ST.MODEL_PALETTE[m]
        patch.set_facecolor(mcolors.to_rgba(c, 0.35))
        patch.set_edgecolor(c)
        patch.set_linewidth(0.8)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        if len(vals) == 0:
            continue
        x = i + rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(x, vals, s=4, color=ST.MODEL_PALETTE[models[i]], alpha=0.85, linewidths=0)
    _model_axis(ax, models, axis="x")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)


def _raw_pca_paired(
    ax: plt.Axes,
    df: pd.DataFrame,
    column: str,
    models: Sequence[str],
    ylabel: str,
) -> None:
    """Paired dot-plot showing per-model *mean* across datasets (raw vs PCA-128)."""
    sub = df[["model", "latent_space", column]].dropna()
    # aggregate to one value per (model, latent_space)
    means = sub.groupby(["model", "latent_space"])[column].mean().unstack("latent_space")
    means = means.reindex(models)

    right_vals: list[tuple[float, str]] = []
    for m in models:
        if m not in means.index:
            continue
        rv = float(means.loc[m, "raw"]) if "raw" in means.columns and not np.isnan(means.loc[m, "raw"]) else None
        pv = float(means.loc[m, "pca128"]) if "pca128" in means.columns and not np.isnan(means.loc[m, "pca128"]) else None
        if rv is None and pv is None:
            continue
        color = ST.MODEL_PALETTE[m]
        if rv is not None and pv is not None:
            ax.plot([0, 1], [rv, pv], color=color, lw=1.2, alpha=0.80, zorder=2)
        if rv is not None:
            ax.scatter([0], [rv], color=color, s=36, zorder=3, edgecolor="white", linewidth=0.5)
        if pv is not None:
            ax.scatter([1], [pv], color=color, s=36, zorder=3, edgecolor="white", linewidth=0.5)
            right_vals.append((pv, m))
        elif rv is not None:
            right_vals.append((rv, m))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["raw", "PCA-128"])
    ax.set_xlim(-0.12, 1.6)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)

    # staggered right-side model labels
    right_vals.sort(key=lambda t: t[0])
    if right_vals:
        ymin, ymax = ax.get_ylim()
        MIN_GAP = (ymax - ymin) * 0.07
        targets = [v for v, _ in right_vals]
        for _ in range(300):
            changed = False
            for i in range(1, len(targets)):
                if targets[i - 1] - targets[i] < MIN_GAP:
                    mid = 0.5 * (targets[i - 1] + targets[i])
                    targets[i - 1] = mid + MIN_GAP / 2
                    targets[i] = mid - MIN_GAP / 2
                    changed = True
            if not changed:
                break
        for ytgt, (yval, m) in zip(targets, right_vals):
            ax.annotate(
                ST.MODEL_DISPLAY.get(m, m),
                xy=(1, yval), xytext=(1.08, ytgt),
                fontsize=5.4, va="center", ha="left",
                color=ST.MODEL_PALETTE[m],
                arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.4),
                annotation_clip=False,
            )


def _dim_vs_metric(ax: plt.Axes, df: pd.DataFrame, column: str, ylabel: str) -> None:
    sub = df[["model", "latent_dim_input", column]].dropna()
    for m, sd in sub.groupby("model"):
        ax.scatter(sd["latent_dim_input"], sd[column], color=ST.MODEL_PALETTE[m],
                   s=10, alpha=0.45, edgecolor="white", linewidth=0.3)
    centroids = (
        sub.groupby("model").agg(
            x=("latent_dim_input", "median"), y=(column, "median")
        ).sort_values("y", ascending=False)
    )
    # numbered markers + inline legend
    for i, (m, row) in enumerate(centroids.iterrows(), start=1):
        ax.scatter(row["x"], row["y"], color=ST.MODEL_PALETTE[m],
                   s=120, edgecolor="black", linewidth=0.6, zorder=5)
        ax.text(row["x"], row["y"], str(i), ha="center", va="center",
                fontsize=6.5, fontweight="bold", color="white", zorder=6)
    ax.set_xscale("log")
    ax.set_xlabel("Native latent dimension")
    ax.set_ylabel(ylabel)
    ax.grid(True, lw=0.3, alpha=0.4, which="both")
    # legend mapping numbers to models
    handles = [plt.Line2D([], [], marker="o", lw=0,
                          markerfacecolor=ST.MODEL_PALETTE[m],
                          markeredgecolor="black", markeredgewidth=0.5,
                          markersize=6, label=f"{i}  {ST.MODEL_DISPLAY.get(m, m)}")
               for i, m in enumerate(centroids.index, start=1)]
    ax.legend(handles=handles, loc="upper left", fontsize=5.6,
              frameon=False, ncol=2, handletextpad=0.3,
              columnspacing=0.7, borderaxespad=0.2)


# ============================ FIG 4: Chempert =============================

def fig4_chempert(df: pd.DataFrame, out_dir: Path,
                  per_pert_df: pd.DataFrame | None = None) -> Tuple[Path, Path]:
    chem = df[df["category"] == "chempert"]
    raw = chem[chem["latent_space"] == "raw"]
    raw_scaled = _add_scale_normalized_perturb_columns(raw, out_dir) if len(raw) else raw.copy()

    fig = plt.figure(figsize=(ST.mm(230), ST.mm(172)))
    # A narrow spacer column between c and d prevents twin-axis / colorbar text
    # from colliding with the right-hand panel.
    gs = GridSpec(2, 5, figure=fig, hspace=0.95, wspace=0.58,
                  left=0.07, right=0.985, top=0.93, bottom=0.24,
                  width_ratios=[1, 1, 0.24, 1, 1])

    order = sorted(df["model"].unique(), key=lambda m: ST.ALL_MODELS.index(m))

    # (a) mean L2 to control per dataset, raw and scale-normalized
    ax_a = fig.add_subplot(gs[0, 0])
    _grouped_bar_chempert(
        ax_a, raw_scaled, column="perturb.centroid_shift.mean_l2_to_control",
        models=order, ylabel="Centroid L2 to control", show_legend=False
    )
    ax_a.set_title("a  Centroid shift: raw L2", loc="left")
    ax_a2 = fig.add_subplot(gs[0, 1])
    _grouped_bar_chempert(
        ax_a2, raw_scaled, column="perturb.centroid_shift.mean_l2_to_control.scale_norm",
        models=order, ylabel="L2 / median ||z||", show_legend=False
    )
    ax_a2.set_title("scale-normalized L2 (/ median ||z||)", loc="left")

    # (b) EMD mean per dataset, raw and scale-normalized
    ax_b = fig.add_subplot(gs[0, 3])
    _grouped_bar_chempert(
        ax_b, raw_scaled, column="perturb.ot_summary.emd_mean",
        models=order, ylabel="EMD mean", show_legend=False
    )
    ax_b.set_title("b  OT EMD: raw", loc="left")
    ax_b2 = fig.add_subplot(gs[0, 4])
    _grouped_bar_chempert(
        ax_b2, raw_scaled, column="perturb.ot_summary.emd_mean.scale_norm",
        models=order, ylabel="EMD / median ||z||", show_legend=True
    )
    ax_b2.set_title("scale-normalized EMD", loc="left")

    # (c) cross cell-line generalization (sciplex3_xCellLine)
    ax_c = fig.add_subplot(gs[1, :2])
    xline = raw[raw["dataset_id"] == "sciplex3_xCellLine"]
    if len(xline):
        x = np.arange(len(order))
        l2 = [xline.loc[xline["model"] == m, "perturb.xcellline.xcellline_mean_l2_across_lines"].mean()
              for m in order]
        emd = [xline.loc[xline["model"] == m, "perturb.xcellline.xcellline_mean_emd_across_lines"].mean()
               for m in order]
        ax_c.bar(x - 0.2, l2, 0.4, color="#4C72B0", alpha=0.85, label="mean L2")
        ax2 = ax_c.twinx()
        ax2.bar(x + 0.2, emd, 0.4, color="#DD8452", alpha=0.85, label="mean EMD")
        _model_axis(ax_c, order, axis="x")
        ax_c.set_ylabel("Mean L2 (across cell lines)", color="#4C72B0")
        ax2.set_ylabel("Mean EMD (across cell lines)", color="#DD8452")
        ax2.spines["right"].set_visible(True)
        ax_c.grid(True, axis="y", lw=0.3, alpha=0.4)
    ax_c.set_title("c  Cross cell-line (sciplex3·xLine)", loc="left")

    # (d) Top-K perturbation ranking fidelity for sciplex3_xCellLine.
    # GT: control vs each perturbation centroid L2 in raw log1p expression
    # space. Per model: rank scale-normalized latent shifts on the GT top-K
    # perturbations and compute Spearman vs GT ranks. K = 10 / 20 / 50.
    ax_d = fig.add_subplot(gs[1, 3:])
    ks_chem = (10, 20, 50)
    n_used_max = 0
    if per_pert_df is not None and len(per_pert_df):
        scfm_root = _scfm_root_from_out_dir(out_dir)
        gt_shifts = compute_raw_pert_shifts(
            scfm_root=scfm_root,
            out_dir=out_dir,
            dataset_id="sciplex3_xCellLine",
        )
        pp_scaled = _scale_normalize_per_pert(
            per_pert_df, out_dir=out_dir, dataset_id="sciplex3_xCellLine",
        )
        if gt_shifts and len(pp_scaled):
            sp_table = _topk_spearman_per_model(
                pp_scaled, gt_shifts, models=order, ks=ks_chem,
            )
            _grouped_bars_spearman(
                ax_d, sp_table, models=order, ks=ks_chem,
                ylabel="Spearman ρ vs raw-expr GT",
            )
            n_used_max = int(sp_table["n_used"].max()) if len(sp_table) else 0
    suffix = f"  (n_perts used ≤ {n_used_max})" if n_used_max else ""
    ax_d.set_title(
        "d  Top-K ranking fidelity vs raw-expr GT (xLine)" + suffix,
        loc="left",
    )
    fig.text(
        0.01,
        0.04,
        "UCE and scGPT outputs are L2-normalized to the unit sphere by model definition; raw Euclidean comparisons are scale-confounded. "
        "Scale-normalized panels remove this confound.",
        fontsize=5.2,
        color="#555555",
        ha="left",
        va="bottom",
    )

    return _save(fig, out_dir, "fig4_chempert")


def _grouped_bar_chempert(ax: plt.Axes, df: pd.DataFrame, *, column: str,
                          models: Sequence[str], ylabel: str,
                          show_legend: bool = True) -> None:
    """Per model: log-scale strip of 4 chempert datasets + black bar at median."""
    datasets = ["sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7", "sciplex3_xCellLine"]
    ds_marker = {"sciplex3_A549": "o", "sciplex3_K562": "^",
                 "sciplex3_MCF7": "s", "sciplex3_xCellLine": "D"}
    rng = np.random.default_rng(1)
    EPS = 1e-3
    all_vals: list[float] = []
    for i, m in enumerate(models):
        ys = []
        for ds in datasets:
            if column not in df.columns:
                continue
            v = df.loc[(df["model"] == m) & (df["dataset_id"] == ds), column]
            val = float(v.mean()) if len(v) else np.nan
            if not np.isnan(val):
                plot_val = max(val, EPS)
                xj = i + rng.uniform(-0.22, 0.22)
                ax.scatter(xj, plot_val, color=ST.MODEL_PALETTE[m],
                           s=38, marker=ds_marker[ds], alpha=0.95,
                           edgecolor="white", linewidth=0.5, zorder=3)
                ys.append(plot_val)
                all_vals.append(plot_val)
        if ys:
            ax.hlines(np.median(ys), i - 0.34, i + 0.34,
                      color="black", lw=1.8, zorder=4)
    ax.set_yscale("log")
    _model_axis(ax, models, axis="x")
    ax.set_ylabel(ylabel + "  (log scale)")
    ax.grid(True, axis="y", lw=0.3, alpha=0.4, which="both")
    # ensure even very small values (e.g. UCE ~0.01) are well above the bottom edge
    if all_vals:
        ymin = max(EPS, min(all_vals)) / 3.0
        ymax = max(all_vals) * 2.0
        ax.set_ylim(ymin, ymax)
    if show_legend:
        legend_handles = [
            plt.Line2D([0], [0], marker=ds_marker[ds], color="grey", lw=0,
                       markerfacecolor="grey", markersize=5, markeredgewidth=0,
                       label=DATASET_DISPLAY.get(ds, ds))
            for ds in datasets
        ]
        legend_handles.append(
            plt.Line2D([0], [0], color="black", lw=1.6, label="median across datasets")
        )
        ax.legend(handles=legend_handles, fontsize=5.3, loc="lower right",
                  frameon=True, framealpha=0.9, edgecolor="#bbb",
                  handlelength=1.1, handletextpad=0.35, labelspacing=0.25,
                  borderpad=0.25)


def _grouped_bar_genepert(
    ax: plt.Axes, df: pd.DataFrame, *, column: str,
    models: Sequence[str], ylabel: str,
    show_legend: bool = True,
) -> None:
    """Genepert strip + median (chempert analogue)."""
    datasets = list(GENEPERT_DATASETS)
    markers = ["o", "^", "s", "D", "v", "P"]
    ds_marker = {ds: markers[i % len(markers)] for i, ds in enumerate(datasets)}
    rng = np.random.default_rng(2)
    EPS = 1e-3
    all_vals: list[float] = []
    for i, m in enumerate(models):
        ys: list[float] = []
        for ds in datasets:
            if column not in df.columns:
                continue
            v = df.loc[(df["model"] == m) & (df["dataset_id"] == ds), column]
            val = float(v.mean()) if len(v) else np.nan
            if not np.isnan(val):
                plot_val = max(val, EPS)
                xj = i + rng.uniform(-0.22, 0.22)
                ax.scatter(
                    xj, plot_val,
                    color=ST.MODEL_PALETTE[m],
                    s=38, marker=ds_marker[ds], alpha=0.95,
                    edgecolor="white", linewidth=0.5, zorder=3,
                )
                ys.append(plot_val)
                all_vals.append(plot_val)
        if ys:
            ax.hlines(np.median(ys), i - 0.34, i + 0.34,
                      color="black", lw=1.8, zorder=4)
    ax.set_yscale("log")
    _model_axis(ax, models, axis="x")
    ax.set_ylabel(ylabel + "  (log scale)")
    ax.grid(True, axis="y", lw=0.3, alpha=0.4, which="both")
    if all_vals:
        ymin = max(EPS, min(all_vals)) / 3.0
        ymax = max(all_vals) * 2.0
        ax.set_ylim(ymin, ymax)
    if show_legend:
        legend_handles = [
            plt.Line2D([0], [0], marker=ds_marker[ds], color="grey", lw=0,
                       markerfacecolor="grey", markersize=5, markeredgewidth=0,
                       label=DATASET_DISPLAY.get(ds, ds))
            for ds in datasets
        ]
        legend_handles.append(
            plt.Line2D([0], [0], color="black", lw=1.6, label="median across datasets"),
        )
        ax.legend(handles=legend_handles, fontsize=5.3, loc="lower right",
                  frameon=True, framealpha=0.9, edgecolor="#bbb",
                  handlelength=1.1, handletextpad=0.35, labelspacing=0.25,
                  borderpad=0.25)


def _genepert_ot_emd_rank_matrix(
    raw_gene: pd.DataFrame,
    *,
    models: Sequence[str],
) -> pd.DataFrame:
    """Rows=datasets; cols=models; OT EMD per-dataset ranks -> [0,1] score."""
    piv = raw_gene.pivot_table(
        index="dataset_id",
        columns="model",
        values="perturb.ot_summary.emd_mean",
        aggfunc="mean",
    ).reindex(GENEPERT_DATASETS).reindex(columns=list(models))
    mat = pd.DataFrame(index=piv.index, columns=piv.columns, dtype=float)
    for ds in piv.index:
        row = piv.loc[ds].dropna()
        n = len(row)
        if n == 0:
            continue
        ranks = row.rank(ascending=False, method="average")
        mat.loc[ds, ranks.index] = (float(n + 1) - ranks) / float(n)
    return mat


def _genepert_composite_model_scores(
    gene_scaled: pd.DataFrame,
    *,
    models: Sequence[str],
) -> pd.Series:
    """Across genepert datasets: mean of dataset-wise min-max (L2_sn, EMD_sn)."""
    datasets = list(GENEPERT_DATASETS)
    col_l2 = "perturb.centroid_shift.mean_l2_to_control.scale_norm"
    col_emd = "perturb.ot_summary.emd_mean.scale_norm"
    per_model: Dict[str, list[float]] = {m: [] for m in models}
    for ds in datasets:
        sub = gene_scaled[gene_scaled["dataset_id"] == ds].set_index("model").reindex(models)
        for col_k in (col_l2, col_emd):
            if col_k not in sub.columns:
                continue
            vals = pd.to_numeric(sub[col_k], errors="coerce").dropna()
            if len(vals) < 2:
                continue
            lo, hi = float(vals.min()), float(vals.max())
            span = hi - lo
            if not np.isfinite(span) or span <= 0:
                continue
            norm = (pd.to_numeric(sub[col_k], errors="coerce") - lo) / span
            for m in models:
                vv = norm.get(m, np.nan)
                if pd.notna(vv):
                    per_model[str(m)].append(float(vv))
    return pd.Series({m: (float(np.mean(vs)) if vs else np.nan) for m, vs in per_model.items()})


def fig4b_genepert(df: pd.DataFrame, out_dir: Path,
                   per_pert_df: pd.DataFrame | None = None) -> Tuple[Path, Path]:
    """CRISPR / gene perturbation panels mirroring Fig.4 chem layout."""
    gene = df[(df["category"] == "genepert") & (df["latent_space"] == "raw")].copy()
    raw_scaled = _add_scale_normalized_perturb_columns(gene, out_dir) if len(gene) else gene.copy()

    fig = plt.figure(figsize=(ST.mm(230), ST.mm(172)))
    gs = GridSpec(2, 5, figure=fig, hspace=0.95, wspace=0.58,
                  left=0.07, right=0.985, top=0.93, bottom=0.24,
                  width_ratios=[1, 1, 0.24, 1, 1])
    order = sorted(df["model"].unique(), key=lambda m: ST.ALL_MODELS.index(m))

    ax_a = fig.add_subplot(gs[0, 0])
    _grouped_bar_genepert(
        ax_a, raw_scaled,
        column="perturb.centroid_shift.mean_l2_to_control",
        models=order, ylabel="Centroid L2 to control", show_legend=False,
    )
    ax_a.set_title("a  Centroid shift: raw L2", loc="left")

    ax_a2 = fig.add_subplot(gs[0, 1])
    col_sn = "perturb.centroid_shift.mean_l2_to_control.scale_norm"
    df_b = raw_scaled.copy()
    if col_sn in df_b.columns and df_b[col_sn].isna().any():
        pcac = df[
            (df["category"] == "genepert")
            & (df["latent_space"] == "pca128")
        ][["model", "dataset_id", "perturb.centroid_shift.mean_l2_to_control"]].copy()
        merged = df_b[["model", "dataset_id"]].merge(
            pcac, how="left", on=["model", "dataset_id"],
            suffixes=("", "_pca"),
        )
        pca_col = "perturb.centroid_shift.mean_l2_to_control_pca"
        if pca_col not in merged.columns:
            pca_col = "perturb.centroid_shift.mean_l2_to_control"
        fb = merged[pca_col].astype(float)
        df_b["_plot_l2sn"] = fb.combine_first(pd.to_numeric(df_b[col_sn], errors="coerce"))

    plot_col_l2sn = "_plot_l2sn" if "_plot_l2sn" in df_b.columns else col_sn
    ylab = (
        "L2 / median ||z||"
        if plot_col_l2sn == col_sn
        else "L2 / ||z||  (fallback: PCA-128 L2)"
    )
    _grouped_bar_genepert(
        ax_a2, df_b, column=plot_col_l2sn,
        models=order, ylabel=ylab, show_legend=False,
    )
    ax_a2.set_title("scale-normalized L2", loc="left")

    ax_bem = fig.add_subplot(gs[0, 3])
    _grouped_bar_genepert(
        ax_bem, raw_scaled, column="perturb.ot_summary.emd_mean",
        models=order, ylabel="EMD mean", show_legend=False,
    )
    ax_bem.set_title("b  OT EMD: raw", loc="left")

    ax_bem2 = fig.add_subplot(gs[0, 4])
    _grouped_bar_genepert(
        ax_bem2, raw_scaled, column="perturb.ot_summary.emd_mean.scale_norm",
        models=order, ylabel="EMD / median ||z||", show_legend=True,
    )
    ax_bem2.set_title("scale-normalized EMD", loc="left")

    ax_c = fig.add_subplot(gs[1, :2])
    if len(raw_scaled):
        rmat = _genepert_ot_emd_rank_matrix(raw_scaled, models=order).T
        rmat_disp = rmat.rename(columns=lambda i: DATASET_DISPLAY.get(str(i), str(i)))
        _heatmap(
            ax_c, rmat_disp, cmap="RdBu_r", vmin=0.0, vmax=1.0,
            cbar_label="rank score\n(OT EMD)",
            annotate=True,
            cbar_fraction=0.018,
        )
        ax_c.set_xlabel("Genepert datasets")
        ax_c.tick_params(axis="x", labelsize=5.5, rotation=70)
        for lbl in ax_c.get_xticklabels():
            lbl.set_ha("right")
    ax_c.set_title("c  OT EMD rank heatmap (datasets × models)", loc="left")

    # (d) Top-K perturbation ranking fidelity vs raw-expression GT, averaged
    # across the 6 genepert datasets. Bars = mean Spearman per (model, K);
    # black dots = per-dataset Spearman so dispersion is visible. K=10/20/30
    # (genepert benchmarks select ~30 conditions per dataset, so K>30 collapses).
    ax_d = fig.add_subplot(gs[1, 3:])
    ks_gene = (10, 20, 30)
    n_used_max = 0
    if per_pert_df is not None and len(per_pert_df):
        scfm_root = _scfm_root_from_out_dir(out_dir)
        gt_per_dataset: Dict[str, Dict[str, float]] = {
            ds: compute_raw_pert_shifts(
                scfm_root=scfm_root, out_dir=out_dir, dataset_id=ds,
            )
            for ds in GENEPERT_DATASETS
        }
        gt_per_dataset = {ds: g for ds, g in gt_per_dataset.items() if g}

        pp_scaled_parts = [
            _scale_normalize_per_pert(per_pert_df, out_dir=out_dir, dataset_id=ds)
            for ds in gt_per_dataset.keys()
        ]
        pp_scaled = (
            pd.concat(pp_scaled_parts, ignore_index=True)
            if pp_scaled_parts else pd.DataFrame()
        )

        if gt_per_dataset and len(pp_scaled):
            per_ds_table = _topk_spearman_per_model_multi(
                pp_scaled, gt_per_dataset, models=order, ks=ks_gene,
            )
            agg_table = (
                per_ds_table.groupby(["model", "K"])["spearman"]
                .mean().reset_index()
            )
            agg_table["n_used"] = (
                per_ds_table.groupby(["model", "K"])["n_used"].mean().values
            )
            _grouped_bars_spearman(
                ax_d, agg_table, models=order, ks=ks_gene,
                ylabel="Spearman ρ vs raw-expr GT  (mean over 6 ds)",
            )
            n_used_max = int(per_ds_table["n_used"].max()) if len(per_ds_table) else 0
    suffix = f"  (≤{n_used_max} perts/ds)" if n_used_max else ""
    ax_d.set_title(
        "d  Top-K ranking fidelity vs raw-expr GT  (genepert)" + suffix,
        loc="left",
    )

    fig.text(
        0.01,
        0.04,
        "Genepert CRISPR screens vs control latent geometry; normalization matches Fig.4 (/ median ||z||). ",
        fontsize=5.2,
        color="#555555",
        ha="left",
        va="bottom",
    )
    return _save(fig, out_dir, "fig4b_genepert")


# ============================ FIG 4-2 / 4b-2: Perturbation similarity =====


def _fig_pert_similarity(
    out_dir: Path,
    *,
    dataset_id: str,
    models: Sequence[str],
    file_stem: str,
    title: str,
    k: int = 10,
) -> Tuple[Path, Path]:
    """Build a perturbation effect-similarity figure for one dataset.

    Top-K perturbations chosen by raw expression centroid L2 (GT). For each
    perturbation, diff vector = mu_pert - mu_control. Cosine similarity among
    diff vectors gives a (K, K) matrix in raw expression and in each model's
    latent space. Bottom panel: Mantel-Spearman of upper-triangle vectors
    between GT and each model's latent matrix (Pearson similarity also shown).
    """
    scfm_root = _scfm_root_from_out_dir(out_dir)
    res = PS.compute_or_load(scfm_root, out_dir, dataset_id, models, k=k)
    perts = res["perts"]
    if not perts:
        LOG.warning("No GT perturbations available for %s; skipping %s",
                    dataset_id, file_stem)
        # still emit an empty placeholder so downstream paths exist
        fig = plt.figure(figsize=(ST.mm(80), ST.mm(40)))
        fig.text(0.5, 0.5, f"No data for {dataset_id}", ha="center", va="center")
        return _save(fig, out_dir, file_stem)

    sim_cos = res["sim"].get("cosine", {})
    consistency = res["consistency"]

    panels = ["GT"] + list(models)
    n_cols = 4
    n_rows_mat = int(np.ceil(len(panels) / n_cols))

    fig = plt.figure(figsize=(ST.mm(210), ST.mm(60 * n_rows_mat + 70)))
    outer = GridSpec(
        2, 1, figure=fig, height_ratios=[n_rows_mat, 1.05],
        hspace=0.35, left=0.05, right=0.90, top=0.93, bottom=0.08,
    )
    mat_gs = outer[0].subgridspec(n_rows_mat, n_cols, wspace=0.40, hspace=0.65)

    vmin, vmax = -1.0, 1.0
    cmap = "RdBu_r"

    for idx, name in enumerate(panels):
        r, c = divmod(idx, n_cols)
        ax = fig.add_subplot(mat_gs[r, c])
        S = sim_cos.get(name)
        if S is None:
            ax.set_axis_off()
            continue
        im = ax.imshow(S, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(perts)))
        ax.set_yticks(range(len(perts)))
        ax.set_xticklabels(perts, rotation=70, ha="right", fontsize=4.6)
        ax.set_yticklabels(perts, fontsize=4.6)
        ax.tick_params(length=1.5, width=0.3)
        if name == "GT":
            ax.set_title("GT (raw expression)", fontsize=7,
                         color="#222", fontweight="bold")
        else:
            cs = consistency.get(name, {})
            sp_c = cs.get("mantel_spearman_cosine", float("nan"))
            ax.set_title(
                f"{ST.MODEL_DISPLAY.get(name, name)}\n"
                f"Mantel ρ_S={sp_c:.2f}",
                fontsize=7, color=ST.MODEL_PALETTE.get(name, "#333"),
            )
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

    cbar_ax = fig.add_axes([0.93, 0.55, 0.012, 0.28])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("cosine similarity\n(diff vectors)", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.5, length=2, width=0.4)

    bar_ax = fig.add_subplot(outer[1])
    rows: list[dict] = []
    for m in models:
        cs = consistency.get(m, {})
        rows.append(dict(
            model=m,
            spear_cos=cs.get("mantel_spearman_cosine", np.nan),
            spear_pear=cs.get("mantel_spearman_pearson", np.nan),
        ))
    cdf = pd.DataFrame(rows)
    cdf["sort_key"] = cdf["spear_cos"].fillna(-2)
    cdf = cdf.sort_values("sort_key", ascending=False).drop(columns="sort_key")

    x = np.arange(len(cdf))
    width = 0.38
    bar_ax.bar(
        x - width / 2, cdf["spear_cos"].values, width=width,
        color=[mcolors.to_rgba(ST.MODEL_PALETTE.get(m, "#888"), 0.95)
               for m in cdf["model"]],
        edgecolor="black", linewidth=0.35, label="cosine sim → Spearman",
    )
    bar_ax.bar(
        x + width / 2, cdf["spear_pear"].values, width=width,
        color=[mcolors.to_rgba(ST.MODEL_PALETTE.get(m, "#888"), 0.55)
               for m in cdf["model"]],
        edgecolor="black", linewidth=0.35, hatch="//",
        label="Pearson sim → Spearman",
    )
    bar_ax.axhline(0, color="grey", lw=0.4, ls="--", alpha=0.6)
    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels(
        [ST.MODEL_DISPLAY.get(m, m) for m in cdf["model"]],
        rotation=45, ha="right",
    )
    bar_ax.set_ylabel("Mantel ρ_S\n(upper-tri Spearman vs GT)")
    bar_ax.set_ylim(-0.2, 1.05)
    bar_ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    bar_ax.legend(fontsize=6, frameon=False, loc="upper right",
                  handlelength=1.5, handletextpad=0.5)
    bar_ax.set_title(
        f"Perturbation-similarity preservation  |  top-{len(perts)} perts in {dataset_id}",
        loc="left", fontsize=8.5,
    )

    fig.suptitle(title, x=0.05, ha="left", y=0.985,
                 fontsize=10, fontweight="bold")
    return _save(fig, out_dir, file_stem)


def fig4_2_chempert_sim(df: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path]:
    """Chempert perturbation-similarity preservation (sciplex3_xCellLine, top-10)."""
    models = sorted(df["model"].unique(), key=lambda m: ST.ALL_MODELS.index(m))
    return _fig_pert_similarity(
        out_dir,
        dataset_id="sciplex3_xCellLine",
        models=models,
        file_stem="fig4_2_chempert_sim",
        title="Fig 4-2  Chempert perturbation-effect similarity (sciplex3 xLine)",
        k=10,
    )


def fig4b_2_genepert_sim(df: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path]:
    """Genepert perturbation-similarity preservation (NormanWeissman2019, top-10)."""
    models = sorted(df["model"].unique(), key=lambda m: ST.ALL_MODELS.index(m))
    return _fig_pert_similarity(
        out_dir,
        dataset_id="NormanWeissman2019_filtered__single",
        models=models,
        file_stem="fig4b_2_genepert_sim",
        title="Fig 4b-2  Genepert perturbation-effect similarity (NormanWeissman2019)",
        k=10,
    )


# ============================ FIG 5: Overall / robustness ==================

def fig5_overall(df: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path]:
    """Rank stability, balanced overall score (chem+gene perturb equal), win rates, PCA gap."""
    long = melt_metrics(df, M.ALL_METRICS)
    normed = normalize_per_dataset(long, method="rank")
    raw = normed[normed["latent_space"] == "raw"]
    pca = normed[normed["latent_space"] == "pca128"]

    order_frame = _balanced_overall_from_normed(normed, latent_space="raw")
    order = order_frame.sort_values("mean_score", ascending=False)["model"].tolist()
    if not order:
        order = sorted(df["model"].unique(), key=lambda m: ST.ALL_MODELS.index(m))

    fig = plt.figure(figsize=(ST.mm(190), ST.mm(120)))
    gs = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.68,
                  left=0.07, right=0.97, top=0.91, bottom=0.16,
                  width_ratios=[1.05, 1])

    # (a) per-dataset rank distribution per model (violin) ---------------
    ax_a = fig.add_subplot(gs[0, 0])
    per_ds = (
        raw.groupby(["model", "dataset_id"])["score"].mean().reset_index()
    )
    data = [per_ds.loc[per_ds["model"] == m, "score"].values for m in order]
    parts = ax_a.violinplot(data, positions=range(len(order)),
                            widths=0.7, showmeans=False, showextrema=False)
    for pc, m in zip(parts["bodies"], order):
        c = ST.MODEL_PALETTE[m]
        pc.set_facecolor(mcolors.to_rgba(c, 0.35))
        pc.set_edgecolor(c)
        pc.set_linewidth(0.6)
    medians = [np.median(d) for d in data]
    ax_a.scatter(range(len(order)), medians, color=[ST.MODEL_PALETTE[m] for m in order],
                 s=14, edgecolor="black", linewidth=0.4, zorder=5)
    _model_axis(ax_a, order, axis="x")
    ax_a.set_ylabel("per-dataset mean rank score")
    ax_a.set_ylim(0, 1.0)
    ax_a.grid(True, axis="y", lw=0.3, alpha=0.4)
    ax_a.set_title("a  Rank stability across datasets (raw)", loc="left")

    # (b) latent dim vs balanced overall rank score (raw) ---------------
    ax_b = fig.add_subplot(gs[0, 1])
    means = _balanced_overall_from_normed(normed, latent_space="raw").copy()
    dims = (
        df[df["latent_space"] == "raw"].groupby("model")["latent_dim_input"].mean()
    )
    means["dim"] = means["model"].map(dims)
    means_sorted = means.sort_values("mean_score", ascending=False).reset_index(drop=True)
    for i, r in means_sorted.iterrows():
        ax_b.scatter(r["dim"], r["mean_score"], color=ST.MODEL_PALETTE[r["model"]],
                     s=140, edgecolor="black", linewidth=0.6, zorder=4)
        ax_b.text(r["dim"], r["mean_score"], str(i + 1),
                  ha="center", va="center", fontsize=6.5,
                  fontweight="bold", color="white", zorder=5)
    ax_b.set_xscale("log")
    ax_b.set_xlabel("Native latent dim (log)")
    ax_b.set_ylabel("balanced mean rank score (raw)")
    ax_b.grid(True, lw=0.3, alpha=0.4, which="both")
    ax_b.set_title("b  Capacity vs balanced overall", loc="left")
    handles_b = [
        plt.Line2D([], [], marker="o", lw=0,
                   markerfacecolor=ST.MODEL_PALETTE[r["model"]],
                   markeredgecolor="black", markeredgewidth=0.5,
                   markersize=6,
                   label=f"{i + 1}  {ST.MODEL_DISPLAY.get(r['model'], r['model'])}")
        for i, r in means_sorted.iterrows()
    ]
    ax_b.legend(handles=handles_b, loc="upper left", fontsize=5.6,
                frameon=False, ncol=2, handletextpad=0.3,
                columnspacing=0.6, borderaxespad=0.2)

    # (c) win-rate matrix
    ax_c = fig.add_subplot(gs[1, 0])
    matrix = _win_rate_matrix(raw, order)
    im = ax_c.imshow(matrix.values, cmap="RdBu_r", vmin=0, vmax=1, aspect="equal")
    ax_c.set_xticks(range(len(order)))
    ax_c.set_xticklabels([ST.MODEL_DISPLAY.get(m, m) for m in order],
                         rotation=45, ha="right")
    ax_c.set_yticks(range(len(order)))
    ax_c.set_yticklabels([ST.MODEL_DISPLAY.get(m, m) for m in order])
    cb = plt.colorbar(im, ax=ax_c, fraction=0.026, pad=0.02)
    cb.set_label("P(row beats col)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6, length=2, width=0.4)
    ax_c.set_title("c  Pairwise win rate (raw, all metrics)", loc="left")

    # (d) raw vs pca128 delta per metric family
    ax_d = fig.add_subplot(gs[1, 1])
    flips = _raw_vs_pca128_family_change(normed, order)
    fams = ["atlas", "geometry", "perturb"]
    fam_colors = {"atlas": "#1F77B4", "geometry": "#2CA02C", "perturb": "#E15759"}
    width = 0.27
    x = np.arange(len(order))
    for k, fam in enumerate(fams):
        vals = [flips.get((m, fam), 0.0) for m in order]
        ax_d.bar(x + (k - 1) * width, vals, width=width * 0.95,
                 color=fam_colors[fam], alpha=0.85,
                 edgecolor="white", linewidth=0.3, label=fam)
    _model_axis(ax_d, order, axis="x")
    ax_d.set_ylabel("Δ rank score (PCA-128 − raw)")
    ax_d.axhline(0, color="black", lw=0.4)
    ax_d.legend(loc="upper right", fontsize=6.0, frameon=False)
    ax_d.grid(True, axis="y", lw=0.3, alpha=0.4)
    ax_d.set_title("d  Latent-space gap by metric family", loc="left")

    fig.text(
        0.01,
        0.01,
        "Balanced overall: mean of atlas pillar, atlas/TS geometry pillar, and perturb pillar; "
        "perturb pillar averages chempert and genepert (CRISPR) equally.",
        fontsize=5.0,
        color="#555555",
        ha="left",
        va="bottom",
    )
    return _save(fig, out_dir, "fig5_overall")


# ============================ FIG 6: Efficiency ===========================

_ATLAS_EFFICIENCY_DATASETS: Tuple[str, ...] = (
    "Blood",
    "BoneMarrow",
    "Heart",
    "Lung",
    "LymphNode",
    "Skin",
    "TS_Immune_xtissue",
)


def _load_throughput_table(scfm_root: Path | None = None) -> pd.DataFrame:
    """Load per-run throughput metadata from the configured output embeddings."""
    root = (scfm_root or Path.cwd()).resolve()
    base = root / "scFM_output" / "embeddings"
    if not base.is_dir():
        base = root / "output" / "embeddings"
    baseline_wall = _load_baseline_wall_times(root)
    rows: list[dict] = []
    skipped_missing_wall = 0

    for f in sorted(base.glob("*/*/raw/meta.json")):
        model = f.parents[2].name
        dataset_id = f.parents[1].name
        if model not in ST.ALL_MODELS:
            continue
        try:
            meta = json.loads(f.read_text())
        except Exception as exc:
            LOG.warning("Skipping unreadable throughput metadata %s: %s", f, exc)
            continue
        wall = meta.get("wall_time_s")
        if wall is None or pd.isna(wall):
            wall = baseline_wall.get((model, dataset_id))
        if wall is None or pd.isna(wall):
            skipped_missing_wall += 1
            continue
        wall = float(wall)
        n_obs = float(meta.get("n_obs", np.nan))
        if not np.isfinite(n_obs) or not np.isfinite(wall) or wall <= 0:
            continue
        if dataset_id in _ATLAS_EFFICIENCY_DATASETS:
            family = "atlas"
        elif dataset_id.startswith("sciplex3_"):
            family = "chempert"
        elif dataset_id in GENEPERT_DATASETS:
            family = "genepert"
        else:
            family = "other"
        rows.append(
            dict(
                model=str(meta.get("model", model)),
                dataset_id=dataset_id,
                n_obs=n_obs,
                latent_dim=float(meta.get("latent_dim", np.nan)),
                wall_time_s=wall,
                cells_per_s=n_obs / wall,
                family=family,
            )
        )

    if skipped_missing_wall:
        LOG.warning("Skipped %d throughput metadata rows without wall_time_s", skipped_missing_wall)
    out = pd.DataFrame(
        rows,
        columns=["model", "dataset_id", "n_obs", "latent_dim", "wall_time_s", "cells_per_s", "family"],
    )
    if len(out):
        out = out[out["model"].isin(ST.ALL_MODELS)].copy()
    return out


def _load_baseline_wall_times(scfm_root: Path) -> Dict[Tuple[str, str], float]:
    """Recover baseline fit+transform timings from the batch-run logs if needed."""
    out: Dict[Tuple[str, str], float] = {}
    log_dir = scfm_root / "output" / "logs"
    if not log_dir.is_dir():
        return out

    # PCA was run serially; the interval to the next dataset start is the
    # elapsed fit+transform time for the previous dataset.
    pca_logs = sorted(log_dir.glob("baseline_pca_*.log"))
    if pca_logs:
        entries = _parse_baseline_log_starts(pca_logs[-1], "PCA")
        for (dataset_id, start), (_, next_start) in zip(entries, entries[1:]):
            out[("pca", dataset_id)] = max((next_start - start).total_seconds(), 1e-6)

    # scVI was dispatched in parallel; match each dataset start to its ok line.
    scvi_logs = sorted(log_dir.glob("baseline_scvi_*.log"))
    if scvi_logs:
        starts = dict(_parse_baseline_log_starts(scvi_logs[-1], "scVI"))
        finishes = _parse_scvi_log_finishes(scvi_logs[-1])
        for dataset_id, start in starts.items():
            finish = finishes.get(dataset_id)
            if finish is not None:
                out[("scvi", dataset_id)] = max((finish - start).total_seconds(), 1e-6)
    return out


def _parse_baseline_log_starts(path: Path, label: str) -> list[tuple[str, datetime]]:
    pat = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) INFO (?:\[gpu=\d+\] )?"
                     + re.escape(label) + r" ([^:]+): ")
    entries: list[tuple[str, datetime]] = []
    for line in path.read_text().splitlines():
        m = pat.match(line)
        if not m:
            continue
        ts = datetime.strptime(f"{m.group(1)}.{m.group(2)}", "%Y-%m-%d %H:%M:%S.%f")
        entries.append((m.group(3), ts))
    return entries


def _parse_scvi_log_finishes(path: Path) -> Dict[str, datetime]:
    pat = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) INFO scVI ([^ ]+) -> ok")
    out: Dict[str, datetime] = {}
    for line in path.read_text().splitlines():
        m = pat.match(line)
        if not m:
            continue
        out[m.group(3)] = datetime.strptime(f"{m.group(1)}.{m.group(2)}", "%Y-%m-%d %H:%M:%S.%f")
    return out


def fig6_efficiency(df: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path]:
    """Efficiency figure: throughput, score trade-offs, and native dimension."""
    scfm_root = out_dir.resolve().parents[1]
    tput = _load_throughput_table(scfm_root)
    atlas = tput[tput["family"] == "atlas"].copy()
    atlas_n = int(len(atlas))
    atlas_ds_n = int(atlas["dataset_id"].nunique()) if len(atlas) else 0
    LOG.info("Fig6 aggregated %d atlas throughput rows across %d datasets", atlas_n, atlas_ds_n)

    atlas_weighted = (
        atlas.groupby("model")
        .agg(n_obs=("n_obs", "sum"), wall_time_s=("wall_time_s", "sum"), latent_dim=("latent_dim", "median"))
        .assign(throughput=lambda x: x["n_obs"] / x["wall_time_s"])
        .reindex(ST.ALL_MODELS)
    )
    order = atlas_weighted["throughput"].sort_values(ascending=False).dropna().index.tolist()

    reg = M.by_column()
    headline_metrics = [
        reg[c] for c in (M.HEADLINE_ATLAS + M.HEADLINE_GEOMETRY + M.HEADLINE_PERTURB)
        if c in reg
    ]
    long_h = melt_metrics(df, headline_metrics)
    normed_headline = normalize_per_dataset(long_h, method="rank")
    raw_agg = _headline_balanced_agg(normed_headline)
    raw_scores_body = raw_agg[raw_agg["latent_space"] == "raw"]
    score = raw_scores_body.set_index("model")["mean_score"].reindex(ST.ALL_MODELS)
    score_order = score.sort_values(ascending=False).dropna().index.tolist()
    number_map = {m: i + 1 for i, m in enumerate(score_order)}

    fig = plt.figure(figsize=(ST.mm(180), ST.mm(145)))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.55,
        wspace=0.35,
        left=0.09,
        right=0.98,
        top=0.93,
        bottom=0.20,
    )

    # (a) atlas-weighted throughput --------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    y = np.arange(len(order))
    vals = atlas_weighted.loc[order, "throughput"].values
    for yi, m, val in zip(y, order, vals):
        color = ST.MODEL_PALETTE[m]
        edge = _darker(color)
        ax_a.barh(
            yi,
            val,
            color=mcolors.to_rgba(color, 0.95 if m not in ST.BASELINE_MODELS else 0.62),
            edgecolor=edge if m in ST.BASELINE_MODELS else "white",
            linewidth=0.5,
            hatch="///" if m in ST.BASELINE_MODELS else None,
            zorder=3,
        )
        ax_a.text(val * 1.06, yi, f"{val:.0f}", ha="left", va="center", fontsize=5.6, color="#333333")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([ST.MODEL_DISPLAY.get(m, m) for m in order])
    ax_a.invert_yaxis()
    ax_a.set_xscale("log")
    ax_a.set_xlabel("Throughput (cells / s on atlas, log scale)")
    ax_a.grid(True, axis="x", lw=0.3, alpha=0.35, which="both", zorder=0)
    ax_a.set_title("a  Atlas encoding throughput", loc="left")

    # (b) throughput vs score --------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    pareto_df = atlas_weighted[["throughput"]].join(score.rename("score")).dropna()
    _draw_pareto_frontier(ax_b, pareto_df)
    for m, r in pareto_df.iterrows():
        _numbered_bubble(ax_b, r["throughput"], r["score"], m, number_map[m])
    ax_b.set_xscale("log")
    ax_b.set_xlabel("Throughput (cells / s, log scale)")
    ax_b.set_ylabel("balanced headline rank score (raw)")
    ax_b.set_ylim(0, 1.02)
    ax_b.grid(True, lw=0.3, alpha=0.35, which="both")
    ax_b.set_title("b  Capability vs efficiency", loc="left")
    handles_b = [
        plt.Line2D(
            [],
            [],
            marker="o",
            lw=0,
            markerfacecolor=mcolors.to_rgba(ST.MODEL_PALETTE[m], 0.65 if m in ST.BASELINE_MODELS else 1.0),
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=5.8,
            label=f"{number_map[m]}  {ST.MODEL_DISPLAY.get(m, m)}",
        )
        for m in score_order
        if m in pareto_df.index
    ]
    ax_b.legend(handles=handles_b, loc="upper left", fontsize=5.2, frameon=False,
                ncol=2, handletextpad=0.3, columnspacing=0.6, borderaxespad=0.2)

    # (c) per-dataset throughput strip -----------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    present_ds = [d for d in _ATLAS_EFFICIENCY_DATASETS if d in set(atlas["dataset_id"])]
    xloc = {d: i for i, d in enumerate(present_ds)}
    rng = np.random.default_rng(6)
    for m in ST.ALL_MODELS:
        md = atlas[atlas["model"] == m].set_index("dataset_id").reindex(present_ds)
        if md["cells_per_s"].notna().sum() == 0:
            continue
        xs = np.array([xloc[d] for d in present_ds], dtype=float)
        ys = md["cells_per_s"].values
        mask = np.isfinite(ys)
        ax_c.plot(xs[mask], ys[mask], color="#999999", lw=0.35, alpha=0.35, zorder=1)
        ax_c.scatter(
            xs[mask] + rng.uniform(-0.18, 0.18, size=int(mask.sum())),
            ys[mask],
            s=22,
            color=ST.MODEL_PALETTE[m],
            alpha=0.90,
            edgecolor="none",
            zorder=3,
        )
    ax_c.set_yscale("log")
    ax_c.set_xticks(range(len(present_ds)))
    ax_c.set_xticklabels([DATASET_DISPLAY.get(d, d) for d in present_ds], rotation=30, ha="right")
    ax_c.set_ylabel("Throughput (cells / s, log scale)")
    ax_c.grid(True, axis="y", lw=0.3, alpha=0.35, which="both")
    ax_c.set_title("c  Throughput across atlas datasets", loc="left")
    handles_c = [
        plt.Line2D([], [], marker="o", color=ST.MODEL_PALETTE[m], lw=0,
                   markerfacecolor=ST.MODEL_PALETTE[m], markeredgewidth=0,
                   markersize=4.2, label=ST.MODEL_DISPLAY.get(m, m))
        for m in ST.ALL_MODELS
        if m in set(atlas["model"])
    ]
    ax_c.legend(handles=handles_c, loc="upper center",
                bbox_to_anchor=(0.5, -0.22),
                fontsize=5.2, frameon=False,
                ncol=4, handletextpad=0.3, columnspacing=0.6, borderaxespad=0.2)

    # (d) latent dimension vs throughput ---------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    dim_df = atlas_weighted[["throughput", "latent_dim"]].dropna()
    for m, r in dim_df.iterrows():
        if m not in number_map:
            continue
        _numbered_bubble(ax_d, r["latent_dim"], r["throughput"], m, number_map[m])
    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlabel("Native latent dim (log)")
    ax_d.set_ylabel("Atlas-weighted throughput (log)")
    ax_d.grid(True, lw=0.3, alpha=0.35, which="both")
    ax_d.text(
        0.98,
        0.04,
        "compact & fast \u2192",
        transform=ax_d.transAxes,
        fontsize=5.0,
        color="#777777",
        fontstyle="italic",
        ha="right",
        va="bottom",
    )
    ax_d.set_title("d  Latent dim vs throughput", loc="left")

    fig.text(
        0.005,
        0.005,
        "Baselines (PCA, scVI-128) wall time includes dataset fit; foundation-model time is encode-only. "
        "Throughput aggregated across atlas datasets.",
        fontsize=4.8,
        color="#555555",
        ha="left",
        va="bottom",
    )

    return _save(fig, out_dir, "fig6_efficiency")


def _darker(color: str, factor: float = 0.65) -> str:
    rgb = np.array(mcolors.to_rgb(color))
    return mcolors.to_hex(np.clip(rgb * factor, 0, 1))


def _numbered_bubble(ax: plt.Axes, x: float, y: float, model: str, number: int) -> None:
    color = ST.MODEL_PALETTE[model]
    alpha = 0.68 if model in ST.BASELINE_MODELS else 1.0
    ax.scatter(
        x,
        y,
        color=mcolors.to_rgba(color, alpha),
        s=140,
        edgecolor="black",
        linewidth=0.6,
        hatch="///" if model in ST.BASELINE_MODELS else None,
        zorder=4,
    )
    ax.text(x, y, str(number), ha="center", va="center",
            fontsize=6.5, fontweight="bold", color="white", zorder=5)


def _draw_pareto_frontier(ax: plt.Axes, points: pd.DataFrame) -> None:
    if len(points) < 2:
        return
    ordered = points.sort_values("throughput", ascending=False)
    frontier: list[tuple[float, float]] = []
    best_score = -np.inf
    for _, r in ordered.iterrows():
        if r["score"] > best_score:
            frontier.append((float(r["throughput"]), float(r["score"])))
            best_score = float(r["score"])
    if len(frontier) < 2:
        return
    frontier = sorted(frontier, key=lambda t: t[0])
    xs, ys = zip(*frontier)
    ax.plot(xs, ys, color="#999999", lw=0.6, ls="--", zorder=2)
    ax.annotate(
        "Pareto frontier",
        xy=(xs[-1], ys[-1]),
        xycoords="data",
        xytext=(0.98, 0.96),
        textcoords="axes fraction",
        ha="right",
        va="top",
        fontsize=5.0,
        color="#777777",
        fontstyle="italic",
        arrowprops=dict(arrowstyle="-", lw=0.4, color="#999999"),
    )


def fig_supp_all_metrics(df: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path]:
    """Supplementary: full metric heatmap for raw and pca128 spaces side by side."""
    geo_and_atlas = [m for m in M.ALL_METRICS
                     if m.family in ("atlas", "geometry") and m.column in df.columns]
    reg = M.by_column()
    order = sorted(df["model"].unique(), key=lambda m: ST.ALL_MODELS.index(m))

    fig, axes = plt.subplots(1, 2, figsize=(ST.mm(220), ST.mm(125)),
                             gridspec_kw={"wspace": 0.10},
                             constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.86, bottom=0.27)

    for ax, sp in zip(axes, ("raw", "pca128")):
        sub = df[df["latent_space"] == sp]
        long = melt_metrics(sub, geo_and_atlas)
        normed = normalize_per_dataset(long, method="rank")
        mat = (
            normed.pivot_table(index="model", columns="short", values="score", aggfunc="mean")
            .reindex(order)
        )
        col_order = [reg[c].short for c in
                     [m.column for m in geo_and_atlas] if reg[c].short in mat.columns]
        mat = mat.reindex(columns=col_order)
        im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels(mat.columns, rotation=60, ha="right", fontsize=6)
        ax.set_yticks(range(mat.shape[0]))
        if ax is axes[0]:
            ax.set_yticklabels([ST.MODEL_DISPLAY.get(m, m) for m in mat.index])
        else:
            ax.set_yticklabels([])
        # color x-tick labels by family
        fam_colors = {"atlas": "#1F77B4", "geometry": "#2CA02C"}
        short_to_fam = {m.short: m.family for m in geo_and_atlas}
        for tick, short in zip(ax.get_xticklabels(), mat.columns):
            tick.set_color(fam_colors.get(short_to_fam.get(short, ""), "#444"))
        ax.set_title(
            f"{'Raw embedding' if sp == 'raw' else 'PCA-128 projection'}",
            loc="center", fontsize=8.5, fontweight="bold", pad=4)
        cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cb.set_label("rank score (1 = best)", fontsize=6.5)
        cb.ax.tick_params(labelsize=6, length=2, width=0.4)

    fig.suptitle(
        "Supplementary Fig. S1  Full atlas + geometry metric scoreboard",
        fontsize=9.5, fontweight="bold", x=0.5, y=0.965)
    # family legend
    handles = [plt.Line2D([], [], marker="s", color=c, lw=0,
                          markersize=7, label=k)
               for k, c in [("Atlas integration", "#1F77B4"),
                            ("Latent geometry", "#2CA02C")]]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.93))
    return _save(fig, out_dir, "figS1_full_metrics")


def _win_rate_matrix(normed: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    pivot = (
        normed.pivot_table(
            index=["dataset_id", "metric"], columns="model", values="score", aggfunc="mean"
        )
        .reindex(columns=list(models))
    )
    n = len(models)
    M_ = np.full((n, n), np.nan)
    for i, ma in enumerate(models):
        for j, mb in enumerate(models):
            if i == j:
                M_[i, j] = 0.5
                continue
            d = pivot[[ma, mb]].dropna()
            if len(d) == 0:
                continue
            wins = (d[ma] > d[mb]).sum()
            ties = (d[ma] == d[mb]).sum()
            M_[i, j] = (wins + 0.5 * ties) / len(d)
    return pd.DataFrame(M_, index=list(models), columns=list(models))


def _raw_vs_pca128_family_change(normed: pd.DataFrame, models: Sequence[str]) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for m in models:
        for fam in ["atlas", "geometry", "perturb"]:
            sub = normed[(normed["model"] == m) & (normed["family"] == fam)]
            r = sub.loc[sub["latent_space"] == "raw", "score"].mean()
            p = sub.loc[sub["latent_space"] == "pca128", "score"].mean()
            if np.isnan(r) or np.isnan(p):
                out[(m, fam)] = 0.0
            else:
                out[(m, fam)] = float(p - r)
    return out
