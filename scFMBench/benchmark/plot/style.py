"""Matplotlib rcParams + categorical palettes for Nature-style figures.

Conventions:
- Helvetica/Arial 7-9pt for labels, 6-7pt for ticks
- Linewidth 0.6-1.0, fig sizes in mm via ``mm`` helper
- Foundation models (FM) colored from a qualitative palette,
  baselines (PCA, scVI) drawn in neutral grey to read as references
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt

MM_PER_INCH = 25.4

# Canonical model order: 9 FM (alphabetical), then 2 baselines at the right.
FM_MODELS: Tuple[str, ...] = (
    "cellnavi",
    "geneformer",
    "nicheformer",
    "scfoundation",
    "scgpt",
    "scldm",
    "stack",
    "state",
    "transcriptformer",
    "uce",
    "xverse",
)
BASELINE_MODELS: Tuple[str, ...] = ("pca", "scvi")
ALL_MODELS: Tuple[str, ...] = FM_MODELS + BASELINE_MODELS

# Pretty display names (kept short and stylistically consistent for tick labels)
MODEL_DISPLAY: Dict[str, str] = {
    "cellnavi": "CellNavi",
    "geneformer": "Geneformer",
    "nicheformer": "NicheFormer",
    "scfoundation": "scFoundation",
    "scgpt": "scGPT",
    "scldm": "scLDM",
    "stack": "Stack",
    "state": "State",
    "transcriptformer": "TranscriptFormer",
    "uce": "UCE",
    "xverse": "xVerse",
    "pca": "PCA",
    "scvi": "scVI",
}

# Categorical palette for FM (tab10-derived, color-vision-safe alternates).
# Baselines drawn in two greys.
_FM_COLORS = (
    "#4C72B0",  # cellnavi   blue
    "#DD8452",  # geneformer orange
    "#009E73",  # nicheformer green-teal
    "#55A868",  # scfoundation green
    "#C44E52",  # scgpt      red
    "#8172B2",  # scldm      purple
    "#937860",  # stack      brown
    "#DA8BC3",  # state      pink
    "#0072B2",  # transcriptformer blue
    "#8C8C8C",  # uce        slate (will reassign below)
    "#CCB974",  # xverse     ochre
)
_BASELINE_COLORS = ("#444444", "#999999")  # PCA dark, scVI light

# Reassign uce a distinctive teal so it doesn't clash with baseline grey
_FM_COLORS = list(_FM_COLORS)
_FM_COLORS[9] = "#17BECF"  # uce teal
_FM_COLORS = tuple(_FM_COLORS)

MODEL_PALETTE: Dict[str, str] = {
    **dict(zip(FM_MODELS, _FM_COLORS)),
    **dict(zip(BASELINE_MODELS, _BASELINE_COLORS)),
}

# Category palette (light backgrounds for facet headers / banding).
CATEGORY_DISPLAY: Dict[str, str] = {
    "atlas": "Atlas (staging)",
    "atlas_TS": "Atlas (TS raw)",
    "chempert": "Chem-perturbation",
    "genepert": "Gene-perturbation",
}
CATEGORY_PALETTE: Dict[str, str] = {
    "atlas": "#1F77B4",
    "atlas_TS": "#9EBCDA",
    "chempert": "#E15759",
    "genepert": "#B07AA1",
}

# Latent-space palette
LATENT_PALETTE: Dict[str, str] = {"raw": "#444444", "pca128": "#1F77B4"}
LATENT_DISPLAY: Dict[str, str] = {"raw": "Raw embedding", "pca128": "PCA-128 projection"}


def mm(value_mm: float) -> float:
    """Convert millimetres to inches (figsize unit)."""
    return value_mm / MM_PER_INCH


def apply_rcparams() -> None:
    """Set Nature-like global matplotlib defaults."""
    base = {
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.5,
        "axes.labelweight": "regular",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.minor.size": 1.2,
        "ytick.minor.size": 1.2,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "lines.linewidth": 0.9,
        "lines.markersize": 3.0,
        "patch.linewidth": 0.5,
        "boxplot.flierprops.markersize": 2.0,
        "errorbar.capsize": 1.8,
        "image.cmap": "viridis",
    }
    mpl.rcParams.update(base)


def model_colors(models: Sequence[str]) -> List[str]:
    return [MODEL_PALETTE.get(m, "#999999") for m in models]


def model_display(models: Iterable[str]) -> List[str]:
    return [MODEL_DISPLAY.get(m, m) for m in models]
