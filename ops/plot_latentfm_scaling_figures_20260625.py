#!/usr/bin/env python3
"""Plot static scaling/failure-map figures from figure-ready CSV tables.

CPU-only. Reads only the figure-data package CSV files generated from completed
reports. It does not read checkpoints, canonical multi, held-out Track C query,
train, infer, or use GPU.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
DATA_DIR = REPORTS / "scaling_figure_data_20260625"
OUT_DIR = REPORTS / "scaling_figures_20260625"
OUT_JSON = REPORTS / "latentfm_scaling_figures_20260625.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_FIGURES_20260625.md"

INPUTS = {
    "s0": DATA_DIR / "s0_provenance_summary.csv",
    "truecell": DATA_DIR / "truecell_budget_curve.csv",
    "exposure": DATA_DIR / "condition_exposure_curve.csv",
    "noharm": DATA_DIR / "canonical_noharm_veto.csv",
    "failure_map": DATA_DIR / "failure_map_axis_summary.csv",
}


COLORS = {
    "blue": "#3b6ea8",
    "teal": "#2f8f83",
    "gold": "#c9952e",
    "red": "#b84a4a",
    "gray": "#666666",
    "lightgray": "#d9d9d9",
}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def savefig(name: str) -> dict[str, str]:
    png = OUT_DIR / f"{name}.png"
    svg = OUT_DIR / f"{name}.svg"
    plt.tight_layout()
    plt.savefig(png, dpi=220)
    plt.savefig(svg)
    plt.close()
    return {"png": str(png), "svg": str(svg)}


def parse_ci(text: Any) -> tuple[float | None, float | None]:
    if not isinstance(text, str) or not text:
        return None, None
    try:
        vals = json.loads(text)
        if isinstance(vals, list) and len(vals) == 2:
            return float(vals[0]), float(vals[1])
    except Exception:
        return None, None
    return None, None


def plot_s0() -> dict[str, str]:
    df = pd.read_csv(INPUTS["s0"])
    keep = df[df["category"].isin(["modality_counts", "perturbation_type_counts"])].copy()
    keep["label"] = keep["category"].str.replace("_counts", "", regex=False) + ":" + keep["name"].astype(str)
    keep["value"] = pd.to_numeric(keep["value"], errors="coerce")
    keep = keep.sort_values("value", ascending=True)
    plt.figure(figsize=(8.0, 4.8))
    colors = [COLORS["teal"] if "modality" in c else COLORS["blue"] for c in keep["category"]]
    plt.barh(keep["label"], keep["value"], color=colors)
    plt.xlabel("Condition rows")
    plt.title("S0 provenance coverage")
    plt.grid(axis="x", alpha=0.25)
    return savefig("FigS_scaling_S0_provenance")


def plot_truecell() -> dict[str, str]:
    df = pd.read_csv(INPUTS["truecell"])
    df["budget"] = pd.to_numeric(df["budget"])
    df["steps"] = pd.to_numeric(df["steps"])
    df["cross_pp_mean"] = pd.to_numeric(df["cross_pp_mean"])
    plt.figure(figsize=(7.0, 4.6))
    for steps, group in df.groupby("steps", sort=False):
        group = group.sort_values("budget")
        yerr_low = []
        yerr_high = []
        for _, row in group.iterrows():
            lo, hi = parse_ci(row.get("cross_pp_ci95"))
            if lo is None:
                yerr_low.append(0.0)
                yerr_high.append(0.0)
            else:
                y = float(row["cross_pp_mean"])
                yerr_low.append(max(0.0, y - lo))
                yerr_high.append(max(0.0, hi - y))
        color = COLORS["blue"] if steps == 3000 else COLORS["teal"]
        plt.errorbar(
            group["budget"],
            group["cross_pp_mean"],
            yerr=np.array([yerr_low, yerr_high]),
            marker="o",
            capsize=4,
            linewidth=2,
            color=color,
            label=f"{int(steps)} steps",
        )
    plt.axhline(0, color=COLORS["gray"], linewidth=1)
    plt.xlabel("Per-condition cell budget")
    plt.ylabel("Cross-background pp delta")
    plt.title("True-cell budget mechanism signal")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    return savefig("Fig_scaling_truecell_budget")


def plot_exposure() -> dict[str, str]:
    df = pd.read_csv(INPUTS["exposure"])
    df["cross_pp_delta"] = pd.to_numeric(df["cross_pp_delta"], errors="coerce")
    order = df.sort_values("cross_pp_delta", ascending=True)
    labels = order["arm"].astype(str)
    vals = order["cross_pp_delta"]
    colors = [
        COLORS["teal"] if v >= 0.01 else COLORS["gold"] if v >= 0 else COLORS["red"]
        for v in vals.fillna(0)
    ]
    plt.figure(figsize=(9.0, 5.6))
    plt.barh(labels, vals, color=colors)
    plt.axvline(0, color=COLORS["gray"], linewidth=1)
    plt.axvline(0.01, color=COLORS["gray"], linestyle="--", linewidth=1)
    plt.xlabel("Cross-background pp delta")
    plt.title("Condition exposure and breadth are non-monotonic")
    plt.grid(axis="x", alpha=0.25)
    return savefig("Fig_scaling_exposure_nonmonotonic")


def plot_noharm() -> dict[str, str]:
    df = pd.read_csv(INPUTS["noharm"])
    pp = df[df["metric"].str.contains("pearson_pert", regex=False)].copy()
    pp["label"] = pp["seed"].astype(str) + " " + pp["metric"].str.replace(":pearson_pert", "", regex=False)
    pp["delta_mean"] = pd.to_numeric(pp["delta_mean"], errors="coerce")
    pp["p_harm"] = pd.to_numeric(pp["p_harm"], errors="coerce")
    pp = pp.sort_values(["seed", "metric"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), sharey=True)
    axes[0].barh(pp["label"], pp["delta_mean"], color=[COLORS["teal"] if v >= 0 else COLORS["red"] for v in pp["delta_mean"]])
    axes[0].axvline(0, color=COLORS["gray"], linewidth=1)
    axes[0].set_xlabel("Delta mean")
    axes[0].set_title("Pearson perturbation delta")
    axes[1].barh(pp["label"], pp["p_harm"], color=[COLORS["red"] if v >= 0.5 else COLORS["gold"] for v in pp["p_harm"]])
    axes[1].axvline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1)
    axes[1].set_xlabel("p_harm")
    axes[1].set_title("Frozen no-harm veto")
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Budget128 6k fails canonical no-harm")
    return savefig("Fig_scaling_noharm_veto")


def plot_failure_map() -> dict[str, str]:
    df = pd.read_csv(INPUTS["failure_map"])
    df = df.copy()
    y = np.arange(len(df))
    color_map = {
        "main_text": COLORS["blue"],
        "supplement_or_failure_map": COLORS["gray"],
    }
    colors = [color_map.get(v, COLORS["gray"]) for v in df["manuscript_use"]]
    plt.figure(figsize=(9.0, 5.0))
    plt.barh(y, np.ones(len(df)), color=colors)
    plt.yticks(y, df["axis"])
    plt.xticks([])
    plt.xlim(0, 1.0)
    plt.title("Scaling axis claim boundary")
    for idx, row in df.iterrows():
        txt = str(row["claim_level"]) + " | promotion=" + str(row["promotion_allowed"])
        plt.text(0.02, idx, txt, va="center", ha="left", color="white" if row["manuscript_use"] == "main_text" else "black", fontsize=8)
    return savefig("FigS_scaling_failure_map")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "FigS_scaling_S0_provenance": plot_s0(),
        "Fig_scaling_truecell_budget": plot_truecell(),
        "Fig_scaling_exposure_nonmonotonic": plot_exposure(),
        "Fig_scaling_noharm_veto": plot_noharm(),
        "FigS_scaling_failure_map": plot_failure_map(),
    }
    payload = {
        "status": "scaling_figures_ready_no_gpu",
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "reads_figure_data_csv": True,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "inputs": {k: {"path": str(v), "sha256": sha256(v)} for k, v in INPUTS.items()},
        "figures": figures,
        "output_dir": str(OUT_DIR),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# LatentFM Scaling Figures",
        "",
        "Status: `scaling_figures_ready_no_gpu`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only plotting from figure-data CSV files.",
        "- Does not read checkpoints, canonical multi, Track C held-out query, train, infer, or use GPU.",
        "",
        "## Figures",
        "",
        "| figure | PNG | SVG |",
        "|---|---|---|",
    ]
    for name, outs in figures.items():
        lines.append(f"| `{name}` | `{outs['png']}` | `{outs['svg']}` |")
    lines.extend(["", "## JSON", "", f"`{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
