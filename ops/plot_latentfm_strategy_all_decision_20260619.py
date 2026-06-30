#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
CSV_IN = ROOT / "reports/latentfm_strategy_all_decision_20260619.csv"
OUT_BASE = ROOT / "reports/latentfm_strategy_all_decision_20260619"


def fnum(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_rows() -> list[dict[str, Any]]:
    if not CSV_IN.is_file():
        return []
    with CSV_IN.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        if str(row.get("complete", "")).lower() != "true":
            continue
        for key in (
            "test_mmd",
            "mmd_ratio_to_ref",
            "test_pp",
            "delta_test_pp",
            "delta_multi_seen_pp",
            "delta_multi_unseen1_pp",
            "delta_multi_unseen2_pp",
            "delta_family_gene_pp",
            "delta_family_drug_pp",
            "score",
        ):
            row[key] = fnum(row.get(key))
        required = (
            "test_mmd",
            "test_pp",
            "delta_multi_seen_pp",
            "delta_multi_unseen1_pp",
            "delta_multi_unseen2_pp",
            "delta_family_gene_pp",
            "delta_family_drug_pp",
            "score",
        )
        if all(row.get(key) is not None for key in required):
            out.append(row)
    return sorted(out, key=lambda r: float(r["score"]), reverse=True)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.5,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "lines.linewidth": 0.9,
            "lines.markersize": 4.0,
        }
    )


def color_for(row: dict[str, Any]) -> str:
    decision = str(row.get("decision", ""))
    if decision == "repeat_candidate":
        return "#009E73"
    if decision == "diagnostic_candidate":
        return "#0072B2"
    if str(row.get("backbone")) == "scfoundation":
        return "#D55E00"
    return "#666666"


def marker_for(row: dict[str, Any]) -> str:
    return "o" if str(row.get("backbone")) == "scfoundation" else "s"


def label(row: dict[str, Any]) -> str:
    return str(row.get("run", "NA")).replace("_", "\n")


def save(fig: plt.Figure) -> None:
    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(OUT_BASE.with_suffix(suffix))
    missing = [
        str(OUT_BASE.with_suffix(suffix))
        for suffix in (".pdf", ".png", ".svg")
        if (not OUT_BASE.with_suffix(suffix).is_file()) or OUT_BASE.with_suffix(suffix).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"missing or empty strategy decision figure outputs: {missing}")
    placeholder = OUT_BASE.with_suffix(".txt")
    if placeholder.is_file():
        placeholder.unlink()


def plot(rows: list[dict[str, Any]]) -> int:
    if not rows:
        OUT_BASE.with_suffix(".txt").write_text(
            "No complete LatentFM strategy rows are available yet.\n",
            encoding="utf-8",
        )
        return 0
    apply_style()
    top = rows[:12]
    x = np.arange(len(top))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0))
    ax = axes[0, 0]
    for row in top:
        ax.scatter(
            row["test_mmd"],
            row["test_pp"],
            s=42,
            marker=marker_for(row),
            color=color_for(row),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
        )
        ax.text(row["test_mmd"], row["test_pp"], str(row.get("run")), fontsize=5.5, ha="left", va="bottom")
    ax.axhline(0, color="#999999", lw=0.6, ls="--")
    ax.set_xlabel("test MMD (lower is better)")
    ax.set_ylabel("test pp / perturbation Pearson")
    ax.set_title("a  Distribution vs perturbation signal")
    ax.grid(alpha=0.2, lw=0.5)

    ax = axes[0, 1]
    width = 0.26
    vals = {
        "seen": [row["delta_multi_seen_pp"] or 0 for row in top],
        "unseen1": [row["delta_multi_unseen1_pp"] or 0 for row in top],
        "unseen2": [row["delta_multi_unseen2_pp"] or 0 for row in top],
    }
    ax.bar(x - width, vals["seen"], width, color="#56B4E9", label="seen")
    ax.bar(x, vals["unseen1"], width, color="#009E73", label="unseen1")
    ax.bar(x + width, vals["unseen2"], width, color="#CC79A7", label="unseen2")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([label(row) for row in top], rotation=0)
    ax.set_ylabel("delta pp vs matched reference")
    ax.set_title("b  Multi-perturbation split movement")
    ax.legend(ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.2, lw=0.5)

    ax = axes[1, 0]
    gene = [row["delta_family_gene_pp"] or 0 for row in top]
    drug = [row["delta_family_drug_pp"] or 0 for row in top]
    ax.bar(x - width / 2, gene, width, color="#4C72B0", label="gene")
    ax.bar(x + width / 2, drug, width, color="#DD8452", label="drug")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([label(row) for row in top], rotation=0)
    ax.set_ylabel("delta family pp")
    ax.set_title("c  Family preservation")
    ax.legend(ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.2, lw=0.5)

    ax = axes[1, 1]
    scores = [row["score"] or 0 for row in top]
    colors = [color_for(row) for row in top]
    ax.barh(np.arange(len(top)), scores[::-1], color=colors[::-1], edgecolor="black", linewidth=0.3)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([str(row.get("run")) for row in top[::-1]])
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("composite strategy score")
    ax.set_title("d  Decision-score ranking")
    ax.grid(axis="x", alpha=0.2, lw=0.5)

    fig.suptitle("LatentFM strategy probe decision summary", fontsize=10, fontweight="bold", x=0.03, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig)
    plt.close(fig)
    return 0


def main() -> int:
    return plot(load_rows())


if __name__ == "__main__":
    raise SystemExit(main())
