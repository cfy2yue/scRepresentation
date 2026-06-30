#!/usr/bin/env python3
"""
Plot baseline model performance as simple bar charts (no scaling-law fit).

Inputs:
  latent/runs/baseline/{run}/eval_results.json
  (fallback: iid_eval_results.json written by latent/train.py)

Output:
  latent/runs/baseline/plots/baseline_metrics_bar.png (or .pdf)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


def _pick_eval_json(run_dir: Path) -> Path | None:
    for name in ("eval_results.json", "iid_eval_results.json"):
        p = run_dir / name
        if p.is_file():
            return p
    return None


def load_metrics(runs_dir: Path, models: list[str]) -> dict[str, dict[str, float]]:
    data: dict[str, dict[str, float]] = {}
    for m in models:
        run_dir = runs_dir / m
        f = _pick_eval_json(run_dir)
        if f is None:
            continue
        with open(f, "r", encoding="utf-8") as fh:
            j = json.load(fh)
        data[m] = {
            "mse": float(j["test_mse"]),
            "mmd": float(j["test_mmd"]),
            "dp": float(j["direct_pearson"]),
            "pc": float(j["pearson_ctrl"]),
            "pp": float(j["pearson_pert"]),
        }
    return data


def _auto_ylim(vals: list[float], metric: str) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    vmin = float(arr.min())
    vmax = float(arr.max())
    spread = vmax - vmin

    if spread < 1e-12:
        pad = max(abs(vmin) * 0.02, 1e-6)
    else:
        pad = spread * 0.35

    low = vmin - pad
    high = vmax + pad
    if metric in {"dp", "pc", "pp"}:
        low = max(-1.0, low)
        high = min(1.0, high)
    return low, high


def _fmt_value(v: float, metric: str) -> str:
    if metric == "mse":
        return f"{v:.7f}"
    if metric == "mmd":
        return f"{v:.5f}"
    return f"{v:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("../runs/baseline"))
    parser.add_argument("--out-dir", type=Path, default=Path("../runs/baseline/plots"))
    parser.add_argument("--format", choices=["png", "pdf"], default="png")
    args = parser.parse_args()

    runs_dir = args.runs_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_order = ["mse_only", "ode_mmd_v2_a", "ode_mmd_v2_b"]
    model_labels = ["MSE only", "MMD v2 A", "MMD v2 B"]
    metrics = load_metrics(runs_dir, model_order)
    available = [m for m in model_order if m in metrics]
    if not available:
        print(f"No eval_results.json found under: {runs_dir}")
        print("Run: bash scripts/run_baseline_eval.sh")
        return

    labels = [model_labels[model_order.index(m)] for m in available]
    x = np.arange(len(available))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 1,
            "axes.edgecolor": "#2F2F2F",
            "figure.dpi": 150,
        }
    )

    # 2x3: Pearson metrics + distance/loss metrics
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 4.8))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.9, wspace=0.32, hspace=0.45)

    metric_specs = [
        ("dp", "Direct Pearson (higher better)"),
        ("pc", "Pearson vs control (higher better)"),
        ("pp", "Pearson vs pert (higher better)"),
        ("mmd", "MMD (lower better)"),
        ("mse", "MSE (lower better)"),
    ]

    bar_color = "#4C78A8"
    edge_color = "#2B2B2B"

    for ax, (k, title) in zip(axes.flat, metric_specs):
        vals = [metrics[m][k] for m in available]
        ax.bar(x, vals, color=bar_color, edgecolor=edge_color, linewidth=0.6)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_title(title, fontsize=8, pad=8)
        ax.grid(axis="y", linestyle="-", alpha=0.2, linewidth=0.8)
        ax.set_ylim(*_auto_ylim(vals, k))

        if k == "mse":
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_scientific(True)
            fmt.set_powerlimits((-3, 3))
            ax.yaxis.set_major_formatter(fmt)

        for xi, v in zip(x, vals):
            ax.text(xi, v, _fmt_value(v, k), ha="center", va="bottom", fontsize=7)

    for ax in axes.flat[len(metric_specs):]:
        ax.axis("off")

    fig.suptitle("Latent Baseline Evaluation (Full Test Set)", fontsize=10)

    out_path = out_dir / f"baseline_metrics_bar.{args.format}"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

