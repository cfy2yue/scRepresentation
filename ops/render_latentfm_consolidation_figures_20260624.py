#!/usr/bin/env python3
"""Render LatentFM consolidation figure candidates from prepared CSV tables."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures" / "latentfm_consolidation_20260624"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def short_name(name: str) -> str:
    replacements = {
        "forbidden_condition_pp_mmd_oracle": "oracle pp+mmd",
        "forbidden_condition_pp_oracle": "oracle pp",
        "forbidden_dataset_outcome_oracle": "dataset oracle",
        "bootstrap_target_noise_cap120": "bootstrap noise",
        "composite_safe_subset_cap120": "composite subset",
        "control_state_support_cap120": "control support",
        "reliability_condition_cap120": "reliability",
        "signed_neighborhood_cap120": "signed neigh.",
        "all_cap120_candidate": "cap120 all",
        "deployable_noop_anchor": "anchor",
        "perturbation_equivariant_prototype": "prototype",
        "factorized_gene_context": "gene x context",
        "xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42": "frozen v2 resfilm",
        "control_state_support_cap120": "control support",
        "composite_safe_subset": "composite subset",
        "bootstrap_target_noise": "bootstrap noise",
        "reliability_condition_cap120": "reliability",
        "signed_neighborhood": "signed neigh.",
    }
    return replacements.get(name, name.replace("_", " "))


def save_both(fig: plt.Figure, stem: str) -> dict[str, str]:
    png = FIG_DIR / f"{stem}.png"
    svg = FIG_DIR / f"{stem}.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "svg": str(svg)}


def plot_oracle_ladder(rows: list[dict[str, str]]) -> dict[str, str]:
    group = "internal_val_cross_background_seen_gene_proxy"
    rows = [r for r in rows if r["group"] == group]
    order = [
        "deployable_noop_anchor",
        "all_cap120_candidate",
        "bootstrap_target_noise_cap120",
        "reliability_condition_cap120",
        "signed_neighborhood_cap120",
        "composite_safe_subset_cap120",
        "control_state_support_cap120",
        "forbidden_dataset_outcome_oracle",
        "forbidden_condition_pp_mmd_oracle",
        "forbidden_condition_pp_oracle",
    ]
    by_name = {r["name"]: r for r in rows}
    rows = [by_name[n] for n in order if n in by_name]
    labels = [short_name(r["name"]) for r in rows]
    pp = [to_float(r["mean_pp_delta"]) or 0.0 for r in rows]
    dmin = [to_float(r["dataset_min_pp_delta"]) or 0.0 for r in rows]
    colors = []
    for r in rows:
        tier = r["tier"]
        if tier == "forbidden_oracle":
            colors.append("#8f4a9f")
        elif tier == "train_only_gate":
            colors.append("#d08a27")
        elif tier == "candidate":
            colors.append("#7a8a99")
        else:
            colors.append("#3b7f6f")

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    y = range(len(rows))
    ax.barh(y, pp, color=colors, alpha=0.88, label="mean pp delta")
    ax.scatter(dmin, list(y), color="#202020", s=34, zorder=3, label="dataset min")
    ax.axvline(0, color="#555555", linewidth=0.9)
    ax.axvline(0.01, color="#2e6da4", linewidth=0.9, linestyle="--", label="+0.010 gate")
    ax.axvline(-0.02, color="#9a3d3d", linewidth=0.9, linestyle=":", label="-0.020 tail gate")
    ax.set_yticks(list(y), labels)
    ax.set_xlabel("Pearson perturbation delta")
    ax.set_title("Track A oracle headroom vs deployable gates")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return save_both(fig, "oracle_headroom_ladder")


def plot_gain_tail(rows: list[dict[str, str]]) -> dict[str, str]:
    group = "internal_val_cross_background_seen_gene_proxy"
    rows = [r for r in rows if r["group"] == group]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    label_offsets = {
        "control_state_support_cap120": (0.004, -0.010),
        "composite_safe_subset": (0.004, 0.006),
        "signed_neighborhood": (0.002, 0.004),
        "bootstrap_target_noise": (0.002, 0.0),
        "reliability_condition_cap120": (0.002, 0.0),
        "perturbation_equivariant_prototype": (0.002, 0.0),
        "factorized_gene_context": (0.002, 0.0),
    }
    for r in rows:
        x = to_float(r["mean_pp_delta"])
        y = to_float(r["dataset_min_pp_delta"])
        if x is None or y is None:
            continue
        ax.scatter(x, y, s=70, alpha=0.9)
        dx, dy = label_offsets.get(r["branch"], (0.002, 0.0))
        ax.text(x + dx, y + dy, short_name(r["branch"]), fontsize=7.5, va="center")
    ax.axvline(0.01, color="#2e6da4", linestyle="--", linewidth=1.0, label="+0.010 mean gate")
    ax.axhline(-0.02, color="#9a3d3d", linestyle=":", linewidth=1.0, label="-0.020 tail gate")
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlim(-0.02, 0.085)
    ax.set_ylim(-0.75, 0.015)
    ax.set_xlabel("Mean pp delta")
    ax.set_ylabel("Worst dataset pp delta")
    ax.set_title("Average gain vs tail risk")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return save_both(fig, "gain_vs_tail_risk")


def plot_trackc(rows: list[dict[str, str]]) -> dict[str, str]:
    labels = [short_name(r["name"]) for r in rows]
    pp = [to_float(r["pearson_delta"]) or 0.0 for r in rows]
    zero = [to_float(r["unseen2_pearson_delta"]) for r in rows]
    zero_vals = [0.0 if v is None else v for v in zero]
    y = range(len(rows))
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.barh([i + 0.18 for i in y], pp, height=0.32, color="#4f7cac", label="support/query pp")
    ax.barh([i - 0.18 for i in y], zero_vals, height=0.32, color="#c46d5e", label="unseen2/zero overlap pp")
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.axvline(0.02, color="#2e6da4", linestyle="--", linewidth=0.9, label="+0.020 expansion gate")
    ax.set_yticks(list(y), labels)
    ax.set_xlabel("Pearson perturbation delta")
    ax.set_title("Track C support signal does not extend to zero-overlap")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return save_both(fig, "trackc_overlap_failure")


def plot_ot(rows: list[dict[str, str]]) -> dict[str, str]:
    lookup = {r["item"]: r for r in rows}
    items = [
        ("multinomial cost", lookup["pairing_signal_multinomial"], "value_a"),
        ("assignment cost", lookup["pairing_signal_assignment"], "value_a"),
        ("multinomial unique", lookup["pairing_signal_multinomial"], "value_b"),
        ("assignment delta err", lookup["pairing_signal_assignment"], "value_b"),
        ("expected corr.", lookup["pairing_quality_reliability"], "value_a"),
        ("contradict. corr.", lookup["pairing_quality_reliability"], "value_b"),
    ]
    labels = [x[0] for x in items]
    vals = [to_float(x[1][x[2]]) or 0.0 for x in items]
    colors = ["#4f7cac", "#4f7cac", "#d08a27", "#d08a27", "#3b7f6f", "#9a3d3d"]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = range(len(items))
    ax.bar(x, vals, color=colors, alpha=0.9)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(list(x), labels, rotation=25, ha="right")
    ax.set_title("OT is wired, but pairing quality does not translate to gain")
    ax.set_ylabel("Audit value")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return save_both(fig, "ot_wired_no_gain")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    oracle_rows = read_csv(REPORTS / "latentfm_oracle_headroom_ladder_20260624.csv")
    risk_rows = read_csv(REPORTS / "latentfm_gain_vs_tail_risk_20260624.csv")
    trackc_rows = read_csv(REPORTS / "latentfm_trackc_overlap_failure_panel_20260624.csv")
    ot_rows = read_csv(REPORTS / "latentfm_ot_wired_no_gain_panel_20260624.csv")

    figures = {
        "oracle_headroom_ladder": plot_oracle_ladder(oracle_rows),
        "gain_vs_tail_risk": plot_gain_tail(risk_rows),
        "trackc_overlap_failure": plot_trackc(trackc_rows),
        "ot_wired_no_gain": plot_ot(ot_rows),
    }
    manifest = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "latentfm_consolidation_figures_ready_no_gpu",
        "boundary": {
            "input_csvs": [
                str(REPORTS / "latentfm_oracle_headroom_ladder_20260624.csv"),
                str(REPORTS / "latentfm_gain_vs_tail_risk_20260624.csv"),
                str(REPORTS / "latentfm_trackc_overlap_failure_panel_20260624.csv"),
                str(REPORTS / "latentfm_ot_wired_no_gain_panel_20260624.csv"),
            ],
            "active_logs": False,
            "raw_canonical_or_query": False,
            "gpu": False,
        },
        "figures": figures,
    }
    manifest_path = FIG_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_path = REPORTS / "LATENTFM_CONSOLIDATION_FIGURES_20260624.md"
    lines = [
        "# LatentFM Consolidation Figures",
        "",
        "Status: `latentfm_consolidation_figures_ready_no_gpu`",
        "",
        "## Boundary",
        "",
        "- Renders only from prepared consolidation CSV tables.",
        "- No active logs, raw canonical/query artifacts, canonical multi selection, training, inference, or GPU.",
        "",
        "## Figures",
        "",
        "| Figure | PNG | SVG |",
        "|---|---|---|",
    ]
    for name, paths in figures.items():
        lines.append(f"| `{name}` | `{paths['png']}` | `{paths['svg']}` |")
    lines.extend(["", "## Manifest", "", f"`{manifest_path}`", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)
    print(manifest_path)
    for paths in figures.values():
        print(paths["png"])
        print(paths["svg"])


if __name__ == "__main__":
    main()
