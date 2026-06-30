#!/usr/bin/env python3
"""Render frozen Track C support-context v2 reporting figure panels.

This script is reporting-only. It reads frozen narrative/manifest/table/caveat
artifacts and writes PNG/SVG figure panels plus a render manifest. It does not
train, evaluate, tune, inspect active logs, or authorize GPU/query reuse.
"""

from __future__ import annotations

import csv
import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures"

OUT_JSON = REPORTS / "latentfm_trackc_support_context_v2_figure_panels_20260623.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_FIGURE_PANELS_20260623.md"

INPUTS = {
    "narrative_json": REPORTS / "latentfm_trackc_support_context_v2_manuscript_narrative_20260623.json",
    "figure_manifest_json": REPORTS / "latentfm_trackc_support_context_v2_figure_manifest_20260623.json",
    "manuscript_table_csv": REPORTS / "latentfm_trackc_support_context_v2_manuscript_table_20260623.csv",
    "caveat_table_csv": REPORTS / "latentfm_trackc_support_context_v2_caveat_table_20260623.csv",
    "final_package_audit_json": REPORTS / "latentfm_trackc_support_context_v2_final_package_audit_20260623.json",
}

EXPECTED_STATUSES = {
    "narrative_json": "support_context_v2_manuscript_narrative_ready",
    "figure_manifest_json": "support_context_v2_figure_manifest_ready",
    "final_package_audit_json": "trackc_support_context_v2_final_package_audit_pass",
}

OUTPUTS = {
    "fig1_png": FIG_DIR / "latentfm_trackc_support_context_v2_fig1_gate_chain.png",
    "fig1_svg": FIG_DIR / "latentfm_trackc_support_context_v2_fig1_gate_chain.svg",
    "fig2_png": FIG_DIR / "latentfm_trackc_support_context_v2_fig2_query_strata_failures.png",
    "fig2_svg": FIG_DIR / "latentfm_trackc_support_context_v2_fig2_query_strata_failures.svg",
    "extfig1_png": FIG_DIR / "latentfm_trackc_support_context_v2_extfig1_claim_boundary.png",
    "extfig1_svg": FIG_DIR / "latentfm_trackc_support_context_v2_extfig1_claim_boundary.svg",
}


PALETTE = {
    "green": "#2E8B57",
    "blue": "#2B6CB0",
    "teal": "#2C7A7B",
    "orange": "#C05621",
    "red": "#B83232",
    "gray": "#4A5568",
    "light_gray": "#EDF2F7",
    "dark": "#1A202C",
    "purple": "#6B46C1",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path),
    }


def row_by_role(rows: list[dict[str, str]], role: str) -> dict[str, str]:
    for row in rows:
        if row.get("role") == role:
            return row
    raise KeyError(role)


def fnum(value: Any) -> float:
    return float(value)


def fmt(value: Any, signed: bool = True) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    return f"{number:+.3f}" if signed else f"{number:.3f}"


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A0AEC0")
    ax.spines["bottom"].set_color("#A0AEC0")
    ax.tick_params(colors=PALETTE["gray"], labelsize=9)
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8, zorder=0)


def metric_bar(ax: plt.Axes, rows: list[dict[str, str]], labels: list[str], colors: list[str], title: str) -> None:
    xs = list(range(len(rows)))
    vals = [fnum(row["delta"]) for row in rows]
    lows = [fnum(row["ci95_low"]) for row in rows]
    highs = [fnum(row["ci95_high"]) for row in rows]
    span = max(highs + [0.0]) - min(lows + [0.0])
    label_offset = max(span * 0.06, 0.001)
    yerr = [[vals[i] - lows[i] for i in xs], [highs[i] - vals[i] for i in xs]]
    ax.axhline(0, color="#2D3748", linewidth=1)
    ax.bar(xs, vals, color=colors, width=0.62, zorder=3)
    ax.errorbar(xs, vals, yerr=yerr, fmt="none", ecolor="#1A202C", elinewidth=1.2, capsize=4, zorder=4)
    for x, val, row in zip(xs, vals, rows):
        offset = label_offset if val >= 0 else -label_offset
        va = "bottom" if val >= 0 else "top"
        ax.text(x, val + offset, f"{fmt(val)}\np_harm {float(row['p_harm']):.3f}", ha="center", va=va, fontsize=8, color=PALETTE["dark"])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Delta vs anchor")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=PALETTE["dark"])
    style_axis(ax)


def save_figure(fig: plt.Figure, png_path: Path, svg_path: Path) -> None:
    fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_fig1(rows: dict[str, dict[str, str]], out_png: Path, out_svg: Path) -> None:
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15])
    ax_top = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    ax_top.set_axis_off()
    stages = [
        ("1", "Support-val\ncapped pass", "trainselect support only\nquery absent"),
        ("2", "Uncapped canonical\nno-harm", "test_single/family\nexact no-op"),
        ("3", "Query-free\nfreeze", "route/checkpoint\nlocked"),
        ("4", "One-shot\nquery diagnostic", "final-only\nno tuning"),
        ("5", "Reporting\nboundary", "diagnostic claim\nnot formal solved"),
    ]
    x0, gap, w, h = 0.03, 0.19, 0.15, 0.35
    for idx, (num, title, subtitle) in enumerate(stages):
        x = x0 + idx * gap
        box = FancyBboxPatch(
            (x, 0.38),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1.4,
            edgecolor=PALETTE["blue"] if idx < 4 else PALETTE["orange"],
            facecolor="#EBF8FF" if idx < 4 else "#FFF5EB",
            transform=ax_top.transAxes,
        )
        ax_top.add_patch(box)
        ax_top.text(x + w / 2, 0.66, num, transform=ax_top.transAxes, fontsize=13, fontweight="bold", color=PALETTE["dark"], ha="center")
        ax_top.text(x + w / 2, 0.57, title, transform=ax_top.transAxes, fontsize=10.2, fontweight="bold", color=PALETTE["dark"], ha="center", va="center")
        ax_top.text(x + w / 2, 0.44, subtitle, transform=ax_top.transAxes, fontsize=8.3, color=PALETTE["gray"], ha="center", va="center")
        if idx < len(stages) - 1:
            start = (x + w + 0.01, 0.55)
            end = (x + gap - 0.015, 0.55)
            ax_top.add_patch(FancyArrowPatch(start, end, transform=ax_top.transAxes, arrowstyle="-|>", mutation_scale=16, color="#718096", linewidth=1.3))

    ax_top.text(
        0.03,
        0.18,
        "Boundary: support-val, canonical no-harm, and held-out query are shown as separate evidence layers. Query rows did not select route, checkpoint, thresholds, or features.",
        transform=ax_top.transAxes,
        fontsize=10,
        color=PALETTE["dark"],
    )

    bar_rows = [rows["support_pp"], rows["support_mmd"], rows["query_pp"], rows["query_mmd"]]
    labels = ["Support\nPearson", "Support\nMMD", "Query\nPearson", "Query\nMMD"]
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["green"], PALETTE["purple"]]
    metric_bar(ax_bar, bar_rows, labels, colors, "Aggregate frozen diagnostic evidence")
    ax_bar.set_ylim(-0.05, 0.16)
    fig.suptitle("Track C support-context v2: frozen gate chain and aggregate evidence", fontsize=16, fontweight="bold", color=PALETTE["dark"])
    save_figure(fig, out_png, out_svg)


def render_fig2(rows: dict[str, dict[str, str]], caveat_rows: list[dict[str, str]], out_png: Path, out_svg: Path) -> None:
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.2], width_ratios=[1.05, 0.95])
    ax_strata = fig.add_subplot(gs[0, 0])
    ax_mmd = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])

    strata_rows = [rows["query_seen"], rows["query_unseen1"], rows["query_unseen2_pp"]]
    metric_bar(
        ax_strata,
        strata_rows,
        ["Seen", "Unseen1", "Unseen2"],
        [PALETTE["green"], PALETTE["blue"], PALETTE["orange"]],
        "Held-out query Pearson by stratum",
    )
    ax_strata.set_ylim(-0.05, 0.22)

    mmd_rows = [rows["query_mmd"], rows["query_unseen2_mmd"]]
    metric_bar(
        ax_mmd,
        mmd_rows,
        ["All query\nMMD", "Unseen2\nMMD"],
        [PALETTE["purple"], PALETTE["teal"]],
        "Clamped MMD remains improved",
    )
    ax_mmd.set_ylim(-0.012, 0.005)

    ax_table.set_axis_off()
    top = caveat_rows[:8]
    table_data = [
        [row["stratum"], row["dataset"].replace("NormanWeissman2019_filtered", "Norman"), row["condition"], fmt(row["pp_delta"]), fmt(row["mmd_delta"])]
        for row in top
    ]
    table = ax_table.table(
        cellText=table_data,
        colLabels=["Stratum", "Dataset", "Condition", "Pearson delta", "MMD delta"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.12, 0.16, 0.28, 0.14, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E0")
        if r == 0:
            cell.set_facecolor(PALETTE["light_gray"])
            cell.set_text_props(weight="bold", color=PALETTE["dark"])
        elif c == 3 and table_data[r - 1][3].startswith("-"):
            cell.set_text_props(color=PALETTE["red"])
    ax_table.set_title("Worst condition-level Pearson rows remain visible", loc="left", fontsize=12, fontweight="bold", color=PALETTE["dark"], pad=10)

    fig.suptitle("Track C support-context v2: query strata and failure cases", fontsize=16, fontweight="bold", color=PALETTE["dark"])
    save_figure(fig, out_png, out_svg)


def render_extfig1(rows: dict[str, dict[str, str]], narrative: dict[str, Any], audit: dict[str, Any], out_png: Path, out_svg: Path) -> None:
    fig = plt.figure(figsize=(14, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.15])
    ax_noharm = fig.add_subplot(gs[0, 0])
    ax_claim = fig.add_subplot(gs[0, 1])
    ax_prov = fig.add_subplot(gs[1, :])

    noharm_rows = [rows["canonical_single"], rows["canonical_family"]]
    metric_bar(
        ax_noharm,
        noharm_rows,
        ["test_single\nPearson", "family_gene\nPearson"],
        [PALETTE["gray"], PALETTE["gray"]],
        "Canonical support-absent no-harm",
    )
    ax_noharm.set_ylim(-0.02, 0.03)

    ax_claim.set_axis_off()
    allowed = narrative.get("allowed_claims", [])[:3]
    disallowed = narrative.get("disallowed_claims", [])[:3]
    y = 0.95
    ax_claim.text(0.0, y, "Allowed wording", fontsize=12, fontweight="bold", color=PALETTE["green"], transform=ax_claim.transAxes)
    y -= 0.1
    for item in allowed:
        ax_claim.text(0.02, y, "- " + wrap(item, 58), fontsize=9, color=PALETTE["dark"], transform=ax_claim.transAxes, va="top")
        y -= 0.16
    y -= 0.02
    ax_claim.text(0.0, y, "Disallowed wording", fontsize=12, fontweight="bold", color=PALETTE["red"], transform=ax_claim.transAxes)
    y -= 0.1
    for item in disallowed:
        ax_claim.text(0.02, y, "- " + wrap(item, 58), fontsize=9, color=PALETTE["dark"], transform=ax_claim.transAxes, va="top")
        y -= 0.16

    ax_prov.set_axis_off()
    split_hashes = audit.get("split_hashes") or {}
    freeze_hashes = audit.get("freeze_hashes") or {}
    prov_rows = [
        ["canonical split", (split_hashes.get("canonical_split") or "NA")[:18]],
        ["safe trainselect split", (split_hashes.get("safe_trainselect_split") or "NA")[:18]],
        ["full v2 split", (split_hashes.get("full_v2_split") or "NA")[:18]],
        ["anchor checkpoint", (freeze_hashes.get("anchor_checkpoint") or "NA")[:18]],
        ["candidate checkpoint", (freeze_hashes.get("candidate_checkpoint") or "NA")[:18]],
        ["CoupledFM commit", ((audit.get("git") or {}).get("coupledfm_commit") or "NA")[:18]],
    ]
    table = ax_prov.table(
        cellText=prov_rows,
        colLabels=["Provenance item", "SHA/commit prefix"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.34, 0.42],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E0")
        if r == 0:
            cell.set_facecolor(PALETTE["light_gray"])
            cell.set_text_props(weight="bold", color=PALETTE["dark"])
    ax_prov.set_title("Artifact provenance for reproducible reporting", loc="left", fontsize=12, fontweight="bold", color=PALETTE["dark"], pad=10)

    fig.suptitle("Track C support-context v2: claim boundary and provenance", fontsize=16, fontweight="bold", color=PALETTE["dark"])
    save_figure(fig, out_png, out_svg)


def pixel_check(path: Path) -> dict[str, Any]:
    arr = mpimg.imread(path)
    return {
        "path": str(path),
        "shape": list(arr.shape),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "nonblank": bool(arr.std() > 0.005),
    }


def render_manifest(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Support-Context V2 Figure Panels",
        "",
        f"Status: `{payload['status']}`",
        "GPU authorization: `none`",
        "Held-out query reuse forbidden: `True`",
        "",
        "## Boundary",
        "",
        "- Reporting-only render from frozen artifacts.",
        "- No active logs, model training, route selection, threshold tuning, or query reuse.",
        "- The claim remains a frozen diagnostic, not formal multi solved.",
        "",
        "## Rendered Panels",
        "",
        "| panel | PNG | SVG | pixel std | nonblank |",
        "|---|---|---|---:|---:|",
    ]
    for panel in payload["panels"]:
        check = panel["pixel_check"]
        lines.append(
            f"| `{panel['panel_id']}` | `{panel['png']}` | `{panel['svg']}` | "
            f"{check['std']:.6f} | `{check['nonblank']}` |"
        )
    lines.extend(["", "## Input Checks", "", "| check | passed | evidence |", "|---|---:|---|"])
    for row in payload["checks"]:
        evidence = row["evidence"]
        if isinstance(evidence, dict):
            evidence = json.dumps(evidence, sort_keys=True)
        lines.append(f"| `{row['name']}` | `{row['passed']}` | `{evidence}` |")
    lines.extend(["", "## Output Hashes", "", "| output | exists | size | sha256 |", "|---|---:|---:|---|"])
    for name, meta in payload["outputs"].items():
        lines.append(f"| `{name}` | `{meta['exists']}` | {meta['size_bytes']} | `{(meta['sha256'] or 'NA')[:16]}` |")
    lines.extend(["", "## Next Action", "", payload["next_action"], ""])
    return "\n".join(lines)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    narrative = load_json(INPUTS["narrative_json"])
    figure_manifest = load_json(INPUTS["figure_manifest_json"])
    audit = load_json(INPUTS["final_package_audit_json"])
    table_rows = load_csv(INPUTS["manuscript_table_csv"])
    caveat_rows = load_csv(INPUTS["caveat_table_csv"])
    rows = figure_manifest["main_metric_rows"]

    checks: list[dict[str, Any]] = []
    for name, path in INPUTS.items():
        checks.append({"name": f"exists:{name}", "passed": path.is_file(), "evidence": str(path)})
    status_objects = {
        "narrative_json": narrative,
        "figure_manifest_json": figure_manifest,
        "final_package_audit_json": audit,
    }
    for name, expected in EXPECTED_STATUSES.items():
        observed = status_objects[name].get("status")
        checks.append({"name": f"status:{name}", "passed": observed == expected, "evidence": {"expected": expected, "observed": observed}})
    checks.append({"name": "csv:manuscript_rows_ge_12", "passed": len(table_rows) >= 12, "evidence": len(table_rows)})
    checks.append({"name": "csv:caveat_rows_ge_12", "passed": len(caveat_rows) >= 12, "evidence": len(caveat_rows)})

    render_fig1(rows, OUTPUTS["fig1_png"], OUTPUTS["fig1_svg"])
    render_fig2(rows, caveat_rows, OUTPUTS["fig2_png"], OUTPUTS["fig2_svg"])
    render_extfig1(rows, narrative, audit, OUTPUTS["extfig1_png"], OUTPUTS["extfig1_svg"])

    panels = [
        {
            "panel_id": "fig1_gate_chain",
            "png": str(OUTPUTS["fig1_png"]),
            "svg": str(OUTPUTS["fig1_svg"]),
            "pixel_check": pixel_check(OUTPUTS["fig1_png"]),
        },
        {
            "panel_id": "fig2_query_strata_failures",
            "png": str(OUTPUTS["fig2_png"]),
            "svg": str(OUTPUTS["fig2_svg"]),
            "pixel_check": pixel_check(OUTPUTS["fig2_png"]),
        },
        {
            "panel_id": "extfig1_claim_boundary",
            "png": str(OUTPUTS["extfig1_png"]),
            "svg": str(OUTPUTS["extfig1_svg"]),
            "pixel_check": pixel_check(OUTPUTS["extfig1_png"]),
        },
    ]
    for panel in panels:
        checks.append({"name": f"pixel:{panel['panel_id']}:nonblank", "passed": panel["pixel_check"]["nonblank"], "evidence": panel["pixel_check"]})

    outputs = {name: artifact(path) for name, path in OUTPUTS.items()}
    for name, meta in outputs.items():
        checks.append({"name": f"exists:{name}", "passed": meta["exists"] and (meta["size_bytes"] or 0) > 1000, "evidence": meta})

    failed = [row for row in checks if not row["passed"]]
    payload = {
        "status": "support_context_v2_figure_panels_ready" if not failed else "support_context_v2_figure_panels_needs_review",
        "timestamp": "2026-06-23 12:48 CST",
        "boundary": {
            "reporting_only": True,
            "gpu_authorization": "none",
            "heldout_query_reuse_forbidden": True,
            "selection_or_tuning": False,
            "active_log_reads": False,
        },
        "panels": panels,
        "checks": checks,
        "failed_checks": failed,
        "outputs": outputs,
        "inputs": {name: artifact(path) for name, path in INPUTS.items()},
        "next_action": "Use the rendered panels with the manuscript narrative; any new modeling still requires a materially new query-free CPU gate.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render_manifest(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
