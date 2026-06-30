#!/usr/bin/env python3
"""Generate the Nature-style benchmark figure set under ``output/figures/``.

Usage (from scFM root):
    python benchmark/cli/build_figures.py [--scfm-root .] [--out-dir output/figures]
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import subprocess
import traceback
from pathlib import Path
from typing import Any, Callable

import sys

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parents[2] / "fm"))

from benchmark.plot import data as D
from benchmark.plot import figures as F
from benchmark.plot import style as ST
import paths


def _try_figure(
    name: str,
    builder: Callable[[], tuple[Path, Path]],
) -> tuple[dict[str, str], dict[str, str] | None]:
    try:
        pdf, png = builder()
    except Exception as exc:
        return {}, {
            "name": name,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
    svg = pdf.with_suffix(".svg")
    meta = pdf.with_suffix(".meta.json")
    record = {"name": name, "pdf": str(pdf), "png": str(png)}
    if svg.is_file():
        record["svg"] = str(svg)
    if meta.is_file():
        record["meta"] = str(meta)
    return record, None


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
    return out or None


def _existing_inputs(paths_to_check: dict[str, Path]) -> dict[str, str]:
    return {k: str(v) for k, v in paths_to_check.items() if v.is_file()}


def _model_coverage(df) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    chempert_only: list[str] = []
    if not {"model", "dataset_id", "category"}.issubset(df.columns):
        return {"by_model": coverage, "chempert_only_models": chempert_only}
    for model, sub in df.groupby("model"):
        categories = sorted(map(str, sub["category"].dropna().unique()))
        datasets = sorted(map(str, sub["dataset_id"].dropna().unique()))
        latent_spaces = sorted(map(str, sub["latent_space"].dropna().unique())) if "latent_space" in sub else []
        coverage[str(model)] = {
            "categories": categories,
            "n_categories": int(len(categories)),
            "n_datasets": int(len(datasets)),
            "latent_spaces": latent_spaces,
        }
        if categories == ["chempert"]:
            chempert_only.append(str(model))
    return {
        "by_model": coverage,
        "chempert_only_models": sorted(chempert_only),
        "coverage_note": (
            "Models listed under chempert_only_models currently contribute only "
            "chemical perturbation rows to aggregate benchmark figures; do not "
            "interpret their blank atlas/genepert blocks as failed metrics."
        ),
    }


def _figure_provenance(df, scfm: Path, out_dir: Path) -> dict[str, Any]:
    metrics_root = paths.output_root() / "metrics"
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scfm_root": str(scfm),
        "git_commit": _git_commit(scfm),
        "out_dir": str(out_dir),
        "input_files": _existing_inputs({
            "summary_all": metrics_root / "summary_all.csv",
            "summary_all_raw": metrics_root / "summary_all_raw.csv",
            "summary_all_pca128": metrics_root / "summary_all_pca128.csv",
            "run_status_transcriptformer_chempert": metrics_root / "run_status_transcriptformer_chempert.jsonl",
            "run_manifest_transcriptformer_chempert": metrics_root / "run_manifest_transcriptformer_chempert.jsonl",
        }),
        "n_rows_summary_all": int(len(df)),
        "models": sorted(map(str, df["model"].dropna().unique())),
        "model_display": {m: ST.MODEL_DISPLAY.get(m, m) for m in sorted(map(str, df["model"].dropna().unique()))},
        "latent_spaces": sorted(map(str, df["latent_space"].dropna().unique())),
        "dataset_ids": sorted(map(str, df["dataset_id"].dropna().unique())),
        "categories": sorted(map(str, df["category"].dropna().unique())) if "category" in df.columns else [],
        "model_coverage": _model_coverage(df),
        "style": {
            "png_dpi": 600,
            "pdf_fonttype": 42,
            "svg_fonttype": "none",
            "palette": "benchmark.plot.style.MODEL_PALETTE",
        },
    }


def _augment_figure_meta(records: list[dict[str, str]], provenance: dict[str, Any]) -> None:
    for rec in records:
        meta_s = rec.get("meta")
        if not meta_s:
            continue
        meta_path = Path(meta_s)
        try:
            obj = json.loads(meta_path.read_text())
        except Exception:
            obj = {}
        obj["provenance"] = provenance
        meta_path.write_text(json.dumps(obj, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scfm-root", type=Path, default=_HERE.parents[2])
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Default: <scfm-root>/output/figures")
    args = ap.parse_args()

    scfm = args.scfm_root.resolve()
    out_dir = (args.out_dir or paths.output_root() / "figures").resolve()

    ST.apply_rcparams()

    df = D.load_wide(scfm)
    per_pert = D.per_perturb_table(scfm)
    df = D.augment_with_topk_spearman(df, scfm, out_dir, per_pert)
    df = D.augment_with_mantel_spearman(df, scfm, out_dir)
    provenance = _figure_provenance(df, scfm, out_dir)

    has_atlas = df["category"].isin(("atlas", "atlas_TS")).any()
    has_chempert = df["category"].eq("chempert").any()
    has_genepert = df["category"].eq("genepert").any()
    has_atlas_efficiency = (
        (paths.output_root() / "embeddings").glob("*/*/raw/meta.json")
    )
    has_atlas_efficiency = any(
        p.parents[1].name in {
            "Blood",
            "BoneMarrow",
            "Heart",
            "Lung",
            "LymphNode",
            "Skin",
            "TS_Immune_xtissue",
        }
        for p in has_atlas_efficiency
    )

    figure_specs = [
        ("fig1_overview", lambda: F.fig1_overview(df, out_dir), True, ""),
        ("fig2_atlas", lambda: F.fig2_atlas(df, out_dir), has_atlas, "no atlas metrics in summary_all.csv"),
        ("fig3_geometry", lambda: F.fig3_geometry(df, out_dir), True, ""),
        ("fig4_chempert", lambda: F.fig4_chempert(df, out_dir, per_pert_df=per_pert), has_chempert, "no chempert rows in summary_all.csv"),
        ("fig4b_genepert", lambda: F.fig4b_genepert(df, out_dir, per_pert_df=per_pert), has_genepert, "no genepert rows in summary_all.csv"),
        ("fig4_2_chempert_sim", lambda: F.fig4_2_chempert_sim(df, out_dir), has_chempert, "no chempert rows in summary_all.csv"),
        ("fig4b_2_genepert_sim", lambda: F.fig4b_2_genepert_sim(df, out_dir), has_genepert, "no genepert rows in summary_all.csv"),
        ("fig5_overall", lambda: F.fig5_overall(df, out_dir), True, ""),
        ("fig6_efficiency", lambda: F.fig6_efficiency(df, out_dir), has_atlas_efficiency, "no atlas throughput metadata under embeddings"),
        ("fig_supp_all_metrics", lambda: F.fig_supp_all_metrics(df, out_dir), True, ""),
    ]
    figure_records: list[dict[str, str]] = []
    failed_figures: list[dict[str, str]] = []
    skipped_figures: list[dict[str, str]] = []
    for name, builder, should_run, skip_reason in figure_specs:
        if not should_run:
            skipped_figures.append({"name": name, "reason": skip_reason})
            continue
        record, failure = _try_figure(name, builder)
        if failure:
            failed_figures.append(failure)
        else:
            figure_records.append(record)
    _augment_figure_meta(figure_records, provenance)

    manifest = {
        **provenance,
        "figures": figure_records,
        "failed_figures": failed_figures,
        "skipped_figures": skipped_figures,
        "n_figures": int(len(figure_records)),
        "n_failed_figures": int(len(failed_figures)),
        "n_skipped_figures": int(len(skipped_figures)),
        "n_models": int(df["model"].nunique()),
        "n_datasets": int(df["dataset_id"].nunique()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
