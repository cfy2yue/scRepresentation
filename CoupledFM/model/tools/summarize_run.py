"""
Summarize a training run into a summary/ folder.

Auto-detects run type:
  - Pretrain run:  contains metrics_log.jsonl  → training curve PNG + metrics CSV + report
  - Sweep run:     contains sweep_*/           → ranked table CSV + curves + markdown report

Usage:
  # Single pretrain run
  python -m model.tools.summarize_run output/cellgene_pretrain_20260521_120000

  # Sweep run (pass the sweep root)
  python -m model.tools.summarize_run output/sweep_gene_pert_32_20260521_120000

  # Glob (auto picks all matching dirs)
  python -m model.tools.summarize_run "output/sweep_gene_pert_32_*"
  python -m model.tools.summarize_run "output/cellgene_pretrain_*"

Output is written to <run_dir>/summary/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import glob as _glob
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"  wrote → {path}")


def _find_train_log(run_dir: Path) -> Path | None:
    """
    train_log.jsonl lives one level below the run dir because train.py puts
    outputs in  <output_dir>/<coupling_mode>/train_log.jsonl
    (coupling_mode is typically 'ot', 'coupled', or 'baseline').
    """
    hits = sorted(run_dir.glob("*/train_log.jsonl"))
    return hits[0] if hits else None


# ─────────────────────────────────────────────────────────────────────────────
# run-type detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_type(run_dir: Path) -> str:
    """Return 'pretrain', 'sweep', 'single', or 'unknown'."""
    if (run_dir / "metrics_log.jsonl").exists():
        return "pretrain"
    # sweep: contains sweep_XX_* subdirs that themselves contain a mode subdir
    if any(run_dir.glob("sweep_*/*/train_log.jsonl")):
        return "sweep"
    # single CoupledFM run: one level of mode subdir
    if any(run_dir.glob("*/train_log.jsonl")):
        return "single"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# pretrain summary
# ─────────────────────────────────────────────────────────────────────────────

def _summarize_pretrain(run_dir: Path) -> None:
    metrics_path = run_dir / "metrics_log.jsonl"
    if not metrics_path.exists():
        print(f"[warn] metrics_log.jsonl not found in {run_dir}", file=sys.stderr)
        return

    out = run_dir / "summary"
    out.mkdir(exist_ok=True)

    records = _read_jsonl(metrics_path)
    if not records:
        print("[warn] metrics_log.jsonl is empty", file=sys.stderr)
        return

    df = pd.DataFrame(records)
    df.to_csv(out / "metrics_log.csv", index=False)
    print(f"  wrote → {out / 'metrics_log.csv'}")

    # ── training curve ────────────────────────────────────────────────────────
    # step-level records (all records with train_loss, for fine-grained curve)
    step_df = df[df["train_loss"].notna()].copy() if "train_loss" in df.columns else pd.DataFrame()
    # epoch-summary records (eval_type == "epoch_summary", avg per epoch)
    epoch_df = df[df.get("eval_type", pd.Series(dtype=str)) == "epoch_summary"].copy() \
        if "eval_type" in df.columns else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Pretrain: {run_dir.name}", fontsize=11, y=1.01)

    # left: step-level train loss
    ax = axes[0]
    if not step_df.empty and "global_step" in step_df.columns:
        ax.plot(step_df["global_step"], step_df["train_loss"],
                lw=0.8, alpha=0.7, color="#2563eb", label="step loss")
    if not epoch_df.empty and "global_step" in epoch_df.columns:
        ax.plot(epoch_df["global_step"], epoch_df["train_loss"],
                marker="o", ms=5, lw=1.5, color="#f59e0b", label="epoch avg", zorder=3)
    if "best_train_loss" in df.columns:
        best_val = df["best_train_loss"].dropna().min()
        if pd.notna(best_val):
            ax.axhline(best_val, ls="--", lw=0.9, color="#dc2626",
                       label=f"best={best_val:.5f}")
    ax.set_xlabel("global_step")
    ax.set_ylabel("train_loss")
    ax.set_title("Training Loss (step-level)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4, alpha=0.5)

    # right: per-epoch avg loss (from epoch_summary records)
    ax2 = axes[1]
    if not epoch_df.empty and "epoch" in epoch_df.columns:
        ax2.plot(epoch_df["epoch"], epoch_df["train_loss"],
                 marker="o", ms=5, lw=1.5, color="#059669", label="epoch avg loss")
        if "best_train_loss" in epoch_df.columns:
            ax2.plot(epoch_df["epoch"], epoch_df["best_train_loss"],
                     ls="--", lw=1.0, color="#dc2626", label="best so far")
        ax2.legend(fontsize=8)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("train_loss")
    ax2.set_title("Loss per Epoch")
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.grid(True, lw=0.4, alpha=0.5)

    plt.tight_layout()
    curve_path = out / "training_curve.png"
    fig.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote → {curve_path}")

    # ── text report ───────────────────────────────────────────────────────────
    total_steps = int(df["global_step"].max()) if "global_step" in df.columns else "?"
    total_epochs = int(epoch_df["epoch"].max()) if not epoch_df.empty and "epoch" in epoch_df.columns else "?"
    final_loss = float(epoch_df["train_loss"].iloc[-1]) if not epoch_df.empty else float("nan")
    best_loss = float(df["best_train_loss"].dropna().min()) \
        if "best_train_loss" in df.columns else float("nan")

    ckpts = sorted(run_dir.glob("backbone_step*.pt"))
    ckpt_steps = [int(p.stem.replace("backbone_step", "")) for p in ckpts]

    config_note = ""
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            lr = cfg.get("lr", cfg.get("learning_rate", "?"))
            epochs = cfg.get("epochs", cfg.get("n_epochs", "?"))
            bs = cfg.get("batch_size", cfg.get("batch_per_gpu", "?"))
            config_note = f"  lr={lr}  epochs={epochs}  batch_size={bs}"
        except Exception:
            pass

    lines = [
        "=" * 64,
        "PRETRAIN RUN SUMMARY",
        "=" * 64,
        f"Run dir   : {run_dir}",
        f"Generated : {_now()}",
        "",
        "── Config ──────────────────────────────────────────────────",
        config_note or "  (see config.json)",
        "",
        "── Training Progress ───────────────────────────────────────",
        f"  Total steps  : {total_steps:,}" if isinstance(total_steps, int)
            else f"  Total steps  : {total_steps}",
        f"  Total epochs : {total_epochs}",
        f"  Final loss   : {final_loss:.6f}" if not pd.isna(final_loss) else "  Final loss   : N/A",
        f"  Best loss    : {best_loss:.6f}"  if not pd.isna(best_loss)  else "  Best loss    : N/A",
        "",
        "── Checkpoints ─────────────────────────────────────────────",
    ]
    if ckpt_steps:
        lines += [f"  backbone_step{s}.pt" for s in sorted(ckpt_steps)]
    else:
        lines.append("  (none found; check backbone.pt / pretrain_adapter.pt)")

    final_ckpts = ([p.name for p in sorted(run_dir.glob("backbone.pt"))]
                 + [p.name for p in sorted(run_dir.glob("pretrain_adapter.pt"))]
                 + [p.name for p in sorted(run_dir.glob("ema.pt"))])
    if final_ckpts:
        lines += ["", "── Final Weights ───────────────────────────────────────────"]
        lines += [f"  {c}" for c in final_ckpts]

    lines += [
        "",
        "── Summary Files ───────────────────────────────────────────",
        "  summary/metrics_log.csv      - 完整 step/epoch 指标（CSV）",
        "  summary/training_curve.png   - 训练曲线图",
        "  summary/report.txt           - 本文件",
        "=" * 64,
    ]
    report = "\n".join(lines) + "\n"
    _write(out / "report.txt", report)
    print()
    print(report)


# ─────────────────────────────────────────────────────────────────────────────
# sweep summary
# ─────────────────────────────────────────────────────────────────────────────

def _parse_grid_line(grid_path: Path) -> dict:
    params: dict = {}
    if grid_path.exists():
        for line in grid_path.read_text().splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                params[k.strip()] = v.strip()
    return params


def _summarize_sweep(run_dir: Path) -> None:
    sweep_dirs = sorted(run_dir.glob("sweep_*/"))
    if not sweep_dirs:
        print(f"[warn] no sweep_* subdirs found in {run_dir}", file=sys.stderr)
        return

    out = run_dir / "summary"
    out.mkdir(exist_ok=True)

    rows = []
    for sd in sweep_dirs:
        # train_log.jsonl is one level deeper: sd/<coupling_mode>/train_log.jsonl
        log = _find_train_log(sd)
        if log is None:
            continue
        records = _read_jsonl(log)
        tests = [r for r in records if r.get("eval_type") == "test"]
        vals  = [r for r in records if r.get("eval_type") == "val"]
        pool  = tests if tests else vals
        if not pool:
            continue

        best = max(pool, key=lambda r: r.get("eval_pearson_delta_ctrl", -999))
        grid = _parse_grid_line(sd / "_grid_line.txt")

        row = {
            "run_id":           sd.name,
            "mode":             log.parent.name,          # "ot" / "coupled" / "baseline"
            "lr":               grid.get("lr", "?"),
            "max_pert_genes":   grid.get("max_pert_genes", "?"),
            "pool":             grid.get("pool", "?"),
            "cfg_drop_prob":    grid.get("cfg_drop_prob", "?"),
            "latent_z_mode":    grid.get("latent_z_mode", "?"),
            "selection_metric": grid.get("selection_metric", "?"),
            "best_epoch":       best.get("epoch", "?"),
            "best_step":        best.get("global_step", "?"),
            "pd_ctrl":          round(float(best.get("eval_pearson_delta_ctrl", float("nan"))), 5),
            "direct_pearson":   round(float(best.get("eval_direct_pearson",      float("nan"))), 5),
            "corr_pert_mean":   round(float(best.get("eval_corr_pert_mean",      float("nan"))), 5),
            "corr_ctrl_mean":   round(float(best.get("eval_corr_ctrl_mean",      float("nan"))), 5),
            "mmd":              round(float(best.get("eval_mmd",                 float("nan"))), 6),
            "train_loss":       round(float(best.get("train_loss",               float("nan"))), 5),
            "n_test_records":   len(tests),
        }
        rows.append(row)

    if not rows:
        print("[warn] no completed runs found (all missing train_log.jsonl or no val/test records)",
              file=sys.stderr)
        return

    df = (pd.DataFrame(rows)
          .sort_values("pd_ctrl", ascending=False)
          .reset_index(drop=True))
    df.index += 1
    df.index.name = "rank"

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = out / "ranked_runs.csv"
    df.to_csv(csv_path)
    print(f"  wrote → {csv_path}")

    # ── training curves ───────────────────────────────────────────────────────
    _plot_sweep_curves(run_dir, sweep_dirs, out)

    # ── markdown report ───────────────────────────────────────────────────────
    header_cols = ["rank", "lr", "max_pert_genes", "pool", "cfg_drop_prob",
                   "pd_ctrl", "direct_pearson", "corr_pert_mean", "mmd", "best_epoch"]
    md_lines = [
        f"# Sweep Summary: {run_dir.name}",
        "",
        f"Generated: {_now()}  |  Total completed runs: {len(df)}",
        "",
        "## Ranked Results (by `pearson_delta_ctrl` ↓)",
        "",
        "| " + " | ".join(header_cols) + " |",
        "| " + " | ".join(["---"] * len(header_cols)) + " |",
    ]
    for rank, row in df.iterrows():
        vals = [str(rank)] + [str(row.get(c, "")) for c in header_cols[1:]]
        md_lines.append("| " + " | ".join(vals) + " |")

    best_row = df.iloc[0]
    md_lines += [
        "",
        "## Best Configuration",
        "",
        "```",
    ]
    for col in ["run_id", "mode", "lr", "max_pert_genes", "pool", "cfg_drop_prob",
                "latent_z_mode", "pd_ctrl", "direct_pearson", "corr_pert_mean",
                "corr_ctrl_mean", "mmd", "best_epoch", "best_step"]:
        md_lines.append(f"{col:<22} = {best_row.get(col, '')}")
    md_lines += [
        "```",
        "",
        "## Output Files",
        "",
        "- `ranked_runs.csv`   — 所有超参组合排名（可 Excel/pandas 打开）",
        "- `curves_top10.png`  — Top-10 run 的 val pearson_delta_ctrl 训练曲线",
        "- `curves_all.png`    — 全部 run 曲线（run 数 > 10 时生成）",
        "- `report.md`         — 本文件",
        "- `report.txt`        — 纯文本版排名表",
    ]

    _write(out / "report.md", "\n".join(md_lines) + "\n")

    # ── plain text table ──────────────────────────────────────────────────────
    sep = "-" * 125
    txt_lines = [
        "=" * 64,
        "SWEEP SUMMARY",
        "=" * 64,
        f"Run dir   : {run_dir}",
        f"Generated : {_now()}",
        f"Completed : {len(df)} / {len(sweep_dirs)} runs",
        "",
        f"{'Rank':<5} {'pd_ctrl':>9} {'direct_p':>9} {'corr_pert':>10} {'mmd':>9} "
        f"{'ep':>4}  {'lr':<8} {'mpg':<4} {'pool':<16} {'cfg':<5}",
        sep,
    ]
    for rank, row in df.iterrows():
        txt_lines.append(
            f"{rank:<5} {row['pd_ctrl']:>9.4f} {row['direct_pearson']:>9.4f} "
            f"{row['corr_pert_mean']:>10.4f} {row['mmd']:>9.5f} "
            f"{str(row['best_epoch']):>4}  "
            f"{str(row['lr']):<8} {str(row['max_pert_genes']):<4} "
            f"{str(row['pool']):<16} {str(row['cfg_drop_prob']):<5}"
        )
    txt_lines += [
        sep,
        "",
        "── Best Configuration ──────────────────────────────────────",
        f"  run_id  : {best_row['run_id']}",
        f"  lr      : {best_row['lr']}",
        f"  mpg     : {best_row['max_pert_genes']}",
        f"  pool    : {best_row['pool']}",
        f"  cfg     : {best_row['cfg_drop_prob']}",
        f"  pd_ctrl : {best_row['pd_ctrl']:.5f}",
        f"  direct_p: {best_row['direct_pearson']:.5f}",
        "",
        "=" * 64,
    ]
    txt = "\n".join(txt_lines) + "\n"
    _write(out / "report.txt", txt)
    print()
    print(txt)


def _plot_sweep_curves(run_dir: Path, sweep_dirs: list[Path], out: Path) -> None:
    """Plot val/test pearson_delta_ctrl over training steps for each sweep run."""
    series: list[tuple[str, list, list, float]] = []
    for sd in sweep_dirs:
        log = _find_train_log(sd)
        if log is None:
            continue
        records = _read_jsonl(log)
        pts = [
            (r["global_step"], r["eval_pearson_delta_ctrl"])
            for r in records
            if r.get("eval_type") in ("val", "test")
            and "eval_pearson_delta_ctrl" in r
            and "global_step" in r
        ]
        if not pts:
            continue
        pts.sort()
        steps, scores = zip(*pts)
        best = max(scores)
        label = sd.name.replace("_lzinterp_selpearson_delta_ctrl", "")
        series.append((label, list(steps), list(scores), best))

    if not series:
        return
    series.sort(key=lambda x: -x[3])

    def _make_fig(selected: list[tuple], title: str, fname: str) -> None:
        n = len(selected)
        fig, ax = plt.subplots(figsize=(11, max(4, min(n * 0.25 + 3, 8))))
        cmap = plt.get_cmap("tab20")
        for i, (name, steps, scores, best) in enumerate(selected):
            ax.plot(steps, scores, lw=1.0, alpha=0.85,
                    color=cmap(i % 20), label=f"{name}  ({best:.4f})")
        ax.set_xlabel("global_step")
        ax.set_ylabel("pearson_delta_ctrl")
        ax.set_title(title)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.grid(True, lw=0.3, alpha=0.5)
        ncol = max(1, n // 20)
        ax.legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left", ncol=ncol)
        plt.tight_layout()
        p = out / fname
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote → {p}")

    _make_fig(series[:10],
              f"Top-10 Runs — val/test pearson_delta_ctrl\n{run_dir.name}",
              "curves_top10.png")
    if len(series) > 10:
        _make_fig(series,
                  f"All {len(series)} Runs — val/test pearson_delta_ctrl\n{run_dir.name}",
                  "curves_all.png")


# ─────────────────────────────────────────────────────────────────────────────
# single CoupledFM run summary (not part of a sweep)
# ─────────────────────────────────────────────────────────────────────────────

def _summarize_single(run_dir: Path) -> None:
    log = _find_train_log(run_dir)
    if log is None:
        print(f"[warn] train_log.jsonl not found in {run_dir}", file=sys.stderr)
        return

    records = _read_jsonl(log)
    if not records:
        print("[warn] train_log.jsonl is empty", file=sys.stderr)
        return

    out = run_dir / "summary"
    out.mkdir(exist_ok=True)

    df = pd.DataFrame(records)
    df.to_csv(out / "train_log.csv", index=False)
    print(f"  wrote → {out / 'train_log.csv'}")

    tests = df[df["eval_type"] == "test"] if "eval_type" in df.columns else pd.DataFrame()
    vals  = df[df["eval_type"] == "val"]  if "eval_type" in df.columns else pd.DataFrame()

    # curve
    if "eval_pearson_delta_ctrl" in df.columns and "global_step" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        if not vals.empty:
            ax.plot(vals["global_step"], vals["eval_pearson_delta_ctrl"],
                    lw=1.2, color="#2563eb", alpha=0.7, label="val pd_ctrl")
        if not tests.empty:
            ax.plot(tests["global_step"], tests["eval_pearson_delta_ctrl"],
                    marker="s", ms=5, lw=1.5, color="#dc2626", label="test pd_ctrl")
        ax.set_xlabel("global_step")
        ax.set_ylabel("pearson_delta_ctrl")
        ax.set_title(f"{run_dir.name} / {log.parent.name}")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.grid(True, lw=0.3, alpha=0.5)
        fig.tight_layout()
        p = out / "training_curve.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote → {p}")

    # best test metrics
    best_test = None
    if not tests.empty and "eval_pearson_delta_ctrl" in tests.columns:
        best_test = tests.loc[tests["eval_pearson_delta_ctrl"].idxmax()]

    lines = [
        "=" * 64,
        f"RUN SUMMARY: {run_dir.name}",
        "=" * 64,
        f"Generated : {_now()}",
        f"Mode      : {log.parent.name}",
        "",
    ]
    if best_test is not None:
        lines += [
            "── Best Test Metrics ───────────────────────────────────────",
            f"  epoch               : {best_test.get('epoch', '?')}",
            f"  global_step         : {int(best_test.get('global_step', 0)):,}",
            f"  pearson_delta_ctrl  : {float(best_test.get('eval_pearson_delta_ctrl', float('nan'))):.5f}",
            f"  direct_pearson      : {float(best_test.get('eval_direct_pearson',      float('nan'))):.5f}",
            f"  corr_pert_mean      : {float(best_test.get('eval_corr_pert_mean',      float('nan'))):.5f}",
            f"  corr_ctrl_mean      : {float(best_test.get('eval_corr_ctrl_mean',      float('nan'))):.5f}",
            f"  mmd                 : {float(best_test.get('eval_mmd',                 float('nan'))):.6f}",
            f"  train_loss          : {float(best_test.get('train_loss',               float('nan'))):.5f}",
        ]
    else:
        lines.append("  (no test records found yet)")

    lines += ["", "=" * 64]
    _write(out / "report.txt", "\n".join(lines) + "\n")
    print("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a pretrain or sweep run into <run_dir>/summary/")
    parser.add_argument("run_dir", nargs="+",
                        help="Path(s) to run directory (glob patterns supported).")
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.run_dir:
        expanded = sorted(_glob.glob(pattern))
        paths.extend(Path(p) for p in expanded) if expanded else paths.append(Path(pattern))

    for run_dir in paths:
        if not run_dir.is_dir():
            print(f"[skip] not a directory: {run_dir}", file=sys.stderr)
            continue
        rtype = _detect_type(run_dir)
        print(f"\n{'='*64}")
        print(f"Summarizing [{rtype}]: {run_dir}")
        print(f"{'='*64}")
        if rtype == "pretrain":
            _summarize_pretrain(run_dir)
        elif rtype == "sweep":
            _summarize_sweep(run_dir)
        elif rtype == "single":
            _summarize_single(run_dir)
        else:
            print(
                f"[warn] cannot detect run type in {run_dir}\n"
                "  Expected: metrics_log.jsonl (pretrain)  OR  sweep_*/*/train_log.jsonl (sweep)  "
                "OR  */train_log.jsonl (single run)",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
