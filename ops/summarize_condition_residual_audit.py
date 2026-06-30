#!/usr/bin/env python3
"""Summarize LatentFM per-condition residual audit CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


NUMERIC_FIELDS = (
    "pred_norm",
    "target_norm",
    "pred_target_cosine",
    "pred_target_pearson",
    "retrieval_rank",
    "retrieval_true_similarity",
)


def _float(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        out = float(s)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _bool(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _read_csv(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            row["checkpoint_label"] = label
            for key in NUMERIC_FIELDS:
                row[key] = _float(row.get(key))
            for key in list(row):
                if key.startswith("retrieval_top"):
                    row[key] = _bool(row[key])
            rows.append(row)
    return rows


def _fmt(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "NA"
    return f"{v:.{digits}f}"


def _safe_mean(vals: list[float | None]) -> float | None:
    clean = [float(v) for v in vals if v is not None]
    return mean(clean) if clean else None


def _hit_rate(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [bool(r.get(key)) for r in rows if key in r]
    return mean(vals) if vals else None


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["checkpoint_label"])].append(row)
    for label in sorted(by_label):
        part = by_label[label]
        out.append(
            {
                "checkpoint": label,
                "n": len(part),
                "cosine": _safe_mean([r.get("pred_target_cosine") for r in part]),
                "pearson": _safe_mean([r.get("pred_target_pearson") for r in part]),
                "rank": _safe_mean([r.get("retrieval_rank") for r in part]),
                "top1": _hit_rate(part, "retrieval_top1"),
                "top5": _hit_rate(part, "retrieval_top5"),
                "target_norm": _safe_mean([r.get("target_norm") for r in part]),
            }
        )
    return out


def _group_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups = [g for g in str(row.get("groups", "")).split(",") if g]
        for group in groups:
            by_key[(str(row["checkpoint_label"]), group)].append(row)
    for (label, group), part in sorted(by_key.items()):
        out.append(
            {
                "checkpoint": label,
                "group": group,
                "n": len(part),
                "cosine": _safe_mean([r.get("pred_target_cosine") for r in part]),
                "pearson": _safe_mean([r.get("pred_target_pearson") for r in part]),
                "rank": _safe_mean([r.get("retrieval_rank") for r in part]),
                "top1": _hit_rate(part, "retrieval_top1"),
                "top5": _hit_rate(part, "retrieval_top5"),
            }
        )
    return out


def _top_rows(
    rows: list[dict[str, Any]],
    *,
    checkpoint: str,
    field: str,
    largest: bool,
    n: int,
) -> list[dict[str, Any]]:
    part = [r for r in rows if r.get("checkpoint_label") == checkpoint and r.get(field) is not None]
    return sorted(part, key=lambda r: float(r[field]), reverse=largest)[:n]


def _table(headers: list[str], data: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in data:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=8)
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for path in sorted(args.input_dir.glob("*.csv")):
        label = path.stem
        rows.extend(_read_csv(path, label))
    if not rows:
        raise SystemExit(f"no CSV rows found in {args.input_dir}")

    summaries = _summaries(rows)
    group_summaries = _group_summaries(rows)
    best_by_cos = sorted(summaries, key=lambda r: (r["cosine"] is not None, r["cosine"]), reverse=True)
    primary = "primary_scfoundation"
    lines = [
        "# LatentFM Condition Residual Audit Report",
        "",
        "This capped diagnostic compares existing checkpoints at condition-level residual geometry.",
        "",
        "## Overall",
    ]
    lines.extend(
        _table(
            ["checkpoint", "n", "mean cosine", "mean Pearson", "mean rank", "top1", "top5", "mean target norm"],
            [
                [
                    str(r["checkpoint"]),
                    str(r["n"]),
                    _fmt(r["cosine"]),
                    _fmt(r["pearson"]),
                    _fmt(r["rank"], 2),
                    _fmt(r["top1"]),
                    _fmt(r["top5"]),
                    _fmt(r["target_norm"]),
                ]
                for r in summaries
            ],
        )
    )
    lines += ["", "## Split And Family Groups"]
    focus_groups = {
        "test_multi_seen",
        "test_multi_unseen1",
        "test_multi_unseen2",
        "family_gene",
        "family_drug",
    }
    focus = [r for r in group_summaries if r["group"] in focus_groups]
    lines.extend(
        _table(
            ["checkpoint", "group", "n", "cosine", "Pearson", "rank", "top1", "top5"],
            [
                [
                    str(r["checkpoint"]),
                    str(r["group"]),
                    str(r["n"]),
                    _fmt(r["cosine"]),
                    _fmt(r["pearson"]),
                    _fmt(r["rank"], 2),
                    _fmt(r["top1"]),
                    _fmt(r["top5"]),
                ]
                for r in focus
            ],
        )
    )
    lines += ["", "## Top Checkpoints By Mean Residual Cosine"]
    for i, r in enumerate(best_by_cos[:5], start=1):
        lines.append(f"{i}. `{r['checkpoint']}` cosine={_fmt(r['cosine'])}, top5={_fmt(r['top5'])}, n={r['n']}")

    if any(r.get("checkpoint_label") == primary for r in rows):
        lines += ["", f"## `{primary}` Best Conditions"]
        lines.extend(
            _table(
                ["dataset", "condition", "groups", "family", "cosine", "Pearson", "rank", "target norm"],
                [
                    [
                        str(r.get("dataset")),
                        str(r.get("condition")),
                        str(r.get("groups")),
                        str(r.get("perturbation_family")),
                        _fmt(r.get("pred_target_cosine")),
                        _fmt(r.get("pred_target_pearson")),
                        _fmt(r.get("retrieval_rank"), 0),
                        _fmt(r.get("target_norm")),
                    ]
                    for r in _top_rows(rows, checkpoint=primary, field="pred_target_cosine", largest=True, n=args.top_n)
                ],
            )
        )
        lines += ["", f"## `{primary}` Worst Conditions"]
        lines.extend(
            _table(
                ["dataset", "condition", "groups", "family", "cosine", "Pearson", "rank", "target norm"],
                [
                    [
                        str(r.get("dataset")),
                        str(r.get("condition")),
                        str(r.get("groups")),
                        str(r.get("perturbation_family")),
                        _fmt(r.get("pred_target_cosine")),
                        _fmt(r.get("pred_target_pearson")),
                        _fmt(r.get("retrieval_rank"), 0),
                        _fmt(r.get("target_norm")),
                    ]
                    for r in _top_rows(rows, checkpoint=primary, field="pred_target_cosine", largest=False, n=args.top_n)
                ],
            )
        )

    payload = {
        "input_dir": str(args.input_dir),
        "n_rows": len(rows),
        "summaries": summaries,
        "group_summaries": group_summaries,
    }
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out_md": str(args.out_md), "out_json": str(args.out_json), "n_rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
