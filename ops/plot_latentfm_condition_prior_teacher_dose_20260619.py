#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
CSV_IN = ROOT / "reports/latentfm_condition_prior_teacher_dose_20260619.csv"
OUT_BASE = ROOT / "reports/latentfm_condition_prior_teacher_dose_20260619"
META_OUT = ROOT / "reports/latentfm_condition_prior_teacher_dose_20260619.figure_meta.json"

METRICS = [
    ("test_pp", "Aggregate pp", True),
    ("multi_seen_pp", "Multi seen pp", True),
    ("multi_unseen1_pp", "Multi unseen1 pp", True),
    ("multi_unseen2_pp", "Multi unseen2 pp", True),
    ("family_gene_pp", "Gene-family pp", True),
    ("test_mmd", "MMD", False),
]

PRIMARY_REFERENCE = {
    "test_mmd": 0.027124,
    "test_pp": 0.0338,
    "family_gene_pp": 0.0437,
    "multi_seen_pp": 0.2112,
    "multi_unseen1_pp": -0.0032,
    "multi_unseen2_pp": -0.1386,
}


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
    for row in rows:
        row["weight_num"] = fnum(row.get("weight"))
        row["complete_bool"] = str(row.get("complete", "")).lower() == "true"
        for key, _label, _higher in METRICS:
            row[f"{key}_num"] = fnum(row.get(key))
    return rows


def main() -> int:
    rows = load_rows()
    complete = [row for row in rows if row.get("complete_bool")]
    missing_reason = ""
    if len(complete) != 3:
        missing_reason = f"need 3 complete rows, found {len(complete)}"
    elif any(row.get("weight_num") is None for row in complete):
        missing_reason = "missing numeric weight"
    elif any(any(row.get(f"{key}_num") is None for key, _label, _higher in METRICS) for row in complete):
        missing_reason = "missing numeric plotted metric"

    if missing_reason:
        META_OUT.write_text(
            json.dumps(
                {
                    "generated": datetime.now().isoformat(timespec="seconds"),
                    "status": "pending",
                    "reason": missing_reason,
                    "csv": str(CSV_IN),
                    "outputs": [],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"pending: {missing_reason}")
        return 2

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    complete = sorted(complete, key=lambda row: float(row["weight_num"]))
    weights = [float(row["weight_num"]) for row in complete]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5), constrained_layout=True)
    color = "#2A6FBB"
    ref_color = "#6B7280"
    for ax, (key, label, higher_is_better) in zip(axes.ravel(), METRICS):
        values = [float(row[f"{key}_num"]) for row in complete]
        ref = PRIMARY_REFERENCE.get(key)
        ax.plot(weights, values, marker="o", color=color, linewidth=1.4, markersize=4.2)
        if ref is not None:
            ax.axhline(ref, color=ref_color, linestyle=(0, (3, 2)), linewidth=0.9)
        for x, y, row in zip(weights, values, complete):
            decision = str(row.get("decision", ""))
            if decision == "repeat_candidate":
                ax.scatter([x], [y], s=42, facecolors="none", edgecolors="#D55E00", linewidths=1.2, zorder=5)
        ax.set_title(label, pad=5)
        ax.set_xlabel("Teacher weight")
        direction = "higher better" if higher_is_better else "lower better"
        ax.set_ylabel(direction)
        ax.set_xticks(weights)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    fig.suptitle("LatentFM condition-prior teacher dose response", fontsize=10, y=1.02)

    outputs = []
    for suffix, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 300}),
    ):
        path = OUT_BASE.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(str(path))
    plt.close(fig)

    META_OUT.write_text(
        json.dumps(
            {
                "generated": datetime.now().isoformat(timespec="seconds"),
                "status": "complete",
                "csv": str(CSV_IN),
                "outputs": outputs,
                "primary_reference": PRIMARY_REFERENCE,
                "rows": complete,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for output in outputs:
        print(output)
    print(META_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
