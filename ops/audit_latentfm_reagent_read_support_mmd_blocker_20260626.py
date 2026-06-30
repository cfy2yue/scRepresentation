#!/usr/bin/env python3
"""Localize the MMD blocker for external reagent/read-support artifacts.

CPU-only diagnostic over already generated train-only/internal proxy rows.
Does not train, infer, read checkpoints, read canonical multi, read Track C
query, or use GPU.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
ROWS_CSV = REPORTS / "latentfm_reagent_read_support_combined_signal_gate_rows_20260626.csv"
SIGNAL_JSON = REPORTS / "latentfm_reagent_read_support_combined_signal_gate_20260626.json"
OUT_JSON = REPORTS / "latentfm_reagent_read_support_mmd_blocker_20260626.json"
OUT_MD = REPORTS / "LATENTFM_REAGENT_READ_SUPPORT_MMD_BLOCKER_20260626.md"
OUT_ROWS = REPORTS / "latentfm_reagent_read_support_mmd_blocker_rows_20260626.csv"


def to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs)
    deny = sum((y - my) ** 2 for y in ys)
    if denx <= 0 or deny <= 0:
        return None
    return num / math.sqrt(denx * deny)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rank(xs), rank(ys))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def load_rows() -> list[dict[str, Any]]:
    with ROWS_CSV.open(newline="", encoding="utf-8") as handle:
        out = []
        for row in csv.DictReader(handle):
            parsed = dict(row)
            for key in ["artifact_value", "artifact_z_within_dataset", "pp_proxy_mean", "mmd_proxy_max"]:
                parsed[key] = to_float(parsed.get(key))
            out.append(parsed)
        return out


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    z = [r["artifact_z_within_dataset"] for r in rows if r.get("artifact_z_within_dataset") is not None]
    pp = [r["pp_proxy_mean"] for r in rows if r.get("pp_proxy_mean") is not None]
    mmd = [r["mmd_proxy_max"] for r in rows if r.get("mmd_proxy_max") is not None]
    paired = [
        (r["artifact_z_within_dataset"], r["pp_proxy_mean"], r["mmd_proxy_max"])
        for r in rows
        if r.get("artifact_z_within_dataset") is not None
        and r.get("pp_proxy_mean") is not None
        and r.get("mmd_proxy_max") is not None
    ]
    if not paired:
        return {"n": len(rows), "status": "no_paired_values"}
    z_med = median([x[0] for x in paired])
    high = [x for x in paired if x[0] > z_med]
    low = [x for x in paired if x[0] <= z_med]
    pp_high = mean([x[1] for x in high]) if high else None
    pp_low = mean([x[1] for x in low]) if low else None
    mmd_high = mean([x[2] for x in high]) if high else None
    mmd_low = mean([x[2] for x in low]) if low else None
    return {
        "n": len(rows),
        "paired_n": len(paired),
        "z_spearman_pp": spearman([x[0] for x in paired], [x[1] for x in paired]),
        "z_spearman_mmd": spearman([x[0] for x in paired], [x[2] for x in paired]),
        "pp_high_minus_low": None if pp_high is None or pp_low is None else pp_high - pp_low,
        "mmd_high_minus_low": None if mmd_high is None or mmd_low is None else mmd_high - mmd_low,
        "pp_mean": mean(pp) if pp else None,
        "mmd_max": max(mmd) if mmd else None,
        "mmd_high_mean": mmd_high,
        "mmd_low_mean": mmd_low,
        "source_files": sorted({str(r.get("source_file")) for r in rows if r.get("source_file")}),
    }


def main() -> int:
    rows = load_rows()
    signal = json.loads(SIGNAL_JSON.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_artifact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("artifact")), str(row.get("dataset")))].append(row)
        by_artifact[str(row.get("artifact"))].append(row)

    detail_rows: list[dict[str, Any]] = []
    for (artifact, dataset), group_rows in sorted(grouped.items()):
        summary = summarize_group(group_rows)
        detail_rows.append({"artifact": artifact, "dataset": dataset, **summary})

    artifact_rows = []
    for artifact, group_rows in sorted(by_artifact.items()):
        summary = summarize_group(group_rows)
        signal_summary = next((s for s in signal.get("summaries", []) if s.get("artifact") == artifact), {})
        mmd_only_blocker = (
            bool(signal_summary)
            and signal_summary.get("reasons") == ["mmd_proxy_max_above_0p001"]
        )
        artifact_rows.append(
            {
                "artifact": artifact,
                **summary,
                "gate_status": signal_summary.get("status"),
                "gate_reasons": signal_summary.get("reasons", []),
                "mmd_only_blocker": mmd_only_blocker,
            }
        )

    pass_like = [
        row["artifact"]
        for row in artifact_rows
        if row.get("mmd_only_blocker")
        and (row.get("pp_high_minus_low") is not None and row["pp_high_minus_low"] > 0.02)
        and (row.get("z_spearman_pp") is not None and row["z_spearman_pp"] > 0.30)
    ]
    payload = {
        "status": "reagent_read_support_mmd_blocker_localized_no_gpu",
        "boundary": {
            "cpu_only": True,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "uses_gpu": False,
        },
        "inputs": {"rows_csv": str(ROWS_CSV), "signal_json": str(SIGNAL_JSON)},
        "artifact_rows": artifact_rows,
        "dataset_rows": detail_rows,
        "pass_like_but_mmd_blocked": pass_like,
        "gpu_authorized": False,
        "decision": (
            "positive signal is retained as mechanism evidence, but MMD veto still blocks GPU; "
            "next step is source/count/tail-MMD diagnostic or external review, not training"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with OUT_ROWS.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "artifact",
            "dataset",
            "n",
            "paired_n",
            "z_spearman_pp",
            "z_spearman_mmd",
            "pp_high_minus_low",
            "mmd_high_minus_low",
            "pp_mean",
            "mmd_max",
            "gate_status",
            "gate_reasons",
            "mmd_only_blocker",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in artifact_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
        for row in detail_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    lines = [
        "# LatentFM Reagent Read-Support MMD Blocker Localization",
        "",
        "Status: `reagent_read_support_mmd_blocker_localized_no_gpu`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only diagnostic over combined signal-gate rows.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "",
        "## Artifact-Level Summary",
        "",
        "| artifact | n | z-Spearman pp | pp high-low | z-Spearman MMD | MMD high-low | MMD max | gate reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in artifact_rows:
        lines.append(
            "| `{artifact}` | {n} | {sp} | {pphl} | {smmd} | {mmdhl} | {mmdmax} | `{reasons}` |".format(
                artifact=row["artifact"],
                n=row.get("paired_n"),
                sp=fmt(row.get("z_spearman_pp")),
                pphl=fmt(row.get("pp_high_minus_low")),
                smmd=fmt(row.get("z_spearman_mmd")),
                mmdhl=fmt(row.get("mmd_high_minus_low")),
                mmdmax=fmt(row.get("mmd_max")),
                reasons=",".join(map(str, row.get("gate_reasons", []))),
            )
        )
    lines.extend(
        [
            "",
            "## Dataset-Level Summary",
            "",
            "| artifact | dataset | n | z-Spearman pp | pp high-low | MMD high-low | MMD max |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in detail_rows:
        lines.append(
            "| `{artifact}` | `{dataset}` | {n} | {sp} | {pphl} | {mmdhl} | {mmdmax} |".format(
                artifact=row["artifact"],
                dataset=row["dataset"],
                n=row.get("paired_n"),
                sp=fmt(row.get("z_spearman_pp")),
                pphl=fmt(row.get("pp_high_minus_low")),
                mmdhl=fmt(row.get("mmd_high_minus_low")),
                mmdmax=fmt(row.get("mmd_max")),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- pass-like but MMD-blocked artifacts: `{pass_like}`",
            "- MMD veto remains active; this diagnostic does not authorize GPU.",
            "- If retained, next step must be a CPU source/count/tail-MMD diagnostic with a predeclared fail-close rule.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- rows: `{OUT_ROWS}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
