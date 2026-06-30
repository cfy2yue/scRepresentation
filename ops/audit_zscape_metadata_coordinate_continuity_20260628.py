#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path


COORD_SPACES = {
    "umap3d": ("umap3d_1", "umap3d_2", "umap3d_3"),
    "subumap3d": ("subumap3d_1", "subumap3d_2", "subumap3d_3"),
}


def now_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def is_control(target: str) -> bool:
    return str(target).lower().startswith("ctrl")


def parse_float(value: str) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def add_sum(store: dict, key: tuple, coords: tuple[float, float, float]) -> None:
    rec = store.setdefault(key, [0.0, 0.0, 0.0, 0])
    rec[0] += coords[0]
    rec[1] += coords[1]
    rec[2] += coords[2]
    rec[3] += 1


def centroid(sumrec: list[float]) -> tuple[float, float, float]:
    n = sumrec[3]
    return (sumrec[0] / n, sumrec[1] / n, sumrec[2] / n)


def dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def median(values: list[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0 or vy <= 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def continuity_metrics(time_to_centroid: dict[float, tuple[float, float, float]]) -> dict:
    times = sorted(time_to_centroid)
    adjacent = []
    nonadjacent = []
    gaps = []
    dists = []
    nearest_adjacent_hits = 0
    nearest_total = 0
    if len(times) < 3:
        return {
            "n_timepoints": len(times),
            "adjacent_median": float("nan"),
            "nonadjacent_median": float("nan"),
            "adjacent_nonadjacent_ratio": float("nan"),
            "distance_timegap_corr": float("nan"),
            "nearest_adjacent_accuracy": float("nan"),
        }
    adjacent_pairs = {(times[i], times[i + 1]) for i in range(len(times) - 1)}
    adjacent_pairs |= {(b, a) for a, b in list(adjacent_pairs)}
    for i, ta in enumerate(times):
        distances_from_t = []
        for j, tb in enumerate(times):
            if i == j:
                continue
            d = dist(time_to_centroid[ta], time_to_centroid[tb])
            gap = abs(tb - ta)
            gaps.append(gap)
            dists.append(d)
            distances_from_t.append((d, tb))
            if (ta, tb) in adjacent_pairs:
                adjacent.append(d)
            else:
                nonadjacent.append(d)
        if distances_from_t:
            nearest = min(distances_from_t, key=lambda x: x[0])[1]
            min_gap = min(abs(tb - ta) for tb in times if tb != ta)
            nearest_adjacent_hits += int(abs(nearest - ta) == min_gap)
            nearest_total += 1
    adj_med = median(adjacent)
    non_med = median(nonadjacent)
    ratio = adj_med / non_med if math.isfinite(adj_med) and math.isfinite(non_med) and non_med > 0 else float("nan")
    acc = nearest_adjacent_hits / nearest_total if nearest_total else float("nan")
    return {
        "n_timepoints": len(times),
        "adjacent_median": adj_med,
        "nonadjacent_median": non_med,
        "adjacent_nonadjacent_ratio": ratio,
        "distance_timegap_corr": pearson(gaps, dists),
        "nearest_adjacent_accuracy": acc,
    }


def bootstrap_continuity(embryo_sums: dict, base_key: tuple, n_boot: int, seed: int) -> dict:
    rng = random.Random(seed)
    embryos = sorted({key[-1] for key in embryo_sums if key[:4] == base_key})
    if len(embryos) < 10:
        return {"n_boot": 0, "pass_fraction": float("nan"), "ratio_median": float("nan")}
    ratios = []
    pass_count = 0
    for _ in range(n_boot):
        sampled = [rng.choice(embryos) for _ in embryos]
        counts = defaultdict(int)
        for emb in sampled:
            counts[emb] += 1
        time_sums: dict[float, list[float]] = {}
        for key, sumrec in embryo_sums.items():
            if key[:4] != base_key:
                continue
            timepoint = key[4]
            embryo = key[5]
            weight = counts.get(embryo, 0)
            if weight <= 0:
                continue
            rec = time_sums.setdefault(timepoint, [0.0, 0.0, 0.0, 0])
            rec[0] += sumrec[0] * weight
            rec[1] += sumrec[1] * weight
            rec[2] += sumrec[2] * weight
            rec[3] += sumrec[3] * weight
        metrics = continuity_metrics({t: centroid(s) for t, s in time_sums.items() if s[3] > 0})
        ratio = metrics["adjacent_nonadjacent_ratio"]
        if math.isfinite(ratio):
            ratios.append(ratio)
            pass_count += int(ratio < 0.85 and metrics["distance_timegap_corr"] > 0.35)
    return {
        "n_boot": len(ratios),
        "pass_fraction": pass_count / len(ratios) if ratios else float("nan"),
        "ratio_median": median(ratios),
    }


def scan_metadata(path: Path, dataset: str, selected: set[str]) -> tuple[dict, dict, dict]:
    group_sums: dict[tuple, list[float]] = {}
    embryo_sums: dict[tuple, list[float]] = {}
    response_sums: dict[tuple, list[float]] = {}
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cell_type = row.get("cell_type_broad", "")
            if cell_type not in selected:
                continue
            timepoint = parse_float(row.get("timepoint", ""))
            if timepoint is None:
                continue
            target = row.get("gene_target", "")
            klass = "control" if is_control(target) else "perturbation"
            embryo = row.get("embryo", "")
            for space, cols in COORD_SPACES.items():
                coords_raw = tuple(parse_float(row.get(col, "")) for col in cols)
                if any(v is None for v in coords_raw):
                    continue
                coords = coords_raw  # type: ignore[assignment]
                add_sum(group_sums, (dataset, space, cell_type, klass, timepoint), coords)
                add_sum(embryo_sums, (dataset, space, cell_type, klass, timepoint, embryo), coords)
                if dataset == "zperturb_full":
                    add_sum(response_sums, (space, cell_type, target, timepoint, klass), coords)
    return group_sums, embryo_sums, response_sums


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=50)
    args = parser.parse_args()

    plan = json.loads(Path(args.plan_json).read_text(encoding="utf-8"))
    selected = {row["cell_type_broad"] for row in plan.get("selected_cell_types", [])}
    metadata_dir = Path(args.metadata_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scans = [
        ("reference", metadata_dir / "GSE202639_reference_cell_metadata.csv.gz"),
        ("zperturb_full", metadata_dir / "GSE202639_zperturb_full_cell_metadata.csv.gz"),
    ]
    all_group_sums: dict[tuple, list[float]] = {}
    all_embryo_sums: dict[tuple, list[float]] = {}
    all_response_sums: dict[tuple, list[float]] = {}
    for dataset, path in scans:
        group_sums, embryo_sums, response_sums = scan_metadata(path, dataset, selected)
        all_group_sums.update(group_sums)
        all_embryo_sums.update(embryo_sums)
        all_response_sums.update(response_sums)

    continuity_rows = []
    for dataset in ["reference", "zperturb_full"]:
        classes = ["control"] if dataset == "reference" else ["control"]
        for space in COORD_SPACES:
            for cell_type in sorted(selected):
                for klass in classes:
                    time_centroids = {}
                    for key, sumrec in all_group_sums.items():
                        if key[:4] == (dataset, space, cell_type, klass):
                            time_centroids[key[4]] = centroid(sumrec)
                    metrics = continuity_metrics(time_centroids)
                    boot = bootstrap_continuity(
                        all_embryo_sums,
                        (dataset, space, cell_type, klass),
                        args.n_bootstrap,
                        seed=42,
                    )
                    gate = (
                        metrics["n_timepoints"] >= (10 if dataset == "reference" else 5)
                        and metrics["adjacent_nonadjacent_ratio"] < 0.85
                        and metrics["distance_timegap_corr"] > 0.35
                        and metrics["nearest_adjacent_accuracy"] >= 0.5
                        and (not math.isfinite(boot["pass_fraction"]) or boot["pass_fraction"] >= 0.7)
                    )
                    continuity_rows.append(
                        {
                            "dataset": dataset,
                            "coord_space": space,
                            "cell_type_broad": cell_type,
                            "condition_class": klass,
                            **metrics,
                            "bootstrap_pass_fraction": boot["pass_fraction"],
                            "bootstrap_ratio_median": boot["ratio_median"],
                            "n_boot": boot["n_boot"],
                            "continuity_gate": gate,
                        }
                    )

    response_rows = []
    control_sums: dict[tuple, list[float]] = {}
    for key, sumrec in all_response_sums.items():
        space, cell_type, target, timepoint, klass = key
        if klass == "control":
            out_key = (space, cell_type, timepoint)
            rec = control_sums.setdefault(out_key, [0.0, 0.0, 0.0, 0])
            rec[0] += sumrec[0]
            rec[1] += sumrec[1]
            rec[2] += sumrec[2]
            rec[3] += sumrec[3]
    control_centroids = {key: centroid(value) for key, value in control_sums.items() if value[3] > 0}
    for key, sumrec in all_response_sums.items():
        space, cell_type, target, timepoint, klass = key
        if klass != "perturbation":
            continue
        ctrl = control_centroids.get((space, cell_type, timepoint))
        if ctrl is None:
            continue
        response_rows.append(
            {
                "coord_space": space,
                "cell_type_broad": cell_type,
                "gene_target": target,
                "timepoint": timepoint,
                "cells": sumrec[3],
                "response_distance_to_matched_control": dist(centroid(sumrec), ctrl),
            }
        )
    response_rows.sort(key=lambda r: r["response_distance_to_matched_control"], reverse=True)

    n_reference_pass = sum(
        1 for r in continuity_rows
        if r["dataset"] == "reference" and r["coord_space"] == "umap3d" and r["continuity_gate"]
    )
    n_zcontrol_pass = sum(
        1 for r in continuity_rows
        if r["dataset"] == "zperturb_full" and r["coord_space"] == "umap3d" and r["continuity_gate"]
    )
    status = "zscape_metadata_coordinate_continuity_gate_pass_no_gpu"
    if n_reference_pass < 2 or n_zcontrol_pass < 2:
        status = "zscape_metadata_coordinate_continuity_gate_fail_no_gpu"

    continuity_csv = out_dir / "metadata_coordinate_continuity_rows.csv"
    response_csv = out_dir / "metadata_coordinate_response_rows.csv"
    with continuity_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(continuity_rows[0].keys()))
        writer.writeheader()
        writer.writerows(continuity_rows)
    with response_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(response_rows[0].keys()))
        writer.writeheader()
        writer.writerows(response_rows)

    payload = {
        "timestamp_utc": now_utc(),
        "status": status,
        "gpu_authorized": False,
        "selected_cell_types": sorted(selected),
        "n_reference_umap3d_gate_pass": n_reference_pass,
        "n_zperturb_control_umap3d_gate_pass": n_zcontrol_pass,
        "continuity_rows": continuity_rows,
        "top_response_rows": response_rows[:100],
        "outputs": {
            "continuity_csv": str(continuity_csv),
            "response_csv": str(response_csv),
        },
    }
    json_path = out_dir / "metadata_coordinate_continuity_gate.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_path = out_dir / "LATENTFM_ZSCAPE_METADATA_COORDINATE_CONTINUITY_GATE_20260628.md"
    lines = [
        "# LatentFM ZSCAPE Metadata-Coordinate Continuity Gate",
        "",
        f"Timestamp: `{payload['timestamp_utc']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only preflight over ZSCAPE metadata coordinates.",
        "- Uses `umap3d_*` and `subumap3d_*` columns already present in metadata.",
        "- No expression matrix/CDS/raw-count download.",
        "- No training, inference, embedding extraction, canonical multi, or Track C query use.",
        "",
        "## Gate Summary",
        "",
        f"- selected cell types: `{sorted(selected)}`",
        f"- reference UMAP3D continuity passes: `{n_reference_pass}`",
        f"- zperturb-control UMAP3D continuity passes: `{n_zcontrol_pass}`",
        "",
        "## UMAP3D Continuity Rows",
        "",
        "| dataset | cell_type_broad | timepoints | adj/nonadj | timegap_corr | nearest_adj_acc | bootstrap_pass | gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in continuity_rows:
        if row["coord_space"] != "umap3d":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["cell_type_broad"],
                    str(row["n_timepoints"]),
                    f"{row['adjacent_nonadjacent_ratio']:.4f}",
                    f"{row['distance_timegap_corr']:.4f}",
                    f"{row['nearest_adjacent_accuracy']:.4f}",
                    f"{row['bootstrap_pass_fraction']:.4f}" if math.isfinite(row["bootstrap_pass_fraction"]) else "nan",
                    str(row["continuity_gate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Top Metadata-Coordinate Perturbation Responses",
            "",
            "| coord_space | cell_type_broad | gene_target | timepoint | cells | distance_to_matched_control |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in response_rows[:20]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["coord_space"],
                    row["cell_type_broad"],
                    row["gene_target"],
                    str(row["timepoint"]),
                    str(row["cells"]),
                    f"{row['response_distance_to_matched_control']:.4f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Decision", ""])
    if status.endswith("pass_no_gpu"):
        lines.append(
            "Proceed to design a CPU expression-subset continuity/OT stability gate for the selected lineages. This still does not authorize GPU."
        )
    else:
        lines.append(
            "Do not download expression matrices for trajectory work yet. Reassess whether metadata coordinates are too distorted or whether a smaller lineage set is needed."
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- JSON: `{json_path}`",
            f"- continuity rows: `{continuity_csv}`",
            f"- response rows: `{response_csv}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)
    print(json_path)
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
