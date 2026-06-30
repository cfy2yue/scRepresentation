#!/usr/bin/env python3
"""JUMP-CP profile-derived aggregate source gate.

This gate streams public assembled JUMP ORF/CRISPR profile parquet files from
S3, aggregates only the train-overlap JCP IDs to gene/modality features, and
tests whether those features have a leakage-safe train/internal outcome signal.
It does not persist full profile matrices.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pyarrow.fs as pafs
import pyarrow.parquet as pq


ROOT = Path("/data/cyx/1030/scLatent")
JUMP_DIR = ROOT / "reports/jump_cp_small_metadata_schema_20260627"
PROFILE_INDEX = JUMP_DIR / "manifests__profile_index.json"
TRAIN_JOIN = ROOT / "reports/jump_cp_trainonly_join_controls_gate_20260627/jump_cp_s0_gene_join_rows.csv"
S0 = ROOT / "reports/latentfm_scaling_s0_provenance_freeze_20260625.tsv"
CRISPR = JUMP_DIR / "metadata__crispr.csv.gz"
ORF = JUMP_DIR / "metadata__orf.csv.gz"
OUTCOME_FILES = {
    "condition_exposure_cross": ROOT / "reports/latentfm_condition_exposure_row_bootstrap_rows_20260625.csv",
    "qc_reliability_cross": ROOT / "reports/latentfm_qc_support_reliability_rows_20260625.csv",
    "response_program": ROOT / "reports/latentfm_response_program_projection_rows_20260625.csv",
    "lodo_domain": ROOT / "reports/latentfm_lodo_domain_conflict_rows_20260625.csv",
    "background_actionability": ROOT / "reports/latentfm_background_target_actionability_rows_20260625.csv",
}

OUT_DIR = ROOT / "reports/jump_cp_profile_aggregate_gate_20260627"
OUT_JSON = ROOT / "reports/latentfm_jump_cp_profile_aggregate_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_JUMP_CP_PROFILE_AGGREGATE_GATE_20260627.md"
OUT_PROFILE = OUT_DIR / "jump_cp_profile_gene_modality_features.csv"
OUT_JOIN = OUT_DIR / "jump_cp_train_profile_join_rows.csv"
OUT_SIGNAL = OUT_DIR / "jump_cp_profile_signal_rows.csv"

PROFILE_FEATURES = [
    "profile_well_count",
    "profiled_jcp_count",
    "profile_source_count",
    "profile_plate_count",
    "mean_profile_norm",
    "mean_profile_norm_sd",
    "mean_centroid_l2",
    "max_centroid_l2",
    "mean_replicate_cosine",
]
CONTROL_FEATURES = [
    "jcp_id_count",
    "source_count",
    "plate_count",
    "batch_count",
    "well_position_count",
    "site_count_sum",
    "well_count_sum",
]


def norm_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def gene_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm_text(value).lower())


def fnum(value: object) -> float | None:
    text = norm_text(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return None if math.isnan(out) or math.isinf(out) else out


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    xm = mean(x)
    ym = mean(y)
    xv = [v - xm for v in x]
    yv = [v - ym for v in y]
    denom = math.sqrt(sum(v * v for v in xv) * sum(v * v for v in yv))
    return None if denom == 0 else sum(a * b for a, b in zip(xv, yv)) / denom


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(rankdata(x), rankdata(y))


def load_s0_membership() -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    with S0.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            out[(norm_text(row.get("dataset")), norm_text(row.get("condition")))] = norm_text(
                row.get("canonical_seed42_membership")
            )
    return out


def load_gene_metadata() -> tuple[dict[tuple[str, str], list[str]], dict[str, tuple[str, str]]]:
    by_gene_mod: dict[tuple[str, str], list[str]] = defaultdict(list)
    jcp_to_gene_mod: dict[str, tuple[str, str]] = {}
    for modality, path in (("crispr", CRISPR), ("orf", ORF)):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                symbol = norm_text(row.get("Metadata_Symbol"))
                jcp = norm_text(row.get("Metadata_JCP2022"))
                key = gene_key(symbol)
                if not key or not jcp:
                    continue
                by_gene_mod[(key, modality)].append(jcp)
                jcp_to_gene_mod[jcp] = (key, modality)
    return by_gene_mod, jcp_to_gene_mod


def load_train_join_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with TRAIN_JOIN.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if norm_text(row.get("membership")) != "train":
                continue
            rows.append(dict(row))
    return rows


def profile_urls() -> dict[str, str]:
    idx = json.loads(PROFILE_INDEX.read_text(encoding="utf-8"))
    return {str(item["subset"]): str(item["url"]) for item in idx if item.get("subset") in {"crispr", "orf"}}


def s3_path_from_url(url: str) -> str:
    key = url.split("https://cellpainting-gallery.s3.amazonaws.com/", 1)[1]
    return "cellpainting-gallery/" + key


def init_acc(dim: int) -> dict[str, Any]:
    return {
        "n": 0,
        "sum_vec": np.zeros(dim, dtype=np.float64),
        "sum_unit_vec": np.zeros(dim, dtype=np.float64),
        "sum_norm": 0.0,
        "sum_norm_sq": 0.0,
        "sources": set(),
        "plates": set(),
        "wells": set(),
    }


def stream_profile_subset(
    subset: str,
    url: str,
    target_jcps: set[str],
    *,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    s3 = pafs.S3FileSystem(anonymous=True, region="us-east-1")
    pf = pq.ParquetFile(s3_path_from_url(url), filesystem=s3)
    names = pf.schema_arrow.names
    meta_cols = ["Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_JCP2022"]
    feature_cols = [name for name in names if name.startswith("X_")]
    cols = meta_cols + feature_cols
    out: dict[str, dict[str, Any]] = {}
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols, use_threads=True):
        bnames = batch.schema.names
        jcp_values = batch.column(bnames.index("Metadata_JCP2022")).to_pylist()
        keep = [i for i, jcp in enumerate(jcp_values) if jcp in target_jcps]
        if not keep:
            continue
        source_values = batch.column(bnames.index("Metadata_Source")).to_pylist()
        plate_values = batch.column(bnames.index("Metadata_Plate")).to_pylist()
        well_values = batch.column(bnames.index("Metadata_Well")).to_pylist()
        feature_arrays = [
            batch.column(bnames.index(col)).to_numpy(zero_copy_only=False)
            for col in feature_cols
        ]
        mat = np.stack(feature_arrays, axis=1).astype(np.float64, copy=False)
        for i in keep:
            jcp = str(jcp_values[i])
            vec = mat[i]
            norm = float(np.linalg.norm(vec))
            rec = out.setdefault(jcp, init_acc(len(feature_cols)))
            rec["n"] += 1
            rec["sum_vec"] += vec
            if norm > 0:
                rec["sum_unit_vec"] += vec / norm
            rec["sum_norm"] += norm
            rec["sum_norm_sq"] += norm * norm
            source = norm_text(source_values[i])
            plate = norm_text(plate_values[i])
            well = norm_text(well_values[i])
            if source:
                rec["sources"].add(source)
            if plate:
                rec["plates"].add(plate)
            if source and plate and well:
                rec["wells"].add((source, plate, well))
    return out


def finalize_jcp_metrics(
    raw: dict[str, dict[str, Any]],
    jcp_to_gene_mod: dict[str, tuple[str, str]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    by_gene: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for jcp, rec in raw.items():
        n = int(rec["n"])
        if n <= 0:
            continue
        centroid = rec["sum_vec"] / n
        centroid_l2 = float(np.linalg.norm(centroid))
        mean_norm = float(rec["sum_norm"] / n)
        var_norm = max(0.0, float(rec["sum_norm_sq"] / n) - mean_norm * mean_norm)
        repl = None
        if centroid_l2 > 0:
            repl = float(np.dot(rec["sum_unit_vec"], centroid) / (n * centroid_l2))
        key = jcp_to_gene_mod.get(jcp)
        if not key:
            continue
        by_gene[key].append(
            {
                "jcp_id": jcp,
                "n_wells": n,
                "profile_norm_mean": mean_norm,
                "profile_norm_sd": math.sqrt(var_norm),
                "centroid_l2": centroid_l2,
                "replicate_cosine": repl,
                "source_count": len(rec["sources"]),
                "plate_count": len(rec["plates"]),
                "well_count": len(rec["wells"]),
            }
        )
    return by_gene


def aggregate_gene_metrics(by_gene_jcp: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (gkey, modality), parts in sorted(by_gene_jcp.items()):
        total_wells = sum(int(p["n_wells"]) for p in parts)
        if total_wells <= 0:
            continue

        def wmean(field: str) -> float:
            vals = [(float(p[field]), int(p["n_wells"])) for p in parts if p.get(field) is not None]
            denom = sum(w for _, w in vals)
            return sum(v * w for v, w in vals) / denom if denom else math.nan

        rows.append(
            {
                "gene_key": gkey,
                "jump_modality": modality,
                "profiled_jcp_count": len(parts),
                "profile_well_count": total_wells,
                "profile_source_count": len({(p["source_count"], p["plate_count"]) for p in parts}),
                "profile_plate_count": sum(int(p["plate_count"]) for p in parts),
                "mean_profile_norm": wmean("profile_norm_mean"),
                "mean_profile_norm_sd": wmean("profile_norm_sd"),
                "mean_centroid_l2": wmean("centroid_l2"),
                "max_centroid_l2": max(float(p["centroid_l2"]) for p in parts),
                "mean_replicate_cosine": wmean("replicate_cosine"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def join_profile_to_train(
    train_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {(str(r["gene_key"]), str(r["jump_modality"])): r for r in profile_rows}
    out: list[dict[str, Any]] = []
    for row in train_rows:
        prof = by_key.get((str(row.get("s0_gene_key")), str(row.get("jump_modality"))))
        if not prof:
            continue
        merged = dict(row)
        merged.update(prof)
        out.append(merged)
    return out


def load_outcome_targets(membership: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, path in OUTCOME_FILES.items():
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = (norm_text(row.get("dataset")), norm_text(row.get("condition")))
                if membership.get(key) != "train":
                    continue
                pp = mmd = None
                if label == "condition_exposure_cross":
                    if row.get("role") != "moderate_exposure_signal" or row.get("group") != "cross":
                        continue
                    pp = fnum(row.get("cross_pp_diff"))
                    mmd = fnum(row.get("cross_mmd_diff"))
                elif label == "qc_reliability_cross":
                    pp = fnum(row.get("cross_pp_diff"))
                    mmd = fnum(row.get("cross_mmd_diff"))
                elif label == "response_program":
                    pp = fnum(row.get("pp_delta"))
                    mmd = fnum(row.get("mmd_delta"))
                elif label == "lodo_domain":
                    pp = fnum(row.get("pp_mean"))
                    mmd = fnum(row.get("mmd_mean"))
                elif label == "background_actionability":
                    pp = fnum(row.get("pp_delta"))
                    mmd = fnum(row.get("mmd_delta"))
                if pp is None or mmd is None:
                    continue
                rows.append(
                    {
                        "target": label,
                        "dataset": key[0],
                        "condition": key[1],
                        "target_pp": pp,
                        "target_mmd": mmd,
                    }
                )
    return rows


def make_signal_rows(profile_join: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cond: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in profile_join:
        by_cond[(str(row["dataset"]), str(row["condition"]))].append(row)
    rows: list[dict[str, Any]] = []
    for out in outcomes:
        for prof in by_cond.get((out["dataset"], out["condition"]), []):
            merged = dict(prof)
            merged.update(out)
            rows.append(merged)
    return rows


def feature_result(rows: list[dict[str, Any]], feature: str, target: str, *, n_perm: int = 1000) -> dict[str, Any]:
    pairs = [(fnum(row.get(feature)), fnum(row.get(target)), norm_text(row.get("dataset"))) for row in rows]
    pairs = [(float(x), float(y), ds) for x, y, ds in pairs if x is not None and y is not None]
    if len(pairs) < 10 or len({x for x, _, _ in pairs}) < 2 or len({y for _, y, _ in pairs}) < 2:
        return {"feature": feature, "target": target, "n": len(pairs), "rho": None, "shuffle_p_abs": None}
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    actual = spearman(x, y)
    if actual is None:
        return {"feature": feature, "target": target, "n": len(pairs), "rho": None, "shuffle_p_abs": None}
    rng = random.Random(20260627)
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for i, (_, _, ds) in enumerate(pairs):
        by_dataset[ds].append(i)
    hits = total = 0
    for _ in range(n_perm):
        shuffled = x[:]
        for idxs in by_dataset.values():
            vals = [shuffled[i] for i in idxs]
            rng.shuffle(vals)
            for i, value in zip(idxs, vals):
                shuffled[i] = value
        rho = spearman(shuffled, y)
        if rho is None:
            continue
        total += 1
        if abs(rho) >= abs(actual):
            hits += 1
    return {
        "feature": feature,
        "target": target,
        "n": len(pairs),
        "rho": actual,
        "shuffle_p_abs": (hits + 1) / (total + 1) if total else 1.0,
    }


def analyze_signal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target_counts = Counter(str(row.get("target")) for row in rows)
    dataset_counts = Counter(str(row.get("dataset")) for row in rows)
    condition_count = len({(row.get("dataset"), row.get("condition")) for row in rows})
    feature_sets = {
        "profile": PROFILE_FEATURES,
        "control": CONTROL_FEATURES,
    }
    results = []
    for kind, features in feature_sets.items():
        for feature in features:
            for target in ("target_pp", "target_mmd"):
                res = feature_result(rows, feature, target)
                res["kind"] = kind
                results.append(res)
    profile_pp = [r for r in results if r["kind"] == "profile" and r["target"] == "target_pp" and r["rho"] is not None]
    control_pp = [r for r in results if r["kind"] == "control" and r["target"] == "target_pp" and r["rho"] is not None]
    profile_mmd = [r for r in results if r["kind"] == "profile" and r["target"] == "target_mmd" and r["rho"] is not None]
    best_profile_pp = max(profile_pp, key=lambda r: abs(float(r["rho"])), default=None)
    best_control_pp = max(control_pp, key=lambda r: abs(float(r["rho"])), default=None)
    max_profile_mmd = max((abs(float(r["rho"])) for r in profile_mmd), default=None)
    return {
        "signal_rows": len(rows),
        "unique_conditions": condition_count,
        "datasets": len(dataset_counts),
        "target_counts": target_counts.most_common(),
        "dataset_counts_top20": dataset_counts.most_common(20),
        "best_profile_pp_signal": best_profile_pp,
        "best_control_pp_signal": best_control_pp,
        "max_abs_profile_mmd_signal_rho": max_profile_mmd,
        "all_results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--subsets", default="crispr,orf")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    boundary = {
        "gpu_used": False,
        "training_or_inference_used": False,
        "full_profile_matrix_persisted": False,
        "profile_access": "streamed public ORF/CRISPR assembled parquet from S3; only aggregate CSV persisted",
        "canonical_multi_tracka_selection_used": False,
        "trackc_heldout_query_used": False,
        "chemical_v2_ack": False,
    }
    missing = [str(p) for p in (PROFILE_INDEX, TRAIN_JOIN, S0, CRISPR, ORF) if not p.is_file()]
    if missing:
        payload = {
            "status": "jump_cp_profile_aggregate_missing_inputs_no_gpu",
            "gpu_authorized": False,
            "boundary": boundary,
            "missing": missing,
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        OUT_MD.write_text("# JUMP-CP Profile Aggregate Gate\n\nMissing inputs; no GPU authorized.\n", encoding="utf-8")
        print(json.dumps({"status": payload["status"], "gpu_authorized": False}, indent=2))
        return 0

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    urls = profile_urls()
    train_rows = load_train_join_rows()
    by_gene_mod, jcp_to_gene_mod = load_gene_metadata()
    needed_jcps: set[str] = set()
    for row in train_rows:
        needed_jcps.update(by_gene_mod.get((str(row["s0_gene_key"]), str(row["jump_modality"])), []))
    raw_by_jcp: dict[str, dict[str, Any]] = {}
    subset_stats: list[dict[str, Any]] = []
    for subset in subsets:
        target = {jcp for jcp in needed_jcps if jcp_to_gene_mod.get(jcp, ("", ""))[1] == subset}
        if not target or subset not in urls:
            subset_stats.append({"subset": subset, "target_jcps": len(target), "profiled_jcps": 0, "skipped": True})
            continue
        raw = stream_profile_subset(subset, urls[subset], target, batch_size=args.batch_size)
        raw_by_jcp.update(raw)
        subset_stats.append(
            {
                "subset": subset,
                "target_jcps": len(target),
                "profiled_jcps": len(raw),
                "profiled_wells": sum(int(v["n"]) for v in raw.values()),
                "url": urls[subset],
            }
        )
    by_gene_jcp = finalize_jcp_metrics(raw_by_jcp, jcp_to_gene_mod)
    profile_rows = aggregate_gene_metrics(by_gene_jcp)
    profile_fields = [
        "gene_key",
        "jump_modality",
        *PROFILE_FEATURES,
    ]
    write_csv(OUT_PROFILE, profile_rows, profile_fields)

    profile_join = join_profile_to_train(train_rows, profile_rows)
    join_fields = list(train_rows[0].keys()) + PROFILE_FEATURES if train_rows else PROFILE_FEATURES
    write_csv(OUT_JOIN, profile_join, join_fields)

    membership = load_s0_membership()
    outcomes = load_outcome_targets(membership)
    signal_rows = make_signal_rows(profile_join, outcomes)
    signal_fields = list(dict.fromkeys([*(profile_join[0].keys() if profile_join else []), "target", "target_pp", "target_mmd"]))
    write_csv(OUT_SIGNAL, signal_rows, signal_fields)

    analysis = analyze_signal(signal_rows)
    reasons: list[str] = []
    if len(profile_join) < 50:
        reasons.append("profile_train_join_rows_below_50")
    if analysis["unique_conditions"] < 50:
        reasons.append("train_internal_signal_condition_count_below_50")
    if analysis["datasets"] < 3:
        reasons.append("train_internal_signal_dataset_count_below_3")
    best_profile = analysis["best_profile_pp_signal"]
    best_control = analysis["best_control_pp_signal"]
    if not best_profile or best_profile.get("rho") is None or abs(float(best_profile["rho"])) < 0.25:
        reasons.append("best_profile_pp_signal_abs_rho_below_0p25")
    if not best_profile or best_profile.get("shuffle_p_abs") is None or float(best_profile["shuffle_p_abs"]) > 0.01:
        reasons.append("best_profile_pp_signal_shuffle_p_above_0p01")
    if best_profile and best_control and best_profile.get("rho") is not None and best_control.get("rho") is not None:
        if abs(float(best_profile["rho"])) <= abs(float(best_control["rho"])):
            reasons.append("profile_signal_not_stronger_than_count_source_control")
    if analysis["max_abs_profile_mmd_signal_rho"] is not None and float(analysis["max_abs_profile_mmd_signal_rho"]) > 0.30:
        reasons.append("profile_mmd_signal_too_large")
    reasons.append("profile_gate_cpu_only_no_training_or_promotion")
    status = (
        "jump_cp_profile_aggregate_gate_pass_review_only_no_gpu"
        if len(reasons) == 1
        else "jump_cp_profile_aggregate_gate_fail_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorized": False,
        "training_authorized": False,
        "promotion_authorized": False,
        "boundary": boundary,
        "subset_stats": subset_stats,
        "summary": {
            "needed_jcps": len(needed_jcps),
            "profiled_jcps": len(raw_by_jcp),
            "profile_gene_modality_rows": len(profile_rows),
            "train_profile_join_rows": len(profile_join),
            **analysis,
        },
        "reasons": reasons,
        "outputs": {
            "profile_features": str(OUT_PROFILE),
            "train_profile_join": str(OUT_JOIN),
            "signal_rows": str(OUT_SIGNAL),
            "markdown": str(OUT_MD),
            "json": str(OUT_JSON),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def fmt_signal(item: dict[str, Any] | None) -> str:
        if not item:
            return "`None`"
        rho = item.get("rho")
        pval = item.get("shuffle_p_abs")
        rho_s = "None" if rho is None else f"{float(rho):+.4f}"
        p_s = "None" if pval is None else f"{float(pval):.4f}"
        return f"`{item['feature']} -> {item['target']}: rho={rho_s}, shuffle_p={p_s}, n={item['n']}`"

    lines = [
        "# JUMP-CP Profile Aggregate Gate",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/network source gate; no GPU, training, inference, checkpoint selection, canonical multi selection, or Track C query.",
        "- Streams public ORF/CRISPR assembled parquet from S3 and persists only aggregate profile features.",
        "- Outcome signal analysis is restricted to S0 `membership=train` rows from existing train/internal artifacts.",
        "",
        "## Profile Materialization",
        "",
        f"- needed JCP ids: `{len(needed_jcps)}`",
        f"- profiled JCP ids: `{len(raw_by_jcp)}`",
        f"- profile gene/modality rows: `{len(profile_rows)}`",
        f"- train profile join rows: `{len(profile_join)}`",
        f"- subset stats: `{subset_stats}`",
        "",
        "## Train/Internal Signal",
        "",
        f"- signal rows: `{analysis['signal_rows']}`",
        f"- unique conditions: `{analysis['unique_conditions']}`",
        f"- datasets: `{analysis['datasets']}`",
        f"- target counts: `{analysis['target_counts']}`",
        f"- best profile pp signal: {fmt_signal(analysis['best_profile_pp_signal'])}",
        f"- best count/source control pp signal: {fmt_signal(analysis['best_control_pp_signal'])}",
        f"- max |profile MMD signal rho|: `{analysis['max_abs_profile_mmd_signal_rho']}`",
        "",
        "## Decision",
        "",
        "This report cannot authorize GPU training or promotion. A pass would only authorize external/protocol review.",
        "",
        "## Reasons",
        "",
        *[f"- `{reason}`" for reason in reasons],
        "",
        "## Outputs",
        "",
        f"- profile features: `{OUT_PROFILE}`",
        f"- train profile join rows: `{OUT_JOIN}`",
        f"- signal rows: `{OUT_SIGNAL}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": False, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
