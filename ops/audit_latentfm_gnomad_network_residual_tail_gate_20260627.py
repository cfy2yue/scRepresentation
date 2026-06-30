#!/usr/bin/env python3
"""Residualized gnomAD/network tail-risk gate on train/internal rows."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
RESIDUAL_ROWS = ROOT / "reports/latentfm_xverse_tracka_residual_forensics_conditions_20260622.csv"
GNOMAD_DIR = ROOT / "reports/gnomad_constraint_artifacts_20260626"
GOA = ROOT / "dataset/external_priors/goa_human_20260519/goa_human_gene_terms.tsv"
REACTOME = ROOT / "dataset/external_priors/reactome_pathways_current_20260623/reactome_gene_pathways.tsv"
CORUM = ROOT / "dataset/external_priors/corum_complexes_20260624/corum_human_gene_complexes.tsv"
OMNIPATH = ROOT / "dataset/external_priors/omnipath_tf_20260623/omnipath_tf_target_gene_features.tsv"

OUT_DIR = ROOT / "reports/gnomad_network_residual_tail_gate_20260627"
OUT_ROWS = OUT_DIR / "gnomad_network_residual_tail_rows.csv"
OUT_JSON = ROOT / "reports/latentfm_gnomad_network_residual_tail_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_GNOMAD_NETWORK_RESIDUAL_TAIL_GATE_20260627.md"

GNOMAD_FILES = [
    "gnomad_lof_constraint_score_neglog10_loeuf.csv",
    "gnomad_pli.csv",
    "gnomad_mis_z.csv",
    "gnomad_oe_lof_upper.csv",
]
CONTROL_FIELDS = [
    "n_go_terms",
    "n_reactome_pathways",
    "n_complexes",
    "tf_out_degree",
    "target_in_degree",
    "tf_activation_out_degree",
    "tf_inhibition_out_degree",
    "target_activation_in_degree",
    "target_inhibition_in_degree",
    "gene_train_count",
]
TARGETS = [
    "bad_pp",
    "anchor_mmd_clamped",
    "target_residual_norm",
    "anchor_minus_gene_raw_mean_abs",
]


def norm_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def gene_key(value: object) -> str:
    return norm_text(value).upper()


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


def load_simple_tsv(path: Path, key_field: str, fields: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            key = gene_key(row.get(key_field))
            if not key:
                continue
            rec = out.setdefault(key, {})
            for field in fields:
                val = fnum(row.get(field))
                if val is not None:
                    rec[field] = val
    return out


def load_network_controls() -> dict[str, dict[str, float]]:
    controls: dict[str, dict[str, float]] = defaultdict(dict)
    for source in [
        load_simple_tsv(GOA, "gene", ["n_go_terms"]),
        load_simple_tsv(REACTOME, "gene", ["n_reactome_pathways"]),
        load_simple_tsv(CORUM, "gene", ["n_complexes"]),
        load_simple_tsv(
            OMNIPATH,
            "gene",
            [
                "tf_out_degree",
                "target_in_degree",
                "tf_activation_out_degree",
                "tf_inhibition_out_degree",
                "target_activation_in_degree",
                "target_inhibition_in_degree",
            ],
        ),
    ]:
        for gene, rec in source.items():
            controls[gene].update(rec)
    return controls


def load_gnomad() -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for name in GNOMAD_FILES:
        path = GNOMAD_DIR / name
        artifact = name.removesuffix(".csv")
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = (norm_text(row.get("dataset")), norm_text(row.get("condition")))
                val = fnum(row.get("artifact_value"))
                if key[0] and key[1] and val is not None:
                    out[key][artifact] = val
                    out[key]["gene"] = gene_key(row.get("target"))
    return out


def build_rows() -> list[dict[str, Any]]:
    controls = load_network_controls()
    gnomad = load_gnomad()
    rows: list[dict[str, Any]] = []
    with RESIDUAL_ROWS.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            group = norm_text(row.get("group"))
            if not group.startswith("internal_val_"):
                continue
            key = (norm_text(row.get("dataset")), norm_text(row.get("condition")))
            grec = gnomad.get(key)
            if not grec:
                continue
            gene = gene_key(row.get("gene") or grec.get("gene"))
            out: dict[str, Any] = {
                "group": group,
                "dataset": key[0],
                "condition": key[1],
                "gene": gene,
            }
            for field in (
                "gene_train_count",
                "anchor_pearson_pert",
                "anchor_mmd_clamped",
                "target_residual_norm",
                "anchor_minus_gene_raw_mean",
            ):
                out[field] = fnum(row.get(field))
            if out["anchor_pearson_pert"] is None:
                continue
            out["bad_pp"] = -float(out["anchor_pearson_pert"])
            out["anchor_minus_gene_raw_mean_abs"] = abs(float(out["anchor_minus_gene_raw_mean"] or 0.0))
            for artifact in [name.removesuffix(".csv") for name in GNOMAD_FILES]:
                out[artifact] = grec.get(artifact)
            for field in CONTROL_FIELDS:
                if field == "gene_train_count":
                    continue
                out[field] = controls.get(gene, {}).get(field, 0.0)
            rows.append(out)
    return rows


def residualize(rows: list[dict[str, Any]], artifact: str) -> None:
    valid = [row for row in rows if fnum(row.get(artifact)) is not None]
    if len(valid) < 10:
        for row in rows:
            row[f"{artifact}_network_resid"] = ""
        return
    y = np.array([float(row[artifact]) for row in valid], dtype=float)
    cols = []
    for field in CONTROL_FIELDS:
        vals = [fnum(row.get(field)) for row in valid]
        cols.append([math.log1p(max(0.0, float(v or 0.0))) for v in vals])
    X = np.array([[1.0, *vals] for vals in zip(*cols)], dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    for row, value in zip(valid, resid):
        row[f"{artifact}_network_resid"] = float(value)


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


def lodo_min_signed(rows: list[dict[str, Any]], feature: str, target: str) -> float | None:
    vals = [(fnum(row.get(feature)), fnum(row.get(target)), norm_text(row.get("dataset"))) for row in rows]
    vals = [(float(x), float(y), ds) for x, y, ds in vals if x is not None and y is not None]
    if len(vals) < 10:
        return None
    full = spearman([x for x, _, _ in vals], [y for _, y, _ in vals])
    if full is None:
        return None
    sign = 1.0 if full >= 0 else -1.0
    outs = []
    for leave in sorted({ds for _, _, ds in vals}):
        part = [(x, y) for x, y, ds in vals if ds != leave]
        if len(part) < 10:
            continue
        rho = spearman([x for x, _ in part], [y for _, y in part])
        if rho is not None:
            outs.append(sign * rho)
    return min(outs) if outs else None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    boundary = {
        "gpu_used": False,
        "training_or_inference_used": False,
        "canonical_multi_selection_used": False,
        "trackc_heldout_query_used": False,
        "selection_scope": "train/internal residual forensics rows only",
        "canonical_tracka_rows_used_for_selection": False,
    }
    missing = [str(p) for p in [RESIDUAL_ROWS, GOA, REACTOME, CORUM, OMNIPATH] if not p.is_file()]
    missing.extend(str(GNOMAD_DIR / name) for name in GNOMAD_FILES if not (GNOMAD_DIR / name).is_file())
    if missing:
        payload = {"status": "gnomad_network_residual_tail_gate_missing_inputs_no_gpu", "gpu_authorized": False, "missing": missing, "boundary": boundary}
        OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        OUT_MD.write_text("# gnomAD Network Residual Tail Gate\n\nMissing inputs; no GPU.\n", encoding="utf-8")
        print(json.dumps({"status": payload["status"], "gpu_authorized": False}, indent=2))
        return 0
    rows = build_rows()
    artifacts = [name.removesuffix(".csv") for name in GNOMAD_FILES]
    for artifact in artifacts:
        residualize(rows, artifact)
    fields = [
        "group",
        "dataset",
        "condition",
        "gene",
        "gene_train_count",
        "anchor_pearson_pert",
        "bad_pp",
        "anchor_mmd_clamped",
        "target_residual_norm",
        "anchor_minus_gene_raw_mean_abs",
        *CONTROL_FIELDS[1:],
        *artifacts,
        *[f"{artifact}_network_resid" for artifact in artifacts],
    ]
    with OUT_ROWS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    results = []
    for artifact in artifacts:
        for feature in (artifact, f"{artifact}_network_resid"):
            for target in TARGETS:
                res = feature_result(rows, feature, target)
                res["artifact"] = artifact
                res["residualized"] = feature.endswith("_network_resid")
                res["lodo_min_signed"] = lodo_min_signed(rows, feature, target)
                results.append(res)
    residual_pp = [r for r in results if r["residualized"] and r["target"] == "bad_pp" and r["rho"] is not None]
    best_resid_badpp = max(residual_pp, key=lambda r: abs(float(r["rho"])), default=None)
    residual_mmd = [r for r in results if r["residualized"] and r["target"] == "anchor_mmd_clamped" and r["rho"] is not None]
    max_abs_mmd = max((abs(float(r["rho"])) for r in residual_mmd), default=None)
    datasets = Counter(row["dataset"] for row in rows)
    reasons: list[str] = []
    if len(rows) < 50:
        reasons.append("joined_internal_rows_below_50")
    if len(datasets) < 3:
        reasons.append("dataset_count_below_3")
    if not best_resid_badpp or abs(float(best_resid_badpp["rho"])) < 0.25:
        reasons.append("best_residualized_badpp_abs_rho_below_0p25")
    if not best_resid_badpp or best_resid_badpp.get("shuffle_p_abs") is None or float(best_resid_badpp["shuffle_p_abs"]) > 0.01:
        reasons.append("best_residualized_badpp_shuffle_p_above_0p01")
    if not best_resid_badpp or best_resid_badpp.get("lodo_min_signed") is None or float(best_resid_badpp["lodo_min_signed"]) <= 0.05:
        reasons.append("best_residualized_badpp_lodo_min_signed_below_0p05")
    if max_abs_mmd is not None and max_abs_mmd > 0.30:
        reasons.append("residualized_constraint_mmd_association_too_large")
    reasons.append("cpu_only_review_not_training_or_promotion")
    status = "gnomad_network_residual_tail_gate_pass_review_only_no_gpu" if len(reasons) == 1 else "gnomad_network_residual_tail_gate_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "training_authorized": False,
        "promotion_authorized": False,
        "boundary": boundary,
        "summary": {
            "joined_rows": len(rows),
            "datasets": len(datasets),
            "dataset_counts_top20": datasets.most_common(20),
            "best_residualized_badpp_signal": best_resid_badpp,
            "max_abs_residualized_mmd_signal_rho": max_abs_mmd,
            "all_results": results,
        },
        "reasons": reasons,
        "outputs": {"rows": str(OUT_ROWS), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def fmt(item: dict[str, Any] | None) -> str:
        if not item:
            return "`None`"
        rho = item.get("rho")
        pval = item.get("shuffle_p_abs")
        lodo = item.get("lodo_min_signed")
        return (
            f"`{item['feature']} -> {item['target']}: "
            f"rho={float(rho):+.4f}, shuffle_p={float(pval):.4f}, "
            f"lodo_min_signed={lodo}, n={item['n']}`"
        )

    lines = [
        "# gnomAD Network Residual Tail Gate",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only train/internal residual gate.",
        "- Joins gnomAD constraint scores with GOA/Reactome/CORUM/OmniPath controls.",
        "- Does not train, infer, select checkpoints, read canonical multi, or read Track C query.",
        "",
        "## Summary",
        "",
        f"- joined rows: `{len(rows)}`",
        f"- datasets: `{len(datasets)}`",
        f"- best residualized bad-pp signal: {fmt(best_resid_badpp)}",
        f"- max |residualized MMD signal rho|: `{max_abs_mmd}`",
        "",
        "## Decision",
        "",
        "A pass would only authorize external/protocol review, not GPU training or promotion.",
        "",
        "## Reasons",
        "",
        *[f"- `{reason}`" for reason in reasons],
        "",
        "## Outputs",
        "",
        f"- rows: `{OUT_ROWS}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": False, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
