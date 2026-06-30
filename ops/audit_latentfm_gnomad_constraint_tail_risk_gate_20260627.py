#!/usr/bin/env python3
"""CPU-only gnomAD constraint tail-risk gate for LatentFM Track A.

This script audits whether public gnomAD gene constraint scores can become a
non-ACK, leakage-safe Track A tail-risk or route prior GPU entry.

Boundary:
- no training, no inference, no checkpoint reads beyond frozen eval JSON text;
- no GPU use;
- no Track C held-out query;
- no canonical multi for Track A selection;
- Chemical V2 is not used because exact ACK is absent.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "gnomad_constraint_tail_risk_gate_20260627"
OUT_MD = REPORTS / "LATENTFM_GNOMAD_CONSTRAINT_TAIL_RISK_GATE_20260627.md"
OUT_JSON = REPORTS / "latentfm_gnomad_constraint_tail_risk_gate_20260627.json"
OUT_ROWS = OUT_DIR / "joined_seed42_seed43_tracka_rows.csv"
OUT_MANIFEST = OUT_DIR / "source_manifest.json"

GNOMAD_SOURCE = (
    REPORTS
    / "external_artifact_sources_20260626/gnomad_constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt"
)
ARTIFACT_DIR = REPORTS / "gnomad_constraint_artifacts_20260626"
ARTIFACTS = {
    "gnomad_lof_constraint_score_neglog10_loeuf": {
        "path": ARTIFACT_DIR / "gnomad_lof_constraint_score_neglog10_loeuf.csv",
        "higher_means_more_constrained": True,
    },
    "gnomad_pli": {
        "path": ARTIFACT_DIR / "gnomad_pli.csv",
        "higher_means_more_constrained": True,
    },
    "gnomad_mis_z": {
        "path": ARTIFACT_DIR / "gnomad_mis_z.csv",
        "higher_means_more_constrained": True,
    },
    "gnomad_oe_lof_upper": {
        "path": ARTIFACT_DIR / "gnomad_oe_lof_upper.csv",
        "higher_means_more_constrained": False,
    },
}

DEPLOYABLE_INPUT_MANIFEST = (
    REPORTS / "tracka_deployable_benchmark_failure_taxonomy_20260627/input_manifest.tsv"
)
REQUIRED_INPUT_KEYS = {
    "seed42_family": ("seed42", ["test_single", "family_gene", "test_all"]),
    "seed43_family": ("seed43", ["test_single", "family_gene", "test_all"]),
}
TRACKA_GROUPS = ["test_single", "family_gene", "test_all"]
MMD_SAFE_THRESHOLD = 0.001
MIN_OVERLAP_ROWS = 20
MIN_DATASETS = 3
MIN_VARYING_DATASETS = 3
MIN_ABS_HIGH_LOW = 0.020
SHUFFLES = 2000


def norm(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "nan", "none", "na", "<na>"}:
        return ""
    return text


def to_float(value: Any) -> float | None:
    text = norm(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for pos in range(i, j + 1):
            out[order[pos]] = avg
        i = j + 1
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rank(xs), rank(ys))


def high_minus_low(rows: list[dict[str, Any]], metric: str = "pearson_pert") -> float | None:
    valid = [r for r in rows if to_float(r.get("risk_value")) is not None and to_float(r.get(metric)) is not None]
    if len(valid) < 3:
        return None
    ordered = sorted(valid, key=lambda r: float(r["risk_value"]))
    k = max(1, len(ordered) // 3)
    low = ordered[:k]
    high = ordered[-k:]
    return sum(float(r[metric]) for r in high) / len(high) - sum(float(r[metric]) for r in low) / len(low)


def read_deployable_inputs() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if not DEPLOYABLE_INPUT_MANIFEST.is_file():
        return paths
    with DEPLOYABLE_INPUT_MANIFEST.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = norm(row.get("input_key"))
            path = Path(norm(row.get("path")))
            if key in REQUIRED_INPUT_KEYS and path.is_file():
                paths[key] = path
    return paths


def read_condition_metrics(eval_json: Path, seed: str, groups: list[str]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    out: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for group in groups:
        group_payload = payload.get("groups", {}).get(group, {})
        for row in group_payload.get("condition_metrics", []):
            ds = norm(row.get("dataset"))
            cond = norm(row.get("condition"))
            if not ds or not cond:
                continue
            out[(seed, group, ds, cond)] = {
                "seed": seed,
                "group": group,
                "dataset": ds,
                "condition": cond,
                "pearson_pert": to_float(row.get("pearson_pert")),
                "pearson_ctrl": to_float(row.get("pearson_ctrl")),
                "direct_pearson": to_float(row.get("direct_pearson")),
                "test_mmd_clamped": to_float(row.get("test_mmd_clamped")),
                "n_src_eval": to_float(row.get("n_src_eval")),
                "n_gt_eval": to_float(row.get("n_gt_eval")),
            }
    return out


def read_artifact_rows() -> dict[str, list[dict[str, Any]]]:
    artifacts: dict[str, list[dict[str, Any]]] = {}
    for artifact, meta in ARTIFACTS.items():
        path = meta["path"]
        rows: list[dict[str, Any]] = []
        if not path.is_file():
            artifacts[artifact] = rows
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = to_float(row.get("artifact_value"))
                ds = norm(row.get("dataset"))
                cond = norm(row.get("condition"))
                if value is None or not ds or not cond:
                    continue
                risk = value if meta["higher_means_more_constrained"] else -value
                rows.append(
                    {
                        "artifact": artifact,
                        "dataset": ds,
                        "condition": cond,
                        "target": norm(row.get("target")) or cond,
                        "cell_background": norm(row.get("cell_background")),
                        "perturbation_type": norm(row.get("perturbation_type")),
                        "artifact_value": value,
                        "risk_value": risk,
                        "source": norm(row.get("source")),
                        "source_file": norm(row.get("source_file")),
                    }
                )
        artifacts[artifact] = rows
    return artifacts


def join_rows(
    artifacts: dict[str, list[dict[str, Any]]],
    metrics: dict[tuple[str, str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    for artifact, rows in artifacts.items():
        for arow in rows:
            for seed in ["seed42", "seed43"]:
                for group in TRACKA_GROUPS:
                    mrow = metrics.get((seed, group, arow["dataset"], arow["condition"]))
                    if not mrow:
                        continue
                    joined.append({**arow, **mrow})
    return joined


def permutation_p(rows: list[dict[str, Any]], observed: float | None, seed: int = 20260627) -> dict[str, float | None]:
    if observed is None or len(rows) < MIN_OVERLAP_ROWS:
        return {"abs_p": None, "less_p": None, "greater_p": None}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    rng = random.Random(seed)
    ge_abs = le = ge = 0
    for _ in range(SHUFFLES):
        shuffled: list[dict[str, Any]] = []
        for ds_rows in grouped.values():
            values = [float(r["risk_value"]) for r in ds_rows]
            rng.shuffle(values)
            for row, value in zip(ds_rows, values):
                new = dict(row)
                new["risk_value"] = value
                shuffled.append(new)
        perm = high_minus_low(shuffled)
        if perm is None:
            continue
        if abs(perm) >= abs(observed):
            ge_abs += 1
        if perm <= observed:
            le += 1
        if perm >= observed:
            ge += 1
    denom = SHUFFLES + 1
    return {
        "abs_p": (ge_abs + 1) / denom,
        "less_p": (le + 1) / denom,
        "greater_p": (ge + 1) / denom,
    }


def lodo(rows: list[dict[str, Any]], observed: float | None) -> dict[str, Any]:
    datasets = sorted({str(r["dataset"]) for r in rows})
    values = []
    sign_flips = 0
    for ds in datasets:
        subset = [r for r in rows if r["dataset"] != ds]
        delta = high_minus_low(subset)
        if delta is None:
            continue
        values.append({"heldout_dataset": ds, "high_minus_low_pp": delta})
        if observed is not None and observed != 0 and (delta > 0) != (observed > 0):
            sign_flips += 1
    return {
        "n_lodo": len(values),
        "sign_flips": sign_flips,
        "min_high_minus_low_pp": min((v["high_minus_low_pp"] for v in values), default=None),
        "max_high_minus_low_pp": max((v["high_minus_low_pp"] for v in values), default=None),
    }


def summarize_subset(rows: list[dict[str, Any]], artifact: str, seed: str, group: str, subset: str) -> dict[str, Any]:
    datasets = sorted({str(r["dataset"]) for r in rows})
    varying = sorted(
        ds for ds in datasets if len({round(float(r["risk_value"]), 8) for r in rows if r["dataset"] == ds}) >= 2
    )
    pp_values = [float(r["pearson_pert"]) for r in rows]
    mmd_values = [float(r["test_mmd_clamped"]) for r in rows if to_float(r.get("test_mmd_clamped")) is not None]
    risk_values = [float(r["risk_value"]) for r in rows]
    delta = high_minus_low(rows)
    pvals = permutation_p(rows, delta)
    lodo_stats = lodo(rows, delta)
    bad_tail = [1.0 if float(r["pearson_pert"]) < 0.0 else 0.0 for r in rows]
    rho = spearman(risk_values, pp_values) if len(rows) >= 3 else None
    rho_bad = spearman(risk_values, bad_tail) if len(set(bad_tail)) > 1 else None

    reasons = []
    if len(rows) < MIN_OVERLAP_ROWS:
        reasons.append("overlap_rows_below_20")
    if len(datasets) < MIN_DATASETS:
        reasons.append("dataset_count_below_3")
    if len(varying) < MIN_VARYING_DATASETS:
        reasons.append("varying_dataset_count_below_3")
    if mmd_values and max(mmd_values) > MMD_SAFE_THRESHOLD:
        reasons.append("mmd_veto_max_above_0p001")
    if delta is None or abs(delta) < MIN_ABS_HIGH_LOW:
        reasons.append("abs_high_low_pp_below_0p020")
    if pvals["abs_p"] is None or pvals["abs_p"] > 0.05:
        reasons.append("within_dataset_gene_label_shuffle_abs_p_gt_0p05")
    if lodo_stats["sign_flips"]:
        reasons.append("lodo_sign_flip")
    status = "pass_review_only_no_gpu" if not reasons else "fail_no_gpu"

    return {
        "artifact": artifact,
        "seed": seed,
        "group": group,
        "subset": subset,
        "status": status,
        "n": len(rows),
        "datasets": len(datasets),
        "varying_datasets": len(varying),
        "pp_mean": sum(pp_values) / len(pp_values) if pp_values else None,
        "pp_min": min(pp_values) if pp_values else None,
        "mmd_max": max(mmd_values) if mmd_values else None,
        "mmd_safe_threshold": MMD_SAFE_THRESHOLD,
        "high_minus_low_pp": delta,
        "spearman_risk_pp": rho,
        "spearman_risk_bad_tail": rho_bad,
        "bad_tail_frac": sum(bad_tail) / len(bad_tail) if bad_tail else None,
        "within_dataset_gene_label_shuffle_abs_p": pvals["abs_p"],
        "within_dataset_gene_label_shuffle_less_p": pvals["less_p"],
        "within_dataset_gene_label_shuffle_greater_p": pvals["greater_p"],
        **{f"lodo_{k}": v for k, v in lodo_stats.items()},
        "reasons": reasons,
    }


def summarize(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for artifact in sorted({str(r["artifact"]) for r in joined}):
        for seed in ["seed42", "seed43"]:
            for group in TRACKA_GROUPS:
                rows = [
                    r
                    for r in joined
                    if r["artifact"] == artifact and r["seed"] == seed and r["group"] == group
                ]
                if not rows:
                    summaries.append(
                        {
                            "artifact": artifact,
                            "seed": seed,
                            "group": group,
                            "subset": "all_rows",
                            "status": "fail_no_gpu",
                            "n": 0,
                            "reasons": ["no_joined_rows"],
                        }
                    )
                    continue
                summaries.append(summarize_subset(rows, artifact, seed, group, "all_rows"))
                safe = [
                    r
                    for r in rows
                    if to_float(r.get("test_mmd_clamped")) is not None
                    and float(r["test_mmd_clamped"]) <= MMD_SAFE_THRESHOLD
                ]
                if safe:
                    summaries.append(summarize_subset(safe, artifact, seed, group, "mmd_safe"))
                else:
                    summaries.append(
                        {
                            "artifact": artifact,
                            "seed": seed,
                            "group": group,
                            "subset": "mmd_safe",
                            "status": "fail_no_gpu",
                            "n": 0,
                            "reasons": ["no_mmd_safe_rows"],
                        }
                    )
    return summaries


def write_joined_rows(rows: list[dict[str, Any]]) -> None:
    fields = [
        "artifact",
        "seed",
        "group",
        "dataset",
        "condition",
        "target",
        "cell_background",
        "perturbation_type",
        "artifact_value",
        "risk_value",
        "pearson_pert",
        "pearson_ctrl",
        "direct_pearson",
        "test_mmd_clamped",
        "n_src_eval",
        "n_gt_eval",
        "source",
        "source_file",
    ]
    with OUT_ROWS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_manifest(eval_paths: dict[str, Path]) -> dict[str, Any]:
    manifest = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M CST"),
        "boundary": {
            "cpu_only": True,
            "no_training": True,
            "no_inference": True,
            "uses_gpu": False,
            "uses_trackc_heldout_query": False,
            "uses_canonical_multi_for_tracka_selection": False,
            "chemical_v2_used": False,
            "chemical_v2_exact_ack": False,
        },
        "sources": {
            "gnomad_source": {
                "path": str(GNOMAD_SOURCE),
                "exists": GNOMAD_SOURCE.is_file(),
                "bytes": GNOMAD_SOURCE.stat().st_size if GNOMAD_SOURCE.is_file() else None,
                "sha256": sha256_file(GNOMAD_SOURCE),
            },
            "deployable_input_manifest": {
                "path": str(DEPLOYABLE_INPUT_MANIFEST),
                "exists": DEPLOYABLE_INPUT_MANIFEST.is_file(),
                "sha256": sha256_file(DEPLOYABLE_INPUT_MANIFEST),
            },
            "constraint_artifacts": {
                name: {
                    "path": str(meta["path"]),
                    "exists": meta["path"].is_file(),
                    "sha256": sha256_file(meta["path"]),
                }
                for name, meta in ARTIFACTS.items()
            },
            "frozen_eval_json": {
                key: {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path)}
                for key, path in sorted(eval_paths.items())
            },
        },
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.{digits}f}"
    return str(value)


def write_report(payload: dict[str, Any]) -> None:
    summaries = payload["summaries"]
    compact = [
        row
        for row in summaries
        if row.get("subset") == "mmd_safe" and row.get("group") in TRACKA_GROUPS
    ]
    lines = [
        "# LatentFM gnomAD Constraint Tail-Risk Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only gate over local gnomAD v2.1.1 gene constraint artifacts and frozen seed42/43 Track A condition metrics.",
        "- Groups used: `test_single`, `family_gene`, `test_all`; canonical multi groups are not read or used for Track A selection.",
        "- Does not train, infer, read Track C held-out query, use Chemical V2, or use GPU.",
        "",
        "## Source And Coverage",
        "",
        f"- gnomAD source exists: `{payload['source_checks']['gnomad_source_exists']}`",
        f"- artifact rows per score: `{payload['source_checks']['artifact_rows']}`",
        f"- joined rows: `{payload['source_checks']['joined_rows']}`",
        f"- joined rows by seed/group: `{payload['source_checks']['joined_by_seed_group']}`",
        f"- degree/pathway proxy definable from allowed sources: `{payload['control_definability']['degree_pathway_proxy_definable']}`",
        f"- gene-label shuffle definable: `{payload['control_definability']['gene_label_shuffle_definable']}`",
        "",
        "## MMD-Safe Gate Summary",
        "",
        "| artifact | seed | group | n | datasets | varying datasets | pp mean | pp min | MMD max | high-low pp | shuffle abs p | LODO flips | status | reasons |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in compact:
        lines.append(
            "| `{artifact}` | `{seed}` | `{group}` | {n} | {datasets} | {varying_datasets} | {pp_mean} | {pp_min} | {mmd_max} | {high_minus_low_pp} | {shuffle_p} | {lodo_flips} | `{status}` | `{reasons}` |".format(
                artifact=row.get("artifact"),
                seed=row.get("seed"),
                group=row.get("group"),
                n=row.get("n", 0),
                datasets=row.get("datasets", 0),
                varying_datasets=row.get("varying_datasets", 0),
                pp_mean=fmt(row.get("pp_mean")),
                pp_min=fmt(row.get("pp_min")),
                mmd_max=fmt(row.get("mmd_max")),
                high_minus_low_pp=fmt(row.get("high_minus_low_pp")),
                shuffle_p=fmt(row.get("within_dataset_gene_label_shuffle_abs_p")),
                lodo_flips=row.get("lodo_sign_flips", "NA"),
                status=row.get("status"),
                reasons=",".join(row.get("reasons", [])),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- gnomAD/constraint is non-ACK and source-local, so it is eligible for CPU audit as a leakage-safe external gene-level prior.",
            "- It does not pass this gate as a GPU entry: mmd-safe exact Track A rows do not show a stable within-dataset shuffled tail-risk association across seed42/43 and requested groups.",
            "- The degree/pathway proxy control is not definable from the allowed local gnomAD constraint artifacts, so even a stronger raw association would need an additional non-duplicative control source before GPU.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- joined rows: `{OUT_ROWS}`",
            f"- source manifest: `{OUT_MANIFEST}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_paths = read_deployable_inputs()
    metrics: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    missing_inputs = []
    for input_key, (seed, groups) in REQUIRED_INPUT_KEYS.items():
        path = eval_paths.get(input_key)
        if not path:
            missing_inputs.append(input_key)
            continue
        metrics.update(read_condition_metrics(path, seed, groups))

    artifacts = read_artifact_rows()
    joined = join_rows(artifacts, metrics)
    write_joined_rows(joined)
    manifest = write_manifest(eval_paths)
    summaries = summarize(joined)

    any_pass = any(row.get("status") == "pass_review_only_no_gpu" for row in summaries)
    hard_blockers = []
    if missing_inputs:
        hard_blockers.append(f"missing_frozen_eval_inputs:{','.join(missing_inputs)}")
    if not GNOMAD_SOURCE.is_file():
        hard_blockers.append("missing_gnomad_source")
    if not joined:
        hard_blockers.append("no_joined_rows")
    if any_pass:
        hard_blockers.append("review_pass_requires_degree_pathway_proxy_control_before_gpu")

    artifact_counts = {name: len(rows) for name, rows in artifacts.items()}
    joined_by_seed_group = Counter((r["seed"], r["group"]) for r in joined)
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M CST"),
        "status": "fail_no_gpu",
        "gpu_authorized": False,
        "boundary": manifest["boundary"],
        "source_checks": {
            "gnomad_source_exists": GNOMAD_SOURCE.is_file(),
            "artifact_rows": artifact_counts,
            "metrics_rows": len(metrics),
            "joined_rows": len(joined),
            "joined_by_seed_group": {f"{k[0]}:{k[1]}": v for k, v in sorted(joined_by_seed_group.items())},
            "missing_inputs": missing_inputs,
        },
        "control_definability": {
            "gene_label_shuffle_definable": bool(joined),
            "degree_pathway_proxy_definable": False,
            "degree_pathway_proxy_note": (
                "Allowed local gnomAD constraint artifacts provide gene-level constraint scores only; "
                "no non-duplicative degree/pathway covariate was materialized in this gate."
            ),
            "dataset_tail_mmd_veto_input_exists": bool(joined)
            and all(to_float(r.get("test_mmd_clamped")) is not None for r in joined),
        },
        "hard_blockers": hard_blockers,
        "summaries": summaries,
        "outputs": {
            "markdown": str(OUT_MD),
            "json": str(OUT_JSON),
            "joined_rows": str(OUT_ROWS),
            "source_manifest": str(OUT_MANIFEST),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    print(json.dumps({"status": payload["status"], "gpu_authorized": False, "out_json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
