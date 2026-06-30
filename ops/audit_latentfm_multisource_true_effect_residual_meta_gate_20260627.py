#!/usr/bin/env python3
"""Multi-source true-effect residual meta-gate for LatentFM.

CPU/report-only. This reconciles local true-effect/reliability artifacts across
sources and asks whether any pooled, source-heldout residual signal is strong
enough to justify a later GPU design. It does not train, infer, select
checkpoints, read canonical multi for Track A selection, or read Track C query.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "multisource_true_effect_residual_meta_gate_20260627"
OUT_JSON = REPORTS / "latentfm_multisource_true_effect_residual_meta_gate_20260627.json"
OUT_MD = REPORTS / "LATENTFM_MULTISOURCE_TRUE_EFFECT_RESIDUAL_META_GATE_20260627.md"
OUT_ROWS = OUT_DIR / "multisource_true_effect_meta_rows.csv"
OUT_ARTIFACTS = OUT_DIR / "multisource_true_effect_artifact_summary.csv"
SEED = 271828


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
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
    if not math.isfinite(out):
        return None
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        for k in range(i, j):
            out[order[k]] = avg
        i = j
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    return pearson(rank(xs), rank(ys))


def finite_pair(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and math.isfinite(x) and math.isfinite(y):
            xs.append(float(x))
            ys.append(float(y))
    return xs, ys


def residualize_by_group(rows: list[dict[str, Any]], key: str, group_key: str, out_key: str) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = row.get(key)
        if isinstance(val, (int, float)) and math.isfinite(val):
            grouped[norm(row.get(group_key))].append(float(val))
    means = {group: mean(vals) for group, vals in grouped.items() if vals}
    for row in rows:
        val = row.get(key)
        group = norm(row.get(group_key))
        row[out_key] = None
        if isinstance(val, (int, float)) and math.isfinite(val) and group in means:
            row[out_key] = float(val) - means[group]


def add_row(
    rows: list[dict[str, Any]],
    *,
    source_family: str,
    dataset: str,
    condition: str,
    artifact: str,
    artifact_role: str,
    artifact_value: float | None,
    pearson_pert: float | None,
    mmd: float | None,
    evidence_scope: str,
    source_file: str,
) -> None:
    if not dataset or not condition or not artifact or artifact_value is None or pearson_pert is None:
        return
    rows.append(
        {
            "source_family": source_family,
            "dataset": dataset,
            "condition": condition,
            "artifact": artifact,
            "artifact_role": artifact_role or "response_candidate",
            "artifact_value": artifact_value,
            "bad_pp": -float(pearson_pert),
            "mmd": None if mmd is None else float(mmd),
            "evidence_scope": evidence_scope,
            "source_file": source_file,
        }
    )


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # Train/internal eligible source. This is the only local true-effect-like
    # train/internal table with explicit internal cross/family rows.
    replogle = REPORTS / "replogle_trainonly_internal_difficulty_gate_20260627/replogle_trainonly_internal_difficulty_joined_rows.csv"
    for rec in read_csv(replogle):
        for col in ("cnv_score_z", "TE_ratio", "std_leverage_score"):
            add_row(
                rows,
                source_family="replogle_traininternal",
                dataset=norm(rec.get("dataset")),
                condition=norm(rec.get("condition")),
                artifact=f"replogle_{col}",
                artifact_role="response_candidate",
                artifact_value=to_float(rec.get(col)),
                pearson_pert=to_float(rec.get("anchor_pearson_pert")),
                mmd=to_float(rec.get("anchor_mmd_clamped")),
                evidence_scope="train_internal",
                source_file=str(replogle),
            )
        for col in ("UMI_count_unfiltered", "num_cells_filtered", "mitopercent", "z_gemgroup_UMI"):
            add_row(
                rows,
                source_family="replogle_traininternal",
                dataset=norm(rec.get("dataset")),
                condition=norm(rec.get("condition")),
                artifact=f"replogle_qc_{col}",
                artifact_role="qc_control",
                artifact_value=to_float(rec.get(col)),
                pearson_pert=to_float(rec.get("anchor_pearson_pert")),
                mmd=to_float(rec.get("anchor_mmd_clamped")),
                evidence_scope="train_internal",
                source_file=str(replogle),
            )

    # Generic train/internal proxy preflight rows from GWT. This source is a
    # reliability proxy and is treated as candidate evidence, but not as a
    # stand-alone true-effect pass.
    gwt = REPORTS / "latentfm_gwt_condition_reliability_artifact_preflight_20260627_rows.csv"
    for rec in read_csv(gwt):
        pp = to_float(rec.get("pp_proxy_mean"))
        if pp is None:
            continue
        artifact = norm(rec.get("artifact"))
        role = "qc_or_reliability_control" if "knockdown" not in artifact and "correlation" not in artifact else "response_candidate"
        add_row(
            rows,
            source_family="gwt_reliability_proxy",
            dataset=norm(rec.get("dataset")),
            condition=norm(rec.get("condition")),
            artifact=artifact,
            artifact_role=role,
            artifact_value=to_float(rec.get("artifact_value")),
            pearson_pert=pp,
            mmd=to_float(rec.get("mmd_proxy_max")),
            evidence_scope="train_internal_proxy",
            source_file=str(gwt),
        )

    # Diagnostic-only test-metric sources. These help assess whether pooling
    # would merely repackage held-out/test associations, but they cannot
    # authorize GPU.
    diagnostic_sources = [
        ("gasperini_author_testmetric", REPORTS / "gasperini_author_self_effect_preview_gate_20260627/gasperini_author_self_effect_preview_joined_rows.csv"),
        ("frangieh_orcs_testmetric", REPORTS / "frangieh_orcs_response_preview_gate_20260627/frangieh_orcs_response_preview_joined_rows.csv"),
        ("replogle_bulk_testmetric", REPORTS / "replogle_bulk_artifact_gate_20260627/replogle_bulk_artifact_gate_joined_rows.csv"),
        ("depmap_dependency_testmetric", REPORTS / "depmap_24q4_dependency_gate_20260627/depmap_24q4_dependency_gate_joined_rows.csv"),
        ("adamson_guide_testmetric", REPORTS / "adamson_author_guide_support_preview_gate_20260627/adamson_author_guide_support_preview_joined_rows.csv"),
        ("papalexi_metadata_testmetric", REPORTS / "papalexi_author_metadata_preview_gate_20260627/papalexi_author_metadata_preview_joined_rows.csv"),
    ]
    for source_family, path in diagnostic_sources:
        for rec in read_csv(path):
            role = norm(rec.get("artifact_role")) or ("qc_control" if "guide" in source_family or "metadata" in source_family else "response_candidate")
            add_row(
                rows,
                source_family=source_family,
                dataset=norm(rec.get("dataset")),
                condition=norm(rec.get("condition")),
                artifact=norm(rec.get("artifact")),
                artifact_role=role,
                artifact_value=to_float(rec.get("artifact_value")),
                pearson_pert=to_float(rec.get("pearson_pert")),
                mmd=to_float(rec.get("test_mmd_clamped")),
                evidence_scope="diagnostic_test_metric_only",
                source_file=str(path),
            )
    return rows


def orient_artifacts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_artifact: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_artifact[(row["evidence_scope"], row["source_family"], row["artifact"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (scope, source_family, artifact), subset in by_artifact.items():
        vals = [float(row["artifact_value"]) for row in subset]
        mu = mean(vals)
        var = sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)
        sd = math.sqrt(var) if var > 0 else 0.0
        for row in subset:
            row["artifact_z"] = 0.0 if sd == 0 else (float(row["artifact_value"]) - mu) / sd
        xs, ys = finite_pair(subset, "artifact_z", "bad_pp")
        rho_bad = spearman(xs, ys)
        direction = 1.0 if (rho_bad is None or rho_bad >= 0) else -1.0
        for row in subset:
            row["oriented_score"] = direction * float(row["artifact_z"])
        sx, sy = finite_pair(subset, "oriented_score", "bad_pp")
        mx, my = finite_pair(subset, "oriented_score", "mmd")
        summaries.append(
            {
                "evidence_scope": scope,
                "source_family": source_family,
                "artifact": artifact,
                "artifact_role": subset[0].get("artifact_role", ""),
                "n": len(subset),
                "datasets": len({row["dataset"] for row in subset}),
                "direction": direction,
                "signed_rho_bad_pp": spearman(sx, sy),
                "abs_rho_mmd": None if len(mx) < 3 else abs(spearman(mx, my) or 0.0),
            }
        )
    return summaries


def aggregate_condition_scores(rows: list[dict[str, Any]], *, scope_filter: set[str], role_prefix: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["evidence_scope"] not in scope_filter:
            continue
        role = norm(row.get("artifact_role"))
        if role_prefix == "candidate" and "control" in role:
            continue
        if role_prefix == "control" and "control" not in role and "qc" not in role:
            continue
        grouped[(row["source_family"], row["dataset"], row["condition"])].append(row)
    out: list[dict[str, Any]] = []
    for (source_family, dataset, condition), subset in grouped.items():
        scores = [float(row["oriented_score"]) for row in subset if math.isfinite(float(row["oriented_score"]))]
        bads = [float(row["bad_pp"]) for row in subset]
        mmds = [float(row["mmd"]) for row in subset if isinstance(row.get("mmd"), (int, float)) and math.isfinite(float(row["mmd"]))]
        if not scores or not bads:
            continue
        out.append(
            {
                "source_family": source_family,
                "dataset": dataset,
                "condition": condition,
                "meta_score": mean(scores),
                "bad_pp": mean(bads),
                "mmd": mean(mmds) if mmds else None,
                "n_artifacts": len(scores),
                "role_bucket": role_prefix,
            }
        )
    return out


def permutation_p(rows: list[dict[str, Any]], *, n_perm: int = 1000) -> float | None:
    xs, ys = finite_pair(rows, "meta_score", "bad_pp")
    obs = spearman(xs, ys)
    if obs is None:
        return None
    rng = random.Random(SEED)
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_dataset[row["dataset"]].append(idx)
    ge = 0
    valid = 0
    for _ in range(n_perm):
        shuffled = [row["meta_score"] for row in rows]
        for idxs in by_dataset.values():
            vals = [shuffled[i] for i in idxs]
            rng.shuffle(vals)
            for i, val in zip(idxs, vals):
                shuffled[i] = val
        rho = spearman(shuffled, [row["bad_pp"] for row in rows])
        if rho is None:
            continue
        valid += 1
        if rho >= obs:
            ge += 1
    return (ge + 1) / (valid + 1) if valid else None


def eval_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    xs, ys = finite_pair(rows, "meta_score", "bad_pp")
    mx, my = finite_pair(rows, "meta_score", "mmd")
    rho = spearman(xs, ys)
    mmd_rho = spearman(mx, my) if len(mx) >= 3 else None
    dataset_rhos = {}
    for ds in sorted({row["dataset"] for row in rows}):
        sub = [row for row in rows if row["dataset"] == ds]
        sx, sy = finite_pair(sub, "meta_score", "bad_pp")
        r = spearman(sx, sy)
        if r is not None:
            dataset_rhos[ds] = r
    source_rhos = {}
    for source in sorted({row["source_family"] for row in rows}):
        sub = [row for row in rows if row["source_family"] == source]
        sx, sy = finite_pair(sub, "meta_score", "bad_pp")
        r = spearman(sx, sy)
        if r is not None:
            source_rhos[source] = r
    high_low_mmd = None
    mmd_rows = [row for row in rows if isinstance(row.get("mmd"), (int, float)) and math.isfinite(float(row["mmd"]))]
    if len(mmd_rows) >= 4:
        med = sorted(row["meta_score"] for row in mmd_rows)[len(mmd_rows) // 2]
        high = [float(row["mmd"]) for row in mmd_rows if row["meta_score"] >= med]
        low = [float(row["mmd"]) for row in mmd_rows if row["meta_score"] < med]
        if high and low:
            high_low_mmd = mean(high) - mean(low)
    return {
        "rows": len(rows),
        "conditions": len({(row["dataset"], row["condition"]) for row in rows}),
        "datasets": len({row["dataset"] for row in rows}),
        "source_families": len({row["source_family"] for row in rows}),
        "signed_rho": rho,
        "shuffle_p": permutation_p(rows) if rows else None,
        "dataset_min_signed_rho": min(dataset_rhos.values()) if dataset_rhos else None,
        "source_min_signed_rho": min(source_rhos.values()) if source_rhos else None,
        "abs_rho_mmd": None if mmd_rho is None else abs(mmd_rho),
        "high_low_mmd": high_low_mmd,
        "dataset_rhos": dataset_rhos,
        "source_rhos": source_rhos,
    }


def status_from_eval(train_eval: dict[str, Any], control_eval: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if train_eval["conditions"] < 50:
        reasons.append("conditions_below_50")
    if train_eval["datasets"] < 4:
        reasons.append("datasets_below_4")
    if train_eval["source_families"] < 3:
        reasons.append("source_families_below_3")
    if train_eval["signed_rho"] is None or train_eval["signed_rho"] < 0.25:
        reasons.append("residual_signed_rho_below_0p25")
    if train_eval["shuffle_p"] is None or train_eval["shuffle_p"] > 0.01:
        reasons.append("within_dataset_shuffle_p_gt_0p01")
    if train_eval["dataset_min_signed_rho"] is None or train_eval["dataset_min_signed_rho"] < 0.15:
        reasons.append("leave_dataset_min_signed_rho_below_0p15")
    if train_eval["source_min_signed_rho"] is None or train_eval["source_min_signed_rho"] < 0.15:
        reasons.append("leave_source_min_signed_rho_below_0p15")
    if train_eval["abs_rho_mmd"] is not None and train_eval["abs_rho_mmd"] >= 0.15:
        reasons.append("mmd_abs_rho_ge_0p15")
    if train_eval["high_low_mmd"] is not None and train_eval["high_low_mmd"] > 0.001:
        reasons.append("high_low_mmd_gt_0p001")
    if control_eval["signed_rho"] is not None and train_eval["signed_rho"] is not None and control_eval["signed_rho"] >= train_eval["signed_rho"] - 0.02:
        reasons.append("qc_or_control_signal_matches_candidate")
    return ("multisource_true_effect_residual_meta_fail_no_gpu" if reasons else "multisource_true_effect_residual_meta_pass_external_audit_next_no_gpu", reasons)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        return f"{value:+.6f}"
    return str(value)


def render_md(payload: dict[str, Any]) -> str:
    train_eval = payload["train_internal_candidate_eval"]
    diagnostic_eval = payload["diagnostic_candidate_eval"]
    control_eval = payload["train_internal_control_eval"]
    lines = [
        "# LatentFM Multi-Source True-Effect Residual Meta-Gate 2026-06-27",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M CST')}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only meta-gate over existing local artifact/gate outputs.",
        "- Train/internal eligible evidence is separated from diagnostic test-metric-only evidence.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C query, or GPU.",
        "",
        "## Train/Internal Candidate Gate",
        "",
        "| rows | conditions | datasets | sources | signed rho | shuffle p | dataset min | source min | abs MMD rho | high-low MMD |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {train_eval['rows']} | {train_eval['conditions']} | {train_eval['datasets']} | {train_eval['source_families']} | {fmt(train_eval['signed_rho'])} | {fmt(train_eval['shuffle_p'])} | {fmt(train_eval['dataset_min_signed_rho'])} | {fmt(train_eval['source_min_signed_rho'])} | {fmt(train_eval['abs_rho_mmd'])} | {fmt(train_eval['high_low_mmd'])} |",
        "",
        "## Train/Internal Control Bucket",
        "",
        f"- signed rho: `{fmt(control_eval['signed_rho'])}`",
        f"- shuffle p: `{fmt(control_eval['shuffle_p'])}`",
        "",
        "## Diagnostic Test-Metric-Only Pool",
        "",
        "| rows | conditions | datasets | sources | signed rho | shuffle p | abs MMD rho |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {diagnostic_eval['rows']} | {diagnostic_eval['conditions']} | {diagnostic_eval['datasets']} | {diagnostic_eval['source_families']} | {fmt(diagnostic_eval['signed_rho'])} | {fmt(diagnostic_eval['shuffle_p'])} | {fmt(diagnostic_eval['abs_rho_mmd'])} |",
        "",
        "## Decision",
        "",
        f"- reasons: `{payload['reasons']}`",
        "- The diagnostic pool cannot authorize GPU because it uses frozen test-metric associations.",
        "- A future GPU route would require the train/internal candidate gate to pass, then external audit and a bounded no-harm design.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- rows: `{OUT_ROWS}`",
        f"- artifact summary: `{OUT_ARTIFACTS}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    rows = load_rows()
    artifact_summaries = orient_artifacts(rows)
    train_scopes = {"train_internal", "train_internal_proxy"}
    train_candidate = aggregate_condition_scores(rows, scope_filter=train_scopes, role_prefix="candidate")
    train_control = aggregate_condition_scores(rows, scope_filter=train_scopes, role_prefix="control")
    diagnostic_candidate = aggregate_condition_scores(rows, scope_filter={"diagnostic_test_metric_only"}, role_prefix="candidate")
    train_eval = eval_bucket(train_candidate)
    control_eval = eval_bucket(train_control)
    diagnostic_eval = eval_bucket(diagnostic_candidate)
    status, reasons = status_from_eval(train_eval, control_eval)
    payload = {
        "status": status,
        "gpu_authorized": False,
        "rows_loaded": len(rows),
        "train_internal_candidate_eval": train_eval,
        "train_internal_control_eval": control_eval,
        "diagnostic_candidate_eval": diagnostic_eval,
        "reasons": reasons,
        "outputs": {
            "rows_csv": str(OUT_ROWS),
            "artifact_summary_csv": str(OUT_ARTIFACTS),
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
        },
    }
    write_csv(
        OUT_ROWS,
        train_candidate + train_control + diagnostic_candidate,
        ["source_family", "dataset", "condition", "meta_score", "bad_pp", "mmd", "n_artifacts", "role_bucket"],
    )
    write_csv(
        OUT_ARTIFACTS,
        artifact_summaries,
        [
            "evidence_scope",
            "source_family",
            "artifact",
            "artifact_role",
            "n",
            "datasets",
            "direction",
            "signed_rho_bad_pp",
            "abs_rho_mmd",
        ],
    )
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
