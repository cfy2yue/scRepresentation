#!/usr/bin/env python3
"""Train-only recurrent gene-harm sentinel feasibility gate."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT_JSON = ROOT / "reports/latentfm_recurrent_gene_harm_sentinel_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_RECURRENT_GENE_HARM_SENTINEL_GATE_20260625.md"

RUN_GLOBS = [
    "runs/latentfm_true_cell_count_nested_smokes_20260624/xverse_truecell_nested_gene_only_fixed256_budget64_128_256_budget*_seed*_3000/posthoc_eval_internal",
    "runs/latentfm_true_cell_count_budget128_tail_stability_6k_20260625/xverse_truecell_nested_budget128_tailstable_seed*_6000/posthoc_eval_internal",
    "runs/latentfm_true_cell_count_budget64_tail_stability_6k_20260625/xverse_truecell_nested_budget64_tailstable_seed*_6000/posthoc_eval_internal",
    "runs/latentfm_true_cell_count_budget128_anchor_replay005_6k_20260625/xverse_truecell_nested_budget128_ar005_seed*_6000/posthoc_eval_internal",
]

CANONICAL_ANCHOR = ROOT / "runs/latentfm_xverse_scaling_canonical_noharm_20260624/xverse_scaling_cap120_all_3k_seed42/posthoc_eval_canonical/condition_family_eval_anchor_ode20_canonical.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def genes(condition: str) -> tuple[str, ...]:
    parts = []
    for token in condition.replace(",", "+").split("+"):
        token = token.strip()
        if token:
            parts.append(token)
    return tuple(parts) if parts else (condition,)


def rows_by_key(blob: dict[str, Any], group: str) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in blob.get("groups", {}).get(group, {}).get("condition_metrics", []):
        ds = str(row.get("dataset") or "")
        cond = str(row.get("condition") or "")
        if ds and cond:
            out[(ds, cond)] = row
    return out


def auc(labels: list[int], scores: list[float]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    total = len(pos) * len(neg)
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1
            elif ps == ns:
                wins += 0.5
    return wins / total


def main() -> None:
    records: list[dict[str, Any]] = []
    for pattern in RUN_GLOBS:
        for posthoc_dir in sorted(ROOT.glob(pattern)):
            anchor_p = posthoc_dir / "condition_family_eval_anchor_internal_ode20.json"
            cand_p = posthoc_dir / "condition_family_eval_candidate_internal_ode20.json"
            if not anchor_p.exists() or not cand_p.exists():
                continue
            anchor = load(anchor_p)
            cand = load(cand_p)
            run = posthoc_dir.parent.name
            group = "family_gene"
            arows = rows_by_key(anchor, group)
            crows = rows_by_key(cand, group)
            for key in sorted(set(arows) & set(crows)):
                ds, cond = key
                pp_a = finite(arows[key].get("pearson_pert"))
                pp_c = finite(crows[key].get("pearson_pert"))
                mmd_a = finite(arows[key].get("test_mmd_clamped"))
                mmd_c = finite(crows[key].get("test_mmd_clamped"))
                if None in (pp_a, pp_c, mmd_a, mmd_c):
                    continue
                pp_delta = float(pp_c) - float(pp_a)
                mmd_delta = float(mmd_c) - float(mmd_a)
                label = int(pp_delta < -0.05 or mmd_delta > 0.010)
                records.append(
                    {
                        "run": run,
                        "dataset": ds,
                        "condition": cond,
                        "genes": genes(cond),
                        "pp_delta": pp_delta,
                        "mmd_delta": mmd_delta,
                        "harm": label,
                    }
                )

    labels: list[int] = []
    scores: list[float] = []
    covered = 0
    retained_pp: list[float] = []
    retained_mmd: list[float] = []
    blocked_pp: list[float] = []
    blocked_mmd: list[float] = []

    for row in records:
        gene_scores = []
        for gene in row["genes"]:
            pool = [
                other["harm"]
                for other in records
                if other is not row
                and gene in other["genes"]
                and other["dataset"] != row["dataset"]
                and other["run"] != row["run"]
            ]
            if pool:
                gene_scores.append(mean(pool))
        score = max(gene_scores) if gene_scores else 0.0
        covered += int(bool(gene_scores))
        labels.append(int(row["harm"]))
        scores.append(score)
        if score >= 0.20:
            blocked_pp.append(float(row["pp_delta"]))
            blocked_mmd.append(float(row["mmd_delta"]))
        else:
            retained_pp.append(float(row["pp_delta"]))
            retained_mmd.append(float(row["mmd_delta"]))

    observed_auc = auc(labels, scores)
    rng = random.Random(20260625)
    perm_aucs: list[float] = []
    if observed_auc is not None:
        for _ in range(500):
            shuffled = labels[:]
            rng.shuffle(shuffled)
            x = auc(shuffled, scores)
            if x is not None:
                perm_aucs.append(x)
    p_value = None
    if observed_auc is not None and perm_aucs:
        p_value = (1 + sum(x >= observed_auc for x in perm_aucs)) / (1 + len(perm_aucs))

    canonical = load(CANONICAL_ANCHOR)
    high_genes = {
        gene
        for row, score in zip(records, scores)
        if score >= 0.20
        for gene in row["genes"]
    }
    footprint: dict[str, int] = {}
    for group in ("test_single", "family_gene"):
        rows = canonical.get("groups", {}).get(group, {}).get("selected_conditions", [])
        count = 0
        for row in rows:
            cond = str(row.get("condition") or "")
            if any(gene in high_genes for gene in genes(cond)):
                count += 1
        footprint[group] = count

    n = len(records)
    positives = sum(labels)
    coverage = covered / n if n else 0.0
    high_score_rows = sum(1 for x in scores if x >= 0.20)
    high_score_frac = high_score_rows / n if n else 0.0
    retained_mean_pp = mean(retained_pp) if retained_pp else None
    retained_max_mmd = max(retained_mmd) if retained_mmd else None
    blocked_harm_rate = mean([row["harm"] for row, score in zip(records, scores) if score >= 0.20]) if high_score_rows else 0.0

    pass_gate = (
        n >= 1000
        and positives >= 50
        and coverage >= 0.25
        and observed_auc is not None
        and (observed_auc >= 0.70 or (p_value is not None and p_value <= 0.05))
        and high_score_frac <= 0.35
        and retained_mean_pp is not None
        and retained_mean_pp >= 0.015
        and retained_max_mmd is not None
        and retained_max_mmd <= 0.010
        and footprint.get("test_single", 0) >= 25
        and footprint.get("family_gene", 0) >= 25
    )

    reasons: list[str] = []
    if n < 1000:
        reasons.append("too_few_trainonly_rows")
    if positives < 50:
        reasons.append("too_few_harm_positive_rows")
    if coverage < 0.25:
        reasons.append("low_cross_dataset_gene_recurrence_coverage")
    if observed_auc is None or not (observed_auc >= 0.70 or (p_value is not None and p_value <= 0.05)):
        reasons.append("sentinel_not_predictive_enough_vs_shuffle")
    if high_score_frac > 0.35:
        reasons.append("sentinel_blocks_too_many_rows")
    if retained_mean_pp is None or retained_mean_pp < 0.015:
        reasons.append("retained_policy_mean_pp_too_low")
    if retained_max_mmd is None or retained_max_mmd > 0.010:
        reasons.append("retained_policy_mmd_tail_too_high")
    if footprint.get("test_single", 0) < 25 or footprint.get("family_gene", 0) < 25:
        reasons.append("canonical_metadata_footprint_too_small")

    status = "recurrent_gene_harm_sentinel_pass_cpu_review_next_no_gpu" if pass_gate else "recurrent_gene_harm_sentinel_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "train_only_internal_labels": True,
            "canonical_performance_used": False,
            "canonical_metadata_footprint_only": True,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
        },
        "summary": {
            "rows": n,
            "harm_positive_rows": positives,
            "cross_dataset_gene_coverage": coverage,
            "auc": observed_auc,
            "shuffle_p_value": p_value,
            "high_score_rows": high_score_rows,
            "high_score_fraction": high_score_frac,
            "blocked_harm_rate": blocked_harm_rate,
            "retained_mean_pp_delta": retained_mean_pp,
            "retained_max_mmd_delta": retained_max_mmd,
            "canonical_metadata_footprint": footprint,
        },
        "reasons": reasons,
        "next_action": (
            "external review and launcher/code design gate before GPU"
            if pass_gate
            else "do not launch recurrent gene-harm sentinel GPU; use as negative evidence or redesign with richer priors"
        ),
        "inputs": {
            "run_globs": RUN_GLOBS,
            "canonical_metadata_source": str(CANONICAL_ANCHOR),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# LatentFM Recurrent Gene-Harm Sentinel Gate",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only train-only/internal row-delta gate.",
        "- Canonical data are used only for metadata footprint, not performance.",
        "- No canonical multi, Track C query, training, inference, or GPU.",
        "",
        "## Summary",
        "",
        f"- rows: `{n}`",
        f"- harm positive rows: `{positives}`",
        f"- cross-dataset gene recurrence coverage: `{coverage:.3f}`",
        f"- AUROC: `{observed_auc}`",
        f"- shuffle p-value: `{p_value}`",
        f"- high-score rows/fraction: `{high_score_rows}` / `{high_score_frac:.3f}`",
        f"- blocked-row harm rate: `{blocked_harm_rate:.3f}`",
        f"- retained mean pp delta: `{retained_mean_pp}`",
        f"- retained max MMD delta: `{retained_max_mmd}`",
        f"- canonical metadata footprint: `{footprint}`",
        "",
        "## Decision",
        "",
        f"- reasons: `{reasons}`",
        f"- next action: `{payload['next_action']}`",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
