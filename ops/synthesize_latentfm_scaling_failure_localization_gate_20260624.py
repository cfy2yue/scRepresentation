#!/usr/bin/env python3
"""CPU-only failure localization for scaling internal-to-canonical transfer."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


ROOT = Path("/data/cyx/1030/scLatent")
IN_JSON = ROOT / "reports/latentfm_scaling_noharm_transfer_calibration_gate_20260624.json"
METAINFO_JSON = ROOT / "reports/latentfm_condition_level_metainfo_scaling_audit_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_scaling_failure_localization_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_SCALING_FAILURE_LOCALIZATION_GATE_20260624.md"


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (vx * vy) ** 0.5


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return corr(rank(xs), rank(ys))


def main() -> int:
    transfer = json.loads(IN_JSON.read_text(encoding="utf-8"))
    meta = json.loads(METAINFO_JSON.read_text(encoding="utf-8")) if METAINFO_JSON.exists() else {}
    candidates = list(transfer.get("candidates", []))
    internal_pass_like = [c for c in candidates if c.get("internal_pass_like")]
    canonical_pass = [c for c in candidates if c.get("canonical_pass")]
    pp_hard = [c for c in candidates if float(c.get("canonical_max_pp_p_harm") or 0.0) >= 0.65]
    any_hard = [c for c in candidates if float(c.get("canonical_max_any_p_harm") or 0.0) >= 0.65]

    by_family: dict[str, list[dict]] = defaultdict(list)
    reason_counts: Counter[str] = Counter()
    for c in candidates:
        by_family[str(c.get("family"))].append(c)
        reason_counts.update(map(str, c.get("canonical_gate_reasons", [])))

    xs = [float(c.get("internal_score") or 0.0) for c in candidates]
    ys = [float(c.get("canonical_max_pp_p_harm") or 0.0) for c in candidates]
    fam_rows = []
    for fam, rows in sorted(by_family.items()):
        fam_rows.append(
            {
                "family": fam,
                "n": len(rows),
                "n_canonical_pass": sum(1 for r in rows if r.get("canonical_pass")),
                "median_internal_score": median(float(r.get("internal_score") or 0.0) for r in rows),
                "median_canonical_max_pp_p_harm": median(float(r.get("canonical_max_pp_p_harm") or 0.0) for r in rows),
                "max_canonical_max_pp_p_harm": max(float(r.get("canonical_max_pp_p_harm") or 0.0) for r in rows),
            }
        )

    status = "scaling_failure_localization_fail_no_gpu"
    reasons = []
    if not canonical_pass:
        reasons.append("no_canonical_noharm_positive_examples")
    if len(pp_hard) >= max(1, len(candidates) // 2):
        reasons.append("canonical_pp_hard_harm_common")
    if len(internal_pass_like) == len(candidates) and not canonical_pass:
        reasons.append("all_internal_pass_like_candidates_failed_canonical")
    if (spearman(xs, ys) or 0.0) < 0.0:
        reasons.append("higher_internal_score_does_not_lower_harm_risk")

    out = {
        "status": status,
        "boundary": {
            "cpu_only": True,
            "reads_completed_reports_only": True,
            "reads_canonical_metrics_as_frozen_veto_context": True,
            "canonical_multi_selection": False,
            "trackc_query_read": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "summary": {
            "n_candidates": len(candidates),
            "n_internal_pass_like": len(internal_pass_like),
            "n_canonical_pass": len(canonical_pass),
            "n_canonical_pp_hard_harm": len(pp_hard),
            "n_canonical_any_hard_harm": len(any_hard),
            "spearman_internal_score_vs_pp_harm": spearman(xs, ys),
            "pearson_internal_score_vs_pp_harm": corr(xs, ys),
            "metainfo_axis_dataset_counts": meta.get("axis_dataset_counts"),
        },
        "family_rows": fam_rows,
        "canonical_reason_counts": dict(reason_counts.most_common()),
        "reasons": reasons,
        "gpu_authorized": False,
        "next_action": "keep scaling as diagnostic; build LODO/mixed-effect evidence matrix only if using train-only outcomes and negative controls",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Scaling Failure Localization Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of completed scaling transfer and metainfo reports.",
        "- Frozen canonical single/family metrics are used only as retrospective no-harm veto context.",
        "- Does not read canonical multi, held-out Track C query, active logs, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- candidates: `{len(candidates)}`",
        f"- internal-pass-like: `{len(internal_pass_like)}`",
        f"- canonical no-harm pass: `{len(canonical_pass)}`",
        f"- canonical pp hard-harm rows: `{len(pp_hard)}`",
        f"- Spearman internal-score vs canonical pp harm: `{out['summary']['spearman_internal_score_vs_pp_harm']}`",
        "",
        "## Family Rows",
        "",
        "| family | n | canonical pass | median internal score | median pp p_harm | max pp p_harm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in fam_rows:
        lines.append(
            f"| `{r['family']}` | {r['n']} | {r['n_canonical_pass']} | "
            f"{r['median_internal_score']:+.6f} | {r['median_canonical_max_pp_p_harm']:+.6f} | "
            f"{r['max_canonical_max_pp_p_harm']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            "- GPU authorized: `False`",
            "- next action: keep scaling diagnostic until a train-only LODO/negative-control matrix identifies a no-harm-safe surrogate.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
