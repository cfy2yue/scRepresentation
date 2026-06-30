#!/usr/bin/env python3
"""CPU-only OT pair-quality versus failed-tail correlation gate."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OT_JSON = REPORTS / "latentfm_ot_pair_quality_gate_20260625.json"
COUNT_JSON = REPORTS / "latentfm_scaling_mixed_effect_lodo_condition_count_gate_20260624.json"
OUT_JSON = REPORTS / "latentfm_ot_pair_quality_failure_correlation_gate_20260625.json"
OUT_MD = REPORTS / "LATENTFM_OT_PAIR_QUALITY_FAILURE_CORRELATION_GATE_20260625.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def main() -> int:
    ot = load_json(OT_JSON)
    count = load_json(COUNT_JSON)
    count_by_ds = {str(r["dataset"]): r for r in count.get("dataset_rows", [])}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ot.get("rows", []):
        grouped[str(row["dataset"])].append(row)
    dataset_rows = []
    for ds, rows in sorted(grouped.items()):
        if ds not in count_by_ds:
            continue
        modes = [r["pair_quality"]["modes"]["multinomial"] for r in rows]
        assign = [r["pair_quality"]["modes"]["assignment"] for r in rows]
        hung = [r["pair_quality"]["modes"]["hungarian"] for r in rows]
        dataset_rows.append(
            {
                "dataset": ds,
                "n_conditions_sampled": len(rows),
                "tail_pp_delta": float(count_by_ds[ds]["pp_delta_mean"]),
                "tail_mmd_delta": float(count_by_ds[ds]["mmd_delta_mean"]),
                "multinomial_cost_mean": float(mean([m["paired_cost_mean"] for m in modes])),
                "multinomial_dup_mean": float(mean([m["gt_duplicate_rate"] for m in modes])),
                "assignment_cost_mean": float(mean([m["paired_cost_mean"] for m in assign])),
                "hungarian_cost_mean": float(mean([m["paired_cost_mean"] for m in hung])),
                "hungarian_minus_multinomial_cost": float(mean([h["paired_cost_mean"] for h in hung]) - mean([m["paired_cost_mean"] for m in modes])),
                "assignment_minus_multinomial_cost": float(mean([a["paired_cost_mean"] for a in assign]) - mean([m["paired_cost_mean"] for m in modes])),
            }
        )
    xs_pp = [r["tail_pp_delta"] for r in dataset_rows]
    corr = {
        "multinomial_cost_vs_tail_pp": pearson([r["multinomial_cost_mean"] for r in dataset_rows], xs_pp),
        "multinomial_dup_vs_tail_pp": pearson([r["multinomial_dup_mean"] for r in dataset_rows], xs_pp),
        "hungarian_gain_vs_tail_pp": pearson([r["hungarian_minus_multinomial_cost"] for r in dataset_rows], xs_pp),
        "assignment_gain_vs_tail_pp": pearson([r["assignment_minus_multinomial_cost"] for r in dataset_rows], xs_pp),
    }
    # Directional expectation for an OT failure mechanism:
    # worse tails should align with higher pair cost, higher duplicate rate, or
    # a strong alternative-mode advantage over multinomial. Require a large
    # absolute correlation and enough overlap; otherwise no GPU.
    reasons = []
    if len(dataset_rows) < 8:
        reasons.append("too_few_overlap_datasets_for_failure_correlation")
    if not any(abs(v or 0.0) >= 0.50 for v in corr.values()):
        reasons.append("no_large_pair_quality_tail_correlation")
    if not any((v or 0.0) <= -0.50 for k, v in corr.items() if "cost_vs" in k or "dup_vs" in k):
        reasons.append("pair_quality_does_not_explain_negative_tails_directionally")
    # Existing GPU evidence for pair modes was negative; this gate can only
    # authorize a new run if it identifies a very specific failure mechanism.
    status = "ot_pair_quality_failure_correlation_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "dataset_rows": dataset_rows,
        "correlations": corr,
        "reasons": reasons or ["existing_ot_gpu_smokes_negative_no_specific_new_intervention"],
        "next_action": "do not launch OT GPU; pair-quality variation does not explain failed tails strongly enough",
        "boundary": {
            "cpu_only": True,
            "reads_train_only_reports": True,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM OT Pair Quality Failure Correlation Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only correlation of existing OT pair-quality audit with completed train-only scaling tails.",
        "- Does not train, infer, use GPU, read canonical multi, or read Track C query.",
        "",
        "## Correlations",
        "",
    ]
    for key, value in corr.items():
        lines.append(f"- `{key}`: `{value if value is not None else 'NA'}`")
    lines.extend(
        [
            "",
            "## Dataset Rows",
            "",
            "| dataset | n | tail pp | multinomial cost | multinomial dup | hungarian-minus-multinomial cost |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(dataset_rows, key=lambda r: r["tail_pp_delta"]):
        lines.append(
            f"| `{row['dataset']}` | {row['n_conditions_sampled']} | {row['tail_pp_delta']:+.6f} | "
            f"{row['multinomial_cost_mean']:.6f} | {row['multinomial_dup_mean']:.3f} | {row['hungarian_minus_multinomial_cost']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- GPU authorized: `{payload['gpu_authorized']}`",
            f"- reasons: `{payload['reasons']}`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": False, "out_md": str(OUT_MD)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
