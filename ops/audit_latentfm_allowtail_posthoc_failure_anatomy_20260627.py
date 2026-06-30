#!/usr/bin/env python3
"""CPU-only anatomy of the allowlisted-tail exact-gate failure."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path("/data/cyx/1030/scLatent")
GATE_DIR = ROOT / "reports/tracka_exact_tail_candidate_gate_20260627"
OUT_JSON = ROOT / "reports/latentfm_allowtail_posthoc_failure_anatomy_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_ALLOWTAIL_POSTHOC_FAILURE_ANATOMY_20260627.md"
RUN_NAME = "xverse_allowtail_hybrid_pertresid_prior_w003_p002_replay1_2k_seed{seed}"
SEEDS = [42, 43]


def read_rows(seed: int) -> list[dict[str, str]]:
    path = GATE_DIR / f"{RUN_NAME.format(seed=seed)}_paired_rows.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return 0.0


def summarize(rows: list[dict[str, str]], group_key: str) -> list[dict[str, object]]:
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[row[group_key]].append(row)
    out = []
    for key, vals in sorted(buckets.items()):
        pp = [f(row, "delta_pearson_pert") for row in vals]
        mmd = [f(row, "delta_test_mmd_clamped") for row in vals]
        nonzero = [abs(x) > 1e-7 for x in pp]
        out.append(
            {
                group_key: key,
                "n": len(vals),
                "pp_mean": mean(pp) if pp else 0.0,
                "pp_min": min(pp) if pp else 0.0,
                "pp_max": max(pp) if pp else 0.0,
                "mmd_mean": mean(mmd) if mmd else 0.0,
                "nonzero_pp_frac_gt_1e_7": sum(nonzero) / len(nonzero) if nonzero else 0.0,
            }
        )
    return out


def main() -> int:
    seed_payloads = {}
    all_group_rows = []
    for seed in SEEDS:
        rows = read_rows(seed)
        group_summary = summarize(rows, "group")
        dataset_summary = summarize([r for r in rows if r["group"] == "recurrent_cross_background_hard_tail"], "dataset")
        seed_payloads[str(seed)] = {
            "n_rows": len(rows),
            "group_summary": group_summary,
            "recurrent_cross_dataset_summary": dataset_summary,
            "max_abs_pp_delta": max(abs(f(row, "delta_pearson_pert")) for row in rows),
            "max_abs_mmd_delta": max(abs(f(row, "delta_test_mmd_clamped")) for row in rows),
            "rows_with_abs_pp_delta_ge_0p001": sum(abs(f(row, "delta_pearson_pert")) >= 0.001 for row in rows),
            "rows_with_abs_pp_delta_ge_0p01": sum(abs(f(row, "delta_pearson_pert")) >= 0.01 for row in rows),
        }
        all_group_rows.extend((seed, row) for row in group_summary)

    status = "allowlisted_tail_effect_near_zero_close_branch"
    payload = {
        "status": status,
        "boundary": {
            "cpu_report_only": True,
            "train_infer_gpu": False,
            "canonical_multi_selection_weight": 0,
            "trackc_query_used": False,
        },
        "interpretation": (
            "Both seeds completed posthoc, but candidate-vs-anchor deltas are near numerical zero. "
            "The branch failed by absence of material recurrent-tail gain, not by a useful localized signal."
        ),
        "seeds": seed_payloads,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# LatentFM Allowlisted-Tail Posthoc Failure Anatomy 2026-06-27",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only over exact-tail paired rows from the completed seed42/43 posthoc rerun.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C query, or GPU.",
        "",
        "## Seed Summary",
        "",
        "| seed | paired rows | max abs pp delta | max abs MMD delta | rows abs pp >=0.001 | rows abs pp >=0.01 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        row = seed_payloads[str(seed)]
        lines.append(
            f"| {seed} | {row['n_rows']} | {row['max_abs_pp_delta']:+.6f} | "
            f"{row['max_abs_mmd_delta']:+.6f} | {row['rows_with_abs_pp_delta_ge_0p001']} | "
            f"{row['rows_with_abs_pp_delta_ge_0p01']} |"
        )
    lines.extend(["", "## Group Means", "", "| seed | group | n | pp mean | pp min | pp max | MMD mean | nonzero pp frac |", "|---:|---|---:|---:|---:|---:|---:|---:|"])
    for seed in SEEDS:
        for row in seed_payloads[str(seed)]["group_summary"]:
            lines.append(
                f"| {seed} | `{row['group']}` | {row['n']} | {row['pp_mean']:+.6f} | "
                f"{row['pp_min']:+.6f} | {row['pp_max']:+.6f} | {row['mmd_mean']:+.6f} | "
                f"{row['nonzero_pp_frac_gt_1e_7']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The allowlisted-tail branch should be closed under the predeclared gate. The result is near-inert rather than a localized partial success, so a cluster-factored GPU branch is not authorized from this evidence alone. A strength/step sweep is also not authorized because seed43 has exact/canonical pp no-harm failures and neither seed shows material recurrent cross-tail gain.",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
