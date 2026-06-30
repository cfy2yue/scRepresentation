#!/usr/bin/env python3
"""Read-only latent-source evidence gate for LatentFM Track A.

This audit formalizes whether existing artifacts justify switching away from
the current xverse/top-latent 8k anchor. It is intentionally conservative:
cross-latent MMD scales are not used as primary evidence, and older capped/IID
artifacts are treated as directional unless they match xverse's condition-
uncapped split/family posthoc standard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_latent_source_evidence_gate_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_LATENT_SOURCE_EVIDENCE_GATE_20260622.md"

RUNS = [
    {
        "label": "xverse_8k_seed42_uncapped",
        "latent": "xverse",
        "evidence_level": "condition_uncapped_split_family",
        "iid": ROOT / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/iid_eval_results.json",
        "split": ROOT
        / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
        "family": ROOT
        / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/posthoc_eval_uncapped_20260621/condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
    },
    {
        "label": "xverse_8k_seed43_uncapped",
        "latent": "xverse",
        "evidence_level": "condition_uncapped_split_family",
        "iid": ROOT / "CoupledFM/output/latentfm_runs/xverse_8k_seed_replicate_20260621/xverse_comp006_endpoint5_8k_seed43_fulleval/iid_eval_results.json",
        "split": ROOT
        / "CoupledFM/output/latentfm_runs/xverse_8k_seed_replicate_20260621/xverse_comp006_endpoint5_8k_seed43_fulleval/posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
        "family": ROOT
        / "CoupledFM/output/latentfm_runs/xverse_8k_seed_replicate_20260621/xverse_comp006_endpoint5_8k_seed43_fulleval/posthoc_eval_uncapped_20260621/condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
    },
    {
        "label": "xverse_2k_uncapped",
        "latent": "xverse",
        "evidence_level": "condition_uncapped_split_family",
        "iid": ROOT / "CoupledFM/output/latentfm_runs/xverse_smoke_20260620/xverse_comp006_endpoint5_2k_smoke/iid_eval_results.json",
        "split": ROOT
        / "CoupledFM/output/latentfm_runs/xverse_smoke_20260620/xverse_comp006_endpoint5_2k_smoke/posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
        "family": ROOT
        / "CoupledFM/output/latentfm_runs/xverse_smoke_20260620/xverse_comp006_endpoint5_2k_smoke/posthoc_eval_uncapped_20260621/condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
    },
    {
        "label": "scfoundation_comp006_endpoint5_12k_fullcap",
        "latent": "scFoundation",
        "evidence_level": "iid_fullcap_directional_only",
        "iid": ROOT / "CoupledFM/output/latentfm_runs/full_scfoundation/20260617_scfoundation_comp006_delta_w5_12k/iid_eval_results.json",
    },
    {
        "label": "scldm_comp006_endpoint5_12k_fullcap",
        "latent": "scLDM",
        "evidence_level": "iid_fullcap_directional_only",
        "iid": ROOT / "CoupledFM/output/latentfm_runs/full_scldm/20260617_scldm_comp006_delta_w5_12k/iid_eval_results.json",
    },
    {
        "label": "stack_comp006_endpoint5_12k_fullcap",
        "latent": "Stack",
        "evidence_level": "iid_fullcap_directional_only",
        "iid": ROOT / "CoupledFM/output/latentfm_runs/full_stack/20260617_stack_comp006_delta_w5_12k/iid_eval_results.json",
    },
    {
        "label": "stack_comp003_endpoint12_12k_fullcap",
        "latent": "Stack",
        "evidence_level": "iid_fullcap_directional_only",
        "iid": ROOT / "CoupledFM/output/latentfm_runs/full_stack/20260617_stack_comp003_delta_w12_12k/iid_eval_results.json",
    },
]


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def group_metric(payload: dict[str, Any] | None, group: str, metric: str) -> float | None:
    if not payload:
        return None
    groups = payload.get("groups") or {}
    row = groups.get(group) or {}
    val = row.get(metric)
    return None if val is None else float(val)


def family_metric(payload: dict[str, Any] | None, group: str, metric: str) -> float | None:
    return group_metric(payload, group, metric)


def iid_metric(payload: dict[str, Any] | None, metric: str) -> float | None:
    if not payload:
        return None
    val = payload.get(metric)
    return None if val is None else float(val)


def collect_rows() -> list[dict[str, Any]]:
    rows = []
    for spec in RUNS:
        iid = load_json(spec.get("iid"))
        split = load_json(spec.get("split"))
        family = load_json(spec.get("family"))
        rows.append(
            {
                "label": spec["label"],
                "latent": spec["latent"],
                "evidence_level": spec["evidence_level"],
                "iid_path": str(spec.get("iid")) if spec.get("iid") else None,
                "split_path": str(spec.get("split")) if spec.get("split") else None,
                "family_path": str(spec.get("family")) if spec.get("family") else None,
                "iid_test_pp": iid_metric(iid, "pearson_pert"),
                "iid_test_pc": iid_metric(iid, "pearson_ctrl"),
                "iid_test_mmd": iid_metric(iid, "test_mmd_clamped") or iid_metric(iid, "test_mmd"),
                "iid_n_conds": iid_metric(iid, "n_conds"),
                "uncapped_test_pp": group_metric(split, "test", "pearson_pert"),
                "uncapped_test_single_pp": group_metric(split, "test_single", "pearson_pert"),
                "uncapped_unseen2_pp": group_metric(split, "test_multi_unseen2", "pearson_pert"),
                "uncapped_test_mmd_clamped": group_metric(split, "test", "test_mmd_clamped"),
                "uncapped_unseen2_mmd_clamped": group_metric(split, "test_multi_unseen2", "test_mmd_clamped"),
                "uncapped_family_gene_pp": family_metric(family, "family_gene", "pearson_pert"),
                "uncapped_family_drug_pp": family_metric(family, "family_drug", "pearson_pert"),
                "uncapped_structure_multi_pp": family_metric(family, "structure_multi", "pearson_pert"),
            }
        )
    return rows


def best_directional_non_xverse(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [r for r in rows if r["latent"] != "xverse" and r.get("iid_test_pp") is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda r: float(r["iid_test_pp"]))


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    xverse = next(r for r in rows if r["label"] == "xverse_8k_seed42_uncapped")
    best_other = best_directional_non_xverse(rows)
    reasons = []
    if xverse.get("uncapped_test_pp") is None or xverse.get("uncapped_family_gene_pp") is None:
        reasons.append("missing_xverse_uncapped_anchor")
    if best_other is None:
        reasons.append("missing_non_xverse_directional_evidence")
    else:
        if float(best_other.get("iid_test_pp") or -999.0) <= float(xverse.get("uncapped_test_pp") or -999.0):
            reasons.append("best_non_xverse_directional_test_pp_below_xverse_uncapped")
        reasons.append("non_xverse_artifacts_not_condition_uncapped_family_comparable")
    status = "keep_xverse_anchor_no_latent_switch_gpu"
    action = "no_top_latent_gpu_without_new_comparable_uncapped_evidence"
    return {
        "status": status,
        "action": action,
        "reasons": reasons,
        "xverse_anchor_label": xverse["label"],
        "best_directional_non_xverse_label": None if best_other is None else best_other["label"],
    }


def fmt(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):.6f}"
    except Exception:
        return str(v)


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Latent-Source Evidence Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
        "",
        "## Evidence Table",
        "",
        "| run | latent | evidence | test pp | test_single pp | family_gene pp | unseen2 pp | test MMD | note |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        test_pp = row.get("uncapped_test_pp")
        if test_pp is None:
            test_pp = row.get("iid_test_pp")
        mmd = row.get("uncapped_test_mmd_clamped")
        if mmd is None:
            mmd = row.get("iid_test_mmd")
        note = "formal Track A anchor evidence" if row["evidence_level"].startswith("condition_uncapped") else "directional only; not family/split comparable"
        lines.append(
            f"| {row['label']} | {row['latent']} | {row['evidence_level']} | "
            f"{fmt(test_pp)} | {fmt(row.get('uncapped_test_single_pp'))} | "
            f"{fmt(row.get('uncapped_family_gene_pp'))} | {fmt(row.get('uncapped_unseen2_pp'))} | "
            f"{fmt(mmd)} | {note} |"
        )
    lines += ["", "## Decision Reasons", ""]
    lines.extend([f"- `{r}`" for r in payload["decision"].get("reasons") or []])
    lines += [
        "",
        "## Gate Rule",
        "",
        "- Do not launch top5/top-latent GPU reruns unless a non-xverse source has condition-uncapped split/family evidence that beats xverse on Track A pp, not just MMD or directional IID.",
        "- Existing non-xverse artifacts are useful negative/context evidence, but not a replacement for xverse seed42/seed43 uncapped support.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    rows = collect_rows()
    payload = {"rows": rows, "decision": decide(rows)}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
