#!/usr/bin/env python3
"""Summarize pairwise-condition LatentFM smoke posthoc results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FOCUS_DATASETS = ("Wessels", "NormanWeissman2019_filtered", "GasperiniShendure2019_lowMOI")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def group(payload: dict[str, Any], name: str) -> dict[str, Any]:
    obj = payload.get("groups", {}).get(name, {})
    return obj if isinstance(obj, dict) else {}


def first_group(payload: dict[str, Any], names: tuple[str, ...]) -> tuple[str, dict[str, Any]]:
    for name in names:
        obj = group(payload, name)
        if obj and not obj.get("skipped", False):
            return name, obj
    return names[0], {}


def metric(g: dict[str, Any], key: str) -> float | None:
    aliases = {
        "pp": "pearson_pert",
        "pc": "pearson_ctrl",
        "dp": "direct_pearson",
        "mmd": "test_mmd",
        "mmd_clamped": "test_mmd_clamped",
        "mmd_biased": "test_mmd_biased",
    }
    return fnum(g.get(aliases.get(key, key)))


def mmd_gate_value(g: dict[str, Any]) -> tuple[str, float | None]:
    for key in ("test_mmd_clamped", "test_mmd_biased", "test_mmd"):
        val = fnum(g.get(key))
        if val is not None:
            return key, val
    return "missing", None


def selected_fingerprint(g: dict[str, Any]) -> tuple[str, ...]:
    rows = g.get("selected_conditions") or []
    out: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(f"{row.get('dataset', '')}\t{row.get('condition', '')}")
    return tuple(sorted(out))


def delta(run: float | None, base: float | None) -> float | None:
    if run is None or base is None:
        return None
    return run - base


def ratio(run: float | None, base: float | None) -> float | None:
    if run is None or base is None or base == 0:
        return None
    return run / base


def per_ds_pp(g: dict[str, Any], dataset: str) -> float | None:
    obj = g.get("per_ds_p_pert") or {}
    return fnum(obj.get(dataset)) if isinstance(obj, dict) else None


def summarize(manifest: dict[str, Any]) -> dict[str, Any]:
    base_split = load_json(Path(manifest["baseline_split_json"]))
    base_family = load_json(Path(manifest["baseline_family_json"]))
    run_split = load_json(Path(manifest["run_split_json"]))
    run_family = load_json(Path(manifest["run_family_json"]))

    b_test = group(base_split, "test")
    r_test = group(run_split, "test")
    b_u2 = group(base_split, "test_multi_unseen2")
    r_u2 = group(run_split, "test_multi_unseen2")
    b_gene = group(base_family, "family_gene")
    r_gene = group(run_family, "family_gene")
    single_group_name_b, b_single = first_group(base_family, ("structure_single", "family_gene_single", "test_single"))
    single_group_name_r, r_single = first_group(run_family, ("structure_single", "family_gene_single", "test_single"))
    drug_group_name_b, b_drug = first_group(base_family, ("type_drug", "family_drug", "test_drug"))
    drug_group_name_r, r_drug = first_group(run_family, ("type_drug", "family_drug", "test_drug"))

    mmd_key_b, b_mmd = mmd_gate_value(b_test)
    mmd_key_r, r_mmd = mmd_gate_value(r_test)
    row: dict[str, Any] = {
        "run_name": manifest["run_name"],
        "anchor_checkpoint": manifest["anchor_checkpoint"],
        "candidate_checkpoint": manifest["candidate_checkpoint"],
        "split_file": manifest["split_file"],
        "pairwise_mode": manifest.get("pairwise_mode", "hadamard_mean"),
        "test_pp_base": metric(b_test, "pp"),
        "test_pp_run": metric(r_test, "pp"),
        "unseen2_pp_base": metric(b_u2, "pp"),
        "unseen2_pp_run": metric(r_u2, "pp"),
        "family_gene_pp_base": metric(b_gene, "pp"),
        "family_gene_pp_run": metric(r_gene, "pp"),
        "single_behavior_group": single_group_name_b if single_group_name_b == single_group_name_r else f"{single_group_name_b}/{single_group_name_r}",
        "single_pp_base": metric(b_single, "pp"),
        "single_pp_run": metric(r_single, "pp"),
        "single_direct_base": metric(b_single, "dp"),
        "single_direct_run": metric(r_single, "dp"),
        "drug_behavior_group": drug_group_name_b if drug_group_name_b == drug_group_name_r else f"{drug_group_name_b}/{drug_group_name_r}",
        "drug_pp_base": metric(b_drug, "pp"),
        "drug_pp_run": metric(r_drug, "pp"),
        "drug_direct_base": metric(b_drug, "dp"),
        "drug_direct_run": metric(r_drug, "dp"),
        "test_mmd_base": b_mmd,
        "test_mmd_run": r_mmd,
        "mmd_gate_metric": mmd_key_b if mmd_key_b == mmd_key_r else f"{mmd_key_b}/{mmd_key_r}",
        "selected_match": {
            "test": selected_fingerprint(b_test) == selected_fingerprint(r_test),
            "test_multi_unseen2": selected_fingerprint(b_u2) == selected_fingerprint(r_u2),
            "family_gene": selected_fingerprint(b_gene) == selected_fingerprint(r_gene),
        },
    }
    row["selected_match_all"] = all(row["selected_match"].values())
    for key in ("test_pp", "unseen2_pp", "family_gene_pp", "single_pp", "single_direct", "drug_pp", "drug_direct"):
        row[f"{key}_delta"] = delta(row.get(f"{key}_run"), row.get(f"{key}_base"))
    row["test_mmd_ratio"] = ratio(r_mmd, b_mmd)
    for ds in FOCUS_DATASETS:
        prefix = ds.replace("NormanWeissman2019_filtered", "Norman").replace(
            "GasperiniShendure2019_lowMOI", "Gasperini"
        )
        b_pp = per_ds_pp(b_u2, ds)
        r_pp = per_ds_pp(r_u2, ds)
        row[f"{prefix}_u2_pp_base"] = b_pp
        row[f"{prefix}_u2_pp_run"] = r_pp
        row[f"{prefix}_u2_pp_delta"] = delta(r_pp, b_pp)

    checks = {
        "selected_match": bool(row["selected_match_all"]),
        "unseen2_pp_rescue": (
            row.get("unseen2_pp_delta") is not None
            and (
                row["unseen2_pp_delta"] >= 0.03
                or (row.get("unseen2_pp_run") is not None and row["unseen2_pp_run"] > 0)
            )
        ),
        "wessels_not_harmed": (
            row.get("Wessels_u2_pp_delta") is None or row["Wessels_u2_pp_delta"] >= -0.01
        ),
        "overall_pp_not_harmed": (
            row.get("test_pp_delta") is not None and row["test_pp_delta"] >= -0.005
        ),
        "family_gene_not_harmed": (
            row.get("family_gene_pp_delta") is not None and row["family_gene_pp_delta"] >= -0.01
        ),
        "mmd_ratio_ok": (
            row.get("test_mmd_ratio") is not None and row["test_mmd_ratio"] <= 1.15
        ),
        "single_gene_behavior_not_harmed": (
            row.get("single_pp_delta") is None or row["single_pp_delta"] >= -0.01
        )
        and (row.get("single_direct_delta") is None or row["single_direct_delta"] >= -0.005),
        "drug_behavior_not_harmed": (
            row.get("drug_pp_delta") is None or row["drug_pp_delta"] >= -0.01
        )
        and (row.get("drug_direct_delta") is None or row["drug_direct_delta"] >= -0.005),
    }
    row["checks"] = checks
    row["triage_status"] = "pairwise_condition_candidate" if all(checks.values()) else "diagnostic_or_fail"
    return row


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_md(row: dict[str, Any], manifest_path: Path) -> str:
    lines = [
        "# LatentFM Pairwise Condition Smoke Summary",
        "",
        f"Manifest: `{manifest_path}`",
        "",
        "Baseline is the unchanged anchor checkpoint evaluated on the same canonical split.",
        "All gates use raw-space split/family metrics with matched selected conditions.",
        "",
        "## Gate",
        "",
        "| status | selected match | unseen2 pp delta | unseen2 pp run | Wessels u2 delta | test pp delta | family_gene pp delta | MMD ratio | single pp delta | drug pp delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| `{status}` | {sel} | {u2d} | {u2r} | {wu2d} | {tpp} | {gpp} | {mmd} | {spp} | {dpp} |".format(
            status=row["triage_status"],
            sel=fmt(row.get("selected_match_all")),
            u2d=fmt(row.get("unseen2_pp_delta")),
            u2r=fmt(row.get("unseen2_pp_run")),
            wu2d=fmt(row.get("Wessels_u2_pp_delta")),
            tpp=fmt(row.get("test_pp_delta")),
            gpp=fmt(row.get("family_gene_pp_delta")),
            mmd=fmt(row.get("test_mmd_ratio")),
            spp=fmt(row.get("single_pp_delta")),
            dpp=fmt(row.get("drug_pp_delta")),
        ),
        "",
        "## Single / Drug Behavior Check",
        "",
        "| stratum | group | base pp | run pp | delta pp | base direct | run direct | delta direct |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        f"| single-gene | `{row.get('single_behavior_group')}` | {fmt(row.get('single_pp_base'))} | {fmt(row.get('single_pp_run'))} | {fmt(row.get('single_pp_delta'))} | {fmt(row.get('single_direct_base'))} | {fmt(row.get('single_direct_run'))} | {fmt(row.get('single_direct_delta'))} |",
        f"| drug/chem-only | `{row.get('drug_behavior_group')}` | {fmt(row.get('drug_pp_base'))} | {fmt(row.get('drug_pp_run'))} | {fmt(row.get('drug_pp_delta'))} | {fmt(row.get('drug_direct_base'))} | {fmt(row.get('drug_direct_run'))} | {fmt(row.get('drug_direct_delta'))} |",
        "",
        "## Focus Dataset Unseen2",
        "",
        "| dataset | base pp | run pp | delta pp |",
        "|---|---:|---:|---:|",
        f"| Wessels | {fmt(row.get('Wessels_u2_pp_base'))} | {fmt(row.get('Wessels_u2_pp_run'))} | {fmt(row.get('Wessels_u2_pp_delta'))} |",
        f"| Norman | {fmt(row.get('Norman_u2_pp_base'))} | {fmt(row.get('Norman_u2_pp_run'))} | {fmt(row.get('Norman_u2_pp_delta'))} |",
        f"| Gasperini | {fmt(row.get('Gasperini_u2_pp_base'))} | {fmt(row.get('Gasperini_u2_pp_run'))} | {fmt(row.get('Gasperini_u2_pp_delta'))} |",
        "",
        "## Interpretation",
        "",
        "- Passing this capped 4k smoke only permits uncapped posthoc and paired bootstrap.",
        "- In zero-multi-train splits, pairwise features are a condition-capacity diagnostic, not proof of supervised interaction learning.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/runs/latentfm_pairwise_condition_20260621/posthoc_manifest.json"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/latentfm_pairwise_condition_smoke_summary_20260621.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/LATENTFM_PAIRWISE_CONDITION_SMOKE_SUMMARY_20260621.md"),
    )
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    row = summarize(manifest)
    payload = {"manifest": str(args.manifest), "row": row}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(row, args.manifest), encoding="utf-8")
    print(json.dumps({"out_md": str(args.out_md), "status": row["triage_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
