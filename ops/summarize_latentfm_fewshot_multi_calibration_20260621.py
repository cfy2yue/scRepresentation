#!/usr/bin/env python3
"""Summarize few-shot multi-calibration posthoc results.

Each candidate is compared against the unchanged anchor checkpoint evaluated on
the same custom split.  This avoids mixing few-shot effects with changed
held-out condition selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FOCUS_DATASETS = (
    "Wessels",
    "NormanWeissman2019_filtered",
    "GasperiniShendure2019_lowMOI",
)


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


def per_ds_pp(g: dict[str, Any], dataset: str) -> float | None:
    obj = g.get("per_ds_p_pert") or {}
    return fnum(obj.get(dataset)) if isinstance(obj, dict) else None


def per_ds_mmd(g: dict[str, Any], dataset: str) -> float | None:
    for key in ("per_ds_mmd_clamped", "per_ds_mmd_biased", "per_ds_mmd"):
        obj = g.get(key) or {}
        if isinstance(obj, dict) and dataset in obj:
            return fnum(obj.get(dataset))
    return None


def delta(run: float | None, base: float | None) -> float | None:
    if run is None or base is None:
        return None
    return run - base


def ratio(run: float | None, base: float | None) -> float | None:
    if run is None or base is None or base == 0:
        return None
    return run / base


def summarize_one(spec: dict[str, Any]) -> dict[str, Any]:
    arm = spec["arm"]
    run_name = spec["run_name"]
    base_split = load_json(Path(spec["baseline_split_json"]))
    base_family = load_json(Path(spec["baseline_family_json"]))
    run_split = load_json(Path(spec["run_split_json"]))
    run_family = load_json(Path(spec["run_family_json"]))

    b_test = group(base_split, "test")
    r_test = group(run_split, "test")
    b_u2 = group(base_split, "test_multi_unseen2")
    r_u2 = group(run_split, "test_multi_unseen2")
    b_gene = group(base_family, "family_gene")
    r_gene = group(run_family, "family_gene")

    mmd_key_b, b_mmd = mmd_gate_value(b_test)
    mmd_key_r, r_mmd = mmd_gate_value(r_test)
    common_mmd_key = mmd_key_b if mmd_key_b == mmd_key_r else f"{mmd_key_b}/{mmd_key_r}"

    selected_match = {
        "test": selected_fingerprint(b_test) == selected_fingerprint(r_test),
        "test_multi_unseen2": selected_fingerprint(b_u2) == selected_fingerprint(r_u2),
        "family_gene": selected_fingerprint(b_gene) == selected_fingerprint(r_gene),
    }

    row: dict[str, Any] = {
        "arm": arm,
        "run_name": run_name,
        "split_file": spec["split_file"],
        "moved_multi": spec.get("moved_multi"),
        "test_n": r_test.get("n_conds"),
        "unseen2_n": r_u2.get("n_conds"),
        "family_gene_n": r_gene.get("n_conds"),
        "test_pp_base": metric(b_test, "pp"),
        "test_pp_run": metric(r_test, "pp"),
        "unseen2_pp_base": metric(b_u2, "pp"),
        "unseen2_pp_run": metric(r_u2, "pp"),
        "family_gene_pp_base": metric(b_gene, "pp"),
        "family_gene_pp_run": metric(r_gene, "pp"),
        "test_mmd_base": b_mmd,
        "test_mmd_run": r_mmd,
        "mmd_gate_metric": common_mmd_key,
        "selected_match_all": all(selected_match.values()),
        "selected_match": selected_match,
    }
    for key in ("test_pp", "unseen2_pp", "family_gene_pp"):
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
        row[f"{prefix}_u2_mmd_base"] = per_ds_mmd(b_u2, ds)
        row[f"{prefix}_u2_mmd_run"] = per_ds_mmd(r_u2, ds)

    checks = {
        "selected_match": bool(row["selected_match_all"]),
        "wessels_u2_rescue": (
            row.get("Wessels_u2_pp_delta") is not None
            and (
                row["Wessels_u2_pp_delta"] >= 0.05
                or (row.get("Wessels_u2_pp_run") is not None and row["Wessels_u2_pp_run"] > 0)
            )
        ),
        "norman_not_harmed": (
            row.get("Norman_u2_pp_delta") is None
            or row["Norman_u2_pp_delta"] >= -0.03
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
    }
    row["checks"] = checks
    row["triage_status"] = (
        "fewshot_wessels_rescue_candidate"
        if all(checks.values())
        else "diagnostic_or_fail"
    )
    return row


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_md(rows: list[dict[str, Any]], manifest_path: Path) -> str:
    lines = [
        "# LatentFM Few-Shot Multi-Calibration Summary",
        "",
        f"Manifest: `{manifest_path}`",
        "",
        "Baseline for each row is the unchanged anchor checkpoint evaluated on the same custom split.",
        "Therefore deltas use matched held-out conditions, not the original canonical split selection.",
        "",
        "## Gate Table",
        "",
        "| arm | status | moved | selected match | unseen2 pp delta | Wessels u2 delta | Norman u2 delta | test pp delta | family_gene pp delta | MMD ratio |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| `{arm}` | `{status}` | {moved} | {sel} | {u2} | {wu2} | {nu2} | {tpp} | {gpp} | {mmd} |".format(
                arm=row["arm"],
                status=row["triage_status"],
                moved=row.get("moved_multi", "NA"),
                sel=fmt(row.get("selected_match_all")),
                u2=fmt(row.get("unseen2_pp_delta")),
                wu2=fmt(row.get("Wessels_u2_pp_delta")),
                nu2=fmt(row.get("Norman_u2_pp_delta")),
                tpp=fmt(row.get("test_pp_delta")),
                gpp=fmt(row.get("family_gene_pp_delta")),
                mmd=fmt(row.get("test_mmd_ratio")),
            )
        )
    lines += [
        "",
        "## Focus Dataset Unseen2",
        "",
        "| arm | dataset | base pp | run pp | delta pp | base MMD | run MMD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for ds, prefix in (
            ("Wessels", "Wessels"),
            ("Norman", "Norman"),
            ("Gasperini", "Gasperini"),
        ):
            lines.append(
                f"| `{row['arm']}` | `{ds}` | "
                f"{fmt(row.get(prefix + '_u2_pp_base'))} | "
                f"{fmt(row.get(prefix + '_u2_pp_run'))} | "
                f"{fmt(row.get(prefix + '_u2_pp_delta'))} | "
                f"{fmt(row.get(prefix + '_u2_mmd_base'))} | "
                f"{fmt(row.get(prefix + '_u2_mmd_run'))} |"
            )
    lines += [
        "",
        "## Interpretation Rules",
        "",
        "- A passing capped result is diagnostic only, not a zero-shot promotion.",
        "- Promotion requires condition-uncapped split/family posthoc and paired condition-level bootstrap.",
        "- If only Norman improves, prefer routed prior or interaction modeling over sampler-only sweeps.",
        "- If Wessels improves with few-shot multi exposure, separate few-shot and zero-shot manuscript claims.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("/data/cyx/1030/scLatent/runs/latentfm_fewshot_multi_calibration_20260621/launch_manifest.json"))
    parser.add_argument("--out-json", type=Path, default=Path("/data/cyx/1030/scLatent/reports/latentfm_fewshot_multi_calibration_summary_20260621.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("/data/cyx/1030/scLatent/reports/latentfm_fewshot_multi_calibration_summary_20260621.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("/data/cyx/1030/scLatent/reports/LATENTFM_FEWSHOT_MULTI_CALIBRATION_SUMMARY_20260621.md"))
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    rows = [summarize_one(spec) for spec in manifest.get("launched_runs", [])]
    payload = {"manifest": str(args.manifest), "rows": rows}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        fieldnames = [
            "arm",
            "run_name",
            "triage_status",
            "moved_multi",
            "selected_match_all",
            "test_pp_delta",
            "unseen2_pp_delta",
            "family_gene_pp_delta",
            "test_mmd_ratio",
            "Wessels_u2_pp_delta",
            "Norman_u2_pp_delta",
            "Gasperini_u2_pp_delta",
            "split_file",
        ]
        with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fieldnames})
    else:
        args.out_csv.write_text("", encoding="utf-8")
    args.out_md.write_text(render_md(rows, args.manifest), encoding="utf-8")
    print(json.dumps({"out_md": str(args.out_md), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
