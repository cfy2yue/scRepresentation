#!/usr/bin/env python3
"""Summarize the xverse condition-prior adapter capped-smoke gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_NAME = "xverse_prior_adapter_global_genemean_w005_add002_replay1_4k"
DEFAULT_RUN_ROOT = (
    ROOT
    / "runs/latentfm_xverse_condition_prior_adapter_smoke_20260621"
    / RUN_NAME
)
DEFAULT_BOOTSTRAP_DIR = (
    ROOT
    / f"reports/latentfm_xverse_condition_prior_adapter_{RUN_NAME}_bootstrap_20260621"
)
DEFAULT_OUT_JSON = (
    ROOT
    / "reports/latentfm_xverse_condition_prior_adapter_decision_20260621.json"
)
DEFAULT_OUT_MD = (
    ROOT
    / "reports/LATENTFM_XVERSE_CONDITION_PRIOR_ADAPTER_DECISION_20260621.md"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    try:
        if value is None:
            return None
        val = float(value)
        if val != val:
            return None
        return val
    except Exception:
        return None


def bootstrap_row(path: Path, group: str, metric: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = load_json(path)
    for row in payload.get("rows") or []:
        if row.get("group") == group and row.get("metric") == metric:
            return row
    return None


def condition_table(payload: dict[str, Any], group: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = ((payload.get("groups") or {}).get(group) or {}).get("condition_metrics") or []
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict):
            key = (str(row.get("dataset")), str(row.get("condition")))
            out[key] = row
    return out


def condition_fraction_delta(
    base_path: Path,
    cand_path: Path,
    group: str,
    metric: str,
    *,
    threshold: float,
    mode: str,
) -> dict[str, Any]:
    if not base_path.is_file() or not cand_path.is_file():
        return {"status": "missing_inputs"}
    base = condition_table(load_json(base_path), group)
    cand = condition_table(load_json(cand_path), group)
    keys = sorted(set(base) & set(cand))
    if not keys:
        return {"status": "no_matched_conditions"}

    def ok(value: Any) -> bool:
        val = fnum(value)
        if val is None:
            return False
        if mode == "ge":
            return val >= threshold
        if mode == "gt":
            return val > threshold
        if mode == "le":
            return val <= threshold
        if mode == "lt":
            return val < threshold
        raise ValueError(f"unknown mode: {mode}")

    b_vals = [ok(base[k].get(metric)) for k in keys]
    c_vals = [ok(cand[k].get(metric)) for k in keys]
    b_frac = sum(b_vals) / len(b_vals)
    c_frac = sum(c_vals) / len(c_vals)
    return {
        "status": "ok",
        "group": group,
        "metric": metric,
        "threshold": threshold,
        "mode": mode,
        "n_matched_conditions": len(keys),
        "baseline_fraction": b_frac,
        "candidate_fraction": c_frac,
        "delta_fraction": c_frac - b_frac,
    }


def fmt(value: Any) -> str:
    val = fnum(value)
    if val is None:
        return "NA"
    return f"{val:+.6f}"


def row_brief(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {"status": "missing"}
    keep = [
        "group",
        "metric",
        "direction",
        "n_matched_conditions",
        "n_matched_datasets",
        "delta_mean",
        "ci95_low",
        "ci95_high",
        "p_improvement",
        "p_harm",
        "status",
    ]
    return {k: row.get(k) for k in keep if k in row}


def assess(payload: dict[str, Any]) -> tuple[str, list[str]]:
    rows = payload["bootstrap_rows"]
    fracs = payload["condition_fraction_deltas"]
    reasons: list[str] = []

    u2_pp = rows["split:test_multi_unseen2:pearson_pert"]
    u2_mmd = rows["split:test_multi_unseen2:test_mmd_clamped"]
    u2_pp_frac = fracs["unseen2_pp_positive"]
    u2_mmd_frac = fracs["unseen2_mmd_gt_005"]

    if fnum(u2_pp.get("delta_mean")) is None or fnum(u2_pp.get("delta_mean")) < 0.03:
        reasons.append("unseen2_pp_delta_below_0.03")
    if fnum(u2_pp.get("p_improvement")) is None or fnum(u2_pp.get("p_improvement")) < 0.90:
        reasons.append("unseen2_pp_p_improve_below_0.90")
    if (
        fnum(u2_pp_frac.get("delta_fraction")) is None
        or fnum(u2_pp_frac.get("delta_fraction")) < 0.10
    ):
        reasons.append("unseen2_pp_positive_fraction_delta_below_0.10")
    if (
        fnum(u2_mmd.get("delta_mean")) is not None
        and fnum(u2_mmd.get("delta_mean")) > 0.0
        and (fnum(u2_mmd.get("p_harm")) is None or fnum(u2_mmd.get("p_harm")) > 0.20)
    ):
        reasons.append("unseen2_mmd_harm")
    if (
        fnum(u2_mmd_frac.get("delta_fraction")) is not None
        and fnum(u2_mmd_frac.get("delta_fraction")) > 0.0
    ):
        reasons.append("unseen2_mmd_gt_005_fraction_increased")

    for key in (
        "split:test:pearson_pert",
        "split:test_single:pearson_pert",
        "family:family_gene:pearson_pert",
        "family:family_drug:pearson_pert",
    ):
        row = rows[key]
        if fnum(row.get("ci95_low")) is None or fnum(row.get("ci95_low")) < -0.005:
            reasons.append(f"{key}:pp_lower_ci_harm")

    for key in (
        "split:test:test_mmd_clamped",
        "family:family_gene:test_mmd_clamped",
    ):
        row = rows[key]
        if fnum(row.get("ci95_high")) is None or fnum(row.get("ci95_high")) > 0.0005:
            reasons.append(f"{key}:mmd_upper_ci_harm")

    return ("pass_capped_smoke" if not reasons else "diagnostic_or_fail"), reasons


def render_md(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# LatentFM xverse Condition-Prior Adapter Decision 2026-06-21",
        "",
        f"Status: `{decision['status']}`",
        "",
        "## Inputs",
        "",
        f"- run_name: `{payload['run_name']}`",
        f"- manifest: `{payload['manifest']}`",
        f"- bootstrap_dir: `{payload['bootstrap_dir']}`",
        "",
        "## Gate Reasons",
        "",
    ]
    if decision["reasons"]:
        lines.extend(f"- `{reason}`" for reason in decision["reasons"])
    else:
        lines.append("- none")

    lines.extend([
        "",
        "## Key Bootstrap Rows",
        "",
        "| key | delta | ci95 low | ci95 high | p_improve | p_harm | n cond | n ds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key, row in payload["bootstrap_rows"].items():
        lines.append(
            "| {key} | {delta} | {lo} | {hi} | {pi} | {ph} | {n} | {nds} |".format(
                key=key,
                delta=fmt(row.get("delta_mean")),
                lo=fmt(row.get("ci95_low")),
                hi=fmt(row.get("ci95_high")),
                pi=fmt(row.get("p_improvement")),
                ph=fmt(row.get("p_harm")),
                n=row.get("n_matched_conditions", "NA"),
                nds=row.get("n_matched_datasets", "NA"),
            )
        )

    lines.extend([
        "",
        "## Condition-Fraction Diagnostics",
        "",
        "| diagnostic | baseline frac | candidate frac | delta frac | n |",
        "|---|---:|---:|---:|---:|",
    ])
    for key, row in payload["condition_fraction_deltas"].items():
        lines.append(
            "| {key} | {base} | {cand} | {delta} | {n} |".format(
                key=key,
                base=fmt(row.get("baseline_fraction")),
                cand=fmt(row.get("candidate_fraction")),
                delta=fmt(row.get("delta_fraction")),
                n=row.get("n_matched_conditions", "NA"),
            )
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `pass_capped_smoke` only permits condition-uncapped posthoc and robustness checks; it is not a promotion claim.",
        "- `diagnostic_or_fail` should be treated as negative or mechanism evidence unless a predeclared follow-up gate is added.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--bootstrap-dir", type=Path, default=DEFAULT_BOOTSTRAP_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    manifest = args.run_root / "posthoc_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest}")
    mp = load_json(manifest)
    rows = mp.get("launched_runs")
    if isinstance(rows, list):
        rows = [row for row in rows if isinstance(row, dict)]
    elif {"baseline_split_json", "run_split_json"}.issubset(mp):
        rows = [mp]
    else:
        rows = []
    if not rows:
        raise ValueError(f"manifest has no launched_runs: {manifest}")
    run_name = str(rows[0].get("run_name") or RUN_NAME)
    split_boot = args.bootstrap_dir / f"{run_name}.split.bootstrap.json"
    family_boot = args.bootstrap_dir / f"{run_name}.family.bootstrap.json"
    base_split = Path(str(rows[0].get("baseline_split_json") or ""))
    cand_split = Path(str(rows[0].get("run_split_json") or ""))

    wanted = {
        "split:test:pearson_pert": (split_boot, "test", "pearson_pert"),
        "split:test:test_mmd_clamped": (split_boot, "test", "test_mmd_clamped"),
        "split:test_single:pearson_pert": (split_boot, "test_single", "pearson_pert"),
        "split:test_multi:pearson_pert": (split_boot, "test_multi", "pearson_pert"),
        "split:test_multi:test_mmd_clamped": (split_boot, "test_multi", "test_mmd_clamped"),
        "split:test_multi_unseen2:pearson_pert": (split_boot, "test_multi_unseen2", "pearson_pert"),
        "split:test_multi_unseen2:test_mmd_clamped": (split_boot, "test_multi_unseen2", "test_mmd_clamped"),
        "family:family_gene:pearson_pert": (family_boot, "family_gene", "pearson_pert"),
        "family:family_gene:test_mmd_clamped": (family_boot, "family_gene", "test_mmd_clamped"),
        "family:family_drug:pearson_pert": (family_boot, "family_drug", "pearson_pert"),
        "family:structure_multi:pearson_pert": (family_boot, "structure_multi", "pearson_pert"),
        "family:structure_multi:test_mmd_clamped": (family_boot, "structure_multi", "test_mmd_clamped"),
    }
    boot_rows = {
        key: row_brief(bootstrap_row(path, group, metric))
        for key, (path, group, metric) in wanted.items()
    }
    frac_rows = {
        "unseen2_pp_positive": condition_fraction_delta(
            base_split,
            cand_split,
            "test_multi_unseen2",
            "pearson_pert",
            threshold=0.0,
            mode="gt",
        ),
        "unseen2_mmd_gt_005": condition_fraction_delta(
            base_split,
            cand_split,
            "test_multi_unseen2",
            "test_mmd_clamped",
            threshold=0.05,
            mode="gt",
        ),
    }
    payload = {
        "run_name": run_name,
        "manifest": str(manifest),
        "bootstrap_dir": str(args.bootstrap_dir),
        "bootstrap_rows": boot_rows,
        "condition_fraction_deltas": frac_rows,
    }
    status, reasons = assess(payload)
    payload["decision"] = {"status": status, "reasons": reasons}
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "status": status}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
