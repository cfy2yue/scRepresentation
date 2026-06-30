#!/usr/bin/env python3
"""Synthesize the active LatentFM few-shot/response posthoc decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_FEWSHOT_SUMMARY = Path("/data/cyx/1030/scLatent/reports/latentfm_fewshot_multi_calibration_summary_20260621.json")
DEFAULT_RESPONSE_SUMMARY = Path("/data/cyx/1030/scLatent/reports/latentfm_response_geometry_smoke_summary_20260621.json")
DEFAULT_FEWSHOT_BOOT = Path("/data/cyx/1030/scLatent/reports/latentfm_fewshot_multi_calibration_bootstrap_20260621/bootstrap_index.json")
DEFAULT_RESPONSE_BOOT = Path("/data/cyx/1030/scLatent/reports/latentfm_response_geometry_smoke_bootstrap_20260621/bootstrap_index.json")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def bootstrap_outputs(index: dict[str, Any] | None) -> list[dict[str, str]]:
    if not index:
        return []
    out = index.get("outputs") or []
    return [x for x in out if isinstance(x, dict)]


def load_bootstrap_rows(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    payload = load_json(p) or {}
    rows = payload.get("rows") or []
    return [r for r in rows if isinstance(r, dict)]


def key_bootstrap_rows(
    outputs: list[dict[str, str]],
    *,
    target_run: str | None,
) -> list[dict[str, Any]]:
    wanted = {
        ("split", "test_multi_unseen2", "pearson_pert"),
        ("split", "test", "test_mmd_clamped"),
        ("family", "family_gene", "pearson_pert"),
    }
    out: list[dict[str, Any]] = []
    for item in outputs:
        run_name = str(item.get("run_name") or "")
        if target_run and run_name != target_run:
            continue
        kind = str(item.get("kind") or "")
        for row in load_bootstrap_rows(item.get("json")):
            key = (kind, str(row.get("group") or ""), str(row.get("metric") or ""))
            if key not in wanted:
                continue
            enriched = dict(row)
            enriched["run_name"] = run_name
            enriched["kind"] = kind
            enriched["source_json"] = item.get("json")
            out.append(enriched)
    return out


def summarize_bootstrap_md(label: str, index: dict[str, Any] | None) -> list[str]:
    outputs = bootstrap_outputs(index)
    if not outputs:
        return [f"- {label}: bootstrap not available yet."]
    lines = [f"- {label}: `{len(outputs)}` bootstrap report files."]
    for row in outputs[:8]:
        md = row.get("md")
        kind = row.get("kind")
        run = row.get("run_name")
        lines.append(f"  - `{run}` {kind}: `{md}`")
    if len(outputs) > 8:
        lines.append(f"  - ... {len(outputs) - 8} additional reports omitted from this summary.")
    return lines


def render_key_bootstrap_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No key bootstrap rows available yet."]
    lines = [
        "| run | kind | group | metric | n conds | delta | 95% CI | p improve | p harm | selected match |",
        "|---|---|---|---|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        ci = f"[{fmt(fnum(row.get('ci95_low')))}, {fmt(fnum(row.get('ci95_high')))}]"
        lines.append(
            "| `{run}` | {kind} | {group} | {metric} | {n} | {delta} | {ci} | {pimp} | {pharm} | {sel} |".format(
                run=row.get("run_name", "NA"),
                kind=row.get("kind", "NA"),
                group=row.get("group", "NA"),
                metric=row.get("metric", "NA"),
                n=row.get("n_matched_conditions", 0),
                delta=fmt(fnum(row.get("delta_mean"))),
                ci=ci,
                pimp=fmt(fnum(row.get("p_improvement"))),
                pharm=fmt(fnum(row.get("p_harm"))),
                sel=fmt(row.get("selected_match")),
            )
        )
    return lines


def fewshot_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    rows = payload.get("rows") or []
    return [r for r in rows if isinstance(r, dict)]


def response_row(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    row = payload.get("row")
    return row if isinstance(row, dict) else None


def best_fewshot(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    def score(row: dict[str, Any]) -> tuple[int, float, float]:
        status = 1 if row.get("triage_status") == "fewshot_wessels_rescue_candidate" else 0
        w = fnum(row.get("Wessels_u2_pp_delta"))
        u2 = fnum(row.get("unseen2_pp_delta"))
        return (status, -999.0 if w is None else w, -999.0 if u2 is None else u2)
    return max(rows, key=score)


def decide(
    fewshot_payload: dict[str, Any] | None,
    response_payload: dict[str, Any] | None,
    fewshot_boot: dict[str, Any] | None,
    response_boot: dict[str, Any] | None,
) -> dict[str, Any]:
    missing = []
    if fewshot_payload is None:
        missing.append("fewshot_summary")
    if response_payload is None:
        missing.append("response_summary")
    if fewshot_boot is None:
        missing.append("fewshot_bootstrap")
    if response_boot is None:
        missing.append("response_bootstrap")

    rows = fewshot_rows(fewshot_payload)
    best = best_fewshot(rows)
    resp = response_row(response_payload)

    fewshot_pass = bool(best and best.get("triage_status") == "fewshot_wessels_rescue_candidate")
    response_pass = bool(resp and resp.get("triage_status") == "response_geometry_candidate")

    if missing:
        status = "waiting_for_inputs"
        next_action = "Wait for active posthoc/bootstraps; do not launch pairwise yet."
    elif fewshot_pass:
        status = "fewshot_rescue_candidate"
        next_action = "Run uncapped few-shot posthoc and bootstrap; keep response/pairwise as ablations."
    elif response_pass:
        status = "response_geometry_candidate"
        next_action = "Run uncapped response-geometry posthoc and bootstrap; pairwise remains queued."
    else:
        status = "launch_pairwise_next"
        next_action = "Launch pairwise condition smoke if GPU/RAM audit passes."

    return {
        "status": status,
        "missing_inputs": missing,
        "next_action": next_action,
        "best_fewshot": best,
        "response": resp,
        "fewshot_rows": rows,
        "fewshot_bootstrap_outputs": bootstrap_outputs(fewshot_boot),
        "response_bootstrap_outputs": bootstrap_outputs(response_boot),
        "fewshot_key_bootstrap": key_bootstrap_rows(
            bootstrap_outputs(fewshot_boot),
            target_run=(str(best.get("run_name")) if best else None),
        ),
        "response_key_bootstrap": key_bootstrap_rows(
            bootstrap_outputs(response_boot),
            target_run=(str(resp.get("run_name")) if resp else None),
        ),
    }


def render_md(decision: dict[str, Any], paths: dict[str, Path]) -> str:
    best = decision.get("best_fewshot") or {}
    resp = decision.get("response") or {}
    lines = [
        "# LatentFM Active Posthoc Decision",
        "",
        f"Status: `{decision['status']}`",
        "",
        f"Next action: {decision['next_action']}",
        "",
        "## Inputs",
        "",
    ]
    for key, path in paths.items():
        lines.append(f"- {key}: `{path}` ({'present' if path.is_file() else 'missing'})")
    if decision.get("missing_inputs"):
        lines += ["", "Missing inputs: `" + "`, `".join(decision["missing_inputs"]) + "`"]

    lines += [
        "",
        "## Best Few-Shot Row",
        "",
        "| arm | status | moved | Wessels u2 delta | Norman u2 delta | unseen2 delta | test pp delta | family_gene delta | MMD ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        "| `{arm}` | `{status}` | {moved} | {w} | {n} | {u2} | {tpp} | {gpp} | {mmd} |".format(
            arm=best.get("arm", "NA"),
            status=best.get("triage_status", "NA"),
            moved=fmt(best.get("moved_multi")),
            w=fmt(fnum(best.get("Wessels_u2_pp_delta"))),
            n=fmt(fnum(best.get("Norman_u2_pp_delta"))),
            u2=fmt(fnum(best.get("unseen2_pp_delta"))),
            tpp=fmt(fnum(best.get("test_pp_delta"))),
            gpp=fmt(fnum(best.get("family_gene_pp_delta"))),
            mmd=fmt(fnum(best.get("test_mmd_ratio"))),
        ),
        "",
        "## Response Geometry Row",
        "",
        "| status | Wessels u2 run | Wessels u2 delta | Norman u2 delta | unseen2 delta | test pp delta | family_gene delta | MMD ratio | direct delta | artifact split |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        "| `{status}` | {wr} | {wd} | {nd} | {u2} | {tpp} | {gpp} | {mmd} | {direct} | {asha} |".format(
            status=resp.get("triage_status", "NA"),
            wr=fmt(fnum(resp.get("Wessels_u2_pp_run"))),
            wd=fmt(fnum(resp.get("Wessels_u2_pp_delta"))),
            nd=fmt(fnum(resp.get("Norman_u2_pp_delta"))),
            u2=fmt(fnum(resp.get("unseen2_pp_delta"))),
            tpp=fmt(fnum(resp.get("test_pp_delta"))),
            gpp=fmt(fnum(resp.get("family_gene_pp_delta"))),
            mmd=fmt(fnum(resp.get("test_mmd_ratio"))),
            direct=fmt(fnum(resp.get("test_direct_delta"))),
            asha=fmt(resp.get("artifact_split_sha_match")),
        ),
        "",
        "## Bootstrap Reports",
        "",
    ]
    lines.extend(summarize_bootstrap_md("few-shot", {"outputs": decision.get("fewshot_bootstrap_outputs", [])}))
    lines.extend(summarize_bootstrap_md("response geometry", {"outputs": decision.get("response_bootstrap_outputs", [])}))
    lines += [
        "",
        "## Key Bootstrap Evidence",
        "",
        "### Few-Shot Best Arm",
        "",
    ]
    lines.extend(render_key_bootstrap_table(decision.get("fewshot_key_bootstrap", [])))
    lines += [
        "",
        "### Response Geometry",
        "",
    ]
    lines.extend(render_key_bootstrap_table(decision.get("response_key_bootstrap", [])))
    lines += [
        "",
        "## Decision Rules",
        "",
        "- Few-shot rescue takes priority if Wessels improves and harm/MMD gates pass.",
        "- If few-shot fails but response geometry passes, promote response geometry to uncapped posthoc.",
        "- If both fail, launch pairwise condition smoke only after GPU/RAM audit.",
        "- No capped result is a final claim without uncapped posthoc and paired bootstrap.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fewshot-summary", type=Path, default=DEFAULT_FEWSHOT_SUMMARY)
    parser.add_argument("--response-summary", type=Path, default=DEFAULT_RESPONSE_SUMMARY)
    parser.add_argument("--fewshot-bootstrap", type=Path, default=DEFAULT_FEWSHOT_BOOT)
    parser.add_argument("--response-bootstrap", type=Path, default=DEFAULT_RESPONSE_BOOT)
    parser.add_argument("--out-json", type=Path, default=Path("/data/cyx/1030/scLatent/reports/latentfm_active_posthoc_decision_20260621.json"))
    parser.add_argument("--out-md", type=Path, default=Path("/data/cyx/1030/scLatent/reports/LATENTFM_ACTIVE_POSTHOC_DECISION_20260621.md"))
    args = parser.parse_args()

    fewshot_payload = load_json(args.fewshot_summary)
    response_payload = load_json(args.response_summary)
    fewshot_boot = load_json(args.fewshot_bootstrap)
    response_boot = load_json(args.response_bootstrap)
    decision = decide(fewshot_payload, response_payload, fewshot_boot, response_boot)
    payload = {
        "status": decision["status"],
        "missing_inputs": decision["missing_inputs"],
        "next_action": decision["next_action"],
        "inputs": {
            "fewshot_summary": str(args.fewshot_summary),
            "response_summary": str(args.response_summary),
            "fewshot_bootstrap": str(args.fewshot_bootstrap),
            "response_bootstrap": str(args.response_bootstrap),
        },
        "decision": decision,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    paths = {
        "fewshot_summary": args.fewshot_summary,
        "response_summary": args.response_summary,
        "fewshot_bootstrap": args.fewshot_bootstrap,
        "response_bootstrap": args.response_bootstrap,
    }
    args.out_md.write_text(render_md(decision, paths), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
