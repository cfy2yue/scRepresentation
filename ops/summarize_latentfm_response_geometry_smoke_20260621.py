#!/usr/bin/env python3
"""Summarize response-geometry LatentFM smoke posthoc results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


FOCUS_DATASETS = ("Wessels", "NormanWeissman2019_filtered", "GasperiniShendure2019_lowMOI")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_artifact_metadata(path_s: str | None) -> dict[str, Any]:
    if not path_s:
        return {}
    path = Path(path_s)
    if not path.is_file():
        return {"artifact_exists": False, "artifact_sha256": None}
    out: dict[str, Any] = {"artifact_exists": True, "artifact_sha256": sha256_file(path)}
    try:
        obj = np.load(str(path), allow_pickle=False)
        metadata = json.loads(str(obj["metadata_json"].item()))
    except Exception as exc:  # noqa: BLE001 - provenance should report parse failure, not hide it.
        out["metadata_error"] = str(exc)
        return out
    out["metadata"] = metadata
    out["split_sha256"] = metadata.get("split_sha256")
    out["fit_scope"] = metadata.get("fit_scope")
    out["forbidden_inputs_used"] = metadata.get("forbidden_inputs_used")
    return out


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

    mmd_key_b, b_mmd = mmd_gate_value(b_test)
    mmd_key_r, r_mmd = mmd_gate_value(r_test)
    row: dict[str, Any] = {
        "run_name": manifest["run_name"],
        "anchor_checkpoint": manifest["anchor_checkpoint"],
        "candidate_checkpoint": manifest["candidate_checkpoint"],
        "split_file": manifest["split_file"],
        "artifact": manifest["response_normalization_artifact"],
        "test_pp_base": metric(b_test, "pp"),
        "test_pp_run": metric(r_test, "pp"),
        "test_direct_base": metric(b_test, "dp"),
        "test_direct_run": metric(r_test, "dp"),
        "unseen2_pp_base": metric(b_u2, "pp"),
        "unseen2_pp_run": metric(r_u2, "pp"),
        "family_gene_pp_base": metric(b_gene, "pp"),
        "family_gene_pp_run": metric(r_gene, "pp"),
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
    for key in ("test_pp", "unseen2_pp", "family_gene_pp"):
        row[f"{key}_delta"] = delta(row.get(f"{key}_run"), row.get(f"{key}_base"))
    row["test_direct_delta"] = delta(row.get("test_direct_run"), row.get("test_direct_base"))
    row["test_mmd_ratio"] = ratio(r_mmd, b_mmd)
    row["artifact_provenance"] = load_artifact_metadata(row.get("artifact"))
    artifact_split_sha = (row["artifact_provenance"].get("split_sha256") if isinstance(row["artifact_provenance"], dict) else None)
    split_sha = sha256_file(Path(manifest["split_file"]))
    row["split_sha256"] = split_sha
    row["artifact_split_sha_match"] = bool(artifact_split_sha and split_sha and artifact_split_sha == split_sha)
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
        "wessels_unseen2_rescue": (
            row.get("Wessels_u2_pp_run") is not None
            and (
                row["Wessels_u2_pp_run"] > -0.05
                or (
                    row.get("Wessels_u2_pp_delta") is not None
                    and row["Wessels_u2_pp_delta"] >= 0.08
                )
            )
        ),
        "overall_pp_not_harmed": (
            row.get("test_pp_delta") is not None and row["test_pp_delta"] >= -0.005
        ),
        "family_gene_not_harmed": (
            row.get("family_gene_pp_delta") is not None and row["family_gene_pp_delta"] >= -0.01
        ),
        "norman_not_harmed": (
            row.get("Norman_u2_pp_delta") is None or row["Norman_u2_pp_delta"] >= -0.03
        ),
        "mmd_ratio_ok": (
            row.get("test_mmd_ratio") is not None and row["test_mmd_ratio"] <= 1.15
        ),
        "direct_pearson_no_collapse": (
            row.get("test_direct_run") is not None
            and row["test_direct_run"] >= 0.95
            and (row.get("test_direct_delta") is None or row["test_direct_delta"] >= -0.005)
        ),
        "artifact_split_sha_match": bool(row.get("artifact_split_sha_match")),
        "artifact_train_only": (
            isinstance(row.get("artifact_provenance"), dict)
            and row["artifact_provenance"].get("fit_scope") == "train_only"
            and not any((row["artifact_provenance"].get("forbidden_inputs_used") or {}).values())
        ),
    }
    row["checks"] = checks
    row["triage_status"] = "response_geometry_candidate" if all(checks.values()) else "diagnostic_or_fail"
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
        "# LatentFM Response Geometry Smoke Summary",
        "",
        f"Manifest: `{manifest_path}`",
        "",
        "Baseline is the unchanged anchor checkpoint evaluated on the same canonical split.",
        "All gates use raw-space split/family metrics, not normalized-space diagnostics.",
        "",
        "## Gate",
        "",
        "| status | selected match | unseen2 pp delta | Wessels u2 run | Wessels u2 delta | Norman u2 delta | test pp delta | family_gene pp delta | MMD ratio | direct delta | artifact split |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        "| `{status}` | {sel} | {u2} | {wu2r} | {wu2d} | {nu2} | {tpp} | {gpp} | {mmd} | {direct} | {asha} |".format(
            status=row["triage_status"],
            sel=fmt(row.get("selected_match_all")),
            u2=fmt(row.get("unseen2_pp_delta")),
            wu2r=fmt(row.get("Wessels_u2_pp_run")),
            wu2d=fmt(row.get("Wessels_u2_pp_delta")),
            nu2=fmt(row.get("Norman_u2_pp_delta")),
            tpp=fmt(row.get("test_pp_delta")),
            gpp=fmt(row.get("family_gene_pp_delta")),
            mmd=fmt(row.get("test_mmd_ratio")),
            direct=fmt(row.get("test_direct_delta")),
            asha=fmt(row.get("artifact_split_sha_match")),
        ),
        "",
        "## Response Artifact Provenance",
        "",
        f"- artifact: `{row.get('artifact')}`",
        f"- artifact sha256: `{(row.get('artifact_provenance') or {}).get('artifact_sha256')}`",
        f"- split sha256: `{row.get('split_sha256')}`",
        f"- artifact split sha match: {fmt(row.get('artifact_split_sha_match'))}",
        f"- fit scope: `{(row.get('artifact_provenance') or {}).get('fit_scope')}`",
        f"- forbidden inputs used: `{(row.get('artifact_provenance') or {}).get('forbidden_inputs_used')}`",
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
        "- Passing this 4k capped smoke only permits uncapped posthoc and paired bootstrap.",
        "- It is a mechanism/optimization branch; it does not replace zero-shot/few-shot exposure analysis.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("/data/cyx/1030/scLatent/runs/latentfm_response_normalization_20260621/posthoc_manifest.json"))
    parser.add_argument("--out-json", type=Path, default=Path("/data/cyx/1030/scLatent/reports/latentfm_response_geometry_smoke_summary_20260621.json"))
    parser.add_argument("--out-md", type=Path, default=Path("/data/cyx/1030/scLatent/reports/LATENTFM_RESPONSE_GEOMETRY_SMOKE_SUMMARY_20260621.md"))
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
