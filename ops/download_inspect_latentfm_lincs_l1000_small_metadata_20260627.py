#!/usr/bin/env python3
"""Download and inspect small LINCS/L1000 metadata tables only.

This intentionally avoids Level5 expression matrices. The output is a schema
gate, not a signal gate.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import time
import urllib.request
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports/external_artifact_sources_20260627/lincs_l1000_geo_small"
REPORT_DIR = ROOT / "reports/lincs_l1000_small_metadata_schema_20260627"
REPORT_JSON = ROOT / "reports/latentfm_lincs_l1000_small_metadata_schema_20260627.json"
REPORT_MD = ROOT / "reports/LATENTFM_LINCS_L1000_SMALL_METADATA_SCHEMA_20260627.md"

SOURCES = [
    {
        "name": "GSE92742_sig_info",
        "kind": "sig_info",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_info.txt.gz",
    },
    {
        "name": "GSE92742_sig_metrics",
        "kind": "sig_metrics",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/GSE92742_Broad_LINCS_sig_metrics.txt.gz",
    },
    {
        "name": "GSE70138_sig_info",
        "kind": "sig_info",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz",
    },
    {
        "name": "GSE70138_sig_metrics",
        "kind": "sig_metrics",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/GSE70138_Broad_LINCS_sig_metrics_2017-03-06.txt.gz",
    },
]

KEY_FIELDS = {
    "sig_id",
    "pert_id",
    "pert_iname",
    "pert_type",
    "cell_id",
    "pert_idose",
    "pert_dose",
    "pert_itime",
    "pert_time",
    "tas",
    "distil_cc_q75",
    "distil_cc",
    "ss_ngene",
}


def _download(url: str, dest: Path) -> dict:
    if dest.exists() and dest.stat().st_size > 0:
        return {"downloaded": False, "status": "exists", "bytes": dest.stat().st_size}
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    h = hashlib.md5()
    total = 0
    start = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "latentfm-schema-audit/20260627"})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            total += len(chunk)
            if time.time() - start > 180:
                raise TimeoutError(f"download exceeded 180s for {url}")
    tmp.replace(dest)
    return {"downloaded": True, "status": "downloaded", "bytes": total, "md5": h.hexdigest()}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _inspect_tsv_gz(path: Path, sample_rows: int = 5000) -> dict:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        columns = reader.fieldnames or []
        n = 0
        nonempty = {c: 0 for c in columns}
        examples: dict[str, list[str]] = {c: [] for c in columns}
        for row in reader:
            n += 1
            if n <= sample_rows:
                for c in columns:
                    val = (row.get(c) or "").strip()
                    if val:
                        nonempty[c] += 1
                        if len(examples[c]) < 3 and val not in examples[c]:
                            examples[c].append(val)
            if n >= sample_rows:
                break
    # Count all rows separately so sample inspection stays cheap.
    full_n = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        next(f, None)
        for _ in f:
            full_n += 1
    lower_map = {c.lower(): c for c in columns}
    present_key_fields = {k: lower_map[k] for k in sorted(KEY_FIELDS & set(lower_map))}
    return {
        "columns": columns,
        "n_columns": len(columns),
        "row_count": full_n,
        "sample_rows": min(sample_rows, full_n),
        "present_key_fields": present_key_fields,
        "nonempty_key_field_counts_in_sample": {
            k: nonempty.get(v, 0) for k, v in present_key_fields.items()
        },
        "example_key_values": {k: examples.get(v, []) for k, v in present_key_fields.items()},
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for src in SOURCES:
        dest = OUT_DIR / Path(src["url"]).name
        rec = dict(src)
        rec["path"] = str(dest)
        try:
            rec["download"] = _download(src["url"], dest)
            rec["bytes"] = dest.stat().st_size
            rec["md5"] = _md5(dest)
            rec["schema"] = _inspect_tsv_gz(dest)
            rec["status"] = "schema_ready"
        except Exception as exc:  # noqa: BLE001 - report exact blocker.
            rec["status"] = "failed"
            rec["error"] = f"{type(exc).__name__}: {exc}"
        records.append(rec)

    required_join_any = {"sig_id", "pert_id", "pert_iname", "pert_type", "cell_id"}
    metric_fields = {"tas", "distil_cc_q75", "distil_cc", "ss_ngene"}
    ready_files = [r for r in records if r.get("status") == "schema_ready"]
    has_sig_info = any(r["kind"] == "sig_info" and r["status"] == "schema_ready" for r in records)
    has_sig_metrics = any(r["kind"] == "sig_metrics" and r["status"] == "schema_ready" for r in records)
    present_union = set()
    for r in ready_files:
        present_union.update((r.get("schema", {}).get("present_key_fields") or {}).keys())

    status = "lincs_small_metadata_schema_ready_no_gpu" if has_sig_info and has_sig_metrics else "lincs_small_metadata_schema_incomplete_no_gpu"
    gpu_authorized = False
    reasons = []
    if has_sig_info and has_sig_metrics:
        reasons.append("small_metadata_downloaded_and_schema_inspected")
    else:
        reasons.append("small_metadata_download_or_schema_incomplete")
    if not (required_join_any & present_union):
        reasons.append("no_join_key_fields_detected")
    if not (metric_fields & present_union):
        reasons.append("no_activity_metric_fields_detected")
    reasons.append("condition_level_join_not_materialized")
    reasons.append("shuffle_source_mmd_tail_gates_not_run")

    out = {
        "status": status,
        "gpu_authorized": gpu_authorized,
        "boundary": {
            "cpu_source_schema_only": True,
            "large_level5_download": False,
            "training_or_inference_used": False,
            "canonical_multi_selection_used": False,
            "trackc_heldout_query_used": False,
            "gpu_used": False,
        },
        "records": records,
        "present_key_fields_union": sorted(present_union),
        "reasons": reasons,
        "next_action": (
            "Build a condition-level LINCS materializer only if sig_info and "
            "sig_metrics can be joined by sig_id and contain perturbagen/type/"
            "cell/dose/time plus activity metrics; then run strict train-only "
            "overlap/shuffle/source/MMD/tail gates before any GPU."
        ),
    }
    REPORT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = REPORT_DIR / "downloaded_files.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["name", "kind", "status", "bytes", "md5", "path", "url", "error"])
        for r in records:
            writer.writerow([
                r["name"],
                r["kind"],
                r.get("status", ""),
                r.get("bytes", ""),
                r.get("md5", ""),
                r.get("path", ""),
                r.get("url", ""),
                r.get("error", ""),
            ])

    lines = [
        "# LINCS/L1000 Small Metadata Schema 2026-06-27",
        "",
        f"Status: `{status}`",
        "",
        f"GPU authorized: `{gpu_authorized}`",
        "",
        "## Boundary",
        "",
        "- Download/inspect only small `sig_info` and `sig_metrics` metadata gzip tables.",
        "- No Level5 GCTX matrices, training, inference, canonical multi selection, Track C held-out query, or GPU.",
        "",
        "## Files",
        "",
        "| name | kind | status | rows | columns | key fields | bytes |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for r in records:
        schema = r.get("schema", {})
        keys = ", ".join(schema.get("present_key_fields", {}).keys())
        lines.append(
            f"| `{r['name']}` | `{r['kind']}` | `{r.get('status')}` | "
            f"{schema.get('row_count', '')} | {schema.get('n_columns', '')} | "
            f"`{keys}` | {r.get('bytes', '')} |"
        )
    lines.extend(
        [
            "",
            "## Reasons",
            "",
            *[f"- `{reason}`" for reason in reasons],
            "",
            "## Decision",
            "",
            "This does not authorize GPU. It only establishes whether small LINCS metadata is locally schema-ready for a future condition-level materializer.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{REPORT_JSON}`",
            f"- manifest: `{manifest}`",
            f"- source dir: `{OUT_DIR}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": gpu_authorized, "out_md": str(REPORT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
