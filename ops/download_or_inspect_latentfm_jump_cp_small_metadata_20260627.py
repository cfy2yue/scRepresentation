#!/usr/bin/env python3
"""Materialize and inspect small JUMP-CP/Cell Painting metadata sources.

CPU-only metadata gate:
- no GPU, training, inference, canonical multi Track A selection, or Track C
  held-out query;
- no profile matrix downloads;
- only small GitHub metadata/stats files and S3/GitHub listings are fetched.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import socket
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports/jump_cp_small_metadata_schema_20260627"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MD = ROOT / "reports/LATENTFM_JUMP_CP_SMALL_METADATA_SCHEMA_20260627.md"
OUT_JSON = ROOT / "reports/latentfm_jump_cp_small_metadata_schema_20260627.json"
OUT_MANIFEST = OUT_DIR / "materialized_file_manifest.tsv"
OUT_SCHEMA = OUT_DIR / "schema_assessment.tsv"
OUT_SOURCE_LISTING = OUT_DIR / "source_listing_manifest.tsv"
OUT_BLOCKERS = OUT_DIR / "blockers.json"

USER_AGENT = "latentfm-jump-cp-small-metadata-gate/20260627"
MAX_DOWNLOAD_BYTES = 12 * 1024 * 1024
NETWORK_TIMEOUT_SECONDS = 20

RAW_BASE = "https://raw.githubusercontent.com/jump-cellpainting/datasets/main"
SMALL_FILES = [
    ("github_metadata", "metadata/README.md", "metadata_readme", 1_000_000),
    ("github_metadata", "metadata/cellprofiler_version.csv", "pipeline_version", 1_000_000),
    ("github_metadata", "metadata/compound.csv.gz", "compound_metadata", MAX_DOWNLOAD_BYTES),
    ("github_metadata", "metadata/compound_source.csv.gz", "compound_source_metadata", 2_000_000),
    ("github_metadata", "metadata/crispr.csv.gz", "crispr_metadata", 1_000_000),
    ("github_metadata", "metadata/microscope_config.csv", "microscope_config", 1_000_000),
    ("github_metadata", "metadata/microscope_filter.csv", "microscope_filter", 1_000_000),
    ("github_metadata", "metadata/orf.csv.gz", "orf_metadata", 2_000_000),
    ("github_metadata", "metadata/perturbation_control.csv", "perturbation_control", 1_000_000),
    ("github_metadata", "metadata/plate.csv.gz", "plate_metadata", 1_000_000),
    ("github_metadata", "metadata/well.csv.gz", "well_metadata", MAX_DOWNLOAD_BYTES),
    ("github_stats", "stats/cpg0016_cell_count_estimate_per_source.csv", "cell_count_by_source", 1_000_000),
    ("github_stats", "stats/cpg0016_site_count.csv", "site_count", 1_000_000),
    ("github_stats", "stats/cpg0016_well_count.csv", "well_count", 1_000_000),
    ("github_manifest", "manifests/profile_index.json", "profile_index_manifest_only", 1_000_000),
]

S3_LISTINGS = [
    ("s3_root_sources", "cpg0016-jump/", "/", 1000),
    ("s3_source4_workspace", "cpg0016-jump/source_4/workspace/", "/", 1000),
    ("s3_source4_load_data_batches", "cpg0016-jump/source_4/workspace/load_data_csv/", "/", 1000),
    ("s3_source4_profile_batches_no_download", "cpg0016-jump/source_4/workspace/profiles/", "/", 1000),
]

CATEGORY_TERMS = {
    "perturbation": ["pert", "compound", "jcp", "inchikey", "inchi", "pubchem", "broad_sample", "target", "gene", "symbol", "orf", "crispr", "control"],
    "cell": ["cell_line", "cellline", "cell_type", "celltype", "background"],
    "dose": ["dose", "dosage", "concentration", "concentration_um", "treatment_conc"],
    "time": ["timepoint", "time_point", "duration", "treatment_time", "hours", "hour"],
    "source": ["source", "provider", "center"],
    "plate": ["plate", "well", "site", "barcode"],
    "batch": ["batch", "run", "source"],
    "activity": ["activity", "phenotypic_activity", "is_active", "active_score", "tas"],
    "reproducibility": ["reproducibility", "replicate_repro", "percent_replicating", "replicate_corr", "replicate_correlation"],
    "profile_norm": ["profile_norm", "profile_magnitude", "profile_distance", "mahalanobis", "distance_from_control"],
}


def request(url: str, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})


def http_probe(url: str) -> dict[str, Any]:
    out: dict[str, Any] = {"url": url, "ok": False, "status": None, "content_length": None, "error": ""}
    try:
        with urllib.request.urlopen(request(url, "HEAD"), timeout=NETWORK_TIMEOUT_SECONDS, context=ssl.create_default_context()) as resp:
            out.update(
                ok=True,
                status=getattr(resp, "status", None),
                content_length=int(resp.headers.get("Content-Length") or 0) or None,
            )
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def download_small(url: str, dest: Path, byte_cap: int) -> dict[str, Any]:
    result = {
        "url": url,
        "path": str(dest),
        "downloaded": False,
        "status": None,
        "content_length": None,
        "size_bytes": 0,
        "sha256": "",
        "error": "",
    }
    try:
        h = hashlib.sha256()
        total = 0
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with urllib.request.urlopen(request(url), timeout=NETWORK_TIMEOUT_SECONDS, context=ssl.create_default_context()) as resp:
            result["status"] = getattr(resp, "status", None)
            content_length = int(resp.headers.get("Content-Length") or 0) or None
            result["content_length"] = content_length
            if content_length is not None and content_length > byte_cap:
                raise RuntimeError(f"content_length_exceeds_cap_{byte_cap}")
            with tmp.open("wb") as handle:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > byte_cap:
                        raise RuntimeError(f"download_exceeds_cap_{byte_cap}")
                    h.update(chunk)
                    handle.write(chunk)
        tmp.replace(dest)
        result.update(downloaded=True, size_bytes=total, sha256=h.hexdigest(), error="")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        try:
            tmp.unlink()
        except Exception:
            pass
    return result


def s3_listing(name: str, prefix: str, delimiter: str, max_keys: int) -> dict[str, Any]:
    params = {
        "list-type": "2",
        "prefix": prefix,
        "delimiter": delimiter,
        "max-keys": str(max_keys),
    }
    url = "https://cellpainting-gallery.s3.amazonaws.com/?" + urllib.parse.urlencode(params)
    row: dict[str, Any] = {
        "name": name,
        "url": url,
        "status": None,
        "bytes_read": 0,
        "prefixes": [],
        "contents": [],
        "error": "",
    }
    try:
        with urllib.request.urlopen(request(url), timeout=30, context=ssl.create_default_context()) as resp:
            data = resp.read(2_000_000)
            row["status"] = getattr(resp, "status", None)
            row["bytes_read"] = len(data)
        root = ET.fromstring(data)
        ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
        row["prefixes"] = [x.findtext("s:Prefix", namespaces=ns) for x in root.findall("s:CommonPrefixes", ns)]
        row["contents"] = [
            {
                "key": x.findtext("s:Key", namespaces=ns),
                "size": int(x.findtext("s:Size", default="0", namespaces=ns) or 0),
            }
            for x in root.findall("s:Contents", ns)
        ]
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def open_table(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8", errors="replace")
    return path.open("r", newline="", encoding="utf-8", errors="replace")


def delimiter_for(path: Path) -> str:
    plain = path.name[:-3] if path.name.endswith(".gz") else path.name
    return "\t" if plain.endswith(".tsv") or plain.endswith(".txt") else ","


def assess_columns(columns: list[str]) -> dict[str, Any]:
    lower = {col: col.lower().replace("-", "_") for col in columns}
    by_category: dict[str, list[str]] = {}
    for category, terms in CATEGORY_TERMS.items():
        hits = []
        for col, low in lower.items():
            if category == "cell" and low in {"metadata_cellprofiler_version", "n_cells"}:
                continue
            if category == "profile_norm" and low in {"metadata_cellprofiler_version", "metadata_distance_between_z_microns"}:
                continue
            if any(term in low for term in terms):
                hits.append(col)
        by_category[category] = hits
    return by_category


def inspect_file(path: Path, logical_name: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "logical_name": logical_name,
        "path": str(path),
        "readable": False,
        "format": "unknown",
        "field_count": 0,
        "fields": [],
        "sample_rows": 0,
        "category_hits": {},
        "error": "",
    }
    try:
        if path.name.endswith(".json"):
            obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            row["readable"] = True
            row["format"] = "json"
            if isinstance(obj, dict):
                row["fields"] = sorted(obj.keys())
                row["field_count"] = len(row["fields"])
            elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
                keys = sorted({k for item in obj[:20] for k in item.keys()})
                row["fields"] = keys
                row["field_count"] = len(keys)
                row["sample_rows"] = min(len(obj), 20)
            row["category_hits"] = assess_columns(list(row["fields"]))
            return row
        if not (path.name.endswith(".csv") or path.name.endswith(".csv.gz") or path.name.endswith(".tsv") or path.name.endswith(".txt")):
            text = path.read_text(encoding="utf-8", errors="replace")
            row["readable"] = True
            row["format"] = "text"
            row["sample_rows"] = min(len(text.splitlines()), 20)
            return row
        with open_table(path) as handle:
            reader = csv.DictReader(handle, delimiter=delimiter_for(path))
            fields = list(reader.fieldnames or [])
            count = 0
            nonempty = Counter()
            for record in reader:
                count += 1
                for field in fields:
                    if str(record.get(field, "")).strip():
                        nonempty[field] += 1
                if count >= 500:
                    break
        row["readable"] = True
        row["format"] = "table"
        row["field_count"] = len(fields)
        row["fields"] = fields
        row["sample_rows"] = count
        row["category_hits"] = assess_columns(fields)
        row["nonempty_preview"] = dict(nonempty)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, sort_keys=True)
                out[field] = value
            writer.writerow(out)


def render_report(payload: dict[str, Any]) -> str:
    schema_rows = payload["schema_rows"]
    downloaded_rows = payload["downloaded_files"]
    global_hits = payload["global_category_hits"]
    status = payload["status"]
    blockers = payload["blockers"]
    lines = [
        "# JUMP-CP Small Metadata Schema / Materializer Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only small metadata/materializer gate.",
        "- No GPU, training, inference, canonical multi Track A selection, or Track C held-out query.",
        "- No Cell Painting profile matrices, parquet profiles, feature CSV matrices, images, or embeddings downloaded.",
        "- Download cap per file: `12 MiB`; all downloaded files are metadata, stats, README, or manifest files.",
        "",
        "## Materialized Sources",
        "",
        "| logical_name | downloaded | size_bytes | content_length | local_path | error |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in downloaded_rows:
        lines.append(
            f"| `{row['logical_name']}` | {row['downloaded']} | {row['size_bytes']} | "
            f"{row.get('content_length') or 'NA'} | `{row.get('local_path', '')}` | {row.get('error') or ''} |"
        )
    lines.extend(
        [
            "",
            "## Schema Assessment",
            "",
            "| logical_name | readable | fields | perturbation | cell | dose | time | source | plate | batch | activity | reproducibility | profile_norm |",
            "|---|---:|---:|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in schema_rows:
        hits = row.get("category_hits", {})
        lines.append(
            f"| `{row['logical_name']}` | {row['readable']} | {row['field_count']} | "
            f"{len(hits.get('perturbation', []))} | {len(hits.get('cell', []))} | "
            f"{len(hits.get('dose', []))} | {len(hits.get('time', []))} | "
            f"{len(hits.get('source', []))} | {len(hits.get('plate', []))} | "
            f"{len(hits.get('batch', []))} | {len(hits.get('activity', []))} | "
            f"{len(hits.get('reproducibility', []))} | {len(hits.get('profile_norm', []))} |"
        )
    lines.extend(["", "## Global Field Coverage", ""])
    for category in CATEGORY_TERMS:
        examples = global_hits.get(category, [])[:12]
        lines.append(f"- `{category}`: `{len(global_hits.get(category, []))}` columns; examples: {examples}")
    lines.extend(
        [
            "",
            "## S3 / Profile Boundary Probe",
            "",
            "| name | status | bytes_read | prefixes | contents | error |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["s3_listings"]:
        lines.append(
            f"| `{row['name']}` | {row.get('status') or 'NA'} | {row.get('bytes_read', 0)} | "
            f"{len(row.get('prefixes', []))} | {len(row.get('contents', []))} | {row.get('error') or ''} |"
        )
    lines.extend(
        [
            "",
            "The S3 profile namespace was listed only to confirm the boundary; no profile object was downloaded.",
            "",
            "## Gate Decision",
            "",
            f"- `can_enter_train_only_condition_join_controls_gate`: `{payload['can_enter_trainonly_join_controls_gate']}`",
            "- `gpu_authorized`: `False`",
            "- `chemical_v2_authorized`: `False`",
            "",
        ]
    )
    if payload["can_enter_trainonly_join_controls_gate"]:
        lines.extend(
            [
                "Conclusion: small JUMP-CP metadata is now locally materialized enough for the next CPU-only train-only condition join + controls gate.",
                "That next gate must still avoid canonical multi selection and Track C held-out query and must not use profile matrices.",
            ]
        )
    else:
        lines.append("Conclusion: remain blocked before train-only condition join + controls gate.")
    lines.extend(["", "## Blockers / Missing Before GPU", ""])
    for blocker in blockers:
        lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- materialized manifest: `{OUT_MANIFEST}`",
            f"- schema TSV: `{OUT_SCHEMA}`",
            f"- listing TSV: `{OUT_SOURCE_LISTING}`",
            f"- downloaded metadata dir: `{OUT_DIR}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    socket.setdefaulttimeout(NETWORK_TIMEOUT_SECONDS)
    downloaded: list[dict[str, Any]] = []
    for source_group, rel_path, logical_name, cap in SMALL_FILES:
        url = f"{RAW_BASE}/{rel_path}"
        dest = OUT_DIR / rel_path.replace("/", "__")
        result = download_small(url, dest, cap)
        result.update(source_group=source_group, rel_path=rel_path, logical_name=logical_name, local_path=str(dest))
        downloaded.append(result)

    s3_rows = [s3_listing(*args) for args in S3_LISTINGS]

    schema_rows = []
    for row in downloaded:
        if row.get("downloaded"):
            schema_rows.append(inspect_file(Path(row["local_path"]), row["logical_name"]))

    global_hits: dict[str, list[str]] = {k: [] for k in CATEGORY_TERMS}
    for row in schema_rows:
        for category, hits in row.get("category_hits", {}).items():
            for hit in hits:
                value = f"{row['logical_name']}:{hit}"
                if value not in global_hits[category]:
                    global_hits[category].append(value)

    downloaded_ok = [row for row in downloaded if row.get("downloaded")]
    core_metadata = {"compound_metadata", "crispr_metadata", "orf_metadata", "plate_metadata", "well_metadata"}
    have_core = core_metadata.issubset({row["logical_name"] for row in downloaded_ok})
    have_join_keys = all(global_hits[k] for k in ["perturbation", "source", "plate", "batch"])
    have_activity = bool(global_hits["activity"])
    have_repro = bool(global_hits["reproducibility"])
    have_profile_norm = bool(global_hits["profile_norm"])

    can_join = bool(have_core and have_join_keys)
    blockers = []
    if not have_core:
        blockers.append("core_jump_cp_metadata_not_fully_materialized")
    if not have_join_keys:
        blockers.append("required_join_or_confound_columns_missing")
    if not have_activity:
        blockers.append("activity_columns_not_confirmed_in_small_metadata")
    if not have_repro:
        blockers.append("reproducibility_columns_not_confirmed_in_small_metadata")
    if not have_profile_norm:
        blockers.append("profile_norm_columns_not_confirmed_in_small_metadata")
    blockers.extend(
        [
            "train_only_condition_join_not_yet_run",
            "within_dataset_shuffle_control_not_yet_run",
            "source_plate_batch_background_controls_not_yet_run",
            "mmd_tail_noharm_veto_not_yet_run",
            "chemical_v2_exact_ack_absent",
        ]
    )

    payload = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "small_metadata_materialized_join_gate_ready_no_gpu" if can_join else "small_metadata_materializer_blocked_no_gpu",
        "gpu_authorized": False,
        "chemical_v2_authorized": False,
        "can_enter_trainonly_join_controls_gate": can_join,
        "download_cap_bytes": MAX_DOWNLOAD_BYTES,
        "downloaded_files": downloaded,
        "schema_rows": schema_rows,
        "global_category_hits": global_hits,
        "s3_listings": s3_rows,
        "blockers": blockers,
        "notes": [
            "profile_index.json is treated as a manifest only; it does not authorize profile matrix download",
            "activity/reproducibility/profile_norm fields were searched by schema terms only",
        ],
    }

    write_tsv(
        OUT_MANIFEST,
        downloaded,
        ["logical_name", "source_group", "rel_path", "url", "downloaded", "status", "content_length", "size_bytes", "sha256", "local_path", "error"],
    )
    write_tsv(
        OUT_SCHEMA,
        schema_rows,
        ["logical_name", "path", "readable", "format", "field_count", "sample_rows", "fields", "category_hits", "error"],
    )
    write_tsv(
        OUT_SOURCE_LISTING,
        s3_rows,
        ["name", "url", "status", "bytes_read", "prefixes", "contents", "error"],
    )
    OUT_BLOCKERS.write_text(json.dumps(blockers, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_report(payload), encoding="utf-8")

    print(json.dumps({"status": payload["status"], "can_join": can_join, "gpu_authorized": False, "out_json": str(OUT_JSON)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.exit(main())
