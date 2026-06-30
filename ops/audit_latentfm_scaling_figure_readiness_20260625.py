#!/usr/bin/env python3
"""Audit scaling figure readiness and provenance consistency.

Short CPU/report task. Reads completed PNG/SVG figure files and the provenance
manifest only. Does not read checkpoints, canonical multi, Track C query,
train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "scaling_figures_20260625"
PROV_TSV = REPORTS / "scaling_nm_provenance_manifest_20260625/artifact_manifest.tsv"
OUT_DIR = REPORTS / "scaling_figure_readiness_20260625"
OUT_CSV = OUT_DIR / "figure_readiness.csv"
OUT_JSON = REPORTS / "latentfm_scaling_figure_readiness_20260625.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_FIGURE_READINESS_20260625.md"

FIGURE_STEMS = [
    "FigS_scaling_S0_provenance",
    "Fig_scaling_truecell_budget",
    "Fig_scaling_exposure_nonmonotonic",
    "Fig_scaling_noharm_veto",
    "FigS_scaling_failure_map",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    if not PROV_TSV.is_file():
        return hashes
    with PROV_TSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            hashes[row["path"]] = row["sha256"]
    return hashes


def audit_png(path: Path, manifest_hashes: dict[str, str]) -> dict[str, Any]:
    rel = str(path.relative_to(ROOT))
    row: dict[str, Any] = {
        "path": rel,
        "format": "png",
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256(path) if path.is_file() else "",
        "manifest_sha256": manifest_hashes.get(rel, ""),
        "hash_matches_manifest": False,
        "width": 0,
        "height": 0,
        "pixel_std_mean": 0.0,
        "nonblank": False,
        "pass": False,
        "reason": "",
    }
    if not path.is_file():
        row["reason"] = "missing_png"
        return row
    row["hash_matches_manifest"] = row["sha256"] == row["manifest_sha256"]
    with Image.open(path) as img:
        img = img.convert("RGB")
        row["width"], row["height"] = img.size
        stat = ImageStat.Stat(img)
        row["pixel_std_mean"] = float(sum(stat.stddev) / len(stat.stddev))
        row["nonblank"] = row["pixel_std_mean"] > 1.0
    reasons = []
    if row["width"] < 800 or row["height"] < 500:
        reasons.append("small_dimensions")
    if not row["nonblank"]:
        reasons.append("blank_or_near_blank")
    if not row["hash_matches_manifest"]:
        reasons.append("manifest_hash_mismatch")
    row["pass"] = not reasons
    row["reason"] = ";".join(reasons) if reasons else "pass"
    return row


def audit_svg(path: Path, manifest_hashes: dict[str, str]) -> dict[str, Any]:
    rel = str(path.relative_to(ROOT))
    exists = path.is_file()
    text = path.read_text(encoding="utf-8", errors="replace") if exists else ""
    draw_tokens = sum(text.count(tok) for tok in ("<path", "<line", "<rect", "<circle", "<text", "<g"))
    row: dict[str, Any] = {
        "path": rel,
        "format": "svg",
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "sha256": sha256(path) if exists else "",
        "manifest_sha256": manifest_hashes.get(rel, ""),
        "hash_matches_manifest": False,
        "width": "",
        "height": "",
        "pixel_std_mean": "",
        "nonblank": draw_tokens > 5,
        "pass": False,
        "reason": "",
    }
    if not exists:
        row["reason"] = "missing_svg"
        return row
    row["hash_matches_manifest"] = row["sha256"] == row["manifest_sha256"]
    reasons = []
    if path.stat().st_size < 1024:
        reasons.append("too_small_svg")
    if not row["nonblank"]:
        reasons.append("few_svg_draw_tokens")
    if not row["hash_matches_manifest"]:
        reasons.append("manifest_hash_mismatch")
    row["pass"] = not reasons
    row["reason"] = ";".join(reasons) if reasons else "pass"
    return row


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "path",
        "format",
        "exists",
        "size_bytes",
        "sha256",
        "manifest_sha256",
        "hash_matches_manifest",
        "width",
        "height",
        "pixel_std_mean",
        "nonblank",
        "pass",
        "reason",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Scaling Figure Readiness",
        "",
        "Timestamp: `2026-06-25 23:42 CST`",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only QA of completed scaling PNG/SVG figures.",
        "- Checks dimensions, nonblank signal, SVG draw tokens, and provenance-manifest SHA256 consistency.",
        "- Does not read checkpoints, canonical multi, Track C query, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- figure files audited: `{payload['summary']['n_files']}`",
        f"- passed: `{payload['summary']['n_pass']}`",
        f"- failed: `{payload['summary']['n_fail']}`",
        f"- manifest hash matches: `{payload['summary']['n_hash_match']}`",
        "",
        "## Rows",
        "",
        "| figure | format | size | dimensions | hash match | pass | reason |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in payload["rows"]:
        dims = f"{row['width']}x{row['height']}" if row["format"] == "png" else "svg"
        lines.append(
            f"| `{Path(row['path']).name}` | `{row['format']}` | {row['size_bytes']} | "
            f"{dims} | `{row['hash_matches_manifest']}` | `{row['pass']}` | {row['reason']} |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        f"- CSV: `{OUT_CSV}`",
        f"- JSON: `{OUT_JSON}`",
        "",
        "## Decision",
        "",
        "Figure QA is provenance/readiness support only. It does not authorize GPU training.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    manifest_hashes = load_manifest_hashes()
    rows: list[dict[str, Any]] = []
    for stem in FIGURE_STEMS:
        rows.append(audit_png(FIG_DIR / f"{stem}.png", manifest_hashes))
        rows.append(audit_svg(FIG_DIR / f"{stem}.svg", manifest_hashes))
    summary = {
        "n_files": len(rows),
        "n_pass": sum(1 for r in rows if r["pass"]),
        "n_fail": sum(1 for r in rows if not r["pass"]),
        "n_hash_match": sum(1 for r in rows if r["hash_matches_manifest"]),
    }
    payload = {
        "status": "scaling_figure_readiness_pass_no_gpu" if summary["n_fail"] == 0 else "scaling_figure_readiness_fail_no_gpu",
        "gpu_authorized": False,
        "summary": summary,
        "rows": rows,
        "outputs": {"csv": str(OUT_CSV), "json": str(OUT_JSON), "md": str(OUT_MD)},
    }
    write_csv(rows)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "gpu_authorized": False, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
