"""Registry of embedding models: Python interpreter paths and weight preflight."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# FM package root: .../scFM/main/fm (parent of tools/)
FM_ROOT = Path(__file__).resolve().parents[1]
if str(FM_ROOT) not in sys.path:
    sys.path.insert(0, str(FM_ROOT))
import paths

SCFM_ROOT = FM_ROOT.parent
# Backwards compatibility: old code and env scripts used "latent_bench" as the adapters root.
LATENT_BENCH_ROOT = FM_ROOT

# CoupledFM pretrained assets (weights, UCE, CellNavi, etc.)
PRETRAINED_ROOT = paths.pretrained_root()

# Unified benchmark output (embeddings, logs, preflight); overrides via env if needed.
OUTPUT_ROOT = paths.output_root()
DEFAULT_EMBEDDING_EXPORT_ROOT = OUTPUT_ROOT / "embeddings"
DEFAULT_EMBEDDING_RUNS_DIR = OUTPUT_ROOT / "embedding_runs"
DEFAULT_EMBEDDING_LOGS_DIR = OUTPUT_ROOT / "logs"

ENV_NAME_BY_MODEL: Dict[str, str] = {
    "scgpt": "scdfm",
    "xverse": "scdfm",
    "scfoundation": "scdfm",
    "uce": "scdfm",
    "scldm": "scdfm",
    "stack": "scdfm",
    "geneformer": "scdfm",
    "cellnavi": "cellnavi",
    "nicheformer": "nicheformer",
    "transcriptformer": "transcriptformer",
}

# Initial GPU slot assignment (plan); actual queue may skip missing models.
MODEL_QUEUE_ORDER: List[str] = [
    "geneformer",
    "uce",
    "scfoundation",
    "cellnavi",
    "scldm",
    "stack",
    "state",
    "nicheformer",
    "transcriptformer",
    "xverse",
    "scgpt",
]


def resolve_state_python() -> str:
    venv_py = paths.third_party_root() / "state" / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return _python_default("state")


def _conda_envs_dirs() -> List[Path]:
    """Return candidate conda-style envs/ roots from the environment, no hardcoding."""
    roots: List[Path] = []
    # 1. SCFM_ENVS_ROOT (explicit project override)
    roots.append(paths.envs_root())
    # 2. From the running conda executable (CONDA_EXE=/path/to/bin/conda → /path/to/envs)
    conda_exe = os.environ.get("CONDA_EXE", "")
    if conda_exe:
        p = Path(conda_exe)
        # miniconda/bin/conda → miniconda/envs
        roots.append(p.parent.parent / "envs")
    # 3. CONDA_ENVS_PATH: colon-separated list conda uses for extra env search
    for r in os.environ.get("CONDA_ENVS_PATH", "").split(os.pathsep):
        if r.strip():
            roots.append(Path(r.strip()))
    # 4. SCFM_CONDA_ROOT: explicit escape hatch
    scfm_conda = os.environ.get("SCFM_CONDA_ROOT", "").strip()
    if scfm_conda:
        roots.append(Path(scfm_conda) / "envs")
    return roots


def _python_default(model: str) -> str:
    m = model.lower().strip()
    for key in (
        f"LATENT_BENCH_{m.upper()}_PYTHON",
        f"SCFM_{m.upper()}_PYTHON",
    ):
        v = os.environ.get(key, "").strip()
        if v:
            return v
    env_name = ENV_NAME_BY_MODEL.get(m, m)
    for root in _conda_envs_dirs():
        p = root / env_name / "bin" / "python"
        if p.is_file():
            return str(p)
    return shutil.which("python3") or shutil.which("python") or "python3"


def python_for_model(model: str) -> str:
    m = model.lower().strip()
    if m == "state":
        return resolve_state_python()
    return _python_default(m)


def check_weights(model: str) -> Tuple[str, str]:
    """Return (status, detail) where status in ready, missing, unknown."""
    m = model.lower().strip()
    if m == "scgpt":
        d = os.environ.get("LATENT_BENCH_SCGPT_MODEL_DIR", "")
        p = Path(d) if d else (PRETRAINED_ROOT / "scgpt")
        bf = p / "best_model.pt"
        if p.is_dir() and bf.is_file():
            return "ready", str(p)
        return "missing", f"scGPT dir missing best_model.pt: {p} (set LATENT_BENCH_SCGPT_MODEL_DIR if elsewhere)"
    if m == "geneformer":
        p = PRETRAINED_ROOT / "geneformer" / "Geneformer-V2-316M"
        if p.is_dir():
            return "ready", str(p)
        return "missing", str(p)
    if m == "uce":
        uce_root = Path(os.environ.get("COUPLEDFM_UCE_ROOT", str(PRETRAINED_ROOT / "uce")))
        ckpt = uce_root / "model_files" / "33layer_model.torch"
        if ckpt.is_file():
            return "ready", str(ckpt)
        return "missing", f"UCE ckpt not found: {ckpt}"
    if m == "state":
        ck = os.environ.get("LATENT_BENCH_STATE_CKPT", "")
        if ck and Path(ck).is_file():
            return "ready", str(ck)
        cand = PRETRAINED_ROOT / "state" / "SE-600M"
        if cand.is_dir():
            for pat in ("*.ckpt", "*.safetensors"):
                hits = list(cand.glob(pat))
                if hits:
                    return "ready", str(hits[0])
        return "missing", "LATENT_BENCH_STATE_CKPT unset or no file in pretrained/state/SE-600M"
    if m == "stack":
        ck = os.environ.get("LATENT_BENCH_STACK_CKPT", str(PRETRAINED_ROOT / "stack" / "bc_large.ckpt"))
        gl = os.environ.get(
            "LATENT_BENCH_STACK_GENELIST",
            str(PRETRAINED_ROOT / "stack" / "basecount_1000per_15000max.pkl"),
        )
        if Path(ck).is_file() and Path(gl).is_file():
            return "ready", str(ck)
        return "missing", f"stack ckpt or genelist missing: {ck}, {gl}"
    if m == "scldm":
        _sd = PRETRAINED_ROOT / "scdlm" / "vae_census"
        ckpt = Path(os.environ.get("LATENT_BENCH_SCLDM_CKPT", str(_sd / "70M.ckpt")))
        cfg = Path(os.environ.get("LATENT_BENCH_SCLDM_CFG", str(_sd / "70M.yaml")))
        parquet = Path(os.environ.get("LATENT_BENCH_SCLDM_GENES", str(_sd / "concatenated_unique_genes.parquet")))
        if ckpt.is_file() and cfg.is_file() and parquet.is_file():
            return "ready", str(ckpt)
        return "missing", f"scldm assets: {ckpt}, {cfg}, {parquet}"
    if m == "xverse":
        ck = os.environ.get("LATENT_BENCH_XVERSE_CKPT", str(PRETRAINED_ROOT / "xVerse" / "xVERSE_384.pth"))
        if Path(ck).is_file():
            return "ready", str(ck)
        return "missing", str(ck)
    if m == "cellnavi":
        ck = os.environ.get("LATENT_BENCH_CELLNAVI_CKPT", "")
        g = PRETRAINED_ROOT / "cellnavi" / "data" / "pretrain" / "pretrain_weights.pth"
        gpkl = PRETRAINED_ROOT / "cellnavi" / "data" / "Nichenet" / "graph.pkl"
        if (ck and Path(ck).is_file()) or g.is_file():
            if gpkl.is_file():
                return "ready", ck or str(g)
            return "missing", f"CellNavi graph.pkl missing: {gpkl}"
        return "missing", f"CellNavi ckpt missing: {ck or g}"
    if m == "scfoundation":
        ck = os.environ.get("LATENT_BENCH_SCFOUNDATION_CKPT", str(PRETRAINED_ROOT / "scFoundation" / "models.ckpt"))
        tsv = os.environ.get(
            "LATENT_BENCH_SCFOUNDATION_GENE_TSV",
            str(paths.third_party_root() / "scFoundation" / "model" / "OS_scRNA_gene_index.19264.tsv"),
        )
        if Path(ck).is_file() and Path(tsv).is_file():
            return "ready", str(ck)
        return "missing", f"{ck}, {tsv}"
    if m == "nicheformer":
        ck = os.environ.get("LATENT_BENCH_NICHEFORMER_CKPT", str(PRETRAINED_ROOT / "nicheformer" / "nicheformer.ckpt"))
        hf_dir = Path(
            os.environ.get(
                "LATENT_BENCH_NICHEFORMER_HF_DIR",
                str(PRETRAINED_ROOT / "nicheformer" / "theislab_Nicheformer"),
            )
        )
        mean_h5ad = os.environ.get(
            "LATENT_BENCH_NICHEFORMER_MEAN_H5AD",
            str(paths.third_party_root() / "nicheformer" / "data" / "model_means" / "model.h5ad"),
        )
        if Path(ck).is_file() and Path(mean_h5ad).is_file():
            return "ready", str(ck)
        if (hf_dir / "config.json").is_file() and (hf_dir / "model.safetensors").is_file() and Path(mean_h5ad).is_file():
            return "ready", str(hf_dir)
        missing = []
        if not Path(ck).is_file() and not (
            (hf_dir / "config.json").is_file() and (hf_dir / "model.safetensors").is_file()
        ):
            missing.append(f"ckpt={ck} or hf_dir={hf_dir}")
        if not Path(mean_h5ad).is_file():
            missing.append(f"model_mean={mean_h5ad}")
        return "missing", "NicheFormer missing " + ", ".join(missing)
    if m == "transcriptformer":
        model = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_MODEL", "tf_sapiens").strip()
        ck = os.environ.get("LATENT_BENCH_TRANSCRIPTFORMER_CKPT", str(PRETRAINED_ROOT / "transcriptformer" / model))
        p = Path(ck)
        if (p / "config.json").is_file() and (p / "model_weights.pt").is_file() and (p / "vocabs").is_dir():
            return "ready", str(p)
        return "missing", f"TranscriptFormer checkpoint dir missing config/model/vocabs: {p}"
    return "unknown", f"unknown model {model}"


def subprocess_env(model: str, base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Merge PYTHONPATH so `import adapters` works; xVERSE needs extra path."""
    env = dict(base or os.environ)
    root = str(LATENT_BENCH_ROOT)
    xverse = str(paths.third_party_root() / "xVERSE_code")
    prev = env.get("PYTHONPATH", "")
    m = model.lower().strip()
    if m == "xverse":
        env["PYTHONPATH"] = f"{xverse}:{root}" + (f":{prev}" if prev else "")
    elif m in {"nicheformer", "transcriptformer"}:
        tp = paths.third_party_root() / m / "src"
        env["PYTHONPATH"] = f"{tp}:{root}" + (f":{prev}" if prev else "")
    else:
        env["PYTHONPATH"] = root + (f":{prev}" if prev else "")
    if m == "scgpt":
        env.setdefault("LATENT_BENCH_SCGPT_MODEL_DIR", str(PRETRAINED_ROOT / "scgpt"))
    elif m == "state":
        state_root = PRETRAINED_ROOT / "state" / "SE-600M"
        if "LATENT_BENCH_STATE_CKPT" not in env:
            for pat in ("*.ckpt", "*.safetensors"):
                hits = sorted(state_root.glob(pat))
                if hits:
                    env["LATENT_BENCH_STATE_CKPT"] = str(hits[0])
                    break
    return env


def import_smoke_cmd(model: str) -> List[str]:
    """One-liner for subprocess import check (same env as encode)."""
    m = model.lower().strip()
    if m == "scgpt":
        return ["-c", "import torch; import adapters.scgpt.encoder as e; print('ok', e.encode.__name__)"]
    mod_map = {
        "geneformer": "adapters.geneformer.encoder",
        "uce": "adapters.uce.encoder",
        "state": "adapters.state.encoder",
        "stack": "adapters.stack.encoder",
        "scldm": "adapters.scldm.encoder",
        "xverse": "adapters.xverse.encoder",
        "cellnavi": "adapters.cellnavi.encoder",
        "scfoundation": "adapters.scfoundation.encoder",
        "nicheformer": "adapters.nicheformer.encoder",
        "transcriptformer": "adapters.transcriptformer.encoder",
    }
    mm = mod_map.get(m, "")
    if not mm:
        return ["-c", "raise SystemExit('bad model')"]
    return ["-c", f"import importlib; m=importlib.import_module('{mm}'); print('ok', m.encode.__name__)"]
