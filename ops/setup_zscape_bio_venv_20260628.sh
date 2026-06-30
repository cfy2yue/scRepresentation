#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
BASE_PYTHON="${BASE_PYTHON:-/data/cyx/software/miniconda3/envs/scdfm/bin/python}"
VENV="${ZSCAPE_BIO_VENV:-${ROOT}/.venvs/zscape_bio_20260628}"
OUT_DIR="${ROOT}/reports/zscape_bio_env_20260628"

mkdir -p "${OUT_DIR}" "$(dirname "${VENV}")"

if [[ ! -x "${VENV}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv "${VENV}"
fi

"${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV}/bin/python" -m pip install \
  requests==2.32.3 \
  gprofiler-official==1.0.0 \
  gseapy==1.1.9 \
  pandas==2.2.2 \
  numpy==1.26.4 \
  scipy==1.13.1 \
  statsmodels==0.14.2

"${VENV}/bin/python" -m pip freeze | sort > "${OUT_DIR}/pip_freeze.txt"

"${VENV}/bin/python" - <<'PY'
import importlib

mods = [
    "requests",
    "gprofiler",
    "gseapy",
    "pandas",
    "numpy",
    "scipy",
    "statsmodels",
]
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{mod}: {exc}")
if missing:
    raise SystemExit("missing modules:\n" + "\n".join(missing))
print("zscape_bio_venv_ok")
PY

cat > "${OUT_DIR}/LATENTFM_ZSCAPE_BIO_VENV_20260628.md" <<EOF
# LatentFM ZSCAPE Bioanalysis Venv

Timestamp: \`$(date '+%F %T %Z')\`

Status: \`ready\`

## Boundary

- Dedicated Python virtual environment for ZSCAPE pathway/enrichment and
  downstream biological interpretation utilities.
- Matrix-level ZSCAPE expression preprocessing still uses the project scdfm
  conda environment, because it already contains scanpy/anndata/sklearn.
- This environment does not authorize model training, GPU use, checkpoint
  selection, canonical multi selection, or Track C query use.

## Environment

- base python: \`${BASE_PYTHON}\`
- venv: \`${VENV}\`
- frozen package list: \`${OUT_DIR}/pip_freeze.txt\`

## Intended Use

- g:Profiler ORA over Danio rerio Ensembl IDs with custom selected-gene
  background and recorded database timestamp/version.
- Optional GSEA/ranked-list utilities after a ranked zebrafish gene-set
  source is frozen.
- QC/log1p sensitivity summaries that consume already-frozen expression-space
  outputs rather than silently renormalizing them.
EOF

echo "${VENV}"
