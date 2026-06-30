#!/usr/bin/env bash
# Fail-fast conda activation for SLURM / batch jobs (no silent `|| true`).
# Usage (from repo):  source "$(dirname "$0")/conda_failfast.sh"  # if path known
# Or:  source "$ROOT/model/env/conda_failfast.sh"
_raw_fm_conda_activate() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH" >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${RAW_CONDA_ENV:-scdfm}"
}
