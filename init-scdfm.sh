# Local runtime environment for CoupledFM + scFMBench.
# Usage: source /data/cyx/1030/scLatent/init-scdfm.sh

export SCDFM_WORKSPACE="/data/cyx/1030/scLatent"
export SCDFM_CONDA_ENV="/data/cyx/1030/software/miniconda3/envs/scdfm"

export SCDFM_DATASET_ROOT="/data/cyx/1030/dataset"
export SCDFM_PRETRAIN_ROOT="${SCDFM_WORKSPACE}/pretrainckpt"
export SCDFM_GENE_CACHE_ROOT="${SCDFM_PRETRAIN_ROOT}/genepert_cache"

export SCFM_DATA_ROOT="${SCDFM_DATASET_ROOT}/scFM_data"
export SCFM_PRETRAINED_ROOT="${SCDFM_WORKSPACE}/scFM_pretrained"
export SCFM_THIRD_PARTY_ROOT="${SCDFM_WORKSPACE}/scFM_third_party"
export SCFM_OUTPUT_ROOT="${SCDFM_WORKSPACE}/scFM_output"
export SCFM_ENVS_ROOT="${SCDFM_WORKSPACE}/.venvs"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    _scdfm_auto_gpus="$(
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
        | sort -t, -k2,2n -k3,3n \
        | head -4 \
        | cut -d, -f1 \
        | tr -d ' ' \
        | paste -sd, -
    )"
    export CUDA_VISIBLE_DEVICES="${_scdfm_auto_gpus:-0,1,2,3}"
    unset _scdfm_auto_gpus
  else
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
  fi
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-4}"
export PYTHONPATH="${SCDFM_WORKSPACE}/CoupledFM:${SCDFM_WORKSPACE}/scFMBench/fm:${PYTHONPATH:-}"

if [[ -f /data/cyx/1030/software/miniconda3/etc/profile.d/conda.sh ]]; then
  source /data/cyx/1030/software/miniconda3/etc/profile.d/conda.sh
  conda activate "${SCDFM_CONDA_ENV}"
else
  echo "conda.sh was not found; activate ${SCDFM_CONDA_ENV} manually." >&2
fi

echo "Activated scdfm workspace: ${SCDFM_WORKSPACE}"
echo "Dataset root: ${SCDFM_DATASET_ROOT}"
echo "GPU visibility: ${CUDA_VISIBLE_DEVICES}"
echo "CPU math threads: OMP=${OMP_NUM_THREADS}, MKL=${MKL_NUM_THREADS}, OPENBLAS=${OPENBLAS_NUM_THREADS}"
