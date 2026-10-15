#!/bin/bash
#SBATCH --job-name=genvectors-cc3
#SBATCH --partition=cpu-e-quick,gpu-ef-quick,rocky
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=logs/cc3-%A_%a.out
#SBATCH --error=logs/cc3-%A_%a.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
source .env 2>/dev/null || true
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$ROOT/work/tmp/inductor-cache TRITON_CACHE_DIR=$ROOT/work/tmp/triton-cache TORCHINDUCTOR_FX_GRAPH_CACHE=1 TORCHINDUCTOR_AUTOGRAD_CACHE=1
mkdir -p logs
PY=.venv/bin/python; [ -f .venv-fa/READY ] && PY=.venv-fa/bin/python
CC=$($PY -c "import torch; print(torch.cuda.get_device_capability()[0])" 2>/dev/null || echo 8)
if [ "$CC" -lt 8 ]; then [ -f .venv-v100/READY ] && PY=.venv-v100/bin/python || { echo "V100 but .venv-v100 not ready" >&2; exit 1; }; fi
echo "GPU compute capability $CC -> $PY" >&2
# shellcheck disable=SC2086
srun $PY scripts/convcurve3.py --model "${MODEL:?}" --presentation "${PRES:?}" \
    --shard "${SLURM_ARRAY_TASK_ID:?}" --shards "${SHARDS:?}" --deadline-min "${DEADLINE:-36}" ${ALL_LAYERS:+--all-layers} ${EXTRA:-} \
    || { echo "convcurve3 failed" >&2; exit 1; }
