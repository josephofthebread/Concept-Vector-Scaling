#!/bin/bash
#SBATCH --job-name=genvectors-convcurve3-test
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00
#SBATCH --output=logs/convcurve3-test-%j.out
#SBATCH --error=logs/convcurve3-test-%j.err
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
srun $PY scripts/convcurve3.py --model "${MODEL:?}" --reverse --deadline-min 26 --presentation "${PRES:-assistant}" ${ALL_LAYERS:+--all-layers} || { echo "convcurve3 failed" >&2; exit 1; }
