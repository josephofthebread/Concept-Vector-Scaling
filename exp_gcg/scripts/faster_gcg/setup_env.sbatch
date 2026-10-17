#!/bin/bash
#SBATCH --job-name=faster-gcg-setup
#SBATCH --account=proj_1887
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup-%j.log
#SBATCH --error=logs/setup-%j.err

set -euo pipefail
cd "${FASTER_GCG:?set FASTER_GCG to a Faster-GCG checkout with faster-gcg.patch applied}" || exit 1
module purge
module load Python/PyTorch_GPU_v2.4
.venv/bin/python -m pip install --no-index --find-links wheelhouse transformers==5.5.0 pydantic==2.12.5 python-dotenv==1.2.1 sentencepiece==0.2.0 openai==2.8.1 accelerate==1.12.0
.venv/bin/python -c "import accelerate, datasets, dotenv, pydantic, sentencepiece, torch, transformers; import tools.runner; assert hasattr(transformers, \"Qwen3Config\") and hasattr(transformers, \"Gemma2Config\"); print(torch.__version__, transformers.__version__, accelerate.__version__)"
