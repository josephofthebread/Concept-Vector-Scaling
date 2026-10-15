# exp_genvectors

Builds concept vectors for the 1036 concept pairs (148 classes) of `AntonKorznikov/feature_stories` across six checkpoints (Qwen3-8B, gemma-2-9b and Llama-3.1-8B, base and instruct), several presentation formats and every layer.
It stores per-layer activation sums rather than raw activations, so new vector definitions need no GPU time.
Downstream analyses read these sums for convergence, per-story drift, cross-concept geometry and lexical effects.

## Run

```bash
uv sync
.venv/bin/genvectors-submit             # chain prepare -> extract array -> vectors array
sbatch slurm/convergence.sbatch      # split-half convergence of each definition
python -m exp_genvectors.vectors work/captures/<tag>/shard-*.safetensors --ontology data/semantic.ontology.json --out work/vectors/<tag>
```

Jobs use the `[project.scripts]` commands in `.venv/bin/` (`genvectors-prepare`, `genvectors-extract`, `genvectors-vectors`, ...) and source a `.env` file.
`scripts/deploy.sh [sync]` syncs the folder to the cluster.

## Pipeline

| Step | Module | Job |
| - | - | - |
| Write datasets, cache checkpoints | `prepare`, `cache_models` | `slurm/prepare.sbatch` |
| Capture per-layer pole, cell and moment sums | `extract` | `slurm/extract.sbatch` |
| Build raw and whitened vectors | `vectors` | `slurm/vectors.sbatch` |
| Submit the three steps above | `submit` | (`genvectors-submit`) |
| Split-half convergence | `convergence` | `slurm/convergence*.sbatch` |
| Per-story vectors and PCA | `perstory`, `warmth_pca` | `slurm/perstory*.sbatch` |
| Language-axis scores | `langscores` | `slurm/langscores*.sbatch` |
| Paired, named and declared variants | `extract`, `vectors` | `slurm/paired-*`, `slurm/naming-control*`, `slurm/decl-*` |
| Learning curves | `scripts/convcurve*.py` | `slurm/convcurve*.sbatch` |
| Out-of-distribution checks | `scripts/ood*.py` | `slurm/ood*.sbatch` |
| Lexical and logit-lens checks | `scripts/lex*.py`, `scripts/logitlens_*.py` | `slurm/lex*`, `slurm/logitlens_*` |
| Other one-off analyses | inline or `scripts/` | `slurm/*` |
| Paper figures | `scripts/paperfigs/` | `slurm/paperfigs-convergence.sbatch` and others |

## Inputs and outputs

- Inputs: the `AntonKorznikov/feature_stories` dataset and ontology, Hugging Face checkpoints, `data/mask_concepts.json`.
- Outputs: `data/semantic.parquet` and `data/syntax.parquet` with their `.ontology.json`; `jobs/*.tsv` task grids; `work/captures/<model>--<format>--<dataset>/shard-NNN.safetensors`; `work/vectors/<tag>/` (vector definitions at every layer); `work/convergence/`, `work/perstory/`, `work/langscores/`.
- Logs go to `logs/`.
