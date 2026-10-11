# exp_suspicious_tests

This experiment tests whether the `exp_gendata` concept vectors connect to the model's own computation, not only to text about the concepts.
It scores every one of the 1036 concept vectors and ranks target concepts against that battery in three assays: MCQA uncertainty (Qwen3-8B instruct and base, gemma-2-9b-it, Llama-3.1-8B), a trashtalk severity sweep (same instruct models), and a head-by-vector attention map on lmsys transcripts (Qwen3-8B).
It also produces the vector-geometry figures (cosine distributions, participation ratio, basis modes).

## Run

```bash
uv sync                                     # or scripts/deploy.sh: sync to the cluster and run uv sync there
python -m exp_suspicious_tests.prepare_mmlu    # MMLU subset + crafted items -> data/mcqa.parquet
python -m exp_suspicious_tests.prepare_lmsys   # stratified lmsys items -> data/lmsys_items.parquet
sbatch slurm/pilot15.sbatch                 # MCQA format and entropy pilot
sbatch slurm/exp2.sbatch                    # MCQA sweep, Qwen3-8B instruct and base (array)
sbatch slurm/exp3-models.sbatch             # trashtalk transcripts and projections, gemma and Llama (array)
sbatch slurm/exp4.sbatch                    # attention-vector rho tensors, 16 chunks (array)
sbatch slurm/report.sbatch                  # reports and figures; EXPERIMENTS picks exp1/exp2/exp3
```

Jobs call the `[project.scripts]` commands in `.venv/bin/`: `suspicious-generate`, `suspicious-measure`, `suspicious-mcqa`, `suspicious-exp4`, `suspicious-report`.

## Pipeline

| Step | Module | Job |
| - | - | - |
| Build MCQA and lmsys item sets | `prepare_mmlu`, `prepare_lmsys` | — |
| MCQA pilot (format, entropy) | `mcqa` | `slurm/pilot15.sbatch` |
| MCQA entropy vs projection sweep | `mcqa` | `slurm/exp2.sbatch`, `slurm/exp2-models.sbatch` |
| Generate trashtalk transcripts | `generate` | `slurm/exp1-generate.sbatch`, `slurm/prefills.sbatch` |
| Project transcripts onto vectors | `measure` | `slurm/exp1-measure.sbatch`, `slurm/exp3-models.sbatch`, `slurm/exp3-prefilled.sbatch` |
| Trashtalk severity plots | inline | `slurm/exp3-compare.sbatch`, `slurm/exp3-monotone.sbatch`, `slurm/exp3-qwen-plot.sbatch` |
| Attention-vector rho tensors | `exp4` | `slurm/pilot35.sbatch`, `slurm/exp4.sbatch` |
| Inspect rho tensors | inline | `slurm/exp4-*.sbatch` |
| Reports and figures | `report` | `slurm/report.sbatch` |
| Vector geometry and basis modes | inline | `slurm/vector-*.sbatch`, `slurm/cosine-*.sbatch`, `slurm/basis-*.sbatch` |
| Paper figures | inline | `slurm/paper-*.sbatch` |

## Inputs and outputs

- Inputs: `../exp_gendata/work/vectors/` (diff and whitened_diff per model and presentation), `../exp_gendata/data/semantic.ontology.json`.
- Inputs: `data/trashtalk_messages.json`, `data/mcqa_extra.json`, `data/lmsys_labels.parquet`.
- Outputs: `data/mcqa.parquet`, `data/lmsys_items.parquet`, `data/prefills-deepseek.json`.
- Outputs: `work/exp2/` — per-model MCQA entropies and projections.
- Outputs: `work/exp3/` — trashtalk transcripts and per-model projections.
- Outputs: `work/exp4/` — per-item transcripts and rho tensors [blocks, heads, vectors] per query position.
- Outputs: `work/figures/`, `work/basis/` — report figures, vector-geometry figures, basis modes.
