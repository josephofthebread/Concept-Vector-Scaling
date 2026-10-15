# exp_attn_interp

This experiment asks which attention heads and MLPs read and build each concept diff vector, and how that relates to probing and steering.
It runs on Qwen3-8B (main), Llama-3.1-8B-Instruct and gemma-2-9b-it, with the assistant-semantic diff vectors, lmsys transcripts and feature stories.
It produces head-by-vector couplings, per-component decompositions of the vectors, ablation and steering results, and paper figures.

## Run

```bash
uv sync                                     # or scripts/deploy.sh: sync to the cluster and run uv sync there
sbatch slurm/markers.sbatch                 # per-concept marker tokens
sbatch slurm/measure.sbatch                 # head-by-vector coupling on lmsys transcripts
sbatch slurm/decomp-all.sbatch              # per-head and per-MLP decomposition of each vector
sbatch slurm/posteer.sbatch                 # steering arms with modified vectors
CMD="..." sbatch slurm/chunk.sbatch         # run any command as a one-off job
python -m exp_attn_interp.<module> --help   # every module; most also have an attn-<module> command
```

Jobs call the `[project.scripts]` commands in `.venv/bin/` (`attn-measure`, `attn-decomp`, `attn-posteer`, ...).

## Pipeline

| Step | Module | Job |
| - | - | - |
| Build marker token sets | `presence`, `markers`, `uniqmarkers` | `slurm/markers.sbatch` |
| Check concept resolution of projections | `ceiling` | `slurm/ceiling.sbatch` |
| Measure and analyze head-by-vector coupling | `measure`, `analyze` | `slurm/measure.sbatch`, `slurm/analyze.sbatch` |
| Per-head write onto vectors | `write`, `writean`, `orthovec` | `slurm/write*.sbatch` |
| Decompose vectors into head and MLP parts | `decomp`, `decompan`, `vdecomp`, `xdecomp` | `slurm/decomp-*.sbatch`, `slurm/vdecomp*.sbatch`, `slurm/xdecomp.sbatch` |
| Ablate and amplify heads | `ablate`, `amp` | `slurm/ablate.sbatch`, `slurm/amp.sbatch` |
| MLP gain, attention sources, logit lens | `mlpgain`, `attnsrc`, `lens2`, `neurlens` | `slurm/mlpgain*.sbatch`, `slurm/attnsrc.sbatch`, `slurm/lens2*.sbatch` |
| Inject vectors and trace generation | `inject`, `gentraj`, `reach`, `gensurv`, `gensurv2` | `slurm/inject*.sbatch`, `slurm/gentraj*.sbatch`, `slurm/reach*.sbatch`, `slurm/gensurv*.sbatch` |
| Steering with modified vectors | `posteer`, `steerabl` | `slurm/posteer*.sbatch`, `slurm/steer*.sbatch` |
| Judge steering arms | `../exp_trigger_prompts` judge | `slurm/pojudge*.sbatch` |
| SAE feature decoding and labels | `saedecode`, `saelabel` | `slurm/sae-*.sbatch` |
| Ablations and follow-ups | various modules and `scripts/*.py` | `slurm/*` |
| Paper figures | `scripts/paperfigs/`, `scripts/figs_cdce.py` | `slurm/figs-cdce.sbatch` |

## Inputs and outputs

- Inputs: `../exp_genvectors/work/vectors/<model>--assistant--semantic/diff.safetensors`, `../exp_genvectors/data/semantic.ontology.json`, stories in `../exp_genvectors/data/semantic.parquet`.
- Inputs: `../exp_suspicious_tests/data/lmsys_items.parquet` (lmsys items).
- Outputs: `data/markers_presence.json`, `data/markers_logitlens.json` — marker token sets.
- Outputs: `work/` (Qwen3-8B) and `work-<model>/` (other models) — one subfolder or JSON per module, e.g. `work/measure/`, `work/decomp/`, `work/posteer/`, `work/pojudge/`.
- Outputs: `figures/` — module figures; `scripts/paperfigs/figdata/` — data behind the paper figures.
