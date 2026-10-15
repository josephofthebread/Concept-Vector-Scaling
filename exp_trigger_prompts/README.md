# exp_trigger_prompts

Builds a trigger-prompt dataset for all 1036 concept pairs and uses it to steer and judge three instruct models: `Qwen/Qwen3-8B`, `google/gemma-2-9b-it` and `unsloth/Meta-Llama-3.1-8B-Instruct`.
gpt-oss-120b drafts 64 prompts per pair (neutral, concept and antagonist), each model answers them greedily as a baseline, and the steering sweep is judged against those baselines.
The folder also holds the probing, lexical and lmsys follow-up analyses built on the same data.

## Run

```bash
uv sync
scripts/deploy.sh fetch                           # cache gpt-oss-120b
scripts/deploy.sh submit                          # sbatch slurm/pipeline.sbatch: prompts, then baselines
sbatch --array=0-N slurm/steer-sweep.sbatch       # steering sweep shards (set SHARDS)
SHARDS=N sbatch slurm/merge-gemma.sbatch          # merge a model's sweep shards
```

## Pipeline

| Step | Module | Job |
| - | - | - |
| Draft trigger prompts | `generate` (`trigger-generate`) | `slurm/pipeline.sbatch` |
| Generate greedy baselines | `baselines` (`trigger-baselines`) | `slurm/pipeline.sbatch` |
| Calibrate steering norms | `calibrate` (`trigger-calibrate`) | `slurm/calibrate-*.sbatch` |
| Steering sweep | `steer` (`trigger-steer`) | `slurm/steer-sweep*.sbatch`, `slurm/steer-validate.sbatch` |
| Merge sweep shards | - | `slurm/merge-gemma.sbatch`, `slurm/merge-llama.sbatch` |
| Judge steered replies | `judge` (`trigger-judge`), `judge_baseline` | `slurm/judge*.sbatch` |
| Summarize judge scores | `scripts/analyze_judge.py` | `slurm/analyze-gemma.sbatch`, `slurm/analysis-llama.sbatch` |
| Multi-concept steering | `mlgen`, `mlsteer`, `mljudge` | `slurm/mlgen.sbatch`, `slurm/mlsteer.sbatch`, `slurm/mljudge.sbatch` |
| Steering demo and scope viewer data | `steer_highlight`, `scopegen` | `slurm/steerdemo.sbatch`, `slurm/scopegen.sbatch` |
| Probing and logit analyses | `probe*`, `depthprobe*`, `logit*`, `yesno*`, `neutralize`, others | matching `slurm/<module>.sbatch` |
| lmsys, lexical and cluster follow-ups | `lmsys*`, `labellmsys`, `lex*`, `*repro`, `conceptclusters`, `umapsweep`, others | matching `slurm/<module>.sbatch` |
| Paper figures | `scripts/paperfigs/` | - |

## Inputs and outputs

- Inputs: concept pairs in `data/features.json` (with `data/ontology.json`), concept vectors in `../exp_genvectors/work/vectors/`, plus `data/inject_concepts.json`, `data/steer_demo.json` and other files in `data/`.
- Outputs: `work/prompts.jsonl` (drafted prompts) and `work/baselines-<model>.jsonl` (baseline replies).
- Outputs: `work/steer/` (norms and sweep replies per model), `work/judge/` (judge scores), and one `work/<analysis>/` folder per follow-up, e.g. `work/mlgen-<model>/`, `work/PROBING_2/`, `work/STEER_DEMO/`, `work/scope-data/`.
