# exp_agentic

Records Qwen3-8B running multi-turn coding workloads in a sandbox and reads every concept at every turn.
Episodes are saved token by token, optionally with a steering vector added at one layer, and then replayed for concept readouts.
This is the predecessor of `exp_reward_hack`.

## Run

```bash
uv sync
QUEUE=<jobfile> scripts/runner.sh                      # record episodes on local GPUs
python -m exp_agentic.agent --workload impossible_tests --variant base --seed 0   # one episode
sbatch slurm/turn.sbatch                               # per-turn readout + turndose
```

The sandbox image `neuralprobes-agentic:1` is built from `scripts/Dockerfile`.
`slurm/turn.sbatch` uses the environment in `../exp_lmsys_probe/.venv`; logs go to `logs/`.

## Pipeline

| Step | Module | Job |
| - | - | - |
| Run the coding sandbox | `sandbox` | `scripts/Dockerfile` |
| Record episodes | `agent`, `runmany` | `scripts/runner.sh`, `scripts/impossible.sh`, `scripts/stage1.sh` |
| Record steered episodes | `agent` (`--alpha`, `--steer-layer`) | `scripts/stage2.sh`, `scripts/dispatch.sh`, `scripts/controls.sh` |
| Score concepts per token | `readout` | — |
| Read concepts per turn | `turnread` | `slurm/turn.sbatch` |
| Find concepts that move monotonically across turns | `turndose` | `slurm/turn.sbatch` |
| Story vectors and gradients | `extract`, `storyvec`, `storygrad`, `bipo`, `mergegrads` | — |
| Analyses and reports | `rank`, `predict`, `families`, `decompose`, `survey`, `report`, `viewer`, others | — |
| Paper figure | `paper` | — |

## Inputs and outputs

- Inputs: workloads in `src/exp_agentic/workloads/`; `../exp_gendata/work/vectors/Qwen_Qwen3-8B--assistant--semantic/diff.safetensors`; `../exp_lmsys_probe/data/features.json`.
- Outputs: `episodes/<run>/` (episode JSON and logs); `turnwork/*.npz` (per-turn readouts) and `turnwork/turndose.json`.
