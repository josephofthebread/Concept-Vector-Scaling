# exp_reward_hack

Runs coding agents on nine impossible coding tasks and reads every concept at every turn of each episode.
Three models (`Qwen/Qwen3-8B`, `unsloth/Meta-Llama-3.1-8B-Instruct`, `google/gemma-2-9b-it`) work through the battery in `src/exp_reward_hack/workloads/` (see its README).
Each episode is recorded, labelled cheat or clean from the trajectory alone, and replayed through HF to produce per-turn and per-token concept readouts.

## Run

```bash
uv sync
python -m exp_reward_hack.validate         # check the workload battery
sbatch slurm/sweep.sbatch                  # full episode sweep (vLLM)
sbatch slurm/label.sbatch                  # label episodes cheat / clean
sbatch slurm/readout.sbatch                # per-turn concept readout
```

Generation and labelling use the vLLM environment in `.venv-vllm`.
The readout jobs use `../exp_lmsys_probe/.venv`.

## Pipeline

| Step | Module | Job |
| - | - | - |
| Validate the battery | `validate`, `sandbox` | - |
| Generate episodes | `runmany` (uses `agent`, `vllm_gen`, `dialects`) | `slurm/sweep.sbatch` |
| Summarize episodes | `report_smoke` | - |
| Label cheat / clean | `label` | `slurm/label.sbatch` |
| Per-turn readout | `turnread` | `slurm/readout.sbatch` |
| Per-CoT-token readout | `turnread_tok` | `slurm/readout_tok.sbatch` |
| Token-distribution AUROC | `tokauroc`, `hist_auroc` | `slurm/tokauroc.sbatch`, `slurm/hist.sbatch` |
| Paper figure | `scripts/paperfigs/agent_fig.py` | - |

## Inputs and outputs

- Inputs: concept vectors in `../exp_genvectors/work/vectors/`, concept pairs in `../exp_lmsys_probe/data/features.json`, tasks in `src/exp_reward_hack/workloads/`.
- Outputs: `episodes/` holds episode JSON files; `labels.json` holds the cheat / clean labels.
- Outputs: `turnwork/` and `turnwork_tok/` hold per-episode readouts (`.npz`); `xresults/` holds `tokauroc.json`, the AUROC histogram and the agent-turns figure.
