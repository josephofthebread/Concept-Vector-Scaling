# exp_dose

Tests which concept readouts change monotonically with medication dose, in the style of Anthropic's emotion-vector "Tylenol" experiment.
It reads all 1036 diff vectors plus 512 random controls at the assistant token, at every layer, for Qwen3-8B, Llama-3.1-8B-Instruct and gemma-2-9b-it.
It ranks concepts against the random null and draws dose-response figures.

## Run

```bash
sbatch slurm/dose.sbatch                                          # readout, array 0-2 (one model each)
PYTHONPATH=src python -m exp_dose.dose_analyze --readout results/dose_<tag>.npz   # rank concepts per model
PYTHONPATH=src python -m exp_dose.dose_fig                        # cross-model figures
PYTHONPATH=src python -m exp_dose.dose_paperfig results results/dose-response.pdf # paper figure
```

The job uses the environment in `../exp_lmsys_probe/.venv`.
Logs go to `logs/`.

## Pipeline

| Step | Module | Job |
| - | - | - |
| Read concepts on four dose ladders | `dose_readout` | `slurm/dose.sbatch` |
| Rank concepts by dose trend vs controls | `dose_analyze` | — |
| Cross-model figures | `dose_fig` | — |
| Paper figure | `dose_paperfig` | — |

## Inputs and outputs

- Inputs: `../exp_genvectors/work/vectors/`, `../exp_lmsys_probe/data/features.json`.
- Outputs: `results/dose_<tag>.npz` (readouts), `results/ranking_<tag>.json` (concept ranking), `results/dose_cross.png`, `results/dose_common.png`, `results/dose-response.pdf` and `.png`.
