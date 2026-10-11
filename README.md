# Concept Vector Scaling

Code for extracting 1036 behavioral concept vectors (148 classes) from three model families and validating them by probing real text, steering generation, and reading them out in the wild.

The three models are Qwen3-8B, gemma-2-9b and Llama-3.1-8B. Vectors come from the [`AntonKorznikov/feature_stories`](https://huggingface.co/datasets/AntonKorznikov/feature_stories) dataset.

## Layout
Each `exp_*` folder is self-contained and has the same layout:
- `README.md`: what the experiment does and how to run it.
- `pyproject.toml`: its environment (`uv sync`).
- `src/exp_<name>/`: the Python package; run modules as `python -m exp_<name>.<module>`.
- `slurm/`: job scripts.
- `scripts/` (optional): analysis scripts, paper-figure scripts (`scripts/paperfigs/`), cluster sync (`scripts/deploy.sh`).
- `data/` (optional): small inputs the code reads.

Submit jobs from the experiment folder (`sbatch slurm/<job>.sbatch`). Each script resolves its paths relative to that folder and writes logs to `logs/`.

`exp_gendata` comes first: it writes the concept vectors to `exp_gendata/work/vectors/`, and the other experiments read them from `../exp_gendata/work/vectors`.

| Experiment | What it does | Paper |
| - | - | - |
| [`exp_gendata`](exp_gendata) | Concept vectors for 6 checkpoints × presentation formats; convergence, held-out transfer, separability, shared modes, lexical content | Sec. 2, App. methodology details, ablations |
| [`exp_lmsys_probe`](exp_lmsys_probe) | Per-conversation concept readout over LMSYS / OpenWebText / Alpaca; span judging; supplementary HTML (`scripts/build_suppl.py`) | Sec. 3.1, App. probing details |
| [`exp_trigger_prompts`](exp_trigger_prompts) | Trigger prompts, baselines, steering sweeps and judging | Sec. 3.2, App. steering details |
| [`exp_attn_interp`](exp_attn_interp) | Where to probe and steer: layer profile, MLP gain, output-word readout | Ablations: where to probe and steer |
| [`exp_suspicious_tests`](exp_suspicious_tests) | Whether vectors connect to the model's own computation; participation-ratio depth figure | Ablations |
| [`exp_dose`](exp_dose) | Tylenol dose–response check | App. dose–response |
| [`exp_reward_hack`](exp_reward_hack) | Agents on impossible coding tasks | Sec. 4.1, App. agentic workloads |
| [`exp_agentic`](exp_agentic) | Original agentic harness and per-turn readout | Sec. 4.1 |

Paper-figure scripts live in each experiment's `scripts/paperfigs/`.
