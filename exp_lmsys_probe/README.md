# exp_lmsys_probe

Reads every concept direction over real text (lmsys conversations, openwebtext, alpaca) and finds where each concept is most active.
The readout runs on `Qwen/Qwen3-8B` by default, with gemma and llama as alternatives, and gpt-oss-120b judges the selected examples.
It produces per-concept candidate examples, judge verdicts, rendered token heatmaps and the supplementary HTML viewer.

`examples.html` is that viewer with the judge-accepted probing examples for Qwen3-8B, gemma-2-9b and Llama-3.1-8B.
It is a single self-contained file: download it and open it in a browser.

## Run

```bash
uv sync
scripts/deploy.sh run                         # sync, then chain readout -> select -> judge -> merge
sbatch slurm/readout.sbatch                   # concept readout over a corpus (CORPUS, MODEL, OUT)
sbatch slurm/render.sbatch                    # render token heatmaps as SVGs
python scripts/build_suppl.py --model qwen    # build the supplementary HTML into suppl/
```

## Pipeline

| Step | Module | Job |
| - | - | - |
| Read concepts over a corpus | `readout` (`lmsys-readout`) | `slurm/readout.sbatch` |
| Select candidate examples | `select` (`lmsys-select`) | `slurm/select.sbatch` |
| Judge candidates | `judge` (`lmsys-judge`) | `slurm/judge.sbatch` |
| Merge judge shards | `judge --merge` | `slurm/merge.sbatch` |
| Per-token highlight readout | `select_examples`, `hl_readout` | `slurm/sweep_readout.sbatch`, `slurm/hl_readout.sbatch` |
| Span judge (pilot and sweep) | `span_judge` | `slurm/span_pilot.sbatch`, `slurm/sweep_judge.sbatch`, `slurm/sweep_merge.sbatch` |
| Tagged study | `study_sample`, `tag_readout`, `judge --study`, `study_report` | `slurm/study_readout.sbatch`, `slurm/study_judge.sbatch` |
| Outcome summary and figures | `outcome`, `render` | `slurm/figures.sbatch` |
| Render heatmap SVGs | `render` | `slurm/render.sbatch` |
| Highlighted paper examples | `hl_examples` | `slurm/hl.sbatch` |
| Viewer data and supplementary HTML | `viewer_prep`, `scripts/build_suppl.py` | - |
| Fetch source texts of fixed examples | `scripts/fetch_examples.py` | - |

## Inputs and outputs

- Inputs: concept vectors in `../exp_gendata/work/vectors/`, concept pairs in `data/features.json`, corpora and models from the Hugging Face cache.
- Outputs: `work/` (with `work_owt/` and `work_alpaca/` for the other corpora) holds readout shards (`cand-*.safetensors`, `ids-*.json`), `candidates.json`, `judged.json`, highlight and span-judge files, `outcome.json` and figures.
- Outputs: `work/PROBING_LMSYS_1/` holds the heatmap SVGs; `suppl/` holds `concept_pairs_*.html`.
