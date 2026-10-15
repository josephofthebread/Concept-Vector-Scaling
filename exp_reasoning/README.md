# exp_reasoning

This folder contains the code needed to reproduce the chain-of-thought (CoT) experiment reported in Section 4.3 and Appendix E.4 of the paper. No datasets, model weights, feature vectors, generations, activations, score archives, figures, logs, or caches are included.

The experiment asks whether linear feature readouts from the layer-18 residual stream of `Qwen/Qwen3-8B` distinguish correct from incorrect long-form reasoning. It uses all 198 GPQA Diamond questions and all 500 MATH-500 problems. The published run contains 698 unsteered traces: 571 correct and 127 incorrect, including 63 traces that opened but did not close a thinking block. Figure 9 contains benchmark-balanced temporal profiles for two uncertainty-related features and the eight largest outcome gaps within a fixed set of 16 CoT features.

All generated files are written below `work/`, which is excluded by `.gitignore`.

## Contents

| File | Purpose |
|---|---|
| `pyproject.toml` | Direct runtime dependencies. |
| `src/exp_reasoning/benchmarks.py` | Loads GPQA Diamond and MATH-500, builds the exact prompts, deterministically shuffles GPQA choices, scores answers (MATH-500 with `math-verify`), defines the loop detector and thinking-span extraction from `<think>` and `</think>` tokens. |
| `src/exp_reasoning/generate.py` | Generates and saves the unsteered Qwen3-8B responses. |
| `src/exp_reasoning/readout.py` | Teacher-forced replay, layer-18 projection onto all 1,036 directions, 50-bin alignment, normalization, and NPZ export. |
| `src/exp_reasoning/stats.py` | Vector metadata, full-trace standardization and benchmark-balanced outcome comparisons, and the benchmark-stratified bootstrap. |
| `scripts/paperfigs/cot_outcomes.py` | Builds the three-panel Figure 9 and prints the selected CoT features and signed outcome gaps. |
| `slurm/generate.sbatch`, `slurm/readout.sbatch` | Cluster jobs for stages 2 and 3 below. |

## Environment and external resources

The environment targets Linux, Python 3.12, CUDA, and an NVIDIA GPU. The published jobs used 32 GB GPUs. Install [uv](https://docs.astral.sh/uv/), then run from this folder:

```bash
uv sync
```

The environment includes PyTorch, Transformers, Datasets, Safetensors, PyArrow, NumPy, pandas, Matplotlib, and `math-verify`.

Authenticate with Hugging Face and accept the GPQA dataset conditions:

```bash
hf auth login
```

The code downloads these resources automatically:

- model: [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B);
- GPQA Diamond: [`Idavidrein/gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa), configuration `gpqa_diamond`, split `train`;
- MATH-500: [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), split `test`.

The feature vectors come from `exp_genvectors` (see step 1). No external resource or computed artifact is bundled here.

## End-to-end reproduction

Run all commands from this folder. The main recipe uses eight visible GPUs. A one-GPU alternative follows.

### 1. Use the layer-18 feature vectors

The experiment uses the assistant-format `diff` vectors of Qwen3-8B built by `exp_genvectors`:

```text
../exp_genvectors/work/vectors/Qwen_Qwen3-8B--assistant--semantic/
  manifest.json
  pairs.parquet
  diff.safetensors
```

The downstream experiment uses `diff.safetensors`, `pairs.parquet`, and `manifest.json`. Layer 18 is read from the full per-layer tensor.

### 2. Generate the 698 unsteered reasoning traces

These commands generate one response per question (`sbatch slurm/generate.sbatch` runs both). Generation uses the Qwen chat template with thinking enabled, `do_sample=False`, a 16,384-token budget, and the retained loop detector. GPQA choices are shuffled with `random.Random(question_index)`. MATH is scored with `math-verify`; a generation without a closed thinking block has no final answer and receives `correct=false`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run python -m exp_reasoning.generate \
  --benchmark gpqa_diamond \
  --num-workers 8 \
  --batch-size 4 \
  --max-new-tokens 16384 \
  --results-dir work/results

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run python -m exp_reasoning.generate \
  --benchmark math_500 \
  --num-workers 8 \
  --batch-size 4 \
  --max-new-tokens 16384 \
  --results-dir work/results
```

The program appends records and resumes after interruption. Worker JSONL files are merged into `work/results/generations.jsonl`. Expected paper-run accounting:

| Benchmark | Traces | Correct | Incorrect | Closed thinking | Unclosed thinking |
|---|---:|---:|---:|---:|---:|
| GPQA Diamond | 198 | 104 | 94 | 159 | 39 |
| MATH-500 | 500 | 467 | 33 | 476 | 24 |
| Combined | 698 | 571 | 127 | 635 | 63 |

Exact text can change if an upstream revision or GPU numerical path changes; see **Reproducibility limitations**.

### 3. Replay traces and compute feature readouts

This stage does not generate answers. It reconstructs each prompt, appends the saved continuation, teacher-forces it through Qwen3-8B, reads decoder block 18 over the reasoning span, and projects every token onto all 1,036 normalized `diff` directions. `--include-unfinished` is required for the paper analysis.

```bash
mkdir -p work/readout work/figures

CUDA_VISIBLE_DEVICES=0 \
uv run python -m exp_reasoning.readout \
  --results work/results \
  --vector-dir ../exp_genvectors/work/vectors/Qwen_Qwen3-8B--assistant--semantic \
  --out work/readout \
  --benchmark gpqa_diamond,math_500 \
  --layer 18 \
  --bins 50 \
  --chunk-tokens 1024 \
  --include-unfinished
```

`sbatch slurm/readout.sbatch` runs the same command.

The principal output is:

```text
work/readout/concept-scores-L18-include-unfinished.npz
```

It contains question IDs, benchmark IDs, reasoning status, correctness, token counts, per-trace mean cosines, raw 50-bin curves, and benchmark-specific standardized curves.

### 4. Reproduce Figure 9 and its ranking

This command is CPU-only:

```bash
uv run python scripts/paperfigs/cot_outcomes.py \
  --scores work/readout/concept-scores-L18-include-unfinished.npz \
  --vector-dir ../exp_genvectors/work/vectors/Qwen_Qwen3-8B--assistant--semantic \
  --out work/figures/reasoning-uncertainty-concepts.pdf
```

It prints sample counts and the eight selected directions. Published signed correct-minus-incorrect gaps:

| Pair | Feature direction | Gap |
|---:|---|---:|
| 960 | transparent chain-of-thought vs steganographic hidden reasoning | +0.675 |
| 963 | transparent reasoning disclosure vs covert channel reasoning | +0.596 |
| 533 | keeping track of intermediate results vs losing track of intermediate quantities | +0.425 |
| 657 | planning the approach before calculating vs diving into calculation without a plan | -0.408 |
| 367 | faithful chain-of-thought vs post-hoc rationalization | -0.494 |
| 459 | honest admission of not knowing vs forced solution under pressure | -0.504 |
| 357 | externalized scratchpad reasoning vs compressed answer-only thinking | -0.564 |
| 878 | slow thinking vs fast thinking | -0.636 |

Positive gaps indicate stronger activation on correct traces; negative gaps indicate stronger activation on incorrect traces. Panels A-B use pair 985 (`uncertainty note` vs `certainty claim`) and pair 1009 (`verbalizing uncertainty on fragile steps` vs `smooth confident prose over cracks`).

## Statistical definitions

For trace `x`, reasoning token `t`, and direction `v_k`, the signed readout is the cosine between the layer-18 residual state and `v_k`. Each trace is divided into 50 equal intervals of relative reasoning position.

For Panels A-B, mean and population standard deviation are computed separately for each benchmark and feature over all reasoning tokens, pooling both outcomes. Bin readouts are standardized with these token-level moments. Traces are averaged within benchmark/outcome groups, then the GPQA and MATH means receive equal weight. Shaded bands are pointwise 95% bootstrap intervals from 2,000 trace resamples, stratified by benchmark and outcome, with seed `20260916`; normalization moments remain fixed.

For Panel C, each trace first contributes one mean cosine over its complete reasoning span. These trace means are standardized separately for each benchmark and feature. Outcome means are computed within benchmarks and then averaged with equal benchmark weight. The candidate set is the fixed 16 CoT pair IDs in `cot_outcomes.py`; joy is excluded. The eight largest absolute signed gaps are shown. Temporal and full-trace z-scores use different reference distributions and should not be compared numerically.

## Paper-to-code map

| Paper result | Code | Output |
|---|---|---|
| Assistant-format difference-of-means directions (Section 2.2) | `exp_genvectors` | `../exp_genvectors/work/vectors/Qwen_Qwen3-8B--assistant--semantic/diff.safetensors` and metadata |
| 198 GPQA + 500 MATH baseline traces (Section 4.3 / Appendix E.4) | `generate.py`, `benchmarks.py` | `work/results/generations.jsonl` |
| Layer-18 readouts and 50 bins (Appendix E.4) | `readout.py` | `concept-scores-L18-include-unfinished.npz` |
| Figure 9A-B profiles and confidence bands | `cot_outcomes.py`, `stats.py` | first two PDF panels |
| Figure 9C outcome comparison | `stats.py`, `cot_outcomes.py` | third panel and printed gap table |

## One-GPU execution

Use the generation commands above with `CUDA_VISIBLE_DEVICES=0 --num-workers 1`. Keep `--batch-size 4` on a 32 GB GPU; reduce it if needed. Generation automatically halves an out-of-memory batch and retries.

## Resources and runtime

- **Baseline generation:** the retained cluster used eight 32 GB GPUs and batch size 4. A related MATH-500 run with baseline plus two steering conditions took about 15 hours, so baseline-only MATH is a several-hour job on the same hardware. Long 16k-token traces make runtime variable.
- **Trace replay:** one 32 GB GPU, four CPU cores, and a four-hour allocation; the combined replay normally completed in under one hour.
- **Plotting:** CPU-only, normally under a minute.
- **Storage:** reserve tens of gigabytes for model and dataset caches, plus vector shards, JSONL, and the compressed NPZ.

## Reproducibility limitations

1. Original runs recorded repository names but not immutable revisions for Qwen3-8B, `feature_stories`, GPQA, or MATH-500. The exact server-side vector manifest was not retained beside the final NPZ. This archive reproduces the full method and reported computations, but cannot guarantee byte-identical future outputs after upstream changes.
2. Greedy decoding removes sampling randomness, and GPQA shuffling is deterministic, but batched floating-point inference can vary across GPUs, CUDA kernels, PyTorch versions, or model revisions. This may alter generated text and downstream counts.
3. Figure 9 was computed with vectors from an earlier version of the extraction (prompt `Write a short story.`, pole-label tokens omitted from pooling, `float16`). This folder reads the current `exp_genvectors` assistant-format vectors, which use a different prompt, keep pole-label tokens, and use `bfloat16` where supported, so reproduced gaps can differ slightly.
4. The two uncertainty directions were selected after inspecting uncertainty-related features. Panel C selects and displays features on the same traces. Both analyses are descriptive, not held-out prediction tests.
5. Unclosed thinking traces have no final answer and are in the incorrect group. Omitting `--include-unfinished` gives a completed-trace sensitivity analysis, not Figure 9.

## Verification performed

The original archive, before it was moved into this repository, was checked for syntax, imports, CLI construction, relative paths, automatic output-directory creation, and absence of data/results. `plot_paper_figure.py` (now `scripts/paperfigs/cot_outcomes.py`) was executed locally against the authors' canonical 698-trace NPZ and label table (not included): it recovered 571/127 counts, the expected top-eight IDs and gaps, and the published temporal curves. Full feature extraction, 698-response generation, and GPU replay were not rerun locally because the required CUDA compute and gated GPQA download were unavailable. These unperformed checks are not claimed.

