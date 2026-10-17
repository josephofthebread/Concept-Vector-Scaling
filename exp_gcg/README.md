# exp_gcg

Replication from existing optimized suffixes: the original `generate`, `judge`, and `aggregate`
commands remain available. The `steer` command adds the article's
generation-boundary steering experiment. Suffix optimization uses Faster-GCG; see **Suffix optimization** below.

Install Python 3.11+ and run `pip install -e '.[test]'` from this directory.
Run `gcg-suffix --help` (or `python -m exp_gcg --help`) and `gcg-suffix steer --help` for arguments.
Models are loaded locally through Transformers; generation generally needs a GPU capable of loading
an 8–9B model in bfloat16. GPT-OSS-20B evaluation dequantizes its weights and needs substantially more
memory. Preparation uses CPU memory for stacked residuals; allow several GB of RAM.
Hugging Face authentication, when required, is handled by its standard runtime configuration.

## Original suffix pipeline

The original `generate` command now uses the reference long-response stopping rule: greedy decoding
until EOS, the remaining model context, or the StrongREJECT rubric reaching 8,191 tokens. `--max-new-tokens` is an optional non-article cap for bounded
runs; omitting it reproduces the long-response rule. `generate` and `judge` accept `--device`.

Original residuals, responses, and scores now have `.meta.json` sidecars binding their contents to
the record, model snapshot, and applicable generation/judge settings. Aggregation and steering
preparation check these sidecars. A missing, modified, or incompatible checkpoint causes an error;
legacy files without metadata are not silently adopted. Regenerate legacy outputs in a fresh
directory; do not reuse a directory from the previous 768-token protocol. Snapshot fingerprints
hash local weight/configuration files and resolved model/tokenizer settings, without storing machine paths.

Judging enumerates the two expected response names (`long_base`, `long_suffix`) from each record, so safe record keys can
have any depth. Missing expected responses fail explicitly. Article ranking outputs use
`signed_change = cosine(mean base, direction) - cosine(mean suffix, direction)`; positive shifts
mean movement toward the antagonist. Absolute ranks and steering-vector orientations are unchanged.
The original analysis manifest is schema version 2 to identify this sign convention.

## Inputs

All data, models, and generated artifacts are external. Paths below are placeholders supplied by the caller.

- `--records`: UTF-8 JSONL with unique safe relative `key` strings, `model`, `dataset`, `index`,
  `forbidden_prompt`, and `input_ids: {base: [...], suffix: [...]}`. The paper's 1,860 records are in
  `data/records.jsonl`. Model names are those shown by `--help`; datasets
  are `advbench` (indices 0–519) and `jbb` (0–99) for each model.
- `--vectors`: the existing model-specific bank directories (`../exp_genvectors/work/vectors`), each containing `pairs.parquet`
  (`pair_id`, `concept`, `antagonist`) and `diff.safetensors` with `diff[layer, feature, hidden]`.
  The 1,036 Parquet rows must follow the tensor's feature-axis order. Layer zero is embedding output.
- `--original`: completed original suffix output root. For every record it must contain
  `records/KEY/hidden_base.npy`, `hidden_suffix.npy`, `responses/long_base.json`,
  `responses/long_suffix.json`, and corresponding `scores/finetuned/long_CONDITION.json` files.
  Saved scores must carry the SHA-256 hash of their response JSON. Every original residual, response,
  and score also requires the `.meta.json` sidecar produced by the original pipeline.
- `--benign`: JSON array or JSONL of exactly the original 100 Alpaca records (not included here), with unique string
  `id` and `prompt`, or `id`, `instruction`, and optional `input`. No prompts are silently sampled:
  identical source prompts and ordering are needed for the article's benign table.
- `--templates`: JSON (`data/judge_templates.json`) containing the `strongreject_finetuned` template with
  `{forbidden_prompt}` and `{response}` placeholders, as used by the original suffix evaluator.
  The benign four-category rubric is included in readable code.

## Steering commands

```sh
gcg-suffix steer prepare --records RECORDS.jsonl --vectors BANKS --original SUFFIX_OUTPUT \
  --benign ALPACA_100.jsonl --output STEERING_OUTPUT

gcg-suffix steer generate --prepared STEERING_OUTPUT/prepared.json --output STEERING_OUTPUT \
  --model qwen3_8b --device cuda:0

gcg-suffix steer judge --prepared STEERING_OUTPUT/prepared.json --output STEERING_OUTPUT \
  --model qwen3_8b --judge finetuned --strongreject qylu4156/strongreject-15k-v1 \
  --strongreject-base google/gemma-2b --templates TEMPLATES.json --device cuda:0

gcg-suffix steer judge --prepared STEERING_OUTPUT/prepared.json --output STEERING_OUTPUT \
  --model qwen3_8b --judge gpt_oss --gpt-oss openai/gpt-oss-20b --device cuda:0

gcg-suffix steer aggregate --prepared STEERING_OUTPUT/prepared.json --output STEERING_OUTPUT
gcg-suffix steer figures --output STEERING_OUTPUT --figures-output FIGURES
```

Repeat generation and both judge commands for `llama3_1_8b_instruct` and `gemma2_9b_it`.
The jobs in `slurm/` run every stage of both pipelines for all three models.
Generation accepts `--qwen3-8b`, `--llama3-1-8b`, and `--gemma-2-9b` local snapshot overrides.
Judges accept local adapter/base/model paths. Generation and judging accept `--shard N --shards TOTAL`
and `--limit N`; limits count assigned response jobs, not input prompts. Shards are zero-based and
partition a fixed job enumeration, so their outputs merge in place without copying files. Use one
worker per shard; concurrent writers to the same shard are unsupported.

## Scientific protocol

Preparation computes base and suffix means of raw residuals within each model/dataset group,
normalizes those means, and subtracts their cosines with unit feature directions. It selects five
features by absolute shift at layers 29 (Qwen), 26 (Llama), and 34 (Gemma), breaking ties by feature
index. Selected directions retain their original orientation. Three random unit controls are drawn
per group using NumPy's default generator with seed `20260914 + model_index * 10 + dataset_index`,
in the original Qwen/Llama/Gemma and AdvBench/JBB order; Gaussian draws are cast to float32 before
normalization, matching the source experiment.

Generation changes the selected block's final input-token residual once:
`x_new = x + alpha * norm(x) * unit_direction`. Feature layer `L` maps to block `L - 1`.
No decoded token is modified. Strengths are `[-.5, -.25, -.1, .1, .25, .5]`, with greedy decoding
for at most 512 new tokens and unchanged chat templates. Harmful prompts use the suffix condition. Benign prompts use the five AdvBench-selected and five JBB-selected concepts, plus one
fresh unsteered baseline per prompt. Random controls are used only for harmful prompts.

StrongREJECT (`qylu4156/strongreject-15k-v1`, a LoRA adapter on `google/gemma-2b`) scores harmful responses from 0 to 1; success uses `score >= .5`. Its template context
fitting matches the existing evaluator. GPT-OSS-20B classifies benign responses as `safety_refusal`,
`capability_limit`, `clarification`, or `substantive_answer`.

Full production coverage is 89,280 harmful and 18,300 benign responses: 107,580 total. Each benign
nonzero-strength model row pools 1,000 responses, and each zero-strength row contains 100. Harmful
concept rows average 5 directions per prompt and random rows average 3. The article's harmful
zero-strength points reuse original suffix baselines, including their original generation length.
They are not regenerated with the 512-token steering cap.

## Checkpoints and outputs

Preparation freezes input hashes, tokenized records, benign prompts, selected vectors, random controls,
and validated baseline scores in `prepared.json` with a hash-validated preparation manifest. It contains
experiment data and must stay outside a source release. Once prepared, later stages use this frozen
copy; editing an original input requires a new preparation directory. All paths can be relocated together.

Responses and scores are atomic JSON checkpoints. Their filenames address the complete job, and
checkpoints bind generation/judge settings and response hashes. Changing settings under an existing
job fails rather than silently reusing a checkpoint. Public model identifiers are not pinned revisions;
use immutable local snapshots for strict reproduction. Model-source paths are hashed, not logged.
Snapshots must not be modified in place during a run.

Aggregation requires every expected job and score, rejects stale or mixed configurations, and writes:

| Article result | Artifact in `steering_analysis/` | Command |
|---|---|---|
| AdvBench/JBB steering table, both direction types | `article_harmful.csv` | `steer aggregate` |
| Base and suffix harmful outcomes and success rates | `harmful_outcomes.csv` | `steer aggregate` |
| 100-prompt Alpaca four-category table | `benign_outcomes.csv` | `steer aggregate` |
| Detailed response scores | `sample_metrics.csv` | `steer aggregate` |
| AdvBench concept/random curves | `gcg-steering-advbench.pdf` and `.png` in `--figures-output` | `steer figures` |

Manifests contain schema versions, hashes, completion flags, and configuration provenance. Figure
rendering validates summary hashes. Bounded generation runs intentionally cannot pass full aggregation.

## Verification and limits

Run `pytest -q tests` for CPU-only selection, hook, parser, checkpoint, sharding, synthetic preparation,
aggregation, and figure checks.
Synthetic tests use no model downloads. Full model generation and local judge reproduction need the
external records, snapshots, and GPU hardware.

The article's original benign baselines used hosted generation for some models. This package generates
them locally with the supplied snapshots, so category percentages may differ. Generated-token steering,
top-100 sweeps, hosted judging, and suffix optimization are excluded. No reference outcomes are embedded
as computed results.

## Suffix optimization

The optimized suffixes in `data/records.jsonl` were produced with [Faster-GCG](https://github.com/weiz0823/Faster-GCG)
at commit `c14694c` with `scripts/faster_gcg/faster-gcg.patch` applied (dataset sharding and skipping,
`HF_TOKEN` for gated datasets, explicit dtype, and gemma-2 routed through the Gemma-3 wrapper without
a system prompt). Model configs are in `scripts/faster_gcg/configs/models/`; the launchers set
`FASTER_GCG` to the patched checkout. Qwen3-8B and Llama-3.1-8B-Instruct used `run_suffix_array.sbatch`
and `run_advbench_remaining_array.sbatch`; gemma-2-9b-it used `gemma_adv.sbatch` and `gemma_jbb.sbatch`
with `gemma2_9b_it.yaml` (the `gemma2_9b.yaml` base-model run was superseded). All other settings are
the upstream `configs/default.yaml`.

## Figure source

The plotting script for the paper's suffix-shift figure (`gcg-concepts-shift.pdf`) is not included.
Each plotted point is a group-mean cosine for one condition, the `base_cosine` and `suffix_cosine`
columns of `analysis/rankings.csv` for the top four AdvBench features per model.
