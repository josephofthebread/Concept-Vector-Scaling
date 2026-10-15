# exp_gendataset

Code that generates the [Feature Stories](https://huggingface.co/datasets/AntonKorznikov/feature_stories) corpus from *From Stories to Features: Scaling Concept Vectors for AI Alignment & Interpretability*.

The only human input per feature is a one- or two-sentence `narrative_guidance`. Everything else is fixed: the ontology, the prompt, the genre contracts, the two demonstrations, and the sampling settings in Appendix B.

## What this reproduces

| | |
|---|---|
| Ontology | 148 classes, 1,036 concept–antagonist pairs |
| Languages | English, Mandarin Chinese, Russian, Spanish, German |
| Genres | third-person narrative, dialogue, monologue, diary, letter, news, memo, fable, speech, case study |
| Setups per cell | 10 (`pair_number` 0–9) |
| Generators | `qwen/qwen3-235b-a22b-2507`, `deepseek/deepseek-v4-flash` |
| Narratives | 2,072,000 (1,000 stories per pole per pair) |
| Released rows | 1,036,000 matched pairs (both poles of one setup in one row) |

One cell is one concept–antagonist pair, one genre, one language, and one generator. The model returns ten setups. Each setup is one shared scene plus two stories, one per pole. Setup metadata (protagonist, shared constraints, setup sentence) is in English. Story bodies are in the language of the cell.

Sampling is temperature **1.0** and top-p **0.95**. The prompt asks for 200–400 words per story (target 300–380), or 600–1,200 characters for Mandarin (target 900–1,100). If a response is missing setups, a follow-up request asks only for those `pair_number`s. Each response is allowed at most ten validation retries.

Every request uses the same two demonstrations: a Russian speech contrasting universal access with privileged access, and a Japanese diary contrasting causal investigation with blame assignment. Japanese is a demonstration language only. It is not one of the five corpus languages.

The full matrix cost about $500 on OpenRouter when the released corpus was generated. A one-cell smoke run is the right first check.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENROUTER_API_KEY=...
```

Optional: `OPENROUTER_PROXY_URL` (or `HTTPS_PROXY`) if the API is reached through a proxy. A local `secrets/openrouter_api_key.txt` is also read when the environment variable is unset. Do not commit that file.

## Check the matrix

```bash
python -m exp_gendataset count
```

Expected last lines:

```text
cells per generator: 51800
narratives per generator: 1036000
generators: 2
narratives total: 2072000
matched pair-rows total: 1036000
```

## Inspect the prompt

The generation prompt is the code in `src/exp_gendataset/generation.py` and `src/exp_gendataset/story_few_shot.py`. Dump the exact chat messages for one cell:

```bash
python -m exp_gendataset dump-prompt \
  --class-id 2 --pair-index 0 \
  --genre speech --language Russian \
  --out prompts/dumped/example.json
```

## Generate

One cell:

```bash
python -m exp_gendataset generate \
  --model qwen/qwen3-235b-a22b-2507 \
  --class-id 2 --pair-index 0 \
  --genre narrative_3rd_person --language English \
  --limit 1
```

One generator, full matrix (51,800 cells). Finished cells are skipped on resume, so the same command can be repeated after a credit or network stop.

```bash
python -m exp_gendataset generate --model qwen/qwen3-235b-a22b-2507
python -m exp_gendataset generate --model deepseek/deepseek-v4-flash
```

Each run writes, under `work/generated/<model-slug>/`:

- `stories.jsonl` — one row per story pole
- `batches.jsonl` — usage and retry metadata
- `errors.jsonl` / `invalid_responses.jsonl` — failed cells
- `manifest.json`, `retry_summary.json`

## Pair into the Hugging Face schema

```bash
python -m exp_gendataset pair \
  --model qwen/qwen3-235b-a22b-2507
```

Each output row has `concept_text` and `antagonist_text` for the same `shared_setup`, plus `narrative_guidance` copied from `data/ontology.json` (not written by the generator).

## Genre ids

| In the paper | `genre` |
|---|---|
| third-person narrative | `narrative_3rd_person` |
| dialogue | `dialogue` |
| monologue | `monologue` |
| diary | `diary` |
| letter | `letter` |
| news | `news` |
| memo | `memo` |
| fable | `fable` |
| speech | `speech` |
| case study | `case_study` |

## Citation

From Stories to Features: Scaling Concept Vectors for AI Alignment & Interpretability. Under review, ICLR 2027.

Dataset: [AntonKorznikov/feature_stories](https://huggingface.co/datasets/AntonKorznikov/feature_stories).
