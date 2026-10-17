---
license: mit
tags:
  - interpretability
  - alignment
  - steering
  - probing
  - concept-vectors
datasets:
  - AntonKorznikov/feature_stories
base_model:
  - Qwen/Qwen3-8B
  - Qwen/Qwen3-8B-Base
  - google/gemma-2-9b-it
  - google/gemma-2-9b
  - unsloth/Meta-Llama-3.1-8B-Instruct
  - unsloth/Meta-Llama-3.1-8B
pretty_name: Concept Vectors
---

# Concept Vectors

Difference-in-means concept vectors for the 1,036 concept-antagonist pairs of [Feature Stories](https://huggingface.co/datasets/AntonKorznikov/feature_stories), at every layer of three model families and five prompt formats.

Each vector belongs to a **concept-antagonist pair**, not a concept label alone. It is the mean residual-stream activation over the stories of the concept pole minus the same mean over the stories of the antagonist pole. Both means use all 1,000 matched rows of the pair, so topic, cast, and scene cancel and the pole difference remains.

One file is one model x one prompt format:

- `diff[layer, k]` is the vector of pair `k` at that layer
- positive projections point toward the concept pole, negative ones toward the antagonist

Example pair, `genuine goal pursuit` vs `reward hacking`: projecting a model's activations on its vector reads how far the text leans toward one pole, and adding the vector pushes generation toward the concept pole.

## Scale

| | |
|---|---|
| Vectors in this repository | **585,340** |
| Vector sets | **15** (one per model family x prompt format) |
| Model families | Qwen3-8B, gemma-2-9b, Llama-3.1-8B |
| Prompt formats | 5 (see the mapping below) |
| Layers | **37** (Qwen3-8B), **43** (gemma-2-9b), **33** (Llama-3.1-8B) |
| Hidden size | **4,096** (Qwen3-8B, Llama-3.1-8B), **3,584** (gemma-2-9b) |
| Size | about **9 GB** in float32 |

A set is one model family x one prompt format. `raw` and `paired` use the base checkpoint; `user`, `assistant` and `paired-assistant` use the instruction-tuned one. Each set holds 1,036 vectors per layer, so the 15 sets are 1,036 x 5 x (37 + 43 + 33) = **585,340** vectors.

## Formats

| Folder | Checkpoint | Story is presented as |
|---|---|---|
| `Qwen3-8B-Base/raw` | `Qwen/Qwen3-8B-Base` | plain text |
| `Qwen3-8B-Base/paired` | `Qwen/Qwen3-8B-Base` | plain text after a line naming the class, both poles, the guidance, language and genre |
| `Qwen3-8B/user` | `Qwen/Qwen3-8B` | a user turn |
| `Qwen3-8B/assistant` | `Qwen/Qwen3-8B` | the assistant's reply to "Generate a short story." |
| `Qwen3-8B/paired-assistant` | `Qwen/Qwen3-8B` | the assistant's reply to the `paired` line |

The same five folders exist under `gemma-2-9b/` and `gemma-2-9b-it/` (`google/gemma-2-9b`, `google/gemma-2-9b-it`), and under `Llama-3.1-8B/` and `Llama-3.1-8B-Instruct/` (`unsloth/Meta-Llama-3.1-8B`, `unsloth/Meta-Llama-3.1-8B-Instruct`).

## Load

```python
import json
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

diff = load_file(hf_hub_download("josephofthebread/Concept-Vectors", "Qwen3-8B/assistant/diff.safetensors"))["diff"]
ontology = json.load(open(hf_hub_download("AntonKorznikov/feature_stories", "ontology.json", repo_type="dataset")))
pairs = sorted(
    ((p["concept"], p["antagonist"], c["name"]) for c in ontology["classes"] for p in c["pairs"]),
    key=lambda p: (p[0], p[1]),
)
k = next(i for i, p in enumerate(pairs) if p[:2] == ("genuine goal pursuit", "reward hacking"))
vector = diff[18, k]
print(diff.shape, pairs[k], vector.norm())
```

Probe text with the same vector:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="auto")
inputs = tokenizer("I found a way to make the tests pass without fixing the bug.", return_tensors="pt").to(model.device)
with torch.no_grad():
    hidden = model(**inputs, output_hidden_states=True).hidden_states[18][0].float().cpu()
print(torch.nn.functional.cosine_similarity(hidden, vector[None], dim=-1))
```

Each file holds all layers of one set (0.56-0.64 GB). Download only the sets you need.

## Files

| Path | Meaning |
|---|---|
| `<checkpoint>/<format>/diff.safetensors` | One float32 tensor `diff` of shape `[layers, 1036, hidden]` |
| `diff[l]` | Vectors read from Transformers' `hidden_states[l]`: layer 0 is the embedding output, layer `l` the output of decoder block `l`, the last layer is after the final norm |
| `diff[:, k]` | Pair `k`: the `k`-th pair of the dataset's `ontology.json` sorted by `(concept, antagonist)` |
| hidden | **4,096** (Qwen3-8B, Llama-3.1-8B), **3,584** (gemma-2-9b) |

## How the vectors were built

Each story is wrapped in its prompt format and run through the model once. The story text is identical across formats; only the wrapping changes.

Activations are averaged over the story tokens only. Stories are truncated to **1,024** tokens, and the first **50** story tokens are left out of the mean, keeping at least 16.

For every pair, layer, and pole, the mean runs over all 1,000 stories of that pole: 2 generators x 5 languages x 10 genres x 10 setups. The vector is the concept mean minus the antagonist mean. Vectors are not normalized; their norms grow with depth.

## Uses

- Probing: cosine of a token's residual state with a pair's vector at the same layer
- Steering: adding a multiple of a vector to the output of the matching decoder block
- Comparing directions across layers, prompt formats, and model families

## Limitations

- The vectors come from synthetic stories. Their projections on other text measure the pole difference the stories carry, not human judgments.
- A vector is tied to its model, layer, and prompt format. Directions from different formats are close but not identical.
- `concept` and `antagonist` are ontology labels, not verified properties of each direction.
- The ontology is a curated set of narratable dichotomies for alignment and interpretability. It is not a complete map of concepts.

## Citation

```
From Stories to Features: Scaling Concept Vectors for AI Alignment & Interpretability.
Under review, ICLR 2027.
```

Vectors: `josephofthebread/Concept-Vectors`.

## License

MIT.
