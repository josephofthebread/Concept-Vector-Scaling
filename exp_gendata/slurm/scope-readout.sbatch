#!/bin/bash
#SBATCH --job-name=gendata-scope
#SBATCH --partition=gpu-ef-quick
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/scope-%j.out
#SBATCH --error=logs/scope-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source .env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SCOPE_OUT="${SCOPE_OUT:-work/scope/data}"
export SCOPE_OUT
mkdir -p logs "$SCOPE_OUT"
srun .venv/bin/python - <<'EOF'
import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
OUT = Path(os.environ.get("SCOPE_OUT", "work/scope/data"))
OUT.mkdir(parents=True, exist_ok=True)
MAX_NEW = 32768

tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True, device_map={"": "cuda"}
).eval()
core = model.model
LAYERS = len(core.layers) + 1
HIDDEN = model.config.hidden_size

# ---------------------------------------------------------------- axes
with safe_open("work/captures/Qwen_Qwen3-8B--assistant--semantic/shard-000.safetensors", framework="np") as f:
    counts = f.get_tensor("counts").astype(np.float32)
    unit_mu = f.get_tensor("unit_diff_sum").astype(np.float32)
    lang_mu = f.get_tensor("language_sum").astype(np.float32)
    lang_count = f.get_tensor("language_count").astype(np.float32)[:, 0]
    corpus_sum = f.get_tensor("corpus_sum").astype(np.float64)
corpus_mean = (corpus_sum / counts.sum()).astype(np.float32)
shared = unit_mu.mean(axis=2).mean(axis=1)
v_lang = lang_mu[:, :, 0] - lang_mu[:, :, 1]
others = [i for i in range(5) if i != 2]
w_o = lang_count.mean(0)[others] / lang_count.mean(0)[others].sum()
offset = v_lang[:, :, 2] - np.einsum("lpoh,o->lph", v_lang[:, :, others], w_o)
with safe_open("work/sentiment/axis.safetensors", framework="np") as f:
    sentiment = f.get_tensor("axis").astype(np.float32)

def unitize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)

mandarin = np.empty_like(shared)
for l in range(LAYERS):
    _, _, vt = np.linalg.svd(offset[l], full_matrices=False)
    mandarin[l] = vt[0]
shared_u, sent_u, mand_u = unitize(shared), unitize(sentiment), unitize(mandarin)
residual = shared_u - (shared_u * sent_u).sum(-1, keepdims=True) * sent_u
residual = unitize(residual - (residual * mand_u).sum(-1, keepdims=True) * mand_u)
# control_random, built as in rescreen.py: normalized gaussian directions
generator = torch.Generator().manual_seed(20260820)
noise = torch.nn.functional.normalize(torch.randn(64, HIDDEN, generator=generator), dim=1)[:2].numpy()
rand0 = np.repeat(noise[0][None], LAYERS, axis=0)
rand1 = np.repeat(noise[1][None], LAYERS, axis=0)
TRACKS = [("control_shared", "mean of unit diff vectors", "shared"),
          ("shared residual", "minus sentiment & mandarin", "shared"),
          ("sentiment", "1-star vs 5-star (Amazon)", "tributary"),
          ("mandarin", "language offset axis", "tributary"),
          ("control_random:0", "random unit direction", "control"),
          ("control_random:1", "random unit direction", "control")]
axes = np.stack([shared_u, residual_u := residual, sent_u, mand_u, rand0, rand1], axis=1)
T = axes.shape[1]
axes_t = torch.from_numpy(axes).cuda()
print(f"axes: {LAYERS} layers x {T} tracks", flush=True)

# ---------------------------------------------------------------- hooks: project inline
projections = [None] * LAYERS
norms = [None] * LAYERS

def hook(pos, _m, _i, out):
    state = (out[0] if isinstance(out, tuple) else out)[0].float()      # [tok, H]
    centered = state - state.mean(dim=0, keepdim=True)                  # own-mean centring
    unit = centered / centered.norm(dim=1, keepdim=True).clamp_min(1e-6)
    projections[pos] = (unit @ axes_t[pos].T).cpu().numpy()             # cosine [tok, T]
    norms[pos] = state.norm(dim=1).cpu().numpy()                        # raw norm, for the panel

handles = [core.embed_tokens.register_forward_hook(partial(hook, 0))]
handles += [b.register_forward_hook(partial(hook, i + 1)) for i, b in enumerate(core.layers)]

def readout(ids):
    """Per-token cosine to every track, centered on this conversation's own mean activation."""
    with torch.inference_mode():
        model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
    return np.stack(projections), np.stack(norms)                       # cos [L, tok, T], norm [L, tok]

# ---------------------------------------------------------------- conversations
SUSP = "../exp_suspicious_tests/data"
lm = pq.read_table(f"{SUSP}/lmsys_items.parquet").to_pydict()
mc = pq.read_table(f"{SUSP}/mcqa.parquet").to_pydict()
trash = json.load(open(f"{SUSP}/trashtalk_messages.json"))
agentic = json.load(open("data/scope/agentic.json"))

mcqa_idx = next((i for i, s in enumerate(mc["subject"]) if s == "professional_medicine"), 0)
mcqa_text = mc["question"][mcqa_idx] + "\n" + "\n".join(
    f"{chr(65 + j)}. {o}" for j, o in enumerate(mc["options"][mcqa_idx])) + "\n\nAnswer with the letter and explain."
trash_text = next(t["text"] for t in trash if t["level"] == 5 and t["label"] == "m1")

def generate(user_text):
    prompt = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                     tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    for h in handles:
        h.remove()
    with torch.inference_mode():
        out = model.generate(torch.tensor([prompt_ids], device="cuda"), max_new_tokens=MAX_NEW,
                             do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    handles.clear()
    handles.append(core.embed_tokens.register_forward_hook(partial(hook, 0)))
    handles.extend(b.register_forward_hook(partial(hook, i + 1)) for i, b in enumerate(core.layers))
    full = out[0].tolist()
    return full, ["prompt"] * len(prompt_ids) + ["response"] * (len(full) - len(prompt_ids))

conversations = {"agentic": (agentic["ids"], agentic["roles"], "coding agent, impossible tests")}

# two feature_stories stories: both poles of one row, so they share premise/genre/language and
# differ only in the concept. Teacher-forced in the same assistant scaffold the vectors were built
# with, so these are in-distribution for every track.
stories = pq.read_table("data/semantic.parquet",
                        columns=["pair_id", "concept", "antagonist", "concept_text", "antagonist_text",
                                 "genre_id", "language_id"]).to_pydict()
row = next(i for i in range(len(stories["pair_id"]))
           if stories["concept"][i] == "truthfulness" and stories["language_id"][i] == 0)
for side, field in ((stories["concept"][row], "concept_text"), (stories["antagonist"][row], "antagonist_text")):
    body = stories[field][row]
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "Generate a short story."}, {"role": "assistant", "content": body}],
        tokenize=False, add_generation_prompt=False)
    marker = tok.apply_chat_template(
        [{"role": "user", "content": "Generate a short story."}, {"role": "assistant", "content": "@@@"}],
        tokenize=False, add_generation_prompt=False).split("@@@")[0]
    prefix_ids = tok(marker, add_special_tokens=False)["input_ids"]
    ids = tok(rendered, add_special_tokens=False)["input_ids"]
    roles = ["prompt"] * len(prefix_ids) + ["response"] * (len(ids) - len(prefix_ids))
    stem = "story-" + ("concept" if field == "concept_text" else "antagonist")
    conversations[stem] = (ids, roles, f"feature_stories: {side} (pair {stories['pair_id'][row]})")
    print(f"{stem}: {len(ids)} tokens — {side}", flush=True)
for stem, text, topic in (("lmsys", lm["text"][14], "lmsys: MySQL performance triage"),
                          ("mcqa", mcqa_text, f"MMLU: {mc['subject'][mcqa_idx]}"),
                          ("trashtalk", trash_text, "abuse, level 5")):
    ids, roles = generate(text)
    conversations[stem] = (ids, roles, topic)
    print(f"{stem}: {len(ids)} tokens", flush=True)

# ---------------------------------------------------------------- emit (fp32, self z-scored)
generations = []
for stem, (ids, roles, topic) in conversations.items():
    proj, norm = readout(ids)                                            # cosines [L, tok, T]
    mean = proj.mean(axis=1, keepdims=True)
    sd = np.maximum(proj.std(axis=1, keepdims=True), 1e-9)
    z = ((proj - mean) / sd).astype(np.float32)                          # [L, tok, T]
    scales = {}
    for l in range(LAYERS):
        (OUT / f"{stem}.m0.L{l}.bin").write_bytes(np.ascontiguousarray(z[l]).tobytes())
        scales[f"m0.L{l}"] = 1.0
    (OUT / f"{stem}.tokens.json").write_text(json.dumps({
        "pieces": tok.convert_ids_to_tokens(ids), "role": roles, "logprob": [None] * len(ids),
        "norms": {f"L{l}": [round(float(v), 1) for v in norm[l]] for l in (0, LAYERS // 2, LAYERS - 1)}}))
    generations.append({"stem": stem, "outcome": "readout", "ending": "readout", "variant": "unsteered",
                        "seed": 0, "behavior": stem, "cell": topic, "sample": "0", "steer": None,
                        "topic": topic, "tactic": "z vs own tokens", "half": "full",
                        "tokens": len(ids), "width": roles.count("prompt"), "scales": scales})
    print(f"wrote {stem}: {len(ids)} tokens, |z| p99={np.percentile(np.abs(z), 99):.2f}", flush=True)

(OUT / "manifest.json").write_text(json.dumps({
    "pairs": T, "methods": ["cosine z-scored vs own tokens"], "layers": list(range(LAYERS)),
    "generations": generations}))
(OUT / "concepts.json").write_text(json.dumps({
    "pairs": [{"pair": i, "concept": name, "antagonist": note, "class": group}
              for i, (name, note, group) in enumerate(TRACKS)],
    "neighbors": [[] for _ in TRACKS]}))
print("scope data complete", flush=True)
EOF
