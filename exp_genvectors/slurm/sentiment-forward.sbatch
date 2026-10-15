#!/bin/bash
#SBATCH --job-name=genvectors-sentfwd
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/sentfwd-%j.out
#SBATCH --error=logs/sentfwd-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source .env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs work/sentiment
srun .venv/bin/python - <<'EOF'
import json
import numpy as np
import torch
import pyarrow.parquet as pq
from functools import partial
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

from exp_genvectors.artifacts import atomic_safetensors
from exp_genvectors.batching import mean_pool, pooling_selector, token_budget_batches
from exp_genvectors.config import SKIP_TOKENS, STORY_MAX_TOKENS
from exp_genvectors.presentation import encode_wrapping

MODEL = "Qwen/Qwen3-8B"
t = pq.read_table("data/sentiment.parquet").to_pydict()
texts, stars, cats = t["text"], t["stars"], t["category"]
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
wrap = encode_wrapping(tokenizer, "assistant", "", "", "")  # assistant scaffold is pole/lang/genre independent
encoded = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=STORY_MAX_TOKENS)["input_ids"]
model = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
                                  local_files_only=True, device_map={"": "cuda"}).eval()
core = model.model if hasattr(model, "model") else model
layers = len(core.layers) + 1
hidden = model.config.hidden_size
weights_holder = [torch.empty(0)]
captured = [None] * layers
def hook(pos, _m, _i, out):
    state = out[0] if isinstance(out, tuple) else out
    captured[pos] = mean_pool(state, weights_holder[0])
core.embed_tokens.register_forward_hook(partial(hook, 0))
for i, block in enumerate(core.layers):
    block.register_forward_hook(partial(hook, i + 1))
pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

pooled_all = np.empty((len(texts), layers, hidden), dtype=np.float16)
lengths = [len(e) for e in encoded]
with torch.inference_mode():
    for batch in token_budget_batches(lengths, 32768):
        seqs = []
        for i in batch:
            seqs.append((wrap.prefix, tuple(encoded[i]), wrap.suffix))
        width = max(len(p) + len(s) + len(x) for p, s, x in seqs)
        ids = np.full((len(seqs), width), pad, dtype=np.int64)
        att = np.zeros_like(ids); wts = np.zeros((len(seqs), 1, width), dtype=np.float32)
        for k, (p, s, x) in enumerate(seqs):
            start = len(p); end = start + len(s) + len(x)
            ids[k, :start] = p; ids[k, start:start + len(s)] = s; ids[k, start + len(s):end] = x
            att[k, :end] = 1
            wts[k, 0, start:start + len(s)] = pooling_selector(len(s), SKIP_TOKENS)
        weights_holder[0] = torch.from_numpy(wts).cuda(non_blocking=True)
        captured[:] = [None] * layers
        model(input_ids=torch.from_numpy(ids).cuda(non_blocking=True),
              attention_mask=torch.from_numpy(att).cuda(non_blocking=True), use_cache=False)
        out = torch.stack(captured).permute(1, 0, 2).cpu().numpy().astype(np.float16)
        for k, i in enumerate(batch):
            pooled_all[i] = out[k]
stars_arr = np.array(stars, dtype=np.int8)
meta = {"manifest": json.dumps({"model": MODEL, "presentation": "assistant", "axes": ["review", "layer", "hidden"],
                                "stars": stars, "category": cats}, sort_keys=True)}
atomic_safetensors({"pooled": pooled_all, "stars": stars_arr}, Path("work/sentiment/pooled.safetensors"), meta)
pos = pooled_all[stars_arr == 5].astype(np.float32).mean(0)
neg = pooled_all[stars_arr == 1].astype(np.float32).mean(0)
atomic_safetensors({"axis": (pos - neg)}, Path("work/sentiment/axis.safetensors"), meta)
print("done:", pooled_all.shape)
EOF
