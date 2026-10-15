from __future__ import annotations

import collections
import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

TAG = {
    "Qwen/Qwen3-8B": "Qwen_Qwen3-8B--assistant--semantic",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "unsloth_Meta-Llama-3.1-8B-Instruct--assistant--semantic",
    "google/gemma-2-9b-it": "google_gemma-2-9b-it--assistant--semantic",
}
TOPN = 60
RANGES = [
    ("Latin", 0x41, 0x5A),
    ("Latin", 0x61, 0x7A),
    ("Latin", 0x00C0, 0x024F),
    ("Han", 0x4E00, 0x9FFF),
    ("Han", 0x3400, 0x4DBF),
    ("Cyrillic", 0x0400, 0x04FF),
    ("Arabic", 0x0600, 0x06FF),
    ("Hangul", 0xAC00, 0xD7A3),
    ("Hangul", 0x1100, 0x11FF),
    ("Hiragana", 0x3040, 0x309F),
    ("Katakana", 0x30A0, 0x30FF),
    ("Devanagari", 0x0900, 0x097F),
    ("Greek", 0x0370, 0x03FF),
    ("Hebrew", 0x0590, 0x05FF),
]
LANG = {
    "Latin": "En/De/Es",
    "Han": "Chinese",
    "Cyrillic": "Russian",
    "Hangul": "Korean",
    "Hiragana": "Japanese",
    "Katakana": "Japanese",
    "Arabic": "Arabic",
    "Hebrew": "Hebrew",
    "Devanagari": "Hindi",
    "Greek": "Greek",
}


def char_script(ch: str) -> str | None:
    o = ord(ch)
    for name, lo, hi in RANGES:
        if lo <= o <= hi:
            return name
    return None


def token_script(text: str) -> str:
    c: collections.Counter = collections.Counter()
    for ch in text:
        s = char_script(ch)
        if s:
            c[s] += 1
    return c.most_common(1)[0][0] if c else "symbol"


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B", choices=sorted(TAG))
    args = parser.parse_args()
    MODEL = args.model
    VEC = Path(f"work/vectors/{TAG[MODEL]}/diff.safetensors")
    attn = "eager" if "gemma" in MODEL else "sdpa"

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation=attn, device_map="auto"
    )
    model.eval()
    LAYER = round(model.config.num_hidden_layers / 2)
    unemb = model.get_output_embeddings().weight.float()
    gamma = model.model.norm.weight.float().to(model.device)
    with safe_open(VEC, framework="pt") as h:
        diff = h.get_tensor("diff")
    npairs = diff.shape[1]
    print(f"{MODEL}: {model.config.num_hidden_layers} layers, lens at L{LAYER}, {npairs} pairs", flush=True)

    total: collections.Counter = collections.Counter()
    dominant: collections.Counter = collections.Counter()
    for pid in range(npairs):
        v = diff[LAYER, pid].float().to(model.device)
        v = v / v.norm().clamp_min(1e-8)
        proj = unemb @ (gamma * v)
        ids = torch.topk(proj, TOPN).indices.tolist()
        local: collections.Counter = collections.Counter()
        for tid in ids:
            s = str(tok.decode([tid]))
            sc = "fragment" if "�" in s else (token_script(s.strip()) if s.strip() else "symbol")
            total[sc] += 1
            local[sc] += 1
        dominant[local.most_common(1)[0][0]] += 1
        if pid % 200 == 0:
            print(f"  pair {pid}/{npairs}", flush=True)

    n = sum(total.values())
    out = Path("work/logitlens_share")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"share-{TAG[MODEL].split('--')[0]}.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "layer": LAYER,
                "npairs": npairs,
                "topn": TOPN,
                "total": dict(total),
                "dominant": dict(dominant),
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"\n=== {MODEL} promoted-token share by script (top-{TOPN}, {npairs} vectors, {n} tokens) ===")
    for sc, c in total.most_common():
        print(f"  {sc:10s} ({LANG.get(sc, '-'):9s}): {c:6d}  {c / n:.1%}")
    print("\n=== per-concept DOMINANT script ===")
    for sc, c in dominant.most_common():
        print(f"  {sc:10s} ({LANG.get(sc, '-'):9s}): {c:5d}  {c / npairs:.1%}")


if __name__ == "__main__":
    main()
