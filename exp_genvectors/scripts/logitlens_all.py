from __future__ import annotations

import collections
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

TAG = {
    "Qwen/Qwen3-8B": "Qwen_Qwen3-8B--assistant--semantic",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "unsloth_Meta-Llama-3.1-8B-Instruct--assistant--semantic",
    "google/gemma-2-9b-it": "google_gemma-2-9b-it--assistant--semantic",
}
TOPN = 60
KEEP_LATIN = 20
KEEP_OTHER = 8
TRAIN_NONLATIN = {"Han", "Cyrillic"}
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


def char_script(ch: str) -> str | None:
    o = ord(ch)
    for name, lo, hi in RANGES:
        if lo <= o <= hi:
            return name
    return None


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B", choices=sorted(TAG))
    MODEL = parser.parse_args().model
    attn = "eager" if "gemma" in MODEL else "sdpa"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation=attn, device_map="auto"
    )
    model.eval()
    device = model.device
    LAYER = round(model.config.num_hidden_layers / 2)
    unemb = model.get_output_embeddings().weight.float()
    gamma = model.model.norm.weight.float().to(device)
    with safe_open(Path(f"work/vectors/{TAG[MODEL]}/diff.safetensors"), framework="pt") as h:
        diff = h.get_tensor("diff")
    npairs = diff.shape[1]
    print(f"{MODEL}: L{LAYER}, diff {tuple(diff.shape)}; {npairs} pairs", flush=True)

    tbl = pq.read_table(
        "data/semantic.parquet", columns=["pair_id", "concept", "antagonist", "concept_text", "antagonist_text"]
    )
    d = tbl.to_pydict()
    names: dict[int, tuple[str, str]] = {}
    for pid, c, a in zip(d["pair_id"], d["concept"], d["antagonist"]):
        names.setdefault(int(pid), (c, a))
    print(f"parquet rows={len(d['pair_id'])}; distinct pair_ids={len(names)}", flush=True)

    corpus_ids: set[int] = set()
    texts = d["concept_text"] + d["antagonist_text"]
    B = 20000
    for i in range(0, len(texts), B):
        for ids in tok(texts[i : i + B], add_special_tokens=False)["input_ids"]:
            corpus_ids.update(ids)
        if i % (B * 20) == 0:
            print(f"  tokenized {i}/{len(texts)}  |corpus_ids|={len(corpus_ids)}", flush=True)
    print(f"corpus token ids: {len(corpus_ids)} of {unemb.shape[0]} vocab", flush=True)

    agg = {"Latin": [0, 0], "TrainNonLatin": [0, 0], "OOD": [0, 0], "fragment": [0, 0]}
    percept = []
    n_ood_concepts = 0
    for pid in range(npairs):
        v = diff[LAYER, pid].float().to(device)
        v = v / v.norm().clamp_min(1e-8)
        proj = unemb @ (gamma * v)
        ids = torch.topk(proj, TOPN).indices.tolist()
        latin: list[str] = []
        other: dict[str, list[str]] = collections.defaultdict(list)
        has_ood = False
        seen: set[str] = set()
        for tid in ids:
            s = str(tok.decode([tid]))
            novel = tid not in corpus_ids
            if "�" in s:
                agg["fragment"][0] += 1
                agg["fragment"][1] += novel
                continue
            w = s.strip()
            if not w:
                continue
            counts: collections.Counter = collections.Counter()
            for ch in w:
                script = char_script(ch)
                if script:
                    counts[script] += 1
            sc = counts.most_common(1)[0][0] if counts else "symbol"
            grp = (
                "Latin"
                if sc == "Latin"
                else "TrainNonLatin"
                if sc in TRAIN_NONLATIN
                else "symbol"
                if sc == "symbol"
                else "OOD"
            )
            if grp in agg:
                agg[grp][0] += 1
                agg[grp][1] += novel
            if sc == "Latin":
                lw = w.lower()
                if lw not in seen and len(lw) > 1:
                    seen.add(lw)
                    if len(latin) < KEEP_LATIN:
                        latin.append(lw)
            elif grp == "OOD":
                has_ood = True
                if len(other[sc]) < KEEP_OTHER and w not in other[sc]:
                    other[sc].append(w)
        n_ood_concepts += has_ood
        c, a = names.get(pid, ("?", "?"))
        percept.append({"pair_id": pid, "concept": c, "antagonist": a, "latin": latin, "ood": dict(other)})
        if pid % 200 == 0:
            print(f"  pair {pid}/{npairs} {c[:30]!r}", flush=True)

    tg = MODEL.split("/")[-1]
    out = Path("work/logitlens_all")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"percept-{tg}.json").write_text(json.dumps(percept, ensure_ascii=False))
    summary: dict[str, Any] = {
        "model": MODEL,
        "npairs": npairs,
        "layer": LAYER,
        "n_concepts_with_ood_script": n_ood_concepts,
        "novelty": {g: {"total": t, "novel": n, "rate": round(n / max(t, 1), 4)} for g, (t, n) in agg.items()},
    }
    (out / f"summary-{tg}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print("\n=== ALL 1036: novelty (token-id absent from corpus) by group ===")
    for g, (t, n) in agg.items():
        print(f"  {g:14s}: {n}/{t}  ({n / max(t, 1):.1%})")
    print(f"concepts promoting >=1 OOD-script token: {n_ood_concepts}/{npairs}")


if __name__ == "__main__":
    main()
