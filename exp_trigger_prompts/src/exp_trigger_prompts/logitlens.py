from __future__ import annotations

import collections
import json
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from .probe import VEC_ROOT, VEC_TAG

TOPN = 200
KEEP_LATIN = 30
KEEP_OTHER = 12
KMARK = 30
STOP = set(
    (
        "a an the to of in on at by for with from as into about over under and or but nor so if then "
        "is am are was were be been being have has had do does did will would can could this that these "
        "those it its there here i you he she they we me him her them us my your his their our not no "
        "s t re ve ll d m".split()
    )
)
WORD = re.compile(r"[a-z']+")

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


def text_scripts(text: str) -> set[str]:
    return {s for ch in text if (s := char_script(ch)) is not None}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B", choices=sorted(VEC_TAG))
    MODEL = parser.parse_args().model
    conc = json.loads(Path("data/inject_concepts.json").read_text())
    want = {(c["concept"], c["antagonist"]): c["pair_id"] for c in conc}

    dfp: dict = collections.defaultdict(collections.Counter)
    dfa: dict = collections.defaultdict(collections.Counter)
    concept_en_vocab: dict = collections.defaultdict(set)
    corpus_en_vocab: set = set()
    langs: set = set()
    corpus_scripts: set = set()
    ds = load_dataset("AntonKorznikov/feature_stories", split="train")
    for r in ds:
        langs.add(r["language"])
        corpus_scripts |= text_scripts(r["concept_text"][:400]) | text_scripts(r["antagonist_text"][:400])
        if r["language"] != "English":
            continue
        k = (r["concept"], r["antagonist"])
        pw = set(WORD.findall(r["concept_text"].lower()))
        aw = set(WORD.findall(r["antagonist_text"].lower()))
        corpus_en_vocab |= pw | aw
        if k in want:
            dfp[k].update(w for w in pw if w not in STOP)
            dfa[k].update(w for w in aw if w not in STOP)
            concept_en_vocab[want[k]] |= pw | aw
    markers = {}
    for k, pid in want.items():
        allw = set(dfp[k]) | set(dfa[k])
        markers[pid] = set(sorted(allw, key=lambda w: dfp[k][w] - dfa[k][w])[-KMARK:])
    print(f"languages={sorted(langs)}  corpus_scripts={sorted(corpus_scripts)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    attn = "eager" if "gemma" in MODEL else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation=attn, device_map="auto"
    )
    model.eval()
    device = model.device
    layer = round(model.config.num_hidden_layers / 2)
    layers, print_layer = [layer], layer
    unemb = model.get_output_embeddings().weight.float()
    gamma = model.model.norm.weight.float().to(device)

    with safe_open(VEC_ROOT / VEC_TAG[MODEL] / "diff.safetensors", framework="pt") as h:
        diff = h.get_tensor("diff")
    assert diff.shape[0] == model.config.num_hidden_layers + 1, diff.shape

    results = []
    for c in conc:
        pid = c["pair_id"]
        pm = markers[pid]
        en_vocab = concept_en_vocab[pid]
        rec: dict[str, Any] = {
            **{k: c[k] for k in ("pair_id", "concept", "antagonist", "steer", "lexauroc")},
            "layers": {},
        }
        for L in layers:
            d = diff[L, pid].float().to(device)
            d = d / d.norm().clamp_min(1e-8)
            proj = unemb @ (gamma * d)
            ids = torch.topk(proj, TOPN).indices.tolist()
            latin: list[str] = []
            other: dict[str, list[str]] = collections.defaultdict(list)
            frag = 0
            scr_counts: collections.Counter = collections.Counter()
            seen_latin: set[str] = set()
            for i in ids:
                s = str(tok.decode([i]))
                if "�" in s:
                    frag += 1
                    scr_counts["fragment"] += 1
                    continue
                w = s.strip()
                if not w:
                    continue
                counts: collections.Counter = collections.Counter()
                for ch in w:
                    cs = char_script(ch)
                    if cs:
                        counts[cs] += 1
                sc = counts.most_common(1)[0][0] if counts else "symbol"
                scr_counts[sc] += 1
                if sc == "Latin":
                    lw = w.lower()
                    if lw not in seen_latin and len(lw) > 1:
                        seen_latin.add(lw)
                        if len(latin) < KEEP_LATIN:
                            latin.append(lw)
                elif sc not in ("symbol",) and len(other[sc]) < KEEP_OTHER and w not in other[sc]:
                    other[sc].append(w)
            pr: dict[str, Any] = {
                "latin": latin,
                "other": dict(other),
                "frag": frag,
                "frag_frac": round(frag / TOPN, 3),
                "script_counts": dict(scr_counts),
            }
            lat = pr["latin"]
            in_concept = [w for w in lat if w in en_vocab]
            in_corpus = [w for w in lat if w not in en_vocab and w in corpus_en_vocab]
            true_ood = [w for w in lat if w not in corpus_en_vocab]
            ood_scripts = [s for s in pr["other"] if s not in corpus_scripts]
            rec["layers"][str(L)] = {
                **pr,
                "overlap_markers": sum(w in pm for w in lat),
                "tier_in_concept": in_concept,
                "tier_in_corpus": in_corpus,
                "tier_true_ood": true_ood,
                "tier_frac": {
                    "in_concept": round(len(in_concept) / max(len(lat), 1), 2),
                    "in_corpus": round(len(in_corpus) / max(len(lat), 1), 2),
                    "true_ood": round(len(true_ood) / max(len(lat), 1), 2),
                },
                "ood_scripts": ood_scripts,
            }
        results.append(rec)
        p = rec["layers"][str(print_layer)]
        other_str = " ".join(f"[{s}:{' '.join(ws[:4])}]" for s, ws in p["other"].items())
        print(
            f"steer={c['steer']:.2f} ood={p['tier_frac']['true_ood']:.2f} frag={p['frag_frac']:.2f}"
            f"  {c['concept'][:30]:30s} | {' '.join(p['latin'][:10])}  {other_str}",
            flush=True,
        )

    tag = MODEL.split("/")[-1]
    out = Path(f"work/logitlens/logitlens-{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": MODEL,
                "layers": layers,
                "print_layer": print_layer,
                "languages": sorted(langs),
                "corpus_scripts": sorted(corpus_scripts),
                "results": results,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    oods = sorted(r["layers"][str(print_layer)]["tier_frac"]["true_ood"] for r in results)
    frs = sorted(r["layers"][str(print_layer)]["frag_frac"] for r in results)
    print(
        f"\nwrote {out}  (L{print_layer}: median true-OOD frac {oods[len(oods) // 2]:.2f}, "
        f"median fragment frac {frs[len(frs) // 2]:.2f})"
    )


if __name__ == "__main__":
    main()
